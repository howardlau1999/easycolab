"""Exercise PE's llama.cpp router progress protocol without Colab or model weights."""
from __future__ import annotations

import contextlib
import io
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

if __package__:
    from .test_image_prompt_enhancement import load_app_code
else:
    from test_image_prompt_enhancement import load_app_code


def router_event(value, current="text_model", stages=("text_model", "mmproj_model")):
    payload = {"current": current, "stages": list(stages), "value": value}
    return "cmd_child_to_router:state:" + json.dumps({"state": "loading", "payload": payload})


class ScriptedProcess:
    """The health responder can wait until the stdout reader has consumed a line."""

    def __init__(self, **options):
        self.options = options
        self.stdin = io.StringIO() if options["stdin"] == subprocess.PIPE else None
        self.returncode = None
        self.lines = queue.Queue()
        self.stdout = self.read_lines()

    def read_lines(self):
        while True:
            item = self.lines.get()
            if item is None:
                return
            line, consumed = item
            yield line
            consumed.set()

    def emit(self, line):
        consumed = threading.Event()
        self.lines.put((line + "\n", consumed))
        if not consumed.wait(2):
            raise AssertionError("stdout reader did not consume the scripted event")

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15
        self.lines.put(None)

    def wait(self, **_kwargs):
        return self.returncode


class PELoadProgressTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_code = load_app_code("_gguf_load_fraction", "_start_qwen_gguf_server", "_stop_gguf_server_bundle")

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.model = Path(temp.name, "model.gguf")
        self.projector = Path(temp.name, "projector.gguf")
        self.model.write_bytes(b"x" * 80)
        self.projector.write_bytes(b"x" * 20)
        self.children = []
        self.opener = Mock()
        self.progress = []
        self.healthy = False
        self.cancel = threading.Event()
        self.env = {
            "os": os, "json": json, "threading": threading, "urllib": urllib,
            "contextlib": contextlib, "time": SimpleNamespace(time=iter(range(2000)).__next__, sleep=lambda _: None),
            "subprocess": SimpleNamespace(Popen=self.spawn, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT,
                                          TimeoutExpired=subprocess.TimeoutExpired),
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
            "_ensure_llama_server_binary": Mock(return_value=("llama-server", {"PATH": os.defpath})),
            "_make_noproxy_opener": lambda: self.opener, "_pick_free_local_port": lambda: 18321,
            "MODEL_DOWNLOAD_STATE": {}, "GenerationAbortedError": type("GenerationAbortedError", (Exception,), {}),
        }
        exec(self.app_code, self.env)

    def spawn(self, _cmd, **kwargs):
        proc = ScriptedProcess(**kwargs)
        self.children.append(proc)
        self.addCleanup(proc.lines.put, None)
        return proc

    def start(self, is_pe=True):
        def record(value):
            self.assertIs(threading.current_thread(), threading.main_thread())
            if value == 1:
                self.assertTrue(self.healthy)
            self.progress.append(value)

        return self.env["_start_qwen_gguf_server"](
            str(self.model), str(self.projector), 2048, 24, for_prompt_expansion=is_pe,
            cancel_event=self.cancel, on_load_progress=record,
        )

    def response(self, status):
        self.healthy = status == 200
        return contextlib.nullcontext(SimpleNamespace(status=status))

    def test_native_event_formats(self):
        parse = self.env["_gguf_load_fraction"]
        sizes = {"text_model": 80, "mmproj_model": 20}
        self.assertAlmostEqual(parse(router_event(0.5), sizes), 0.4)
        self.assertAlmostEqual(parse(router_event(0.5, "mmproj_model"), sizes), 0.9)
        self.assertAlmostEqual(parse(router_event(0.25, stages=["text_model"]), {"text_model": 80}), 0.25)
        for line in [
            "loading tensors 84%", "cmd_child_to_router:state:not JSON",
            'cmd_child_to_router:state:{"state":"ready","payload":{}}',
            'cmd_child_to_router:state:{"state":"loading","payload":{"stage":"mmproj_model"}}',
            router_event(0.5, "other"), router_event(0.5, stages=["text_model"]),
            router_event(-0.1), router_event(1.1), router_event(True), router_event(float("nan")),
        ]:
            with self.subTest(line=line):
                self.assertIsNone(parse(line, sizes))

    def test_partial_load_then_retry_drops_late_output_and_gates_completion_on_health(self):
        events = iter(["first", "fail", "retry", "final", "wait", "ready"])

        def health(*_args, **_kwargs):
            step = next(events)
            if step == "first":
                self.children[0].emit(router_event(0.8))
            elif step == "fail":
                self.children[0].returncode = 1
            elif step == "retry":
                self.assertTrue(self.children[0].stdin.closed)
                self.assertFalse(self.children[1].stdin.closed)
                self.children[1].emit(router_event(0.5))
                self.children[0].emit(router_event(0.99, "mmproj_model"))
            elif step == "final":
                self.children[1].emit(router_event(0.6, "mmproj_model"))
                self.children[1].emit(router_event(1, "mmproj_model"))
            return self.response(200 if step == "ready" else 503)

        self.opener.open.side_effect = health
        bundle = self.start()
        self.addCleanup(self.env["_stop_gguf_server_bundle"], bundle)
        self.assertEqual(len(self.children), 2)
        for proc in self.children:
            self.assertEqual(proc.options["env"]["LLAMA_SERVER_ROUTER_PORT"], "18321")
        self.assertEqual(self.progress, [0.64, 0.0, 0.4, 0.92, 1.0])
        self.children[1].emit(router_event(0.97, "mmproj_model"))
        self.assertEqual(self.progress[-1], 1.0)

    def test_cancel_during_health_check_reaps_child_before_any_ready_callback(self):
        def health(*_args, **_kwargs):
            self.cancel.set()
            return self.response(200)

        self.opener.open.side_effect = health
        with self.assertRaises(self.env["GenerationAbortedError"]):
            self.start()
        self.assertEqual(self.progress, [])
        self.assertTrue(self.children[0].stdin.closed)
        self.assertIsNotNone(self.children[0].returncode)

    def test_regular_qwen_keeps_standalone_process_and_ignores_pe_callback(self):
        self.opener.open.return_value = self.response(200)
        bundle = self.start(is_pe=False)
        self.addCleanup(self.env["_stop_gguf_server_bundle"], bundle)
        self.assertNotIn("LLAMA_SERVER_ROUTER_PORT", self.children[0].options["env"])
        self.assertIsNone(self.children[0].options["stdin"])
        self.assertEqual(self.progress, [])

    def test_legacy_server_stays_indeterminate_and_exits_when_parent_pipe_closes(self):
        def spawn(_cmd, **kwargs):
            # Mirrors llama.cpp's stdin EOF monitor without any load events.
            code = "import os, sys; assert os.environ['LLAMA_SERVER_ROUTER_PORT'] == '18321'; sys.stdin.read()"
            return subprocess.Popen([sys.executable, "-c", code], **kwargs)

        self.env["subprocess"].Popen = spawn
        statuses = iter([503, 503, 200])
        self.opener.open.side_effect = lambda *_args, **_kwargs: self.response(next(statuses))
        bundle = self.start()
        proc = bundle["server_proc"]
        self.addCleanup(proc.stdout.close)
        self.addCleanup(self.env["_stop_gguf_server_bundle"], bundle)
        self.assertIsNone(proc.poll())
        self.assertEqual(self.progress, [1.0])
        proc.stdin.close()
        self.assertEqual(proc.wait(timeout=5), 0)


if __name__ == "__main__":
    unittest.main()
