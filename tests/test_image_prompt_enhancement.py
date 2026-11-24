"""Exercise the image workflow and HTTP handler without loading Colab or CUDA."""
from __future__ import annotations

import ast
import asyncio
import base64
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import weakref
from concurrent.futures import Future
from contextlib import ExitStack, nullcontext
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch


APP = Path(__file__).resolve().parents[1] / "app.py"
ORIGINAL = "画一只在窗台上晒太阳的猫"
EXPANDED = "An orange cat stretches in the afternoon light on a windowsill."
SIZE = (1344, 768)


def load_app_code(*names):
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
    ]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            node.decorator_list.clear()
            nodes.append(node)
    return compile(ast.Module(body=nodes, type_ignores=[]), str(APP), "exec")


class ImagePromptEnhancementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_code = load_app_code(
            "GenerateVideoRequest", "_run_async_generation_job", "_execute_generation_job", "openai_images_generations", "_generate_gateway_images",
            "_edit_reference_sources", "_decode_edit_references", "EDIT_REFERENCE_LIMITS", "api_generate",
            "api_cancel_generation",
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.lock = threading.Lock()
        def expand(_prompt, *, on_expanded=None, **kwargs):
            if on_expanded:
                on_expanded(EXPANDED)
            return EXPANDED

        self.expand = Mock(side_effect=expand)
        image = Mock()
        image.save.side_effect = self.save_image
        self.generate = Mock(return_value=image)
        self.job = {}
        self.engine = "qwen_image_21"
        self.env = {
            "BaseModel": SimpleNamespace, "JSONResponse": SimpleNamespace,
            "threading": threading, "os": os, "time": time, "datetime": datetime, "contextlib": contextlib,
            "base64": base64, "io": io, "torch": Mock(),
            "touch_activity": Mock(), "_verify_external_api_auth": Mock(return_value=None),
            "_JOB_CANCEL_EVENTS": {}, "ABORT_GENERATION_EVENT": threading.Event(),
            "GenerationAbortedError": type("GenerationAbortedError", (RuntimeError,), {}),
            "CONSOLE_LOCK": threading.Lock(), "GPU_LOCK": self.lock, "ACTIVE_TQDM_BARS": {},
            "MODEL_CATALOG": {self.engine: {"modality": "image", "name": "Qwen"}, "minimax_h3": {}},
            "LOADED_RUNTIMES": {self.engine: {"pipeline": self.generate}}, "JOBS": {"job": self.job},
            "_is_llm_model_id": lambda model_id: False,
            "decode_base64_image": lambda data: data or None, "Image": SimpleNamespace(DecompressionBombError=RuntimeError),
            "_refresh_drive_mount_state": Mock(), "DEFAULT_OUTPUT_DIR": self.temp.name,
            "_reclaim_transient_gpu_memory": Mock(), "_pipeline_job_execution_guard": lambda **kw: nullcontext(),
            "_sample_live_hardware_telemetry_once": Mock(),
            "_persist_generated_output_to_drive": lambda path: path,
            "expand_image_prompt": self.expand, "API_REQUEST_COUNT": 0,
        }
        self.env["torch"].randint.return_value.item.return_value = 42
        exec(self.app_code, self.env)

    @staticmethod
    def save_image(target, format=None):
        data = b"mock image" * 32
        if isinstance(target, io.BytesIO):
            target.write(data)
        else:
            Path(target).write_bytes(data)

    def run_job(self, **kwargs):
        request = self.env["GenerateVideoRequest"](
            model_id=self.engine, visual_prompt=ORIGINAL, randomize_seed=False,
            width=SIZE[0], height=SIZE[1], **kwargs,
        )
        self.env["_run_async_generation_job"]("job", request)

    def request_images(self, **kwargs):
        body = {"prompt": ORIGINAL, "size": f"{SIZE[0]}x{SIZE[1]}", **kwargs}
        request = SimpleNamespace(json=AsyncMock(return_value=body))
        return asyncio.run(self.env["openai_images_generations"](request))

    def test_job_preserves_original_prompt_when_flag_omitted(self):
        self.run_job(task_mode="t2i")
        self.assertEqual(self.job["status"], "completed", self.job["message"])
        self.expand.assert_not_called()
        self.assertEqual(self.generate.call_args.kwargs["prompt"], ORIGINAL)
        self.assertNotIn("expanded_prompt", self.job)

    def test_batch_edit_expands_once_with_the_generators_ordered_references(self):
        refs = [f"reference {i}" for i in range(10)]
        self.run_job(
            task_mode="i2i_edit", enhance_prompt=True, batch_count=2, seed=73,
            reference_images_base64=refs, first_frame_base64="stale slot", negative_prompt="blur",
        )
        self.assertEqual(self.job["status"], "completed", self.job["message"])
        self.expand.assert_called_once_with(
            ORIGINAL, images=refs, width=SIZE[0], height=SIZE[1], seed=73, cancel_event=ANY, on_stage=ANY,
            on_expanded=ANY, on_progress=ANY,
        )
        self.assertEqual(self.job["expanded_prompt"], EXPANDED)
        self.assertEqual(self.job["completed_count"], 2)
        self.assertEqual([call.kwargs["seed"] for call in self.generate.call_args_list], [73, 74])
        for call in self.generate.call_args_list:
            args = call.kwargs
            self.assertEqual(args["prompt"], EXPANDED)
            self.assertEqual(args["edit_image"], refs)
            self.assertEqual(args["negative_prompt"], "blur")
            self.assertEqual((args["width"], args["height"]), SIZE)

    def test_edit_preserves_legacy_frame_order_without_pe(self):
        self.run_job(task_mode="i2i_edit", first_frame_base64="first", last_frame_base64="last")
        self.assertEqual(self.generate.call_args.kwargs["edit_image"], ["first", "last"])
        self.expand.assert_not_called()

    def test_cancel_retains_prompt_and_reclaims_after_inference_frame_is_gone(self):
        class Transient:
            pass

        refs = []
        self.env.update(SELECTED_MODEL_ID=self.engine, get_gpu_status_dict=Mock(return_value={}))

        def sample(**kwargs):
            activation = Transient()
            refs.append(weakref.ref(activation))
            self.assertEqual(self.job["expanded_prompt"], EXPANDED)
            self.env["api_cancel_generation"]()
            self.assertEqual(self.job["status"], "running")
            raise self.env["GenerationAbortedError"]("cancelled")

        def reclaim(*, trim_cpu=False):
            if trim_cpu:
                self.assertIsNone(refs[0]())
                self.assertEqual(self.job["status"], "running")

        self.generate.side_effect = sample
        self.env["_reclaim_transient_gpu_memory"].side_effect = reclaim
        self.run_job(task_mode="t2i", enhance_prompt=True)
        self.assertEqual(self.job["status"], "failed")
        self.assertEqual(self.job["expanded_prompt"], EXPANDED)
        self.env["_reclaim_transient_gpu_memory"].assert_called_with(trim_cpu=True)
        self.env["_sample_live_hardware_telemetry_once"].assert_called_once()
        self.assertFalse(self.env["_JOB_CANCEL_EVENTS"])

    def test_api_rejects_overflow_before_superseding_an_active_job(self):
        class HTTPError(Exception):
            def __init__(self, status_code, detail):
                super().__init__(detail)
                self.status_code = status_code

        self.env.update(HTTPException=HTTPError, _normalize_builtin_model_alias=lambda model_id: model_id)
        self.job["status"] = "running"
        request = self.env["GenerateVideoRequest"](model_id=self.engine, task_mode="i2i_edit", reference_images_base64=["ref"] * 11)
        with self.assertRaisesRegex(HTTPError, "10") as raised:
            self.env["api_generate"](request)
        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(self.job["status"], "running")

    def test_superseded_task_stays_active_until_its_worker_releases_resources(self):
        self.job.update(status="running", progress=0.4)
        old_event = threading.Event()
        self.env["_JOB_CANCEL_EVENTS"]["job"] = old_event
        self.env.update(
            _normalize_builtin_model_alias=lambda value: value,
            uuid=SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="nextjob")),
            threading=SimpleNamespace(Thread=Mock(), Event=threading.Event),
        )
        req = self.env["GenerateVideoRequest"](model_id=self.engine, visual_prompt=ORIGINAL)
        new_job = self.env["api_generate"](req)
        self.assertTrue(old_event.is_set())
        self.assertEqual(self.job["status"], "running")
        self.assertIn("正在停止", self.job["message"])
        self.assertEqual(new_job["status"], "queued")

    def test_minimax_edit_passes_sixteen_references_to_pe_and_generator(self):
        self.env["MODEL_CATALOG"][self.engine]["pipeline_arch"] = "minimax_h3_image"
        self.env["run_minimax_h3_image_inference"] = self.generate
        refs = [f"photo {i}" for i in range(16)]
        self.run_job(task_mode="i2i_edit", reference_images_base64=refs, enhance_prompt=True)
        self.assertEqual(self.job["status"], "completed", self.job["message"])
        self.assertEqual(self.expand.call_args.kwargs["images"], refs)
        self.assertIs(self.generate.call_args.kwargs["ref_images"], self.expand.call_args.kwargs["images"])

    def test_invalid_reference_lists_fail_before_expansion(self):
        for model, refs, message in [
            ("minimax_h3_image", ["ref"] * 17, "16"),
            ("qwen_image_21", ["ref"] * 11, "10"),
            ("z_image_turbo", ["ref"] * 2, "1"),
            ("qwen_image_21", [], "至少一张"),
            ("qwen_image_21", ["first", "second", ""], "第 3 张"),
        ]:
            with self.subTest(model=model, count=len(refs)):
                self.env["MODEL_CATALOG"][self.engine]["pipeline_arch"] = model
                self.run_job(task_mode="i2i_edit", reference_images_base64=refs, first_frame_base64="old", enhance_prompt=True)
                self.assertEqual(self.job["status"], "failed")
                self.assertIn(message, self.job["message"])
                self.expand.assert_not_called()
                self.generate.assert_not_called()

    def test_job_failure_before_sampling_releases_gpu(self):
        def fail(_prompt, *, on_progress, **kwargs):
            on_progress("loading", 0.54)
            self.assertEqual((self.job["pe_stage"], self.job["pe_progress"]), ("loading", 0.54))
            raise RuntimeError("expansion failed")

        self.expand.side_effect = fail
        self.run_job(task_mode="t2i", enhance_prompt=True)
        self.assertEqual(self.job["status"], "failed")
        self.assertIn("expansion failed", self.job["message"])
        self.generate.assert_not_called()
        self.assertFalse(self.lock.locked())
        self.assertFalse(self.env["_JOB_CANCEL_EVENTS"])
        self.assertNotIn("pe_progress", self.job)
        self.assertNotIn("pe_stage", self.job)

    def test_gateway_batch_returns_the_prompt_used_for_each_image(self):
        response = self.request_images(enhance_prompt=True, n=2)
        self.expand.assert_called_once_with(ORIGINAL, images=[], width=SIZE[0], height=SIZE[1], seed=42)
        self.assertEqual(self.generate.call_count, 2)
        for call in self.generate.call_args_list:
            self.assertEqual(call.kwargs["prompt"], EXPANDED)
            self.assertEqual((call.kwargs["width"], call.kwargs["height"]), SIZE)
        self.assertEqual([item["revised_prompt"] for item in response["data"]], [EXPANDED] * 2)
        self.assertTrue(all(item["b64_json"] for item in response["data"]))

    def test_gateway_omitted_flag_keeps_original_request_and_response(self):
        result = self.request_images()["data"][0]
        self.expand.assert_not_called()
        self.assertEqual(self.generate.call_args.kwargs["prompt"], ORIGINAL)
        self.assertEqual(set(result), {"b64_json"})

    def test_gateway_reports_expansion_failure_and_releases_gpu(self):
        self.expand.side_effect = RuntimeError("expansion failed")
        error = self.request_images(enhance_prompt=True)
        self.assertEqual(error.status_code, 503)
        self.assertIn("expansion failed", error.content["error"]["message"])
        self.generate.assert_not_called()
        self.assertFalse(self.lock.locked())


class PromptExpansionBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_code = load_app_code(
            "_build_image_pe_request", "_image_pe_prompt_directory", "IMAGE_PE_TEMPLATE_BLOBS",
            "IMAGE_PE_TEMPLATE_REVISION", "_request_image_pe_caption", "expand_image_prompt", "_prepare_image_pe_inference_paths",
        )

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.hub = Mock()
        self.hub.snapshot_download.return_value = root
        client = MagicMock()
        self.post = AsyncMock(return_value=Mock())
        client.__aenter__.return_value.post = self.post
        httpx = SimpleNamespace(AsyncClient=Mock(return_value=client))
        self.stack.enter_context(patch.dict(sys.modules, huggingface_hub=self.hub, httpx=httpx))
        templates = Path(root) / "prompt"
        (templates / "t2i").mkdir(parents=True)
        (templates / "edit").mkdir()
        (templates / "t2i/system_prompt.txt").write_text("System")
        (templates / "t2i/task_prompt.txt").write_text("Expand {}")
        (templates / "edit/task_prompt.txt").write_text("Edit with references")
        (templates / "t2i/request_params.json").write_text('{"response_format":{"json_schema":{}}}')
        self.env = {
            "json": json, "os": os, "threading": threading, "MODEL_BASE_ROOT": root,
            "DRIVE_MOUNTED": False,
            "MINIMAX_IMAGE_PE_REPO": "MiniMaxAI/MiniMax-H3-Image-PE", "_pil_to_jpeg_data_url": lambda im: im.url,
            "torch": Mock(), "_start_qwen_gguf_server": Mock(return_value={"base_url": "http://localhost:30000"}),
            "_queue_image_pe_weights": Mock(return_value=(["test.gguf"], [])),
            "_stop_gguf_server_bundle": Mock(), "_reclaim_transient_gpu_memory": Mock(),
            "GenerationAbortedError": type("GenerationAbortedError", (RuntimeError,), {}),
        }
        self.env["torch"].cuda.mem_get_info.return_value = (20 * 1024**3, 24 * 1024**3)
        exec(self.app_code, self.env)

    def test_pinned_prompt_templates_are_reused_from_drive(self):
        self.env.update(DRIVE_MOUNTED=True, LEGACY_DRIVE_MODEL_ROOTS=[self.env["MODEL_BASE_ROOT"]])
        source = Path(self.env["MODEL_BASE_ROOT"]) / "prompt"
        cached = Path(self.env["MODEL_BASE_ROOT"]) / self.env["MINIMAX_IMAGE_PE_REPO"] / "prompt"
        cached.parent.mkdir(parents=True)
        source.rename(cached)
        for name in self.env["IMAGE_PE_TEMPLATE_BLOBS"]:
            data = (cached / name).read_bytes()
            self.env["IMAGE_PE_TEMPLATE_BLOBS"][name] = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        self.assertEqual(self.env["_image_pe_prompt_directory"](), cached)
        self.hub.snapshot_download.assert_not_called()
        (cached / "edit/task_prompt.txt").write_text("stale revision")
        self.env["_image_pe_prompt_directory"]()
        self.hub.snapshot_download.assert_called_once()

    def test_remote_edit_uses_ordered_images_and_only_returns_caption(self):
        self.stack.enter_context(patch.dict(os.environ, EASYCOLAB_IMAGE_PE_BASE_URL="http://remote.test/v1"))
        message = {"content": json.dumps({"expanded_caption": EXPANDED, "target_aspect_ratio": "7:4"}), "reasoning_content": "private"}
        self.post.return_value.json.return_value = {"choices": [{"finish_reason": "stop", "message": message}]}
        refs = [SimpleNamespace(size=(60, 40), url="first"), SimpleNamespace(size=(40, 60), url="second"),
                SimpleNamespace(size=(40, 40), url="third")]
        result = self.env["expand_image_prompt"](ORIGINAL, images=refs, width=SIZE[0], height=SIZE[1], seed=17)
        self.assertEqual(result, EXPANDED)
        self.hub.hf_hub_download.assert_not_called()
        self.env["_queue_image_pe_weights"].assert_not_called()
        self.env["_start_qwen_gguf_server"].assert_not_called()
        self.assertEqual(self.post.await_args.args, ("http://remote.test/v1/chat/completions",))
        payload = self.post.await_args.kwargs["json"]
        parts = payload["messages"][0]["content"]
        self.assertEqual(parts[0]["text"], "Edit with references")
        self.assertIn("<Picture 1> [aspect_ratio 3:2]", parts[2]["text"])
        self.assertIn("<Picture 2> [aspect_ratio 2:3]", parts[4]["text"])
        self.assertIn("<Picture 3> [aspect_ratio 1:1]", parts[6]["text"])
        self.assertEqual([part["image_url"]["url"] for part in parts if part["type"] == "image_url"], ["first", "second", "third"])
        self.assertIn("User Aspect Ratio Selection:\n7:4", parts[-1]["text"])
        schema = payload["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["target_aspect_ratio"]["enum"], ["7:4"])

    def test_local_edit_budgets_vision_context_for_all_sixteen_images(self):
        self.stack.enter_context(patch.dict(os.environ, EASYCOLAB_IMAGE_PE_BASE_URL=""))
        self.env["_queue_image_pe_weights"].return_value = (["test.gguf", "projector.gguf"], [])
        content = json.dumps({"expanded_caption": EXPANDED})
        self.post.return_value.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        refs = [SimpleNamespace(size=(1000, 1000), url=f"image-{i}") for i in range(16)]
        received = []
        progress = []
        self.env["_stop_gguf_server_bundle"].side_effect = lambda _: self.assertEqual(received, [EXPANDED])
        def load(*args, on_load_progress, **kwargs):
            on_load_progress(0.64)
            on_load_progress(1.0)
            return {"base_url": "http://localhost:30000"}

        self.env["_start_qwen_gguf_server"].side_effect = load
        result = self.env["expand_image_prompt"](
            ORIGINAL, images=refs, width=SIZE[0], height=SIZE[1], seed=17, on_expanded=received.append,
            on_progress=lambda stage, value: progress.append((stage, value)),
        )
        self.assertEqual(result, EXPANDED)
        self.assertEqual(progress, [("loading", None), ("loading", 0.64), ("loading", 1.0), ("expanding", None)])
        start = self.env["_start_qwen_gguf_server"].call_args
        self.assertEqual(start.args[1], "projector.gguf")
        self.assertEqual(start.kwargs["image_max_tokens"], 1024)
        parts = self.post.await_args.kwargs["json"]["messages"][0]["content"]
        self.assertEqual([part["image_url"]["url"] for part in parts if part["type"] == "image_url"], [ref.url for ref in refs])

    def test_local_server_is_released_after_invalid_or_truncated_text(self):
        self.stack.enter_context(patch.dict(os.environ, EASYCOLAB_IMAGE_PE_BASE_URL=""))
        for reason, content in (("length", "{}"), ("stop", "invalid JSON"), ("stop", '{"expanded_caption":"  "}')):
            with self.subTest(reason=reason, content=content):
                self.post.return_value.json.return_value = {"choices": [{"finish_reason": reason, "message": {"content": content}}]}
                self.env["_stop_gguf_server_bundle"].reset_mock()
                stages = []
                with self.assertRaises(ValueError):
                    self.env["expand_image_prompt"](
                        ORIGINAL, images=[], width=SIZE[0], height=SIZE[1], seed=17, on_stage=stages.append,
                    )
                self.assertIn("加载", stages[0])
                self.assertIn("增强", stages[-1])
                self.env["_stop_gguf_server_bundle"].assert_called_once_with(self.env["_start_qwen_gguf_server"].return_value)
                self.assertIsNone(self.env["_start_qwen_gguf_server"].call_args.args[1])
                self.assertEqual(self.post.await_args.kwargs["json"]["messages"], [
                    {"role": "system", "content": "System"}, {"role": "user", "content": f"Expand {ORIGINAL}"},
                ])

    def test_stopping_pending_shared_download_does_not_start_local_server(self):
        self.stack.enter_context(patch.dict(os.environ, EASYCOLAB_IMAGE_PE_BASE_URL=""))
        transfer = Future()
        self.env["_queue_image_pe_weights"].return_value = (["test.gguf"], [transfer])
        cancelled = threading.Event()
        with self.assertRaises(self.env["GenerationAbortedError"]):
            self.env["expand_image_prompt"](
                ORIGINAL, images=[], width=640, height=640, seed=42,
                cancel_event=cancelled, on_stage=lambda _: cancelled.set(),
            )
        self.assertFalse(transfer.done())
        self.env["_start_qwen_gguf_server"].assert_not_called()

    def test_stopping_caption_request_tears_down_both_http_and_local_server(self):
        self.stack.enter_context(patch.dict(os.environ, EASYCOLAB_IMAGE_PE_BASE_URL=""))
        cancelled = threading.Event()
        request_closed = threading.Event()

        async def pending(*args, **kwargs):
            cancelled.set()
            try:
                await asyncio.Event().wait()
            finally:
                request_closed.set()

        self.post.side_effect = pending
        callback = Mock()
        with self.assertRaises(self.env["GenerationAbortedError"]):
            self.env["expand_image_prompt"](
                ORIGINAL, images=[], width=640, height=640, seed=42,
                cancel_event=cancelled, on_expanded=callback,
            )
        self.assertTrue(request_closed.is_set())
        callback.assert_not_called()
        self.env["_stop_gguf_server_bundle"].assert_called_once_with(self.env["_start_qwen_gguf_server"].return_value)


class ImageJobResourceLifecycleTest(unittest.TestCase):
    def test_stopping_while_another_job_installs_the_engine_releases_only_the_waiter(self):
        lock = threading.Lock()
        cancelled = threading.Event()
        cancelled.set()
        env = {
            "_LLAMA_SERVER_INSTALL_LOCK": lock, "_install_or_reuse_llama_server": Mock(),
            "GenerationAbortedError": type("GenerationAbortedError", (RuntimeError,), {}),
        }
        exec(load_app_code("_ensure_llama_server_binary"), env)
        with lock:
            with self.assertRaises(env["GenerationAbortedError"]):
                env["_ensure_llama_server_binary"](cancel_event=cancelled)
            self.assertTrue(lock.locked())
        env["_install_or_reuse_llama_server"].assert_not_called()

    def test_modular_pipeline_stops_at_block_and_restores_its_managed_cuda_component(self):
        class Module:
            def register_forward_pre_hook(self, hook, **kwargs):
                self.abort_hook = hook
                return self.handle

        transformer = Module()
        block = Module()
        transformer.handle = Mock()
        block.handle = Mock()
        transformer.named_modules = lambda: [("layers", [block])]
        transformer.device = SimpleNamespace(type="cuda")
        managed = Mock(model=transformer)
        unrelated = Mock()
        pipe = SimpleNamespace(
            components={"transformer": transformer},
            _components_manager=SimpleNamespace(model_hooks=[managed, unrelated]),
        )
        cancelled = threading.Event()
        env = {
            "torch": SimpleNamespace(nn=SimpleNamespace(Module=Module, ModuleList=list)),
            "threading": threading, "ABORT_GENERATION_EVENT": threading.Event(),
            "GenerationAbortedError": type("GenerationAbortedError", (RuntimeError,), {}),
        }
        exec(load_app_code("_pipeline_job_execution_guard"), env)
        guard = contextlib.contextmanager(env["_pipeline_job_execution_guard"])
        with self.assertRaises(env["GenerationAbortedError"]):
            with guard(pipe, "test", cancelled, time.time(), 0, 1, "", "MiniMax", "image"):
                cancelled.set()
                block.abort_hook(block, ())
        managed.offload.assert_called_once_with()
        unrelated.offload.assert_not_called()
        block.handle.remove.assert_called_once_with()
        transformer.handle.remove.assert_called_once_with()

    def test_startup_exception_reaps_the_temporary_server(self):
        # Fail after Popen but before the first successful health check. A real child verifies wait/reap.
        procs = []

        def spawn(_cmd, **kwargs):
            proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
            procs.append(proc)
            return proc

        thread = Mock()
        thread.start.side_effect = RuntimeError("cannot start output reader")
        env = {
            "os": os, "time": time, "threading": SimpleNamespace(Thread=Mock(return_value=thread)),
            "contextlib": contextlib,
            "subprocess": SimpleNamespace(Popen=spawn, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT,
                                          TimeoutExpired=subprocess.TimeoutExpired),
            "_ensure_llama_server_binary": Mock(return_value=(sys.executable, os.environ.copy())),
            "_make_noproxy_opener": Mock(), "_pick_free_local_port": Mock(return_value=18080),
        }
        exec(load_app_code("_start_qwen_gguf_server", "_stop_gguf_server_bundle"), env)
        try:
            with self.assertRaisesRegex(RuntimeError, "output reader"):
                env["_start_qwen_gguf_server"]("test.gguf", None, 32768, 64, for_prompt_expansion=True)
            self.assertEqual(len(procs), 1)
            self.assertIsNotNone(procs[0].returncode)
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                proc.stdout.close()


class ImagePEWeightDownloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_code = load_app_code(
            "_queue_image_pe_weights", "_download_image_pe_file", "_get_image_pe_status",
            "_image_pe_weight_path", "_image_pe_inference_path", "_image_pe_file_cached", "_image_pe_progress_bar", "_image_pe_download_fraction",
            "_prepare_image_pe_inference_paths", "_stage_image_pe_file", "_verified_image_pe_stage",
            "DownloadImagePERequest", "api_download_image_pe",
        )

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.hub = Mock()
        stack.enter_context(patch.dict(sys.modules, huggingface_hub=self.hub))
        stack.enter_context(patch.dict(os.environ, EASYCOLAB_IMAGE_PE_BASE_URL=""))
        self.env = {
            "os": os, "threading": threading, "Future": Future, "BaseModel": SimpleNamespace,
            "IMAGE_PE_WEIGHT": "weights", "IMAGE_PE_PROJECTOR": "projector", "IMAGE_PE_FILE_SIZES": {"weights": 7, "projector": 7},
            "MODEL_BASE_ROOT": str(self.root), "IMAGE_PE_GGUF_REPO": "pe", "IMAGE_PE_GGUF_REVISION": "revision",
            "_IMAGE_PE_DOWNLOAD_LOCK": threading.Lock(), "_IMAGE_PE_DOWNLOADS": {}, "_IMAGE_PE_ERRORS": {},
            "_IMAGE_PE_DOWNLOAD_BYTES": {},
            "DRIVE_MOUNTED": False, "shutil": Mock(), "GenerationAbortedError": type("GenerationAbortedError", (RuntimeError,), {}),
            "_refresh_drive_mount_state": Mock(), "touch_activity": Mock(), "SELECTED_MODEL_ID": "generator",
            "get_gpu_status_dict": lambda **kw: {"image_pe": self.env["_get_image_pe_status"]()},
        }
        exec(self.app_code, self.env)

    def download(self, include_projector):
        req = self.env["DownloadImagePERequest"](include_projector=include_projector)
        return self.env["api_download_image_pe"](req)["image_pe"]

    def test_manual_download_and_generation_share_files_and_report_cache(self):
        release = threading.Event()

        def fetch(filename, local_dir, **kwargs):
            if not release.wait(3):
                raise TimeoutError("test transfer timed out")
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / filename).write_bytes(b"weights")

        self.hub.hf_hub_download.side_effect = fetch
        try:
            self.assertTrue(self.download(False)["downloading"])
            self.assertTrue(self.download(True)["downloading"])
            _, transfers = self.env["_queue_image_pe_weights"](True)
        finally:
            release.set()
        for transfer in transfers:
            transfer.result(timeout=3)
        status = self.download(True)
        self.assertTrue(status["weights_cached"] and status["projector_cached"])
        self.assertFalse(status["downloading"])
        self.assertCountEqual([call.kwargs["filename"] for call in self.hub.hf_hub_download.call_args_list], ["weights", "projector"])

    def test_incomplete_file_can_be_retried_and_remote_mode_skips_downloads(self):
        destination = self.root / "pe" / "weights"
        destination.parent.mkdir()
        destination.write_bytes(b"part")
        self.hub.hf_hub_download.side_effect = IOError("unavailable")
        queue = self.env["_queue_image_pe_weights"]
        _, transfers = queue(False)
        with self.assertRaisesRegex(IOError, "unavailable"):
            transfers[0].result(timeout=3)
        status = self.env["_get_image_pe_status"]()
        self.assertFalse(status["weights_cached"])
        self.assertIn("unavailable", status["error"])
        self.hub.hf_hub_download.side_effect = lambda **kw: destination.write_bytes(b"weights")
        queue(False)[1][0].result(timeout=3)
        self.assertEqual(self.env["_get_image_pe_status"]()["error"], "")
        with patch.dict(os.environ, EASYCOLAB_IMAGE_PE_BASE_URL="http://pe.test/v1"):
            self.assertEqual(self.download(True)["mode"], "remote")
        self.assertEqual(self.hub.hf_hub_download.call_count, 2)

    def test_native_mount_reuses_weights_saved_in_older_rclone_root(self):
        old_root = self.root / "EasyColab_Drive" / "EasyColab_Models"
        self.env.update(DRIVE_MOUNTED=True, LEGACY_DRIVE_MODEL_ROOTS=[str(old_root)])
        for root, name in ((old_root, "weights"), (self.root, "projector")):
            (root / "pe").mkdir(parents=True)
            (root / "pe" / name).write_bytes(b"weights")
        paths, pending = self.env["_queue_image_pe_weights"](True)
        self.assertEqual(paths, [str(old_root / "pe" / "weights"), str(self.root / "pe" / "projector")])
        self.assertEqual(pending, [])
        self.assertTrue(self.env["_get_image_pe_status"]()["weights_cached"])
        self.hub.hf_hub_download.assert_not_called()

    def test_pinned_local_replica_is_used_for_inference_only(self):
        self.env.update(DRIVE_MOUNTED=True, LEGACY_DRIVE_MODEL_ROOTS=[], IMAGE_PE_FILE_SHA256={"weights": "a" * 64})
        drive = self.root / "pe" / "weights"
        drive.parent.mkdir()
        drive.write_bytes(b"weights")
        local = self.root / "models" / "pe" / "weights"
        metadata = local.parent / ".cache" / "huggingface" / "download" / "weights.metadata"
        metadata.parent.mkdir(parents=True)
        read_path = self.env["_image_pe_inference_path"]
        with patch("os.getcwd", return_value=str(self.root)):
            self.assertEqual(read_path(str(drive)), str(drive))
            local.write_bytes(b"weights")
            fresh = str(local.stat().st_mtime + 1)
            for revision, digest, timestamp in (
                ("old", "a"*64, fresh), ("revision", "b"*64, fresh), ("revision", "a"*64, "0"),
                ("revision", "a"*64, "incomplete sidecar"),
            ):
                metadata.write_text(f"{revision}\n{digest}\n{timestamp}\n")
                self.assertEqual(read_path(str(drive)), str(drive))
            metadata.write_text(f"revision\n{'a' * 64}\n{fresh}\n")
            self.assertEqual(read_path(str(drive)), str(local))
            self.assertEqual(self.env["_queue_image_pe_weights"](False), ([str(drive)], []))
            self.assertTrue(self.env["_get_image_pe_status"]()["weights_cached"])
            local.write_bytes(b"short")
            self.assertEqual(read_path(str(drive)), str(drive))
        self.hub.hf_hub_download.assert_not_called()

    def drive_sources(self, data_by_name):
        self.env.update(
            DRIVE_MOUNTED=True, LEGACY_DRIVE_MODEL_ROOTS=[],
            IMAGE_PE_FILE_SHA256={name: hashlib.sha256(data).hexdigest() for name, data in data_by_name.items()},
            IMAGE_PE_FILE_SIZES={name: len(data) for name, data in data_by_name.items()},
        )
        self.env["shutil"].disk_usage.return_value.free = 40 * 1024**3
        sources = []
        for name, data in data_by_name.items():
            source = self.root / "drive" / name
            source.parent.mkdir(exist_ok=True)
            source.write_bytes(data)
            sources.append(str(source))
        cwd = patch("os.getcwd", return_value=str(self.root))
        cwd.start()
        self.addCleanup(cwd.stop)
        return sources

    def test_staging_reports_actual_bytes_and_reuses_only_the_committed_cache(self):
        chunk = 4 * 1024**2
        sources = self.drive_sources({"weights": b"x" * (chunk + 13), "projector": b"vision"})
        total = chunk + 19
        progress = []
        stages = []
        prepare = self.env["_prepare_image_pe_inference_paths"]
        chosen, remaining = prepare(sources, None, stages.append, lambda stage, fraction: progress.append((stage, fraction)))
        self.assertEqual(remaining, 0)
        self.assertEqual(progress, [("staging", n / total) for n in (0, chunk, chunk + 13, total)])
        self.assertEqual(len(stages), 1)
        for local, src in zip(chosen, sources):
            self.assertEqual(Path(local).read_bytes(), Path(src).read_bytes())
            self.assertTrue(self.env["_verified_image_pe_stage"](Path(local)))
        self.env["_stage_image_pe_file"] = Mock(side_effect=AssertionError("Cache was not reused"))
        no_progress = Mock()
        self.assertEqual(prepare(sources, None, None, no_progress), (chosen, 0))
        no_progress.assert_not_called()
        self.env["shutil"].disk_usage.assert_called()

    def test_cancel_or_corrupt_file_leaves_no_partially_staged_cache(self):
        sources = self.drive_sources({"weights": b"x" * (4 * 1024**2 + 1)})
        cancelled = threading.Event()
        def stop_after_bytes(stage, progress):
            if progress > 0:
                cancelled.set()
        prepare = self.env["_prepare_image_pe_inference_paths"]
        with self.assertRaises(self.env["GenerationAbortedError"]):
            prepare(sources, cancelled, None, stop_after_bytes)
        cache = self.root / "models/_image_pe_cache/revision"
        self.assertEqual(list(cache.iterdir()), [])
        with open(sources[0], "r+b") as src:
            src.write(b"!")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            prepare(sources, None, None, None)
        self.assertEqual(list(cache.iterdir()), [])

    def test_low_disk_and_unrecognized_entries_fall_back_to_drive_per_file(self):
        sources = self.drive_sources({"weights": b"weights", "projector": b"lens"})
        self.env["shutil"].disk_usage.return_value.free = 22 * 1024**3 + 4
        prepare = self.env["_prepare_image_pe_inference_paths"]
        chosen, remaining = prepare(sources, None, None, None)
        self.assertEqual(chosen[0], sources[0])
        self.assertNotEqual(chosen[1], sources[1])
        self.assertEqual(remaining, 7)
        foreign = self.root / "models/_image_pe_cache/revision" / self.env["IMAGE_PE_FILE_SHA256"]["weights"] / "weights"
        foreign.parent.mkdir()
        foreign.write_bytes(b"unknown")
        self.env["shutil"].disk_usage.return_value.free = 40 * 1024**3
        self.assertEqual(prepare(sources, None, None, None), (chosen, 7))
        self.assertEqual(foreign.read_bytes(), b"unknown")

    def test_download_bytes_follow_file_reconstruction_and_cached_dependencies(self):
        class BaseBar:
            def __init__(self, initial=0, **kwargs):
                self.n = initial

        path = str(self.root / "pe" / "weights")
        projector = self.root / "pe" / "projector"
        projector.parent.mkdir()
        projector.write_bytes(b"weights")
        get_status = self.env["_get_image_pe_status"]
        with patch.dict(sys.modules, {"tqdm.auto": SimpleNamespace(tqdm=BaseBar)}):
            bar = self.env["_image_pe_progress_bar"](path)
            file_bar = bar(initial=2, total=7)
            self.assertEqual(get_status()["weights_bytes"], 2)
            wire_bar = bar(name="huggingface_hub.xet_get.transfer", total=7)
            wire_bar.update(6)
            file_bar.update(1)
        self.assertEqual(get_status()["weights_bytes"], 3)
        self.assertEqual(get_status()["projector_bytes"], get_status()["projector_total_bytes"])
        self.assertAlmostEqual(self.env["_image_pe_download_fraction"]([path, str(projector)]), 10 / 14)


if __name__ == "__main__":
    unittest.main()
