import ast
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from contextlib import redirect_stdout
from typing import Any, Dict
from unittest import mock

import colab_bridge

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PY_PATH = os.path.join(REPO_ROOT, "app.py")


def _load_app_drive_helpers(tmp_root: str):
    with open(APP_PY_PATH, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=APP_PY_PATH)

    wanted_funcs = {
        "_is_drive_mounted_live",
        "_has_rclone_drive_remote",
        "_refresh_drive_mount_state",
        "_flush_drive_file",
        "_persist_generated_output_to_drive",
    }
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
            selected.append(node)

    mod = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(mod)

    ns: Dict[str, Any] = {
        "os": os,
        "shutil": shutil,
        "subprocess": subprocess,
        "time": time,
        "Any": Any,
        "Dict": Dict,
        "DRIVE_MOUNTED": False,
        "MODEL_BASE_ROOT": os.path.join(tmp_root, "models"),
        "DEFAULT_OUTPUT_DIR": os.path.join(tmp_root, "outputs"),
        "_RCLONE_DRIVE_PROBE_CACHE": {"mtime": -1.0, "ts": 0.0, "ok": False},
    }
    exec(compile(mod, APP_PY_PATH, "exec"), ns)
    return ns


class ColabBridgeDriveMountTest(unittest.TestCase):
    def test_native_wake_waits_for_consent_and_requires_a_real_mount(self):
        # Exercise the actual detached stdin / stdout boundary, using a fake
        # Colab kernel and filesystem for each case.
        worker = """
import contextlib, io, os, sys, types
from unittest.mock import Mock, patch
import colab_bridge as bridge
from colab_cli.commands import automation

mode = sys.argv[1]
session = types.SimpleNamespace(endpoint="test-vm", url="unused", token="unused", kernel_id="k", session_id="s")
bridge.state.get_session = lambda name: session
bridge.state._store = Mock()
bridge.ColabRuntime = Mock()
drive = types.ModuleType("google.colab.drive")
colab = types.ModuleType("google.colab")
colab.drive = drive
sys.modules["google.colab"] = colab
mounted = mode == "existing"

def mount(path):
    global mounted
    assert path == "/content/drive"
    assert mode != "existing"
    uri = "https://accounts.google.com/o/oauth2/v2/auth?state=test"
    if mode == "bad-link":
        uri = "https://example.org/o/oauth2/v2/auth?state=test"
    bridge.state.history.log_event("test", "drive_auth_needed", {"uri": uri})
    assert sys.stdin.readline() == "\\n"
    mounted = mode != "plain-directory"

def fake_automation(name, op, code, **kwargs):
    assert name == "test" and op == "drivemount" and kwargs["allow_stdin"]
    print("private CLI output", flush=True)
    outputs = io.StringIO()
    def stat(path):
        return types.SimpleNamespace(st_dev=2 if mounted and path.endswith("MyDrive") else 1)
    with patch.object(os.path, "isdir", return_value=True), patch.object(os, "stat", stat), contextlib.redirect_stdout(outputs):
        exec(code, {})
    bridge.state.history.log_event("test", "automation_result", {"outputs": [{"text": outputs.getvalue()}]})

drive.mount = mount
automation.run_automation = fake_automation
sys.exit(bridge._native_drive_worker("test", "test-vm"))
"""
        cases = [
            ("existing", "", 0, False),
            ("new", "DRIVE_GRANTED\n", 0, True),
            ("new", "", 2, True),
            ("new", "\n", 2, True),
            ("bad-link", "DRIVE_GRANTED\n", 2, False),
            ("plain-directory", "DRIVE_GRANTED\n", 1, True),
        ]
        for mode, response, status, requested in cases:
            with self.subTest(mode=mode, response=response):
                result = subprocess.run(
                    [sys.executable, "-c", worker, mode], cwd=REPO_ROOT, input=response,
                    capture_output=True, text=True, timeout=20, start_new_session=True,
                )
                self.assertEqual(result.returncode, status)
                self.assertEqual(result.stderr, "")
                expected = "WAKE_DRIVE_AUTH:https://accounts.google.com/o/oauth2/v2/auth?state=test\n"
                self.assertEqual(result.stdout, expected if requested else "")

    def test_wake_requires_drive_only_when_requested(self):
        for flag in ([], ["--require-drive"]):
            with self.subTest(flag=flag), mock.patch.object(colab_bridge, "cmd_wake") as wake:
                colab_bridge.main(["wake", "--app-py", "app.py", *flag])
                wake.assert_called_once_with("G4", "app.py", require_drive=bool(flag))

    def test_failed_drive_mount_stops_deployment_before_upload(self):
        session = mock.Mock()
        with mock.patch.dict(os.environ, {"EASYCOLAB_ADMIN_PASSWORD": "test-only"}), \
             mock.patch.object(colab_bridge, "_has_valid_token", return_value=True), \
             mock.patch.object(colab_bridge, "_safe_sync_sessions", return_value=({}, [])), \
             mock.patch.object(colab_bridge, "_allocate_fresh_session", return_value=session), \
             mock.patch.object(colab_bridge, "_wait_for_runtime_ready", return_value=True), \
             mock.patch.object(colab_bridge, "_ensure_wake_drive", side_effect=RuntimeError("mount failed")) as mount, \
             mock.patch.object(colab_bridge, "ContentsClient") as contents, redirect_stdout(io.StringIO()) as output:
            colab_bridge.cmd_wake("G4", APP_PY_PATH, require_drive=True)
        mount.assert_called_once_with(session)
        contents.assert_not_called()
        self.assertIn("WAKE_ERROR: mount failed", output.getvalue())

    def test_build_rclone_conf_rejects_public_adc_without_quota_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok_file = os.path.join(tmp, "token.json")
            adc_payload = {
                "token": "ya29.test",
                "refresh_token": "1//refresh",
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_id": "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",
                "client_secret": "d-secret",
                "scopes": list(colab_bridge.PUBLIC_SCOPES),
                "expiry": "2099-01-01T00:00:00Z",
            }
            with open(tok_file, "w", encoding="utf-8") as f:
                json.dump(adc_payload, f)

            self.assertEqual(colab_bridge._build_rclone_drive_conf(tok_file), "")

            custom_payload = {
                **adc_payload,
                "client_id": "1234567890-customapp.apps.googleusercontent.com",
            }
            with open(tok_file, "w", encoding="utf-8") as f:
                json.dump(custom_payload, f)

            conf = colab_bridge._build_rclone_drive_conf(tok_file)
            self.assertIn("[easycolab_gdrive]", conf)
            self.assertIn("1234567890-customapp.apps.googleusercontent.com", conf)


class AppDrivePersistenceHelpersTest(unittest.TestCase):
    def test_is_drive_mounted_live_distinguishes_mount_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            ns = _load_app_drive_helpers(tmp)

            def fake_isdir(p):
                return p == "/content/drive/MyDrive"

            def fake_exists(p):
                return p in ("/content", "/content/drive/MyDrive")

            # Same st_dev -> unmounted local overlay directory -> False
            with (
                mock.patch("os.path.isdir", side_effect=fake_isdir),
                mock.patch("os.path.exists", side_effect=fake_exists),
                mock.patch("os.stat", side_effect=lambda p: types.SimpleNamespace(st_dev=100)),
            ):
                self.assertFalse(ns["_is_drive_mounted_live"]())

            # Different st_dev -> real FUSE DriveFS mount -> True
            def fake_stat_mounted(p):
                return types.SimpleNamespace(st_dev=200 if p == "/content/drive/MyDrive" else 100)

            with (
                mock.patch("os.path.isdir", side_effect=fake_isdir),
                mock.patch("os.path.exists", side_effect=fake_exists),
                mock.patch("os.stat", side_effect=fake_stat_mounted),
            ):
                self.assertTrue(ns["_is_drive_mounted_live"]())

    def test_rclone_remote_rejects_broken_adc_client_and_syncs_when_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            ns = _load_app_drive_helpers(tmp)
            rc_file = os.path.join(tmp, "rclone.conf")
            with open(rc_file, "w", encoding="utf-8") as f:
                f.write("[easycolab_gdrive]\ntype = drive\nclient_id = 764086051850-abc.apps.googleusercontent.com\n")

            with (
                mock.patch("os.path.expanduser", return_value=rc_file),
                mock.patch("shutil.which", return_value="/usr/local/bin/rclone"),
            ):
                self.assertFalse(ns["_has_rclone_drive_remote"]())

            # Write a custom OAuth client config and verify rclone probe + output copy
            with open(rc_file, "w", encoding="utf-8") as f:
                f.write("[easycolab_gdrive]\ntype = drive\nclient_id = 998877-custom.apps.googleusercontent.com\n")

            local_img = os.path.join(tmp, "outputs", "sample.png")
            os.makedirs(os.path.dirname(local_img), exist_ok=True)
            with open(local_img, "wb") as wf:
                wf.write(b"PNGDATA" * 100)

            ran_cmds = []

            def fake_run(cmd, **kwargs):
                ran_cmds.append(cmd)
                return types.SimpleNamespace(returncode=0, stdout="{}", stderr="")

            with (
                mock.patch("os.path.expanduser", return_value=rc_file),
                mock.patch("shutil.which", return_value="/usr/local/bin/rclone"),
                mock.patch("subprocess.run", side_effect=fake_run),
                mock.patch.dict(ns, {"_is_drive_mounted_live": lambda: False}),
            ):
                self.assertTrue(ns["_has_rclone_drive_remote"]())
                saved = ns["_persist_generated_output_to_drive"](local_img)
                self.assertEqual(saved, local_img)

            self.assertTrue(
                any(
                    cmd[:2] == ["rclone", "copy"]
                    and "easycolab_gdrive:EasyColab_Drive/EasyColab_Outputs" in cmd
                    for cmd in ran_cmds
                )
            )

    def test_refresh_drive_mount_state_upgrades_output_dir_and_flushes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ns = _load_app_drive_helpers(tmp)
            fake_drive_outputs = os.path.join(tmp, "mounted_drive_outputs")
            local_video = os.path.join(tmp, "outputs", "clip.mp4")
            os.makedirs(os.path.dirname(local_video), exist_ok=True)
            with open(local_video, "wb") as wf:
                wf.write(b"MP4DATA" * 128)

            flushed = []
            with (
                mock.patch.dict(ns, {"_is_drive_mounted_live": lambda: True}),
                mock.patch("os.makedirs"),
            ):
                self.assertTrue(ns["_refresh_drive_mount_state"]())
                self.assertEqual(ns["DEFAULT_OUTPUT_DIR"], "/content/drive/MyDrive/EasyColab_Outputs")
                self.assertEqual(ns["MODEL_BASE_ROOT"], "/content/drive/MyDrive/EasyColab_Models")

            drive_file = os.path.join(fake_drive_outputs, "clip.mp4")
            os.makedirs(fake_drive_outputs, exist_ok=True)
            shutil.copy2(local_video, drive_file)
            with mock.patch("os.fsync", side_effect=lambda fd: flushed.append(fd)):
                ns["_flush_drive_file"](drive_file)
            self.assertEqual(len(flushed), 1)


if __name__ == "__main__":
    unittest.main()
