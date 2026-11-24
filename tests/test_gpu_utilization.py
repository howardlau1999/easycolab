"""GPU utilization sampling without a CUDA runtime."""
from __future__ import annotations

import subprocess
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

if __package__:
    from .test_image_prompt_enhancement import load_app_code
else:
    from test_image_prompt_enhancement import load_app_code


class GPUUtilizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_code = load_app_code(
            "_read_gpu_utilization_percent", "_sample_live_hardware_telemetry_once",
            "get_live_telemetry_snapshot", "api_job_status",
        )

    def setUp(self):
        self.cuda = Mock()
        self.cuda.is_available.return_value = True
        self.cuda.get_device_name.return_value = "test GPU"
        self.cuda.get_device_properties.return_value = SimpleNamespace(uuid="GPU-demo")
        self.cuda.mem_get_info.return_value = (4 * 1024**3, 8 * 1024**3)
        self.clock = Mock(return_value=100.0)
        self.env = {
            "torch": SimpleNamespace(cuda=self.cuda), "subprocess": subprocess,
            "time": SimpleNamespace(time=self.clock), "estimate_cu_burn_rate": Mock(return_value=1),
            "_get_storage_metrics_dict": Mock(return_value={}), "_TELEMETRY_LOCK": threading.Lock(),
            "_CACHED_LIVE_TELEMETRY": {}, "JOBS": {"job": {"status": "completed"}},
            "get_live_console_Lines": Mock(return_value=[]),
        }
        exec(self.app_code, self.env)

    def test_nvml_reports_busy_and_idle_independently_of_memory(self):
        with patch.object(subprocess, "run") as cli:
            for percent in (0, 83, 100):
                self.cuda.utilization.return_value = percent
                sample = self.env["_sample_live_hardware_telemetry_once"](sample_utilization=True)
                self.assertEqual(sample["gpu_utilization_percent"], percent)
                self.assertEqual(sample["vram_used_gb"], 4)
                self.assertEqual(sample["gpu_utilization_sampled_at"], 100.0)
            cli.assert_not_called()

    def test_cli_fallback_selects_cuda_device_and_rejects_unknown_or_multiple_readings(self):
        self.cuda.utilization.side_effect = ModuleNotFoundError("pynvml")
        read = self.env["_read_gpu_utilization_percent"]
        with patch.object(subprocess, "run") as cli:
            cli.return_value.stdout = "63\n"
            self.assertEqual(read(), 63)
            self.assertEqual(cli.call_args.args[0][-2:], ["--id", "GPU-demo"])
            self.assertLessEqual(cli.call_args.kwargs["timeout"], 2)
            self.cuda.get_device_properties.return_value = SimpleNamespace()
            for stdout in ("N/A\n", "0\n63\n", "-1", "101"):
                cli.return_value.stdout = stdout
                self.assertIsNone(read())
            cli.side_effect = subprocess.TimeoutExpired("nvidia-smi", 1.5)
            self.assertIsNone(read())

    def test_job_polls_and_terminal_memory_refresh_reuse_background_sample(self):
        sample = self.env["_sample_live_hardware_telemetry_once"]
        first = self.env["get_live_telemetry_snapshot"]()
        self.assertIsNone(first["gpu_utilization_sampled_at"])
        self.cuda.utilization.return_value = 72
        sample(sample_utilization=True)
        self.clock.return_value = 102.0
        self.cuda.mem_get_info.return_value = (7 * 1024**3, 8 * 1024**3)
        self.assertEqual(sample()["vram_used_gb"], 1)
        cached = self.env["api_job_status"]("job")["telemetry"]
        self.assertEqual(cached["gpu_utilization_percent"], 72)
        self.assertEqual(cached["gpu_utilization_sampled_at"], 100.0)
        self.cuda.utilization.assert_called_once_with(0)
        self.cuda.is_available.return_value = False
        cpu = sample(sample_utilization=True)
        self.assertIsNone(cpu["gpu_utilization_percent"])
        self.assertEqual(cpu["gpu_utilization_sampled_at"], 102.0)
        self.cuda.utilization.assert_called_once()
