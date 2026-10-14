import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RuntimeHardeningTests(unittest.TestCase):
    def test_bootstrap_package_import_does_not_preimport_cpu_opencv(self):
        script = r'''
import json, sys
from megaglitchnode.opencv_cuda_bootstrap import bootstrap_for_dev_defaults
before = "cv2" in sys.modules
bootstrap_for_dev_defaults()
import cv2
print(json.dumps({"before": before, "path": cv2.__file__, "devices": cv2.cuda.getCudaEnabledDeviceCount()}))
'''
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT,
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        details = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertFalse(details["before"])
        self.assertIn("opencv-cuda", details["path"].lower())
        self.assertGreaterEqual(details["devices"], 1)

    def test_missing_custom_paths_fall_back_to_installed_opencv(self):
        script = r'''
import os, sys
os.environ["MEGAGLITCH_OPENCV_BIN"] = r"Z:\missing\opencv-bin"
os.environ["MEGAGLITCH_OPENCV_PYTHON"] = r"Z:\missing\opencv-python"
from megaglitchnode.opencv_cuda_bootstrap import bootstrap_for_dev_defaults
bootstrap_for_dev_defaults()
import cv2
print(cv2.__file__)
'''
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT,
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cv2", result.stdout.lower())

    def test_committed_diagnostics_identify_custom_opencv_and_native_artifacts(self):
        diagnostics = json.loads(
            (ROOT / "tools" / "diagnostics" / "latest.json").read_text(encoding="utf-8")
        )
        checks = diagnostics["checks"]
        self.assertEqual(checks["native_temporal"]["status"], "ok")
        self.assertEqual(checks["rust_runtime"]["status"], "ok")
        self.assertEqual(checks["opencv"]["status"], "ok")
        self.assertIn("opencv-cuda", checks["opencv"]["module_path"].lower())
        self.assertTrue(checks["opencv"]["cuda_selftest"]["success"])

    def test_pyinstaller_spec_is_syntactically_valid_and_collects_native_artifacts(self):
        spec = ROOT / "packaging" / "pyinstaller" / "megaglitch.spec"
        source = spec.read_text(encoding="utf-8")
        compile(source, str(spec), "exec")
        self.assertIn("_native_temporal", source)
        self.assertIn("megaglitch_runtime.dll", source)
        self.assertIn("opencv_*.dll", source)
        self.assertIn("node_presets", source)
        self.assertIn("opencv-preload-manifest.json", source)
        self.assertIn("cuda-runtime-manifest.json", source)
        self.assertIn("MEGAGLITCH_FFMPEG_BIN", source)
        self.assertIn("quantum_gestalt_model.pt", source)


if __name__ == "__main__":
    unittest.main()
