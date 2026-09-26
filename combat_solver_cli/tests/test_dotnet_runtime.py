"""Selection tests for SDK and game-compatible .NET runtime hosts."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model.dotnet_runtime import find_runtime, find_sdk, runtime_command


class DotnetRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name).resolve()
        self.preferred = self._host(self.home / ".dotnet-spire/dotnet")
        self.system = self._host(self.home / "system/dotnet")
        self.home_patch = patch("model.dotnet_runtime.Path.home", return_value=self.home)
        self.which_patch = patch("model.dotnet_runtime.shutil.which", return_value=self.system)
        self.environ_patch = patch.dict(os.environ, {}, clear=True)
        for item in (self.home_patch, self.which_patch, self.environ_patch):
            item.start()
            self.addCleanup(item.stop)

    @staticmethod
    def _host(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o755)
        path.chmod(0o755)
        return str(path)

    def _probe(self, outputs):
        def run(command, **kwargs):
            self.assertEqual(kwargs["timeout"], 5)
            self.assertTrue(kwargs["capture_output"])
            self.assertTrue(kwargs["text"])
            value = outputs.get(tuple(command), "")
            if isinstance(value, BaseException):
                raise value
            return subprocess.CompletedProcess(command, 0, value, "")

        return patch("model.dotnet_runtime.subprocess.run", side_effect=run)

    def test_only_net10_runtime_is_rejected_even_with_net10_sdk(self):
        with self._probe(
            {
                (self.system, "--list-runtimes"): "Microsoft.NETCore.App 10.0.7 [/x]",
                (self.system, "--list-sdks"): "10.0.100 [/x]",
            }
        ):
            self.assertIsNone(find_runtime())
            self.assertEqual(find_sdk(), self.system)
            with self.assertRaisesRegex(RuntimeError, "Microsoft.NETCore.App 9"):
                runtime_command("game.dll")

    def test_mixed_versions_choose_host_with_net9_and_pin_minor_roll_forward(self):
        with self._probe(
            {
                (self.preferred, "--list-runtimes"): (
                    "Microsoft.NETCore.App 10.0.1 [/x]\n"
                    "Microsoft.NETCore.App 9.0.7 [/x]\n"
                ),
                (self.system, "--list-runtimes"): "Microsoft.NETCore.App 9.0.1 [/x]",
            }
        ):
            self.assertEqual(find_runtime(), self.preferred)
            self.assertEqual(
                runtime_command(Path("game.dll")),
                [self.preferred, "--roll-forward", "Minor", "game.dll"],
            )
            self.assertEqual(
                runtime_command("worker.dll", runtime=self.system),
                [self.system, "--roll-forward", "Minor", "worker.dll"],
            )

    def test_runtime_only_host_runs_but_sdk_comes_from_other_host(self):
        with self._probe(
            {
                (self.preferred, "--list-runtimes"): "Microsoft.NETCore.App 9.0.7 [/x]",
                (self.preferred, "--list-sdks"): "",
                (self.system, "--list-sdks"): "10.0.100 [/x]",
            }
        ):
            self.assertEqual(find_runtime(), self.preferred)
            self.assertEqual(find_sdk(), self.system)

    def test_unstable_net9_is_not_accepted(self):
        with self._probe(
            {
                (self.system, "--list-runtimes"): (
                    "Microsoft.NETCore.App 9.0.0-preview.1 [/x]"
                ),
                (self.system, "--list-sdks"): "9.0.100-preview.1 [/x]",
            }
        ):
            self.assertIsNone(find_runtime())
            self.assertIsNone(find_sdk())

    def test_failed_or_timed_out_probe_is_skipped(self):
        with self._probe(
            {
                (self.preferred, "--list-runtimes"): subprocess.TimeoutExpired(
                    [self.preferred, "--list-runtimes"], 5
                ),
                (self.preferred, "--list-sdks"): OSError("bad executable"),
                (self.system, "--list-runtimes"): "Microsoft.NETCore.App 9.0.7 [/x]",
                (self.system, "--list-sdks"): "9.0.100 [/x]",
            }
        ):
            self.assertEqual(find_runtime(), self.system)
            self.assertEqual(find_sdk(), self.system)

    def test_invalid_explicit_override_fails_without_fallback(self):
        os.environ["STS2_DOTNET"] = str(self.home / "missing-dotnet")
        with self._probe(
            {(self.system, "--list-runtimes"): "Microsoft.NETCore.App 9.0.7 [/x]"}
        ), self.assertRaisesRegex(RuntimeError, "STS2_DOTNET"):
            find_runtime()

    def test_empty_explicit_override_fails(self):
        os.environ["STS2_DOTNET"] = " "
        with self.assertRaisesRegex(RuntimeError, "STS2_DOTNET is set but empty"):
            find_runtime()

    def test_incompatible_explicit_host_fails_without_fallback(self):
        os.environ["STS2_DOTNET"] = self.preferred
        with self._probe(
            {
                (self.preferred, "--list-runtimes"): "Microsoft.NETCore.App 10.0.7 [/x]",
                (self.system, "--list-runtimes"): "Microsoft.NETCore.App 9.0.7 [/x]",
            }
        ), self.assertRaisesRegex(RuntimeError, "STS2_DOTNET"):
            find_runtime()

    def test_explicit_runtime_only_host_does_not_hide_system_sdk(self):
        os.environ["STS2_DOTNET"] = self.preferred
        with self._probe(
            {
                (self.preferred, "--list-runtimes"): "Microsoft.NETCore.App 9.0.7 [/x]",
                (self.preferred, "--list-sdks"): "",
                (self.system, "--list-sdks"): "10.0.100 [/x]",
            }
        ):
            self.assertEqual(find_runtime(), self.preferred)
            self.assertEqual(find_sdk(), self.system)


if __name__ == "__main__":
    unittest.main()
