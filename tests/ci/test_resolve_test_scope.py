import tempfile
import unittest
from pathlib import Path

from tools.ci.resolve_test_scope import (
    ScopeDecision,
    classify_changed_paths,
    force_run,
    write_github_output,
    write_github_summary,
)


class ScopeDecisionTests(unittest.TestCase):
    def test_known_path_categories_select_expected_suites(self):
        cases = (
            (
                "compiler only",
                [
                    "./sima_lmm/config/vlm_config.py",
                    "tests/ci/test_prepare_model_inputs.py",
                ],
                (True, False),
            ),
            (
                "DevKit only",
                ["sima_lmm/devkit/cpp/chat.cpp", "CMakeLists.txt"],
                (False, True),
            ),
            ("shared", ["deps/manifest.json"], (True, True)),
            ("test neutral", ["README.md", "docs/runtime.md"], (False, False)),
            ("empty", [], (False, False)),
        )

        for name, paths, expected in cases:
            with self.subTest(name=name):
                decision = classify_changed_paths(
                    paths,
                    head_ref="feature/test-scope",
                    base_ref="develop",
                )
                self.assertEqual(
                    (decision.run_compiler, decision.run_devkit),
                    expected,
                )
                self.assertEqual(decision.unknown_paths, ())

    def test_unknown_path_runs_both_suites_fail_safe(self):
        decision = classify_changed_paths(
            ["new_runtime_surface/implementation.py"],
            head_ref="feature/new-surface",
            base_ref="develop",
        )

        self.assertTrue(decision.run_compiler)
        self.assertTrue(decision.run_devkit)
        self.assertEqual(
            decision.unknown_paths,
            ("new_runtime_surface/implementation.py",),
        )

    def test_mixed_known_paths_run_both_without_unknowns(self):
        decision = classify_changed_paths(
            [
                "sima_lmm/devkit/cpp/chat.cpp",
                "sima_lmm/config/vlm_config.py",
                "docs/runtime.md",
            ],
            head_ref="feature/mixed-change",
            base_ref="develop",
        )

        self.assertTrue(decision.run_compiler)
        self.assertTrue(decision.run_devkit)
        self.assertEqual(decision.unknown_paths, ())

    def test_force_run_is_fail_safe_for_both_suites(self):
        decision = force_run("Branch comparison failed; running all tests.")

        self.assertTrue(decision.run_compiler)
        self.assertTrue(decision.run_devkit)
        self.assertEqual(decision.devkit_reason, decision.compiler_reason)

    def test_github_output_and_summary(self):
        decision = ScopeDecision(
            run_compiler=True,
            run_devkit=False,
            compiler_reason="Compiler tests required.",
            devkit_reason="DevKit runtime tests skipped.",
            changed_paths=("README.md", "sima_lmm/model/base.py"),
            compiler_paths=("sima_lmm/model/base.py",),
            neutral_paths=("README.md",),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "output"
            summary = Path(temp_dir) / "summary"
            write_github_output(output, decision)
            write_github_summary(summary, decision)

            output_text = output.read_text(encoding="utf-8")
            summary_text = summary.read_text(encoding="utf-8")

        self.assertIn("run_compiler=true", output_text)
        self.assertIn("run_devkit=false", output_text)
        self.assertIn(
            'compiler_paths_json=["sima_lmm/model/base.py"]',
            output_text,
        )
        self.assertIn("| Compiler | `run` | 1 |", summary_text)
        self.assertIn("| DevKit runtime | `skip` | 0 |", summary_text)


if __name__ == "__main__":
    unittest.main()
