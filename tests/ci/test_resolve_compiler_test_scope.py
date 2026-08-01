import tempfile
import unittest
from pathlib import Path

from tools.ci.resolve_compiler_test_scope import (
    COMPILER_PATHS,
    ScopeDecision,
    classify_changed_paths,
    force_run,
    is_compiler_path,
    write_github_output,
    write_github_summary,
)


class CompilerPathTests(unittest.TestCase):
    def test_compiler_implementation_paths_match(self):
        paths = (
            "sima_lmm/__init__.py",
            "sima_lmm/utils.py",
            "sima_lmm/config/vlm_config.py",
            "sima_lmm/gguf/parser.py",
            "sima_lmm/hf/hf_transformer.py",
            "sima_lmm/model/language_model.py",
            "sima_lmm/preproc/vlm_helper.py",
            "sima_lmm/tokenizer/whisper_tokenizer.py",
            "sima_lmm/host/compile_lmm.py",
        )
        self.assertTrue(all(is_compiler_path(path) for path in paths))

    def test_compiler_support_paths_match(self):
        paths = (
            "tests/compilation/unit/test_config.py",
            "pytest.ini",
            "pyproject.toml",
            "build_compiler_wheel.sh",
            "tools/ci/prepare_model_inputs.py",
            "tools/hf-safetensors/selection-policy.json",
            ".github/workflows/model-compiler-tests.yml",
            "tools/ci/resolve_compiler_test_scope.py",
        )
        self.assertTrue(all(is_compiler_path(path) for path in paths))

    def test_runtime_and_mixed_purpose_paths_do_not_match(self):
        paths = (
            "sima_lmm/devkit/cpp/chat.cpp",
            "sima_lmm/assets/sjc.jpg",
            "sima_lmm/mole/mole.py",
            "sima_lmm/host/benchmark.py",
            "sima_lmm/host/deploy_lmm.py",
            "tests/runtime/test_http_black_box.py",
            "CMakeLists.txt",
            "cmake/SimaLMMConfig.cmake.in",
            "toolchain-sima.cmake",
            "third_party/minja",
            ".gitmodules",
            "tools/install_llima.sh",
            "build.sh",
            ".github/workflows/vulcan-ci.yml",
        )
        self.assertFalse(any(is_compiler_path(path) for path in paths))

    def test_every_exact_path_is_normalized(self):
        for path in COMPILER_PATHS:
            with self.subTest(path=path):
                self.assertTrue(is_compiler_path(f"./{path}"))


class ScopeDecisionTests(unittest.TestCase):
    def test_matching_branch_runs_compiler_tests(self):
        decision = classify_changed_paths(
            [
                "sima_lmm/devkit/cpp/chat.cpp",
                "sima_lmm/config/vlm_config.py",
            ],
            head_ref="feature/compiler-change",
            base_ref="develop",
        )

        self.assertTrue(decision.run_compiler)
        self.assertEqual(
            decision.matching_paths,
            ("sima_lmm/config/vlm_config.py",),
        )
        self.assertIn("branch feature/compiler-change", decision.reason)

    def test_runtime_only_branch_skips_compiler_tests(self):
        decision = classify_changed_paths(
            [
                "third_party/minja",
                ".gitmodules",
                "sima_lmm/devkit/cpp/chat.cpp",
            ],
            head_ref="feature/runtime-change",
            base_ref="develop",
        )

        self.assertFalse(decision.run_compiler)
        self.assertEqual(decision.matching_paths, ())
        self.assertIn("skipped", decision.reason)

    def test_empty_branch_diff_skips_compiler_tests(self):
        decision = classify_changed_paths(
            [],
            head_ref="feature/no-changes",
            base_ref="develop",
        )

        self.assertFalse(decision.run_compiler)
        self.assertEqual(decision.changed_paths, ())
        self.assertIn("skipped", decision.reason)

    def test_renamed_compiler_source_path_runs_compiler_tests(self):
        decision = classify_changed_paths(
            [
                "docs/language_model.py",
                "sima_lmm/model/language_model.py",
            ],
            head_ref="feature/rename",
            base_ref="develop",
        )

        self.assertTrue(decision.run_compiler)
        self.assertEqual(
            decision.matching_paths,
            ("sima_lmm/model/language_model.py",),
        )

    def test_force_run_is_fail_safe(self):
        decision = force_run("Branch comparison failed; running by default.")

        self.assertTrue(decision.run_compiler)
        self.assertEqual(
            decision.reason,
            "Branch comparison failed; running by default.",
        )

    def test_github_output_and_summary(self):
        decision = ScopeDecision(
            run_compiler=True,
            reason="Compiler tests required.",
            changed_paths=("README.md", "sima_lmm/model/base.py"),
            matching_paths=("sima_lmm/model/base.py",),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "output"
            summary = Path(temp_dir) / "summary"
            write_github_output(output, decision)
            write_github_summary(summary, decision)

            output_text = output.read_text(encoding="utf-8")
            summary_text = summary.read_text(encoding="utf-8")

        self.assertIn("run_compiler=true", output_text)
        self.assertIn(
            'matching_paths_json=["sima_lmm/model/base.py"]',
            output_text,
        )
        self.assertIn("Decision: `run`", summary_text)
        self.assertIn("`sima_lmm/model/base.py`", summary_text)


if __name__ == "__main__":
    unittest.main()
