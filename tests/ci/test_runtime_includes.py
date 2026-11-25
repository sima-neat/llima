"""Exercise allowed and forbidden runtime include edges without a compiler."""

from pathlib import Path
import tempfile
import unittest

from tools.ci.check_runtime_includes import check


class RuntimeIncludesTest(unittest.TestCase):
    def errors(self, files):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, contents in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
            return check(root)

    def test_language_can_use_shared_backend_and_package_headers(self):
        self.assertEqual(self.errors({
            "runtime/src/language/model.cpp": '#include "mla/runtime_context.hpp"\n#include "vlm_config.hpp"',
            "runtime/src/mla/runtime_context.hpp": "",
            "runtime/include/sima_lmm/vlm_config.hpp": "",
        }), [])

    def test_reverse_edges_and_private_model_headers_are_rejected(self):
        for source, header in [
            ("runtime/src/text/parser.cpp", "runtime/src/mla/runtime_context.hpp"),
            ("runtime/src/mla/loader.cpp", "runtime/include/sima_lmm/vision_language_model.hpp"),
            ("runtime/include/sima_lmm/chat.hpp", "runtime/src/language/state.hpp"),
            ("runtime/src/frontends/zmq_server.cpp", "runtime/src/vlm/helper.hpp"),
        ]:
            with self.subTest(source=source, header=header):
                include = header.removeprefix("runtime/src/").removeprefix("runtime/include/sima_lmm/")
                self.assertTrue(self.errors({source: f'#include "{include}"', header: ""}))

    def test_startup_exception_is_limited_to_its_file_and_target(self):
        header = "runtime/include/sima_lmm/image_processor.hpp"
        self.assertEqual(self.errors({"runtime/src/setup.cpp": '#include "image_processor.hpp"', header: ""}), [])
        self.assertTrue(self.errors({"runtime/src/utils.cpp": '#include "image_processor.hpp"', header: ""}))
        self.assertTrue(self.errors({"runtime/src/setup.cpp": '#include "vision_language_model.hpp"',
                                     "runtime/include/sima_lmm/vision_language_model.hpp": ""}))

    def test_backend_exception_only_allows_generic_file_provider(self):
        source = "runtime/src/mla/mla_model.cpp"
        self.assertEqual(self.errors({source: '#include "file_provider.hpp"',
                                     "runtime/include/sima_lmm/file_provider.hpp": ""}), [])
        self.assertTrue(self.errors({source: '#include "package/model_resources.hpp"',
                                    "runtime/src/package/model_resources.hpp": ""}))

    def test_new_public_headers_require_an_explicit_component(self):
        self.assertTrue(self.errors({"runtime/src/utils.cpp": '#include "new_api.hpp"',
                                    "runtime/include/sima_lmm/new_api.hpp": ""}))

    def test_private_includes_must_use_the_src_root(self):
        self.assertTrue(self.errors({"runtime/src/language/model.cpp": '#include "state.hpp"',
                                    "runtime/src/language/state.hpp": ""}))


if __name__ == "__main__":
    unittest.main()
