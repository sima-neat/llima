"""SDK package coordination must succeed before the build container command."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SdkPackageSetup(unittest.TestCase):
    def test_helper_failure_prevents_build_and_success_precedes_it(self):
        workflow = (ROOT/'.github/workflows/vulcan-ci.yml').read_text()
        script = textwrap.dedent(workflow.split('      build_command: |\n', 1)[1]
                                 .split('      post_build_command:', 1)[0])
        script = re.sub(r'\$\{\{.*?\}\}', 'fixture', script)
        for code in (0, 23):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                docker = root/'docker'
                docker.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALLS'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1] == 'ps': print('chosen-sdk')
if sys.argv[1] == 'shared-helper': sys.exit(int(os.environ['STATUS']))
''')
                docker.chmod(0o755)
                helper = root/'helper'
                helper.write_text('#!/bin/sh\nexec docker shared-helper "$@"\n')
                helper.chmod(0o755)
                env = dict(os.environ, PATH=str(root)+os.pathsep+os.environ['PATH'],
                           SDK_PACKAGE_HELPER=str(helper), CALLS=str(root/'calls'), STATUS=str(code))
                env.pop('GITHUB_WORKSPACE', None)
                result = subprocess.run(['bash', '-c', script], cwd=root, env=env,
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, code, result.stderr)
                calls = [json.loads(line) for line in (root/'calls').read_text().splitlines()]
                self.assertEqual(calls[1], ['shared-helper', 'chosen-sdk', 'configure'])
                self.assertEqual(len(calls), 2 if code else 3)
                if not code:
                    self.assertEqual(calls[2][0], 'exec')
