from __future__ import annotations

import os
import stat
import subprocess
import textwrap


def test_official_remote_launcher_blocks_before_vast_when_readiness_fails(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import sys
            from pathlib import Path

            if len(sys.argv) > 1 and sys.argv[1] == "scripts/check_official_execution_readiness.py":
                output = Path(sys.argv[sys.argv.index("--output-json") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({
                    "remote_launch_ready": False,
                    "route_blockers": {"remote_launch": ["remote API host console.vast.ai does not resolve"]},
                    "status": "blocked",
                }), encoding="utf-8")
                print("readiness blocked")
                raise SystemExit(2)

            if len(sys.argv) > 1 and sys.argv[1] == "-":
                code = sys.stdin.read()
                sys.argv = sys.argv[1:]
                namespace = {"__name__": "__main__", "__file__": "<stdin>"}
                exec(compile(code, "<stdin>", "exec"), namespace)
                raise SystemExit(0)

            raise SystemExit("unexpected python invocation: " + " ".join(sys.argv))
            """
        ),
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "DRY_RUN": "0",
            "GIT_REF": "codex/test",
            "REMOTE_SCRIPT": "scripts/run_remote_official_hydration.sh",
            "READINESS_JSON": str(tmp_path / "official_execution_readiness.json"),
        }
    )

    proc = subprocess.run(
        ["bash", "scripts/launch_remote_transport_shift_candidate_vast.sh"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert proc.returncode == 2
    assert "Remote official hydration is not launch-ready" in proc.stderr
    assert "vast_launch.py" not in proc.stderr
