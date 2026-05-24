from __future__ import annotations

import os
import subprocess
import tarfile


def test_package_demo_artifacts_includes_existing_and_records_missing(tmp_path):
    included = tmp_path / "included.txt"
    missing = tmp_path / "missing.txt"
    output = tmp_path / "artifacts.tar.gz"
    manifest = tmp_path / "artifacts.manifest.txt"
    included.write_text("payload\n", encoding="utf-8")

    env = os.environ.copy()
    env.update({"OUTPUT": str(output), "MANIFEST": str(manifest)})
    proc = subprocess.run(
        ["bash", "scripts/package_demo_artifacts.sh", str(included), str(missing)],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert str(output) in proc.stdout
    assert str(manifest) in proc.stdout
    manifest_text = manifest.read_text(encoding="utf-8")
    assert f"- {included}" in manifest_text
    assert f"- {missing}" in manifest_text
    with tarfile.open(output, "r:gz") as archive:
        names = archive.getnames()
    assert any(name.endswith("/included.txt") for name in names)
