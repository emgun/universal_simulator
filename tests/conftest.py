import os
import sys


def _ensure_src_on_path():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(repo_root)
    src_dir = os.path.join(repo_root, "src")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


_ensure_src_on_path()
