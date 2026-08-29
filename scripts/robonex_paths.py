import os
import subprocess
from pathlib import Path


def resolve_repo(name, environment, explicit=None):
    configured = explicit or os.environ.get(environment)
    if configured:
        root = Path(configured).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        return root
    for anchor in (Path.cwd().resolve(), Path(__file__).resolve()):
        for parent in (anchor, *anchor.parents):
            candidate = parent / name
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError(f"{name} checkout not found; set {environment}")


def description_model(relative_path, root=None):
    description_root = resolve_repo("robonex_description", "ROBONEX_DESCRIPTION_ROOT", root)
    model = description_root / relative_path
    if not model.is_file():
        raise FileNotFoundError(model)
    return model


def git_commit(path):
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
