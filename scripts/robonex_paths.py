import os
import subprocess
from pathlib import Path

DESCRIPTION_REPO_NAMES = ("robonex-description", "robonex_description")


def resolve_repo(name, environment, explicit=None):
    configured = explicit or os.environ.get(environment)
    if configured:
        root = Path(configured).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        return root
    names = (name,) if isinstance(name, str) else tuple(name)
    for anchor in (Path.cwd().resolve(), Path(__file__).resolve()):
        for parent in (anchor, *anchor.parents):
            for candidate_name in names:
                candidate = parent / candidate_name
                if candidate.is_dir():
                    return candidate
    raise FileNotFoundError(f"{names[0]} checkout not found; set {environment}")


def description_model(relative_path, root=None):
    description_root = resolve_repo(DESCRIPTION_REPO_NAMES, "ROBONEX_DESCRIPTION_ROOT", root)
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
