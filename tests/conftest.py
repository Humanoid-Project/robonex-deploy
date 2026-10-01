import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("scripts", "scripts/sim_to_real"):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)
