import hashlib
from pathlib import Path
import yaml


def load_config(path=None):
    path = Path(path) if path else Path(__file__).resolve().parent.parent / "config.yaml"
    text = path.read_text()
    cfg = yaml.safe_load(text)
    cfg["_hash"] = hashlib.sha256(text.encode()).hexdigest()[:12]
    return cfg
