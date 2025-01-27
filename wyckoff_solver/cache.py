import os
from pathlib import Path


def get_cache_dir() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    cachedir = Path(cache_home) / "calypso"
    cachedir.mkdir(parents=True, exist_ok=True)
    return cachedir


PROJECT_CACHEDIR = get_cache_dir()
