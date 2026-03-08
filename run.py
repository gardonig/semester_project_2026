import os
import sys
from pathlib import Path

# Add the 'src' directory to the Python path so it can find your package
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Set Qt plugin paths on macOS *before* any PySide6 import (Qt reads these when loaded)
if sys.platform == "darwin":
    _prefix = Path(sys.executable).resolve().parent.parent
    _site = _prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    _plugin_root = _site / "PySide6" / "Qt" / "plugins"
    _platforms = _plugin_root / "platforms"
    if _plugin_root.exists():
        os.environ.setdefault("QT_PLUGIN_PATH", str(_plugin_root))
    if _platforms.exists():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(_platforms))

from src.anatomy_poset.main import main

if __name__ == "__main__":
    main()