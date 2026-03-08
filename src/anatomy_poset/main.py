import sys
from pathlib import Path
from typing import Optional

# To ensure Python can find your new package structure even if you haven't 
# formally "installed" it via pip, we add the 'src' directory to the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anatomy_poset.gui.utils import _ensure_qt_platform_plugin_path

# Set Qt plugin path on macOS before any Qt import (Qt reads it when loaded)
_ensure_qt_platform_plugin_path()

from PySide6.QtWidgets import QApplication
from anatomy_poset.gui.main_window import MainWindow


def main() -> None:
    """
    Optional usage:
      python main.py path/to/structures.json
    """

    # Optional first positional argument = input JSON with anatomical structures
    input_path: Optional[str] = sys.argv[1] if len(sys.argv) > 1 else None

    app = QApplication(sys.argv)
    window = MainWindow(input_path=input_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()