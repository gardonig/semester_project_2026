import json
import sys
from typing import List

from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from anatomy_poset_gui import (
    _ensure_qt_platform_plugin_path,
    Structure,
    load_structures_from_json,
)


class InputEditor(QWidget):
    """
    Small helper GUI for creating JSON input files of anatomical
    structures (name + CoM), to be consumed by anatomy_poset_gui.py.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Anatomical Structures Editor")
        self.resize(800, 500)

        self._init_ui()

    def _init_ui(self) -> None:
        root = QHBoxLayout(self)

        # Left: table editor
        left_group = QGroupBox("Structures (Name + CoM)")
        left_layout = QVBoxLayout(left_group)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Name", "CoM (Z-axis)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        left_layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Row")
        add_btn.clicked.connect(self.add_row)
        rm_btn = QPushButton("Remove Selected")
        rm_btn.clicked.connect(self.remove_selected)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(rm_btn)
        left_layout.addLayout(btn_row)

        # Right: actions / log
        right_group = QGroupBox("Actions")
        right_layout = QVBoxLayout(right_group)

        self.save_btn = QPushButton("Save Structures to JSON…")
        self.save_btn.clicked.connect(self.save_json)
        right_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("Load Structures from JSON…")
        self.load_btn.clicked.connect(self.load_json)
        right_layout.addWidget(self.load_btn)

        right_layout.addWidget(QLabel("Hint: Use the saved file as\n"
                                      "argument to anatomy_poset_gui.py"))

        self.log = QListWidget()
        right_layout.addWidget(self.log)

        root.addWidget(left_group, stretch=3)
        root.addWidget(right_group, stretch=2)

    def add_row(self) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)

    def remove_selected(self) -> None:
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        for row in sorted(rows, reverse=True):
            self.table.removeRow(row)

    def _collect_structures(self) -> List[Structure]:
        items: List[Structure] = []
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            com_item = self.table.item(row, 1)
            if not name_item or not com_item:
                continue
            name = name_item.text().strip()
            com_text = com_item.text().strip()
            if not name or not com_text:
                continue
            try:
                com_z = float(com_text)
            except ValueError:
                raise ValueError(f"Row {row + 1}: CoM must be numeric")
            items.append(Structure(name=name, com_z=com_z))
        return items

    def save_json(self) -> None:
        try:
            structures = self._collect_structures()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Input", str(exc))
            return

        if not structures:
            QMessageBox.information(
                self,
                "No Data",
                "Please enter at least one structure before saving.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Structures as JSON",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return

        data = {
            "structures": [
                {"name": s.name, "com_z": s.com_z} for s in structures
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self.log.addItem(f"Saved {len(structures)} structures to {path}")

    def load_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Structures from JSON",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return

        try:
            structures = load_structures_from_json(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Load Failed",
                f"Could not load structures from:\n{path}\n\n{exc}",
            )
            return

        self.table.setRowCount(0)
        for s in structures:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(s.name))
            self.table.setItem(row, 1, QTableWidgetItem(str(s.com_z)))

        self.log.addItem(f"Loaded {len(structures)} structures from {path}")


def main() -> None:
    """
    Usage:
      python anatomy_input_editor.py
    """
    _ensure_qt_platform_plugin_path()
    app = QApplication(sys.argv)
    win = InputEditor()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

