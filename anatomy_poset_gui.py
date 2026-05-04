import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

def _ensure_qt_platform_plugin_path() -> None:
    """
    macOS fix for:
      qt.qpa.plugin: Could not find the Qt platform plugin "cocoa" in ""

    PySide6 bundles the plugin at:
      <site-packages>/PySide6/Qt/plugins/platforms/libqcocoa.dylib
    """
    if sys.platform != "darwin":
        return

    try:
        import PySide6
    except Exception:
        return

    base = Path(PySide6.__file__).resolve().parent
    platforms_dir = base / "Qt" / "plugins" / "platforms"
    if platforms_dir.exists():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platforms_dir))


@dataclass
class Structure: # this is a node in the poset which is an anatomical structure (organ, bone, muscle, etc.)
    name: str 
    com_z: float  # Center of Mass along superior–inferior axis #TODO: Y axis better?
# e.g. Structure(name="Skull", com_z=90.0)


def load_structures_from_json(path: str) -> List[Structure]:
    """
    Load a list of structures from a JSON file.

    Expected format:
    {
      "structures": [
        {"name": "Skull", "com_z": 90.0},
        ...
      ]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("structures", [])
    structures: List[Structure] = []
    for item in items:
        try:
            name = str(item["name"])
            com_z = float(item["com_z"])
        except (KeyError, TypeError, ValueError):
            continue
        structures.append(Structure(name=name, com_z=com_z))
    return structures


def save_poset_to_json(
    path: str,
    structures: List[Structure],
    edges: Set[Tuple[int, int]],
) -> None:
    """
    Save a fully constructed poset (structures + Hasse edges) to JSON.

    Format:
    {
      "structures": [{"name": ..., "com_z": ...}, ...],
      "edges": [[u, v], ...]
    }
    """
    payload = {
        "structures": [asdict(s) for s in structures],
        "edges": [[int(u), int(v)] for (u, v) in sorted(edges)],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


class HasseDiagramView(QGraphicsView):
    """
    Interactive drawing surface for the Hasse diagram:
    - circular nodes containing structure names
    - straight edges for cover relations
    - mouse wheel zoom, drag to pan
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHints(
            self.renderHints()
            | QPainter.Antialiasing
            | QPainter.TextAntialiasing
        )
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def clear(self) -> None:
        self._scene.clear()

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        """
        Zoom in/out with the mouse wheel, centered under the cursor.
        """
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            factor = zoom_in_factor
        else:
            factor = zoom_out_factor

        self.scale(factor, factor)

    def draw_diagram(
        self,
        structures: List[Structure],
        edges: Set[Tuple[int, int]],
    ) -> None:
        """
        Lay out nodes in levels (superior at the top, inferior at the bottom),
        then draw nodes with labels and connecting edges.
        """
        self.clear()

        n = len(structures)
        if n == 0:
            return

        # Build adjacency and indegree for level computation
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        indeg: Dict[int, int] = {i: 0 for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            indeg[v] += 1

        # Longest-path style levels from sources (indeg == 0)
        levels: Dict[int, int] = {i: 0 for i in range(n)}

        # Simple topological order to propagate levels
        from collections import deque

        q: deque[int] = deque(i for i in range(n) if indeg[i] == 0)
        seen: Set[int] = set(q)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if levels[v] < levels[u] + 1:
                    levels[v] = levels[u] + 1
                if v not in seen:
                    seen.add(v)
                    q.append(v)

        # Group nodes by level (0 = most superior)
        level_nodes: Dict[int, List[int]] = {}
        max_level = 0
        for node, lvl in levels.items():
            level_nodes.setdefault(lvl, []).append(node)
            if lvl > max_level:
                max_level = lvl

        # Compute positions
        node_radius = 35.0
        h_spacing = 140.0
        v_spacing = 140.0

        positions: Dict[int, QPointF] = {}

        for lvl in range(0, max_level + 1):
            nodes_at_level = level_nodes.get(lvl, [])
            if not nodes_at_level:
                continue
            count = len(nodes_at_level)
            total_width = (count - 1) * h_spacing
            start_x = -total_width / 2.0
            y = lvl * v_spacing

            for idx, node in enumerate(sorted(nodes_at_level)):
                x = start_x + idx * h_spacing
                positions[node] = QPointF(x, y)

        # Draw edges first so they appear behind nodes
        edge_pen = QPen(QColor(80, 80, 80))
        edge_pen.setWidth(2)
        for u, v in edges:
            p1 = positions.get(u)
            p2 = positions.get(v)
            if p1 is None or p2 is None:
                continue
            self._scene.addLine(
                p1.x(),
                p1.y(),
                p2.x(),
                p2.y(),
                edge_pen,
            )

        # Draw nodes with labels
        node_brush = QBrush(QColor(84, 160, 255))
        node_pen = QPen(QColor(20, 60, 120))
        node_pen.setWidth(2)

        for idx, structure in enumerate(structures):
            pos = positions.get(idx, QPointF(0.0, 0.0))

            # Circle centered at (pos.x, pos.y)
            ellipse = QGraphicsEllipseItem(
                -node_radius,
                -node_radius,
                2 * node_radius,
                2 * node_radius,
            )
            ellipse.setBrush(node_brush)
            ellipse.setPen(node_pen)
            ellipse.setPos(pos)
            self._scene.addItem(ellipse)

            # Label centered within the node
            label_item = QGraphicsTextItem(structure.name)
            label_item.setDefaultTextColor(QColor(255, 255, 255))
            br = label_item.boundingRect()
            label_item.setPos(
                pos.x() - br.width() / 2.0,
                pos.y() - br.height() / 2.0,
            )
            self._scene.addItem(label_item)

        # Fit everything in view initially
        self.fitInView(self._scene.itemsBoundingRect(), Qt.KeepAspectRatio)

class PosetBuilder:
    """
    Implements Algorithm 1 from the prompt.
    Structures are first sorted by CoM (descending).
    The user is then queried step‑by‑step via Q(x, y) provided by the GUI.
    """

    def __init__(self, structures: List[Structure]) -> None:
        # Sort descending by CoM
        self.structures: List[Structure] = sorted(
            structures, key=lambda s: s.com_z, reverse=True
        )
        self.n = len(self.structures)

        # Graph represented as adjacency list using indices into self.structures
        self.edges: Set[Tuple[int, int]] = set()

        # Iteration state for gap-based strategy
        self.current_gap = 1
        self.current_i = 0
        self.finished = self.n <= 1

    # -------- Core graph helpers -------- #
    def path_exists(self, start: int, end: int, edges: Set[Tuple[int, int]] | None = None) -> bool:
        if start == end:
            return True
        if edges is None:
            edges = self.edges

        adjacency: Dict[int, List[int]] = {}
        for u, v in edges:
            adjacency.setdefault(u, []).append(v)

        stack = [start]
        visited = set()
        while stack: #seems slow, any alternatives? (i guess not bad for small graphs)
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for v in adjacency.get(u, []):
                if v == end:
                    return True
                if v not in visited:
                    stack.append(v)
        return False

    def transitive_reduction(self) -> Set[Tuple[int, int]]:
        """
        Naive O(V * E * (V + E)) transitive reduction.
        Fine for small N (e.g., <= 50).
        """
        reduced: Set[Tuple[int, int]] = set(self.edges)
        for u, v in list(self.edges):
            # Temporarily remove edge and test if an alternative path still exists
            temp_edges = set(reduced)
            temp_edges.discard((u, v))
            if self.path_exists(u, v, temp_edges):
                # Edge is redundant
                reduced.discard((u, v))
        return reduced

    # -------- Gap‑based query iteration -------- #
    def next_pair(self) -> Tuple[int, int] | None:
        """
        Advance the (gap, i) loops until the next pair requiring a human query is found.
        Returns (i, j) or None when finished.
        """
        if self.finished:
            return None

        while self.current_gap <= self.n - 1:
            while self.current_i <= self.n - 1 - self.current_gap:
                i = self.current_i
                j = i + self.current_gap
                self.current_i += 1

                # Skip if relation already implied by transitivity
                if self.path_exists(i, j):
                    continue

                # We have a new pair to query
                return i, j

            # Move to next gap
            self.current_gap += 1
            self.current_i = 0

        # No more pairs
        self.finished = True
        return None

    def record_response(self, i: int, j: int, is_above: bool) -> None:
        """
        Called by the GUI after the clinician/user answers Q(si, sj).
        """
        if is_above:
            self.edges.add((i, j))

    def get_final_relations(self) -> Tuple[List[Structure], Set[Tuple[int, int]]]:
        """
        Run transitive reduction and return the sorted structures
        and the minimal set of cover relations (Hasse diagram edges).
        """
        reduced_edges = self.transitive_reduction()
        return self.structures, reduced_edges


class MainWindow(QMainWindow):
    def __init__(self, input_path: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("Anatomical Poset Builder (Head → Toes)")
        self.resize(1000, 600)

        # Remember optional input path for use during UI setup
        self._input_path: Optional[str] = input_path
        # Where to auto-save intermediate poset snapshots
        if self._input_path is not None:
            self._autosave_path: Optional[Path] = Path(self._input_path).with_suffix(
                ".poset_autosave.json"
            )
        else:
            self._autosave_path = Path.cwd() / "poset_autosave.json"

        self.poset_builder: PosetBuilder | None = None
        self.pending_pair: Tuple[int, int] | None = None
        # History of answered queries: list of (i, j, is_above)
        self._answer_history: List[Tuple[int, int, bool]] = []

        self._init_ui()

    def _init_ui(self) -> None:
        central = QWidget()
        root_layout = QHBoxLayout(central)

        # Left: structure definition
        left_group = QGroupBox("Anatomical Structures (Input)")
        left_layout = QVBoxLayout(left_group)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Name", "CoM (Z-axis, arbitrary units)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        left_layout.addWidget(self.table)

        add_row_btn = QPushButton("Add Structure")
        add_row_btn.clicked.connect(self.add_structure_row)
        left_layout.addWidget(add_row_btn)

        remove_row_btn = QPushButton("Remove Selected Structure")
        remove_row_btn.clicked.connect(self.remove_selected_row)
        left_layout.addWidget(remove_row_btn)

        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Number of structures (for reference):"))
        self.count_spin = QSpinBox()
        self.count_spin.setMinimum(1)
        self.count_spin.setMaximum(100)
        self.count_spin.setValue(20)
        controls_layout.addWidget(self.count_spin)

        self.start_btn = QPushButton("Start Poset Construction")
        self.start_btn.clicked.connect(self.start_poset_construction)
        controls_layout.addWidget(self.start_btn)

        left_layout.addLayout(controls_layout)

        # Right: querying and results
        right_layout = QVBoxLayout()

        # Query box
        query_group = QGroupBox("Expert Query (Q(x, y))")
        query_layout = QVBoxLayout(query_group)

        self.query_label = QLabel(
            "Configure your structures and click 'Start Poset Construction'."
        )
        self.query_label.setWordWrap(True)
        query_layout.addWidget(self.query_label)

        btn_row = QHBoxLayout()
        # Back button to undo last answer
        self.back_btn = QPushButton("Back (undo last answer)")
        self.back_btn.setEnabled(False)
        self.back_btn.clicked.connect(self.go_back_one_question)
        btn_row.addWidget(self.back_btn)

        self.yes_btn = QPushButton("Yes (X is strictly above Y)")
        self.no_btn = QPushButton("No (X is not strictly above Y)")
        self.yes_btn.setEnabled(False)
        self.no_btn.setEnabled(False)
        self.yes_btn.clicked.connect(lambda: self.answer_query(True))
        self.no_btn.clicked.connect(lambda: self.answer_query(False))
        btn_row.addWidget(self.yes_btn)
        btn_row.addWidget(self.no_btn)
        query_layout.addLayout(btn_row)

        right_layout.addWidget(query_group)

        # Results box
        results_group = QGroupBox("Resulting Poset (Hasse Diagram Edges)")
        results_layout = QVBoxLayout(results_group)

        self.results_list = QListWidget()
        results_layout.addWidget(self.results_list)

        # Visual Hasse diagram
        self.hasse_view = HasseDiagramView()
        results_layout.addWidget(self.hasse_view)

        right_layout.addWidget(results_group)

        # Layout split
        root_layout.addWidget(left_group, stretch=3)
        root_layout.addLayout(right_layout, stretch=2)

        self.setCentralWidget(central)

        # If an input file is provided, load structures from it,
        # otherwise seed with some example structures.
        if self._input_path is not None:
            try:
                structures = load_structures_from_json(self._input_path)
                for s in structures:
                    self.add_structure_row(s.name, str(s.com_z))
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self,
                    "Failed to load input",
                    f"Could not load structures from:\n{self._input_path}\n\n{exc}",
                )
                self._add_example_structures()
        else:
            self._add_example_structures()

    def _autosave_poset(self) -> None:
        """
        Persist the current best-known poset (transitively reduced)
        to a JSON file after each question / correction.
        """
        if not self.poset_builder or not self._autosave_path:
            return
        try:
            structures, edges = self.poset_builder.get_final_relations()
            save_poset_to_json(str(self._autosave_path), structures, edges)
        except Exception:
            # Autosave failures should not break the interactive session
            pass

    # -------- Structure table helpers -------- #
    def add_structure_row(self, name: str = "", com: str = "") -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        name_item = QTableWidgetItem(name)
        com_item = QTableWidgetItem(com)
        self.table.setItem(row, 0, name_item)
        self.table.setItem(row, 1, com_item)

    def remove_selected_row(self) -> None:
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        for row in sorted(rows, reverse=True):
            self.table.removeRow(row)

    def _add_example_structures(self) -> None:
        """
        Pre-populate with a few canonical structures and approximate CoM values
        (larger Z = more superior / closer to the head).
        """
        examples = [
            ("Skull", "90"),
            ("Cervical Spine", "80"),
            ("Thoracic Spine", "70"),
            ("Lumbar Spine", "60"),
            ("Pelvis", "50"),
            ("Femur", "40"),
            ("Tibia", "30"),
            ("Foot", "20"),
        ]
        for name, com in examples:
            self.add_structure_row(name, com)

    def _collect_structures(self) -> List[Structure] | None:
        structures: List[Structure] = []

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
                QMessageBox.warning(
                    self,
                    "Invalid Input",
                    f"Row {row + 1}: CoM must be a number.",
                )
                return None

            structures.append(Structure(name=name, com_z=com_z))

        if not structures:
            QMessageBox.warning(
                self,
                "No Structures",
                "Please define at least one structure with a valid CoM value.",
            )
            return None

        return structures

    # -------- Poset construction flow -------- #
    def start_poset_construction(self) -> None:
        structures = self._collect_structures()
        if structures is None:
            return

        # Reset history for a fresh run
        self._answer_history = []

        self.poset_builder = PosetBuilder(structures)
        self.results_list.clear()

        self.start_btn.setEnabled(False)
        self.yes_btn.setEnabled(True)
        self.no_btn.setEnabled(True)
        self.back_btn.setEnabled(False)

        self.query_label.setText(
            "Answer the following anatomical questions.\n\n"
            "Definition: X is 'strictly above' Y if the lowest point of X "
            "is higher than the highest point of Y along the superior–inferior (Z) axis."
        )

        # Immediately fetch the first pair to query
        self._advance_to_next_query()

    def _advance_to_next_query(self) -> None:
        if not self.poset_builder:
            return

        pair = self.poset_builder.next_pair()
        self.pending_pair = pair

        if pair is None:
            # Done; compute final poset
            structures, edges = self.poset_builder.get_final_relations()
            self._display_results(structures, edges)
            self.query_label.setText(
                "Poset construction complete.\n"
                "You can adjust the structures and click 'Start Poset Construction' "
                "again to recompute."
            )
            self.yes_btn.setEnabled(False)
            self.no_btn.setEnabled(False)
            self.start_btn.setEnabled(True)
            # Also auto-save the final reduced poset snapshot
            self._autosave_poset()
            return

        i, j = pair
        si = self.poset_builder.structures[i]
        sj = self.poset_builder.structures[j]

        self.query_label.setText(
            f"Is structure X strictly above structure Y?\n\n"
            f"X: {si.name} (CoM Z = {si.com_z})\n"
            f"Y: {sj.name} (CoM Z = {sj.com_z})\n\n"
            "Answer according to: the lowest point of X is higher than the highest point of Y."
        )

    def answer_query(self, is_above: bool) -> None:
        if not self.poset_builder or self.pending_pair is None:
            return
        i, j = self.pending_pair
        # Record history so we can go back
        self._answer_history.append((i, j, is_above))
        # Enable back button once at least one answer exists
        self.back_btn.setEnabled(True)
        self.poset_builder.record_response(i, j, is_above)
        # Auto-save after each answered question
        self._autosave_poset()
        # Move to next needed query
        self._advance_to_next_query()

    def go_back_one_question(self) -> None:
        """
        Undo the last answered question and show it again
        so the user can correct their response.
        """
        if not self.poset_builder or not self._answer_history:
            return

        last_i, last_j, last_answer = self._answer_history.pop()
        # Remove the effect of the last answer from the graph, if it added an edge
        if last_answer:
            try:
                self.poset_builder.edges.discard((last_i, last_j))
            except Exception:
                pass

        # Update pending_pair to re-ask this question
        self.pending_pair = (last_i, last_j)
        si = self.poset_builder.structures[last_i]
        sj = self.poset_builder.structures[last_j]
        self.query_label.setText(
            f"(Correcting) Is structure X strictly above structure Y?\n\n"
            f"X: {si.name} (CoM Z = {si.com_z})\n"
            f"Y: {sj.name} (CoM Z = {sj.com_z})\n\n"
            "Answer according to: the lowest point of X is higher than the highest point of Y."
        )

        # If no more history, disable back again
        if not self._answer_history:
            self.back_btn.setEnabled(False)

        # Auto-save the reverted state as well
        self._autosave_poset()

    def _display_results(
        self, structures: List[Structure], edges: Set[Tuple[int, int]]
    ) -> None:
        """
        Display Hasse diagram edges and a linear extension (topological order).
        """
        self.results_list.clear()

        # Show structures sorted by CoM (already the internal order)
        self.results_list.addItem("Structures (sorted superior → inferior):")
        for idx, s in enumerate(structures):
            self.results_list.addItem(f"  {idx}: {s.name} (CoM Z = {s.com_z})")
        self.results_list.addItem("")

        # Show edges (text)
        if not edges:
            self.results_list.addItem("No strict 'above' relations recorded.")
        else:
            self.results_list.addItem("Cover relations (Hasse diagram edges):")
            for u, v in sorted(edges):
                su = structures[u]
                sv = structures[v]
                item = QListWidgetItem(f"{su.name}  ≻  {sv.name}")
                self.results_list.addItem(item)

        # Optional: simple topological order
        topo_order = self._topological_sort(len(structures), edges)
        if topo_order:
            self.results_list.addItem("")
            self.results_list.addItem("One possible total order (topological sort):")
            # This gives a head → toes sequence
            ordered_names = "  →  ".join(structures[i].name for i in topo_order)
            self.results_list.addItem(ordered_names)

        # Draw / update the visual Hasse diagram
        # Note: even if there are no edges, the view will still show nodes.
        self.hasse_view.draw_diagram(structures, edges)
        # (Display only; auto-saving is handled elsewhere)

    @staticmethod
    def _topological_sort(n: int, edges: Set[Tuple[int, int]]) -> List[int] | None:
        from collections import deque

        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        indeg: Dict[int, int] = {i: 0 for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            indeg[v] += 1

        q: deque[int] = deque(i for i in range(n) if indeg[i] == 0)
        order: List[int] = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        if len(order) != n:
            # Graph should be a DAG; if not, return None
            return None
        return order


def main() -> None:
    """
    Optional usage:
      python anatomy_poset_gui.py path/to/structures.json

    Where structures.json has:
    {
      "structures": [
        {"name": "Skull", "com_z": 90.0},
        ...
      ]
    }
    """
    _ensure_qt_platform_plugin_path()

    # Optional first positional argument = input JSON with anatomical structures
    input_path: Optional[str] = sys.argv[1] if len(sys.argv) > 1 else None

    app = QApplication(sys.argv)
    window = MainWindow(input_path=input_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

