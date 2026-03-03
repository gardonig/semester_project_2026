from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QListWidget,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
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
class Structure:
    """A node in the poset: an anatomical structure (organ, bone, muscle, etc.)."""
    name: str
    com_vertical: float   # Center of Mass along vertical (superior–inferior, top–bottom) axis
    com_lateral: float  # Center of Mass along mediolateral (left–right) axis; left = smaller, right = larger; patient's perspective
    com_anteroposterior: float  # Center of Mass along anteroposterior (front–back) axis; anterior = smaller, posterior = larger


def load_structures_from_json(path: str) -> List[Structure]:
    """
    Load a list of structures from a JSON file.

    Expected format (newest version):
    {
      "structures": [
        {
          "name": "Skull",
          "com_vertical": 90.0,          # top–bottom (superior–inferior)
          "com_lateral": 0.0,            # left–right (mediolateral), patient's perspective
          "com_anteroposterior": 0.0     # front–back (anteroposterior), anterior < posterior
        },
        ...
      ]
    }
    Older files may omit com_lateral and/or com_anteroposterior; both default to 0.0.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("structures", [])
    structures: List[Structure] = []
    for item in items:
        try:
            name = str(item["name"])
            com_vertical = float(item["com_vertical"])
            com_lateral = float(item.get("com_lateral", 0.0))
            com_ap = float(item.get("com_anteroposterior", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        structures.append(
            Structure(
                name=name,
                com_vertical=com_vertical,
                com_lateral=com_lateral,
                com_anteroposterior=com_ap,
            )
        )
    return structures


def save_poset_to_json(
    path: str,
    structures: List[Structure],
    edges_vertical: Set[Tuple[int, int]],
    edges_mediolateral: Set[Tuple[int, int]] | None = None,
    edges_anteroposterior: Set[Tuple[int, int]] | None = None,
) -> None:
    """
    Save poset(s) to JSON. All axes stored in one file.

    Format (keys added over time; extra keys are ignored by loaders):
    {
      "structures": [
        {
          "name": "...",
          "com_vertical": ...,
          "com_lateral": ...,
          "com_anteroposterior": ...
        },
        ...
      ],
      "edges_vertical": [[u, v], ...],          # top–bottom (superior–inferior)
      "edges_mediolateral": [[u, v], ...],      # left–right (mediolateral, patient's view)
      "edges_anteroposterior": [[u, v], ...],   # front–back (anteroposterior)
      "adjacency_vertical": [[0/1, ...], ...],          # optional
      "adjacency_mediolateral": [[0/1, ...], ...],      # optional
      "adjacency_anteroposterior": [[0/1, ...], ...]    # optional
    }
    """
    if edges_mediolateral is None:
        edges_mediolateral = set()
    if edges_anteroposterior is None:
        edges_anteroposterior = set()

    n = len(structures)

    def _edges_to_adj_matrix(num_nodes: int, edges: Set[Tuple[int, int]]) -> list[list[int]]:
        mat = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        for u, v in edges:
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                mat[u][v] = 1
        return mat

    payload = {
        "structures": [asdict(s) for s in structures],
        "edges_vertical": [[int(u), int(v)] for (u, v) in sorted(edges_vertical)],
        "edges_mediolateral": [[int(u), int(v)] for (u, v) in sorted(edges_mediolateral)],
        "edges_anteroposterior": [[int(u), int(v)] for (u, v) in sorted(edges_anteroposterior)],
        "adjacency_vertical": _edges_to_adj_matrix(n, edges_vertical),
        "adjacency_mediolateral": _edges_to_adj_matrix(n, edges_mediolateral),
        "adjacency_anteroposterior": _edges_to_adj_matrix(n, edges_anteroposterior),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_poset_from_json(
    path: str,
) -> Tuple[List[Structure], Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Load poset(s) from JSON.

    Returns:
      (structures, edges_vertical, edges_mediolateral, edges_anteroposterior)

    Backward compatibility:
    - Older files may only contain "edges" (treated as vertical) or lack anteroposterior edges.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    structures_data = data.get("structures", [])
    edges_v_data = data.get("edges_vertical", data.get("edges", []))
    # Backward compat: older files used edges_frontal for the left–right axis
    edges_ml_data = data.get("edges_mediolateral", data.get("edges_frontal", []))
    edges_ap_data = data.get("edges_anteroposterior", [])

    structures: List[Structure] = []
    for item in structures_data:
        try:
            name = str(item["name"])
            com_vertical = float(item["com_vertical"])
            com_lateral = float(item.get("com_lateral", 0.0))
            com_ap = float(item.get("com_anteroposterior", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        structures.append(
            Structure(
                name=name,
                com_vertical=com_vertical,
                com_lateral=com_lateral,
                com_anteroposterior=com_ap,
            )
        )

    def parse_edges(edges_data: list) -> Set[Tuple[int, int]]:
        out: Set[Tuple[int, int]] = set()
        for item in edges_data:
            try:
                u, v = int(item[0]), int(item[1])
                out.add((u, v))
            except (TypeError, ValueError, IndexError):
                continue
        return out

    return (
        structures,
        parse_edges(edges_v_data),
        parse_edges(edges_ml_data),
        parse_edges(edges_ap_data),
    )


class PosetViewerWindow(QWidget):
    """View saved poset(s): tabular data + Hasse diagram per axis (Vertical / Mediolateral / Anteroposterior)."""

    def __init__(self, poset_path: str) -> None:
        super().__init__()
        self.setWindowTitle("Anatomical Poset Viewer")
        self.resize(900, 600)

        self._path = poset_path
        self._structures: List[Structure] = []
        self._edges_vertical: Set[Tuple[int, int]] = set()
        self._edges_mediolateral: Set[Tuple[int, int]] = set()
        self._edges_anteroposterior: Set[Tuple[int, int]] = set()

        self._tabs = QTabWidget()
        root = QVBoxLayout(self)
        root.addWidget(self._tabs)

        self._load(poset_path)

    def _fill_tab(
        self,
        list_widget: QListWidget,
        hasse_view: HasseDiagramView,
        structures: List[Structure],
        edges: Set[Tuple[int, int]],
        axis_label: str,
        relation_label: str,
    ) -> None:
        list_widget.clear()
        list_widget.addItem(f"Loaded from: {self._path}")
        list_widget.addItem("")
        list_widget.addItem(f"Structures ({axis_label}):")
        for idx, s in enumerate(structures):
            if axis_label == "Vertical":
                list_widget.addItem(f"  {idx}: {s.name} (CoM vertical = {s.com_vertical})")
            elif axis_label == "Mediolateral":
                list_widget.addItem(f"  {idx}: {s.name} (CoM mediolateral = {s.com_lateral})")
            else:  # Anteroposterior
                list_widget.addItem(
                    f"  {idx}: {s.name} (CoM anteroposterior = {s.com_anteroposterior})"
                )
        list_widget.addItem("")
        if not edges:
            list_widget.addItem(f"No strict '{relation_label}' relations recorded.")
        else:
            list_widget.addItem("Cover relations (Hasse diagram edges):")
            for u, v in sorted(edges):
                su, sv = structures[u], structures[v]
                list_widget.addItem(f"{su.name}  ≻  {sv.name}")

        if axis_label == "Vertical":
            axis = AXIS_VERTICAL
        elif axis_label == "Mediolateral":
            axis = AXIS_MEDIOLATERAL
        else:
            axis = AXIS_ANTERIOR_POSTERIOR
        hasse_view.draw_diagram(structures, edges, axis=axis)

    def _load(self, path: str) -> None:
        try:
            (
                self._structures,
                self._edges_vertical,
                self._edges_mediolateral,
                self._edges_anteroposterior,
            ) = load_poset_from_json(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Failed to Load",
                f"Could not load poset from:\n{path}\n\n{exc}",
            )
            return

        # Tab: Vertical (top–bottom)
        vert_widget = QWidget()
        vert_layout = QHBoxLayout(vert_widget)
        vert_list_group = QGroupBox("Poset Data")
        vert_list_layout = QVBoxLayout(vert_list_group)
        vert_list = QListWidget()
        vert_list_layout.addWidget(vert_list)
        vert_diagram_group = QGroupBox("Hasse Diagram")
        vert_diagram_layout = QVBoxLayout(vert_diagram_group)
        vert_hasse = HasseDiagramView()
        vert_hasse.setMinimumSize(400, 300)
        vert_diagram_layout.addWidget(vert_hasse)
        vert_layout.addWidget(vert_list_group, stretch=1)
        vert_layout.addWidget(vert_diagram_group, stretch=2)
        self._fill_tab(
            vert_list,
            vert_hasse,
            self._structures,
            self._edges_vertical,
            "Vertical",
            "above",
        )
        self._tabs.addTab(vert_widget, "Vertical (top–bottom)")

        # Tab: Mediolateral (left–right)
        ml_widget = QWidget()
        ml_layout = QHBoxLayout(ml_widget)
        ml_list_group = QGroupBox("Poset Data")
        ml_list_layout = QVBoxLayout(ml_list_group)
        ml_list = QListWidget()
        ml_list_layout.addWidget(ml_list)
        ml_diagram_group = QGroupBox("Hasse Diagram")
        ml_diagram_layout = QVBoxLayout(ml_diagram_group)
        ml_hasse = HasseDiagramView()
        ml_hasse.setMinimumSize(400, 300)
        ml_diagram_layout.addWidget(ml_hasse)
        ml_layout.addWidget(ml_list_group, stretch=1)
        ml_layout.addWidget(ml_diagram_group, stretch=2)
        self._fill_tab(
            ml_list,
            ml_hasse,
            self._structures,
            self._edges_mediolateral,
            "Mediolateral",
            "to the left of",
        )
        self._tabs.addTab(ml_widget, "Mediolateral (left–right, patient's view)")

        # Tab: Anteroposterior (front–back)
        ap_widget = QWidget()
        ap_layout = QHBoxLayout(ap_widget)
        ap_list_group = QGroupBox("Poset Data")
        ap_list_layout = QVBoxLayout(ap_list_group)
        ap_list = QListWidget()
        ap_list_layout.addWidget(ap_list)
        ap_diagram_group = QGroupBox("Hasse Diagram")
        ap_diagram_layout = QVBoxLayout(ap_diagram_group)
        ap_hasse = HasseDiagramView()
        ap_hasse.setMinimumSize(400, 300)
        ap_diagram_layout.addWidget(ap_hasse)
        ap_layout.addWidget(ap_list_group, stretch=1)
        ap_layout.addWidget(ap_diagram_group, stretch=2)
        self._fill_tab(
            ap_list,
            ap_hasse,
            self._structures,
            self._edges_anteroposterior,
            "Anteroposterior",
            "in front of",
        )
        self._tabs.addTab(ap_widget, "Anteroposterior (front–back)")


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
        axis: str = "vertical",
    ) -> None:
        """
        Lay out nodes and draw edges. Vertical axis: levels (top to bottom), x spread by index.
        Frontal axis: y by level, x by com_lateral so left is left of spine is left of right.
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

        level_nodes: Dict[int, List[int]] = {}
        max_level = 0
        for node, lvl in levels.items():
            level_nodes.setdefault(lvl, []).append(node)
            if lvl > max_level:
                max_level = lvl

        node_radius = 35.0
        h_spacing = 140.0
        v_spacing = 140.0

        positions: Dict[int, QPointF] = {}

        # Layout like a standard layered Hasse diagram for both axes:
        # - y coordinate encodes level
        # - x coordinate spreads nodes within each level
        # The axis only affects which relation is encoded in the edges; the geometry is identical.
        for lvl in range(0, max_level + 1):
            nodes_at_level = level_nodes.get(lvl, [])
            if not nodes_at_level:
                continue
            nodes_sorted = sorted(nodes_at_level)
            count = len(nodes_sorted)
            total_width = (count - 1) * h_spacing
            start_x = -total_width / 2.0
            y = lvl * v_spacing
            for idx, node in enumerate(nodes_sorted):
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

AXIS_VERTICAL = "vertical"  # top–bottom (superior–inferior)
AXIS_MEDIOLATERAL = "mediolateral"  # left–right (patient's view)
AXIS_ANTERIOR_POSTERIOR = "anteroposterior"  # front–back (anteroposterior)


class PosetBuilder:
    """
    Implements Algorithm 1 using the gap-based CoM strategy.
    Structures are sorted by the chosen axis CoM (descending).
    The user is then queried step‑by‑step via Q(x, y) provided by the GUI.
    """

    def __init__(self, structures: List[Structure], axis: str = AXIS_VERTICAL) -> None:
        if axis == AXIS_MEDIOLATERAL:
            key = lambda s: s.com_lateral
        elif axis == AXIS_ANTERIOR_POSTERIOR:
            key = lambda s: s.com_anteroposterior
        else:
            key = lambda s: s.com_vertical
        self.structures: List[Structure] = sorted(structures, key=key, reverse=True)
        self.n = len(self.structures)
        self.axis = axis

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

    def edge_redundancy_reduction(self) -> Set[Tuple[int, int]]:
        """
        Remove redundant edges implied by transitivity (naive O(V * E * (V + E)) algorithm).
        This is the transitive reduction of the current directed acyclic graph and yields
        exactly the cover relations used in the (directed) Hasse diagram.
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

    def get_iteration_progress(self) -> float:
        """
        Progress through the fixed (gap, i) iteration, 0.0 to 1.0.
        One answer can advance this by a lot when many pairs are skipped by transitivity.
        """
        if self.finished or self.n <= 1:
            return 1.0
        total = self.n * (self.n - 1) // 2
        if total == 0:
            return 1.0
        # Pairs already passed: gaps 1..current_gap-1 fully, plus current_i in current_gap
        steps_done = (self.current_gap - 1) * self.n - (self.current_gap - 1) * self.current_gap // 2 + self.current_i
        return min(1.0, steps_done / total)

    def record_response(self, i: int, j: int, is_above: bool) -> None:
        """
        Called by the GUI after the clinician/user answers Q(si, sj).
        """
        if is_above:
            self.edges.add((i, j))

    def get_final_relations(self) -> Tuple[List[Structure], Set[Tuple[int, int]]]:
        """
        Run edge redundancy reduction (transitive reduction) and return the sorted structures
        and the minimal set of cover relations (Hasse diagram edges).
        """
        reduced_edges = self.edge_redundancy_reduction()
        return self.structures, reduced_edges


class DefinitionDialog(QDialog):
    """
    Shown before the query window. Displays the definition (and examples for vertical).
    User presses "Proceed" to proceed to questions.
    """

    def __init__(self, axis: str = AXIS_VERTICAL) -> None:
        super().__init__()
        self.setWindowTitle("Instructions — Poset Construction")
        self.resize(560, 600)
        self.setModal(True)
        self._axis = axis
        self.setStyleSheet("background-color: #ffffff;")

        layout = QVBoxLayout(self)

        # All text: dark on white (no grey backgrounds)
        _text_style = "color: #1a1a1a; font-size: 14px; padding: 6px 0;"

        # ---- 1. Welcome + anatomical position (side-by-side) ----
        welcome_heading = QLabel("Welcome to the Anatomical Structure Questionnaire")
        welcome_heading.setStyleSheet("color: #1a1a1a; font-weight: bold; font-size: 18px; padding: 0 0 12px 0;")
        layout.addWidget(welcome_heading)

        intro_text = (
            "Thank you for taking part. In this questionnaire you will be asked to compare pairs of "
            "anatomical structures and indicate whether one is strictly above the other (vertical axis) or "
            "strictly to the left of the other (mediolateral axis). Your answers help us build a spatial ordering "
            "that can be used to check and correct automatic segmentations. There are no wrong answers—we need "
            "your clinical judgement.\n\n"
            "We assume the patient is in standard anatomical position, as shown in the figure. "
            "If you have any questions, please do not hesitate to reach out to Gian or Güney."
        )
        intro_label = QLabel(intro_text)
        intro_label.setWordWrap(True)
        intro_label.setStyleSheet(_text_style)

        anatomy_img = QLabel()
        anatomy_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        anatomy_img.setFixedHeight(220)
        anatomy_path = Path(__file__).resolve().parent / "assets" / "definition_images" / "anatomical_postion.png"
        if anatomy_path.exists():
            anatomy_pix = QPixmap(str(anatomy_path))
            if not anatomy_pix.isNull():
                anatomy_img.setPixmap(
                    anatomy_pix.scaledToHeight(220, Qt.SmoothTransformation)
                )
        if anatomy_img.pixmap() is None or anatomy_img.pixmap().isNull():
            anatomy_img.setText("[Anatomical position diagram missing]")
        # Put text and image side-by-side
        intro_row = QHBoxLayout()
        intro_row.addWidget(intro_label, stretch=3)
        intro_row.addWidget(anatomy_img, stretch=2)
        layout.addLayout(intro_row)

        anatomy_ref = QLabel("Images in this window are captured from Complete Anatomy.")
        anatomy_ref.setWordWrap(True)
        anatomy_ref.setStyleSheet("color: #555; font-size: 10px; margin-top: 2px;")
        layout.addWidget(anatomy_ref)

        layout.addStretch(1)

        # Proceed button, right-bottom
        proceed_btn = QPushButton("Understood")
        proceed_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #007aff; color: white; border: none; border-radius: 8px;
                padding: 12px 24px; font-size: 15px;
            }
            QPushButton:hover { background-color: #5ac8fa; }
            QPushButton:pressed { background-color: #0051d5; }
            """
        )
        proceed_btn.clicked.connect(self.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(proceed_btn)
        layout.addLayout(btn_row)


class VerticalDefinitionDialog(QDialog):
    """
    Dedicated window for the vertical 'strictly above' definition and examples.
    Shown only when the vertical axis is selected; user presses 'Proceed' to start questions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Definition — Vertical \"strictly above\"")
        self.resize(820, 520)
        self.setModal(True)
        self.setStyleSheet("background-color: #ffffff;")

        root = QHBoxLayout(self)

        _text_style = "color: #1a1a1a; font-size: 14px; padding: 4px 0;"
        _heading_style = "color: #1a1a1a; font-weight: bold; font-size: 16px; padding: 4px 0 4px 0;"

        # Left column: text (task, definition, question form)
        left_col = QVBoxLayout()
        root.addLayout(left_col, stretch=3)

        heading = QLabel("Vertical relation: \"strictly above\"")
        heading.setStyleSheet(_heading_style)
        left_col.addWidget(heading)

        text1 = QLabel(
            "You will be asked whether one anatomical structure is strictly above another "
            "along the superior–inferior (head–to–toes) axis."
        )
        text1.setWordWrap(True)
        text1.setStyleSheet(_text_style)
        left_col.addWidget(text1)

        text2 = QLabel(
            "Definition: one structure is strictly above another if the lowest point of the upper structure "
            "is higher than the highest point of the lower one."
        )
        text2.setWordWrap(True)
        text2.setStyleSheet(_text_style)
        left_col.addWidget(text2)

        q_form_heading = QLabel("What you will be asked")
        q_form_heading.setStyleSheet(_heading_style)
        left_col.addWidget(q_form_heading)

        q_form = QLabel(
            "For each pair of structures you will answer \"Yes\" or \"No\" to:\n"
            "  \"Is the first structure strictly above the second?\""
        )
        q_form.setWordWrap(True)
        q_form.setStyleSheet(_text_style)
        left_col.addWidget(q_form)

        left_col.addStretch(1)

        proceed_btn = QPushButton("Understood")
        proceed_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #007aff; color: white; border: none; border-radius: 8px;
                padding: 10px 22px; font-size: 15px;
            }
            QPushButton:hover { background-color: #5ac8fa; }
            QPushButton:pressed { background-color: #0051d5; }
            """
        )
        proceed_btn.clicked.connect(self.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(proceed_btn)
        left_col.addLayout(btn_row)

        # Right column: large example images (No example first, then Yes)
        right_col = QVBoxLayout()
        root.addLayout(right_col, stretch=4)

        _script_dir = Path(__file__).resolve().parent
        _img_dir = _script_dir / "assets" / "definition_images"
        img_height = 320

        img_style = (
            "border: 1px solid #cccccc; border-radius: 8px; margin-top: 6px; "
            "background: #ffffff;"
        )
        placeholder_style = (
            "border: 1px dashed #bbbbbb; border-radius: 8px; margin-top: 6px; "
            "color: #444; font-size: 13px; background: #ffffff;"
        )

        # No example first: text above image (Femur–Tibia)
        no_text = QLabel("Example 1 (No): \"Is the Femur strictly above the Tibia?\" → Answer: No.")
        no_text.setWordWrap(True)
        no_text.setStyleSheet(_text_style)
        right_col.addWidget(no_text)

        no_label = QLabel()
        no_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        no_label.setFixedHeight(img_height)
        no_label.setStyleSheet(img_style)
        no_path = _img_dir / "example_vert_No.png"
        if no_path.exists():
            pix = QPixmap(str(no_path))
            if not pix.isNull():
                no_label.setPixmap(
                    pix.scaledToHeight(img_height, Qt.SmoothTransformation)
                )
        if no_label.pixmap() is None or no_label.pixmap().isNull():
            no_label.setText("[Add vertical No example image here]")
            no_label.setStyleSheet(placeholder_style)
        right_col.addWidget(no_label)

        # Yes example second: text above image (Femur–Fibula)
        yes_text = QLabel("Example 2 (Yes): \"Is the Femur strictly above the Fibula?\" → Answer: Yes.")
        yes_text.setWordWrap(True)
        yes_text.setStyleSheet(_text_style)
        yes_text.setContentsMargins(0, 12, 0, 0)
        right_col.addWidget(yes_text)

        yes_label = QLabel()
        yes_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        yes_label.setFixedHeight(img_height)
        yes_label.setStyleSheet(img_style)
        yes_path = _img_dir / "eample_vert_Yes.png"
        if yes_path.exists():
            pix = QPixmap(str(yes_path))
            if not pix.isNull():
                yes_label.setPixmap(
                    pix.scaledToHeight(img_height, Qt.SmoothTransformation)
                )
        if yes_label.pixmap() is None or yes_label.pixmap().isNull():
            yes_label.setText("[Add vertical Yes example image here]")
            yes_label.setStyleSheet(placeholder_style)
        right_col.addWidget(yes_label)

        source = QLabel("Images in this window are captured from Complete Anatomy.")
        source.setWordWrap(True)
        source.setStyleSheet("color: #555; font-size: 10px; margin-top: 4px;")
        right_col.addWidget(source)


class MediolateralDefinitionDialog(QDialog):
    """
    Dedicated window for the mediolateral (left–right) 'strictly to the left of' definition and examples.
    Shown only when the mediolateral axis is selected; user presses 'Proceed' to start questions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Definition — Mediolateral \"strictly to the left of\"")
        self.resize(820, 520)
        self.setModal(True)
        self.setStyleSheet("background-color: #ffffff;")

        root = QHBoxLayout(self)

        _text_style = "color: #1a1a1a; font-size: 14px; padding: 4px 0;"
        _heading_style = "color: #1a1a1a; font-weight: bold; font-size: 16px; padding: 4px 0 4px 0;"

        # Left column: text (task, definition, question form)
        left_col = QVBoxLayout()
        root.addLayout(left_col, stretch=3)

        heading = QLabel("Mediolateral relation: \"strictly to the left of\"")
        heading.setStyleSheet(_heading_style)
        left_col.addWidget(heading)

        text1 = QLabel(
            "You will be asked whether one anatomical structure is strictly to the left of another "
            "along the left–right (mediolateral) axis, from the patient's perspective."
        )
        text1.setWordWrap(True)
        text1.setStyleSheet(_text_style)
        left_col.addWidget(text1)

        text2 = QLabel(
            "Definition: one structure is strictly to the left of another if the rightmost point of the first "
            "is to the left of the leftmost point of the second."
        )
        text2.setWordWrap(True)
        text2.setStyleSheet(_text_style)
        left_col.addWidget(text2)

        patient_note = QLabel(
            "Left and right are always defined from the patient's view: "
            "the patient's right femur is to the right of the patient's left femur."
        )
        patient_note.setWordWrap(True)
        patient_note.setStyleSheet(_text_style)
        left_col.addWidget(patient_note)

        q_form_heading = QLabel("What you will be asked")
        q_form_heading.setStyleSheet(_heading_style)
        left_col.addWidget(q_form_heading)

        q_form = QLabel(
            "For each pair of structures you will answer \"Yes\" or \"No\" to:\n"
            "  \"Is the first structure strictly to the left of the second?\""
        )
        q_form.setWordWrap(True)
        q_form.setStyleSheet(_text_style)
        left_col.addWidget(q_form)

        left_col.addStretch(1)

        proceed_btn = QPushButton("Proceed")
        proceed_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #007aff; color: white; border: none; border-radius: 8px;
                padding: 10px 22px; font-size: 15px;
            }
            QPushButton:hover { background-color: #5ac8fa; }
            QPushButton:pressed { background-color: #0051d5; }
            """
        )
        proceed_btn.clicked.connect(self.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(proceed_btn)
        left_col.addLayout(btn_row)

        # Right column: textual examples + placeholder images
        right_col = QVBoxLayout()
        root.addLayout(right_col, stretch=4)

        _script_dir = Path(__file__).resolve().parent
        _img_dir = _script_dir / "assets" / "definition_images"
        img_height = 260

        img_style = (
            "border: 1px solid #cccccc; border-radius: 8px; margin-top: 6px; "
            "background: #ffffff;"
        )
        placeholder_style = (
            "border: 1px dashed #bbbbbb; border-radius: 8px; margin-top: 6px; "
            "color: #444; font-size: 13px; background: #ffffff;"
        )

        # Example 1 (Yes): Left vs Right femur
        ex1_text = QLabel(
            "Example 1 (Yes): \"Is the Left femur strictly to the left of the Right femur?\" "
            "→ Answer: Yes (from the patient's perspective)."
        )
        ex1_text.setWordWrap(True)
        ex1_text.setStyleSheet(_text_style)
        right_col.addWidget(ex1_text)

        ex1_img = QLabel()
        ex1_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ex1_img.setFixedHeight(img_height)
        ex1_img.setStyleSheet(img_style)
        # Placeholder only for now
        ex1_img.setText("[Add mediolateral Yes example image here]")
        ex1_img.setStyleSheet(placeholder_style)
        right_col.addWidget(ex1_img)

        # Example 2 (No): Left femur vs pelvis (overlap)
        ex2_text = QLabel(
            "Example 2 (No): \"Is the Left femur strictly to the left of the pelvis?\" "
            "→ Answer: No (they overlap in the mediolateral direction)."
        )
        ex2_text.setWordWrap(True)
        ex2_text.setStyleSheet(_text_style)
        ex2_text.setContentsMargins(0, 12, 0, 0)
        right_col.addWidget(ex2_text)

        ex2_img = QLabel()
        ex2_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ex2_img.setFixedHeight(img_height)
        ex2_img.setStyleSheet(img_style)
        # Placeholder only for now
        ex2_img.setText("[Add mediolateral No example image here]")
        ex2_img.setStyleSheet(placeholder_style)
        right_col.addWidget(ex2_img)

        source = QLabel("Images in this window are captured from Complete Anatomy (mediolateral examples to be added).")
        source.setWordWrap(True)
        source.setStyleSheet("color: #555; font-size: 10px; margin-top: 4px;")
        right_col.addWidget(source)


class AnteroposteriorDefinitionDialog(QDialog):
    """
    Dedicated window for the anteroposterior (front–back) 'strictly in front of' definition and examples.
    Shown only when the anteroposterior axis is selected; user presses 'Proceed' to start questions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Definition — Anteroposterior \"strictly in front of\"")
        self.resize(820, 520)
        self.setModal(True)
        self.setStyleSheet("background-color: #ffffff;")

        root = QHBoxLayout(self)

        _text_style = "color: #1a1a1a; font-size: 14px; padding: 4px 0;"
        _heading_style = "color: #1a1a1a; font-weight: bold; font-size: 16px; padding: 4px 0 4px 0;"

        # Left column: text (task, definition, question form)
        left_col = QVBoxLayout()
        root.addLayout(left_col, stretch=3)

        heading = QLabel("Anteroposterior relation: \"strictly in front of\"")
        heading.setStyleSheet(_heading_style)
        left_col.addWidget(heading)

        text1 = QLabel(
            "You will be asked whether one anatomical structure is strictly in front of another "
            "along the front–back (anteroposterior) axis."
        )
        text1.setWordWrap(True)
        text1.setStyleSheet(_text_style)
        left_col.addWidget(text1)

        text2 = QLabel(
            "Definition: one structure is strictly in front of another if the posterior-most point of the first "
            "is anterior to the anterior-most point of the second."
        )
        text2.setWordWrap(True)
        text2.setStyleSheet(_text_style)
        left_col.addWidget(text2)

        q_form_heading = QLabel("What you will be asked")
        q_form_heading.setStyleSheet(_heading_style)
        left_col.addWidget(q_form_heading)

        q_form = QLabel(
            "For each pair of structures you will answer \"Yes\" or \"No\" to:\n"
            "  \"Is the first structure strictly in front of the second?\""
        )
        q_form.setWordWrap(True)
        q_form.setStyleSheet(_text_style)
        left_col.addWidget(q_form)

        left_col.addStretch(1)

        proceed_btn = QPushButton("Proceed")
        proceed_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #007aff; color: white; border: none; border-radius: 8px;
                padding: 10px 22px; font-size: 15px;
            }
            QPushButton:hover { background-color: #5ac8fa; }
            QPushButton:pressed { background-color: #0051d5; }
            """
        )
        proceed_btn.clicked.connect(self.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(proceed_btn)
        left_col.addLayout(btn_row)

        # Right column: textual examples + placeholder images
        right_col = QVBoxLayout()
        root.addLayout(right_col, stretch=4)

        img_height = 260

        placeholder_style = (
            "border: 1px dashed #bbbbbb; border-radius: 8px; margin-top: 6px; "
            "color: #444; font-size: 13px; background: #ffffff;"
        )

        ex1_text = QLabel(
            "Example 1 (Yes): \"Is the sternum strictly in front of the spine?\" → Answer: Yes."
        )
        ex1_text.setWordWrap(True)
        ex1_text.setStyleSheet(_text_style)
        right_col.addWidget(ex1_text)

        ex1_img = QLabel("[Add anteroposterior Yes example image here]")
        ex1_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ex1_img.setFixedHeight(img_height)
        ex1_img.setStyleSheet(placeholder_style)
        right_col.addWidget(ex1_img)

        ex2_text = QLabel(
            "Example 2 (No): \"Is the spine strictly in front of the sternum?\" → Answer: No."
        )
        ex2_text.setWordWrap(True)
        ex2_text.setStyleSheet(_text_style)
        ex2_text.setContentsMargins(0, 12, 0, 0)
        right_col.addWidget(ex2_text)

        ex2_img = QLabel("[Add anteroposterior No example image here]")
        ex2_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ex2_img.setFixedHeight(img_height)
        ex2_img.setStyleSheet(placeholder_style)
        right_col.addWidget(ex2_img)

        source = QLabel("Images in this window are captured from Complete Anatomy (anteroposterior examples to be added).")
        source.setWordWrap(True)
        source.setStyleSheet("color: #555; font-size: 10px; margin-top: 4px;")
        right_col.addWidget(source)


def _relation_verb(axis: str) -> str:
    if axis == AXIS_VERTICAL:
        return "strictly above"
    if axis == AXIS_MEDIOLATERAL:
        return "strictly to the left of"
    return "strictly in front of"


class QueryDialog(QDialog):
    """
    Standalone dialog for expert queries only.
    Clinicians focus on answering questions; no structure input.
    """

    def __init__(
        self,
        poset_builder: PosetBuilder,
        autosave_path: Path,
        axis: str,
        save_callback: Callable[[str, List[Structure], Set[Tuple[int, int]]], None],
    ) -> None:
        super().__init__()
        self.setWindowTitle("Expert Query")
        self.resize(520, 420)
        self.setModal(False)

        self.poset_builder = poset_builder
        self._autosave_path = autosave_path
        self._axis = axis
        self._save_callback = save_callback
        self.pending_pair: Tuple[int, int] | None = None
        self._answer_history: List[Tuple[int, int, bool]] = []

        layout = QVBoxLayout(self)

        # Question card
        self.question_card = QFrame()
        self.question_card.setMinimumHeight(160)
        self.question_card.setStyleSheet(
            """
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 12px;
                padding: 24px;
                margin: 12px 0;
            }
            """
        )
        card_layout = QVBoxLayout(self.question_card)
        self.query_label = QLabel("")
        self.query_label.setWordWrap(True)
        self.query_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.query_label.setStyleSheet(
            "color: #1a1a1a; font-size: 22px; font-weight: 500; line-height: 1.4;"
        )
        card_layout.addWidget(self.query_label)
        layout.addWidget(self.question_card)

        # Back, Yes, No
        btn_row = QHBoxLayout()
        self.back_btn = QPushButton("← Undo")
        self.back_btn.setEnabled(False)
        self.back_btn.setStyleSheet("padding: 10px 16px; font-size: 14px;")
        self.back_btn.clicked.connect(self.go_back_one_question)
        btn_row.addWidget(self.back_btn)

        self.yes_btn = QPushButton("Yes")
        self.yes_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2e7d32; color: white; border: none; border-radius: 8px;
                padding: 14px 28px; font-size: 18px; font-weight: 600;
            }
            QPushButton:hover:enabled { background-color: #388e3c; }
            QPushButton:pressed:enabled { background-color: #1b5e20; }
            QPushButton:disabled { background-color: #bdbdbd; color: #757575; }
            """
        )
        self.yes_btn.clicked.connect(lambda: self.answer_query(True))

        self.no_btn = QPushButton("No")
        self.no_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #c62828; color: white; border: none; border-radius: 8px;
                padding: 14px 28px; font-size: 18px; font-weight: 600;
            }
            QPushButton:hover:enabled { background-color: #d32f2f; }
            QPushButton:pressed:enabled { background-color: #b71c1c; }
            QPushButton:disabled { background-color: #bdbdbd; color: #757575; }
            """
        )
        self.no_btn.clicked.connect(lambda: self.answer_query(False))
        btn_row.addStretch()
        btn_row.addWidget(self.no_btn)
        btn_row.addWidget(self.yes_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Progress bar (no numbers): reflects iteration progress; can jump when answers imply many skips
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: none;
                border-radius: 4px;
                background: #e0e0e0;
            }
            QProgressBar::chunk {
                border-radius: 4px;
                background: #007aff;
            }
            """
        )
        layout.addWidget(self.progress_bar)

        # Finish and Close (shown when done)
        self.finish_btn = QPushButton("Done")
        self.finish_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #007aff; color: white; border: none; border-radius: 8px;
                padding: 12px 24px; font-size: 15px;
            }
            QPushButton:hover { background-color: #5ac8fa; }
            QPushButton:pressed { background-color: #0051d5; }
            """
        )
        self.finish_btn.clicked.connect(self.accept)
        self.finish_btn.hide()
        layout.addWidget(self.finish_btn)

        self._advance_to_next_query()

    def _autosave_poset(self) -> None:
        if not self._autosave_path or not self._save_callback:
            return
        try:
            structures, edges = self.poset_builder.get_final_relations()
            self._save_callback(self._axis, structures, edges)
        except Exception:
            pass

    def _advance_to_next_query(self) -> None:
        pair = self.poset_builder.next_pair()
        self.pending_pair = pair
        if pair is None:
            self._autosave_poset()
            self.query_label.setText(
                "Thank you for your participation!\n\nEnjoy the pizza 🍕"
            )
            self.yes_btn.hide()
            self.no_btn.hide()
            self.back_btn.hide()
            self.finish_btn.show()
            self.progress_bar.setValue(100)
            return
        i, j = pair
        si, sj = self.poset_builder.structures[i], self.poset_builder.structures[j]
        verb = _relation_verb(self._axis)
        self.query_label.setText(f"Is the {si.name} {verb} the {sj.name}?")
        self.progress_bar.setValue(int(self.poset_builder.get_iteration_progress() * 100))

    def answer_query(self, is_above: bool) -> None:
        if self.pending_pair is None:
            return
        i, j = self.pending_pair
        self._answer_history.append((i, j, is_above))
        self.back_btn.setEnabled(True)
        self.poset_builder.record_response(i, j, is_above)
        self._autosave_poset()
        self._advance_to_next_query()

    def go_back_one_question(self) -> None:
        if not self._answer_history:
            return
        last_i, last_j, last_answer = self._answer_history.pop()
        if last_answer:
            self.poset_builder.edges.discard((last_i, last_j))
        self.poset_builder.finished = False
        self.poset_builder.current_gap = last_j - last_i
        self.poset_builder.current_i = last_i + 1
        self.pending_pair = (last_i, last_j)
        si, sj = self.poset_builder.structures[last_i], self.poset_builder.structures[last_j]
        verb = _relation_verb(self._axis)
        self.query_label.setText(f"(Correcting) Is the {si.name} {verb} the {sj.name}?")
        self.yes_btn.setEnabled(True)
        self.no_btn.setEnabled(True)
        if not self._answer_history:
            self.back_btn.setEnabled(False)
        self.progress_bar.setValue(int(self.poset_builder.get_iteration_progress() * 100))
        self._autosave_poset()

class MainWindow(QMainWindow):
    def __init__(self, input_path: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("Anatomical Poset Builder")
        self.resize(520, 500)

        # Remember optional input path for use during UI setup
        self._input_path: Optional[str] = input_path
        # Where to auto-save; set after we know the actual load path in _init_ui
        output_dir = Path(__file__).resolve().parent / "Output_constructed_posets"
        self._autosave_path: Optional[Path] = output_dir / "poset_autosave.json"

        self.poset_builder: PosetBuilder | None = None
        self._viewer_windows: List[QWidget] = []
        self._edges_vertical: Set[Tuple[int, int]] = set()
        self._edges_mediolateral: Set[Tuple[int, int]] = set()
        self._edges_anteroposterior: Set[Tuple[int, int]] = set()

        self._init_ui()

    def _init_ui(self) -> None:
        central = QWidget()
        root_layout = QVBoxLayout(central)

        # Structure definition only (no query UI until Start is clicked)
        left_group = QGroupBox("Anatomical Structures (Input)")
        left_layout = QVBoxLayout(left_group)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([
            "Name",
            "CoM vertical",
            "CoM mediolateral",
            "CoM anteroposterior",
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        left_layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load Structures")
        load_btn.clicked.connect(self.load_structures_dialog)
        add_row_btn = QPushButton("+ Add Structure")
        add_row_btn.clicked.connect(self.add_structure_row)
        remove_row_btn = QPushButton("− Remove Selected")
        remove_row_btn.clicked.connect(self.remove_selected_row)
        view_btn = QPushButton("View Posets")
        view_btn.clicked.connect(self._open_viewer)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(add_row_btn)
        btn_row.addWidget(remove_row_btn)
        btn_row.addWidget(view_btn)
        left_layout.addLayout(btn_row)

        # Axis choice: run vertical, mediolateral, or anteroposterior poset construction
        axis_group = QGroupBox("Axis for this run:")
        axis_layout = QVBoxLayout(axis_group)
        self.axis_vertical_rb = QRadioButton("Vertical (top–bottom, superior–inferior) — \"strictly above\"")
        self.axis_vertical_rb.setChecked(True)
        self.axis_frontal_rb = QRadioButton("Mediolateral (left–right, patient's view) — \"strictly to the left of\"")
        self.axis_ap_rb = QRadioButton("Anteroposterior (front–back) — \"strictly in front of\"")
        axis_layout.addWidget(self.axis_vertical_rb)
        axis_layout.addWidget(self.axis_frontal_rb)
        axis_layout.addWidget(self.axis_ap_rb)
        left_layout.addWidget(axis_group)

        self.start_btn = QPushButton("▶  Start Poset Construction")
        self.start_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #007aff;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-size: 15px;
            }
            QPushButton:hover { background-color: #5ac8fa; }
            QPushButton:pressed { background-color: #0051d5; }
            QPushButton:disabled { background-color: #bdbdbd; color: #757575; }
            """
        )
        self.start_btn.clicked.connect(self.start_poset_construction)
        left_layout.addWidget(self.start_btn)

        root_layout.addWidget(left_group)

        self.setCentralWidget(central)

        # Load structures from file: CLI arg, or default test_structures.json
        load_path = self._input_path
        if load_path is None:
            default_file = Path(__file__).resolve().parent / "Input_CoM_structures" / "test_structures.json"
            if default_file.exists():
                load_path = str(default_file)
        if load_path is not None:
            self._autosave_path = self._builtposet_output_path(Path(load_path))
            try:
                structures = load_structures_from_json(load_path)
                self._edges_vertical = set()
                self._edges_anteroposterior = set()
                self._edges_mediolateral = set()
                for s in structures:
                    self.add_structure_row(
                        s.name,
                        str(s.com_vertical),
                        str(s.com_lateral),
                        str(s.com_anteroposterior),
                    )
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self,
                    "Failed to load input",
                    f"Could not load structures from:\n{load_path}\n\n{exc}",
                )

    def _builtposet_output_path(self, input_path: Path) -> Path:
        """Autosave goes to Output_constructed_posets folder, not the input folder."""
        output_dir = Path(__file__).resolve().parent / "Output_constructed_posets"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{input_path.stem}.poset_autosave.json"

    def _on_poset_autosave(
        self, axis: str, structures: List[Structure], edges: Set[Tuple[int, int]]
    ) -> None:
        """Called by QueryDialog on each answer; updates the correct edge set and saves all axes."""
        if not self._autosave_path:
            return
        try:
            self._autosave_path.parent.mkdir(parents=True, exist_ok=True)
            if axis == AXIS_VERTICAL:
                self._edges_vertical = edges
            elif axis == AXIS_MEDIOLATERAL:
                self._edges_mediolateral = edges
            else:
                self._edges_anteroposterior = edges
            save_poset_to_json(
                str(self._autosave_path),
                structures,
                self._edges_vertical,
                self._edges_mediolateral,
                self._edges_anteroposterior,
            )
        except Exception:
            pass

    # -------- Structure table helpers -------- #
    def load_structures_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load structures from JSON",
            str(Path(__file__).resolve().parent),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            structures = load_structures_from_json(path)
            self.table.setRowCount(0)
            self._edges_vertical = set()
            self._edges_mediolateral = set()
            self._edges_anteroposterior = set()
            for s in structures:
                self.add_structure_row(
                    s.name,
                    str(s.com_vertical),
                    str(s.com_lateral),
                    str(s.com_anteroposterior),
                )
            self._autosave_path = self._builtposet_output_path(Path(path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Failed to load",
                f"Could not load structures from:\n{path}\n\n{exc}",
            )

    def add_structure_row(
        self,
        name: str = "",
        com_vertical: str = "",
        com_lateral: str = "",
        com_anteroposterior: str = "",
    ) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(name))
        self.table.setItem(row, 1, QTableWidgetItem(com_vertical))
        self.table.setItem(row, 2, QTableWidgetItem(com_lateral))
        self.table.setItem(row, 3, QTableWidgetItem(com_anteroposterior))

    def remove_selected_row(self) -> None:
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        for row in sorted(rows, reverse=True):
            self.table.removeRow(row)

    def _collect_structures(self) -> List[Structure] | None:
        structures: List[Structure] = []

        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            com_v_item = self.table.item(row, 1)
            com_l_item = self.table.item(row, 2)
            com_ap_item = self.table.item(row, 3)
            if not name_item:
                continue
            name = name_item.text().strip()
            com_v_text = (com_v_item.text() if com_v_item else "").strip()
            com_l_text = (com_l_item.text() if com_l_item else "").strip()
            com_ap_text = (com_ap_item.text() if com_ap_item else "").strip()
            if not name:
                continue
            if not com_v_text:
                QMessageBox.warning(
                    self,
                    "Invalid Input",
                    f"Row {row + 1}: CoM vertical is required.",
                )
                return None
            try:
                com_vertical = float(com_v_text)
                com_lateral = float(com_l_text) if com_l_text else 0.0
                com_ap = float(com_ap_text) if com_ap_text else 0.0
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Invalid Input",
                    f"Row {row + 1}: CoM values must be numbers.",
                )
                return None
            structures.append(
                Structure(
                    name=name,
                    com_vertical=com_vertical,
                    com_lateral=com_lateral,
                    com_anteroposterior=com_ap,
                )
            )

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

        if self.axis_vertical_rb.isChecked():
            axis = AXIS_VERTICAL
        elif self.axis_frontal_rb.isChecked():
            axis = AXIS_MEDIOLATERAL
        else:
            axis = AXIS_ANTERIOR_POSTERIOR
        self.poset_builder = PosetBuilder(structures, axis=axis)
        self.start_btn.setEnabled(False)

        # 1) Generic welcome/instructions window (always shown)
        welcome_dialog = DefinitionDialog(axis=axis)
        if welcome_dialog.exec() != QDialog.DialogCode.Accepted:
            self.start_btn.setEnabled(True)
            return

        # 2) Axis-specific definition window
        if axis == AXIS_VERTICAL:
            axis_dialog: QDialog = VerticalDefinitionDialog()
        elif axis == AXIS_MEDIOLATERAL:
            axis_dialog = MediolateralDefinitionDialog()
        else:
            axis_dialog = AnteroposteriorDefinitionDialog()
        if axis_dialog.exec() != QDialog.DialogCode.Accepted:
            self.start_btn.setEnabled(True)
            return

        # 3) Start the query dialog
        query_dialog = QueryDialog(
            self.poset_builder,
            self._autosave_path,
            axis=axis,
            save_callback=self._on_poset_autosave,
        )
        query_dialog.finished.connect(lambda: self.start_btn.setEnabled(True))
        query_dialog.show()

    def _open_viewer(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose poset file to view",
            str(Path(__file__).resolve().parent),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            win = PosetViewerWindow(path)
            win.setWindowFlags(Qt.Window)
            self._viewer_windows.append(win)
            win.show()
            win.raise_()
            win.activateWindow()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Failed to open viewer",
                f"Could not open viewer:\n{exc}",
            )


def main() -> None:
    """
    Optional usage:
      python anatomy_poset_gui.py path/to/structures.json

    Where structures.json has:
    {
      "structures": [
        {"name": "Skull", "com_vertical": 90.0, "com_lateral": 0.0},
        ...
      ]
    }
    com_lateral is optional (default 0).
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

