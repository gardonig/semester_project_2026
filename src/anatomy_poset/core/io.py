import json
from dataclasses import asdict
from typing import List, Set, Tuple

from .models import Structure

def load_structures_from_json(path: str) -> List[Structure]:
    """
    Load a list of structures from a JSON file.
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