import pytest

from src.anatomy_poset.core.models import (
    AXIS_VERTICAL,
    AXIS_MEDIOLATERAL,
    Structure,
)
from src.anatomy_poset.core.builder import PosetBuilder

# Helper to create basic structures for testing
def create_mock_structures():
    return [
        Structure("Skull", com_vertical=90.0, com_lateral=50.0, com_anteroposterior=50.0),
        Structure("Thorax", com_vertical=70.0, com_lateral=50.0, com_anteroposterior=50.0),
        Structure("Pelvis", com_vertical=40.0, com_lateral=50.0, com_anteroposterior=50.0),
        Structure("Femur", com_vertical=20.0, com_lateral=50.0, com_anteroposterior=50.0),
    ]

def test_poset_builder_sorting():
    """Test that the builder sorts structures descending by the chosen axis CoM."""
    structures = [
        Structure("Pelvis", com_vertical=40.0, com_lateral=50.0, com_anteroposterior=0.0),
        Structure("Skull", com_vertical=90.0, com_lateral=50.0, com_anteroposterior=0.0),
    ]
    builder = PosetBuilder(structures, axis=AXIS_VERTICAL)
    
    # Skull (90) should come before Pelvis (40)
    assert builder.structures[0].name == "Skull"
    assert builder.structures[1].name == "Pelvis"

def test_path_exists_transitivity():
    """Test the graph traversal path_exists function."""
    builder = PosetBuilder(create_mock_structures(), axis=AXIS_VERTICAL)
    
    # Add explicit edges: 0->1 (Skull->Thorax) and 1->2 (Thorax->Pelvis)
    builder.edges.add((0, 1))
    builder.edges.add((1, 2))
    
    # Path from 0 to 2 should exist due to transitivity
    assert builder.path_exists(0, 2) is True
    # Path backward should not exist
    assert builder.path_exists(2, 0) is False

def test_edge_redundancy_reduction():
    """Test that direct edges implied by transitivity are removed (Hasse diagram)."""
    builder = PosetBuilder(create_mock_structures(), axis=AXIS_VERTICAL)
    
    # 0->1, 1->2, and a redundant direct edge 0->2
    builder.edges.update([(0, 1), (1, 2), (0, 2)])
    
    reduced_edges = builder.edge_redundancy_reduction()
    
    # 0->2 should be gone
    assert (0, 1) in reduced_edges
    assert (1, 2) in reduced_edges
    assert (0, 2) not in reduced_edges

def test_symmetry_vertical_axis():
    """Test that answering a query for 'Left X' applies to 'Right X' automatically."""
    structures = [
        Structure("Skull", com_vertical=90.0, com_lateral=50.0, com_anteroposterior=0.0),
        Structure("Left Femur", com_vertical=20.0, com_lateral=80.0, com_anteroposterior=0.0),
        Structure("Right Femur", com_vertical=20.0, com_lateral=20.0, com_anteroposterior=0.0),
    ]
    builder = PosetBuilder(structures, axis=AXIS_VERTICAL)
    
    # Indices after sorting by vertical: 0=Skull, 1=Left Femur, 2=Right Femur
    # Record answer: Skull (0) is above Left Femur (1)
    builder.record_response(0, 1, is_above=True)
    
    # It should have automatically added Skull (0) above Right Femur (2)
    assert (0, 2) in builder.edges

def test_gap_iteration_skips_implied_relations():
    """Test that the next_pair query skips pairs if their relation is already implied."""
    builder = PosetBuilder(create_mock_structures(), axis=AXIS_VERTICAL)
    # Sorted: 0=Skull, 1=Thorax, 2=Pelvis, 3=Femur
    
    # We manually add edges to simulate previous answers
    builder.edges.add((0, 1)) # Skull > Thorax
    builder.edges.add((1, 2)) # Thorax > Pelvis
    
    # We are looking for gap 2 (e.g., comparing 0 and 2)
    builder.current_gap = 2
    builder.current_i = 0
    
    # next_pair() should skip (0, 2) because path_exists(0, 2) is True.
    # It should immediately jump to (1, 3) which is the next pair at gap 2.
    pair = builder.next_pair()
    assert pair == (1, 3)