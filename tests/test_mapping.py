import numpy as np
import sys
import os

# Add project root to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from area_map import AreaMap
from behaviors.mapping_scan_algorithm import MappingScanAlgorithm
from drone import Drone


class TestAreaMapInit:
    """Tests for AreaMap initialization and grid dimensions."""

    def test_default_dimensions(self):
        """Grid dimensions should match width/resolution × height/resolution."""
        am = AreaMap(width=100, height=100, resolution=1.0)
        assert am.grid_width == 100
        assert am.grid_height == 100

    def test_custom_resolution(self):
        """Higher resolution means smaller cells and more grid cells."""
        am = AreaMap(width=50, height=30, resolution=0.5)
        assert am.grid_width == 100  # 50 / 0.5
        assert am.grid_height == 60   # 30 / 0.5

    def test_discovered_map_starts_unmapped(self):
        """All cells in the discovered map should start as UNMAPPED (-1)."""
        am = AreaMap(width=20, height=20, resolution=1.0)
        assert np.all(am.discovered_map == AreaMap.UNMAPPED)


class TestAreaMapTerrain:
    """Tests for terrain generation."""

    def test_terrain_values_in_range(self):
        """Ground truth values should be 0-5 (elevation) or 8 (obstacle)."""
        am = AreaMap(width=50, height=50, resolution=1.0)
        unique_vals = set(np.unique(am.ground_truth))
        valid_vals = {0, 1, 2, 3, 4, 5, 8}
        assert unique_vals.issubset(valid_vals), f"Unexpected values: {unique_vals - valid_vals}"

    def test_seeded_terrain_is_reproducible(self):
        """Using the same seed should produce the same terrain."""
        am1 = AreaMap(width=30, height=30, resolution=1.0)
        am1.generate_terrain(seed=42)
        am2 = AreaMap(width=30, height=30, resolution=1.0)
        am2.generate_terrain(seed=42)
        assert np.array_equal(am1.ground_truth, am2.ground_truth)

    def test_terrain_has_obstacles(self):
        """Generated terrain should contain at least some obstacles."""
        am = AreaMap(width=50, height=50, resolution=1.0)
        am.generate_terrain(obstacle_density=0.10, seed=7)
        assert np.any(am.ground_truth == AreaMap.OBSTACLE)


class TestAreaMapScanning:
    """Tests for cell scanning and coverage."""

    def test_scan_cell_returns_ground_truth(self):
        """scan_cell should return the ground truth value."""
        am = AreaMap(width=10, height=10, resolution=1.0)
        val = am.scan_cell(0, 0)
        assert val == am.ground_truth[0, 0]

    def test_update_cell_updates_discovered_map(self):
        """update_cell should mark a cell in the discovered map."""
        am = AreaMap(width=10, height=10, resolution=1.0)
        am.update_cell(3, 3, 2.0)
        assert am.discovered_map[3, 3] == 2.0

    def test_scan_area_discovers_cells(self):
        """scan_area should discover cells around the given position."""
        am = AreaMap(width=20, height=20, resolution=1.0)
        newly = am.scan_area(10.0, 10.0, radius=2)
        assert newly > 0
        # The center cell should now not be UNMAPPED
        result = am.world_to_grid(10.0, 10.0)
        assert result is not None
        col, row = result
        assert am.discovered_map[row, col] != AreaMap.UNMAPPED

    def test_coverage_starts_at_zero(self):
        """Coverage should be 0% initially."""
        am = AreaMap(width=10, height=10, resolution=1.0)
        assert am.get_coverage_percent() == 0.0

    def test_coverage_increases_after_scan(self):
        """Coverage should increase after scanning cells."""
        am = AreaMap(width=10, height=10, resolution=1.0)
        am.scan_area(5.0, 5.0, radius=1)
        assert am.get_coverage_percent() > 0.0

    def test_full_scan_reaches_100(self):
        """Scanning every cell should give 100% coverage."""
        am = AreaMap(width=5, height=5, resolution=1.0)
        # Scan every cell
        for r in range(am.grid_height):
            for c in range(am.grid_width):
                val = am.scan_cell(c, r)
                am.update_cell(c, r, val)
        assert am.get_coverage_percent() == 100.0


class TestAreaMapCoordinates:
    """Tests for coordinate conversion."""

    def test_world_to_grid(self):
        """World coordinates should map to correct grid indices."""
        am = AreaMap(width=100, height=100, resolution=1.0)
        result = am.world_to_grid(5.5, 10.3)
        assert result == (5, 10)

    def test_world_to_grid_out_of_bounds(self):
        """Out-of-bounds world coordinates should return None."""
        am = AreaMap(width=10, height=10, resolution=1.0)
        assert am.world_to_grid(-1.0, 5.0) is None
        assert am.world_to_grid(5.0, 15.0) is None

    def test_grid_to_world_center(self):
        """Grid-to-world should return center of cell."""
        am = AreaMap(width=100, height=100, resolution=1.0)
        x, y = am.grid_to_world(0, 0)
        assert x == 0.5
        assert y == 0.5


class TestMappingScanAlgorithm:
    """Tests for the mapping scan algorithm."""

    def test_waypoints_generated_for_all_drones(self):
        """Waypoints should be generated for each drone."""
        am = AreaMap(width=50, height=50, resolution=1.0)
        alg = MappingScanAlgorithm(am, num_drones=5)
        for i in range(5):
            assert i in alg.waypoints
            assert len(alg.waypoints[i]) > 0

    def test_all_drones_start_in_scanning_phase(self):
        """All drones should start in the SCANNING phase."""
        am = AreaMap(width=50, height=50, resolution=1.0)
        alg = MappingScanAlgorithm(am, num_drones=3)
        for i in range(3):
            assert alg.get_phase(i) == MappingScanAlgorithm.PHASE_SCANNING

    def test_apply_moves_drone_toward_waypoint(self):
        """apply() should move the drone closer to its waypoint."""
        am = AreaMap(width=50, height=50, resolution=1.0)
        alg = MappingScanAlgorithm(am, num_drones=1, origin=[0, 0, 5])
        drone = Drone(np.array([0.0, 0.0, 5.0]), 0)

        target = alg.get_current_waypoint(0)
        initial_distance = np.linalg.norm(drone.position - target)

        new_pos = alg.apply(drone, [], drone.position.copy())
        new_distance = np.linalg.norm(new_pos - target)

        assert new_distance < initial_distance, "Drone should move closer to waypoint"

    def test_progress_reports_coverage(self):
        """get_progress() should include coverage info."""
        am = AreaMap(width=10, height=10, resolution=1.0)
        alg = MappingScanAlgorithm(am, num_drones=2)
        progress = alg.get_progress()
        assert "coverage_percent" in progress
        assert "drones_active" in progress
        assert "drones_done" in progress

    def test_last_waypoint_is_origin(self):
        """The final waypoint for each drone should be the origin (return to base)."""
        am = AreaMap(width=50, height=50, resolution=1.0)
        origin = np.array([0.0, 0.0, 5.0])
        alg = MappingScanAlgorithm(am, num_drones=3, origin=origin)
        for i in range(3):
            last_wp = alg.waypoints[i][-1]
            assert np.allclose(last_wp, origin), f"Drone {i} last waypoint {last_wp} != origin {origin}"


class TestDroneWaypoints:
    """Tests for drone waypoint extensions."""

    def test_drone_has_waypoint_attributes(self):
        """Drone should have waypoint tracking attributes."""
        d = Drone(np.array([0.0, 0.0, 0.0]), 0)
        assert hasattr(d, 'waypoints')
        assert hasattr(d, 'current_waypoint_idx')
        assert hasattr(d, 'sensor_data')

    def test_has_reached_waypoint_no_waypoints(self):
        """With no waypoints, has_reached_waypoint should return True."""
        d = Drone(np.array([0.0, 0.0, 0.0]), 0)
        assert d.has_reached_waypoint()

    def test_advance_waypoint(self):
        """advance_waypoint should increment the index."""
        d = Drone(np.array([0.0, 0.0, 0.0]), 0)
        d.waypoints = [np.array([1, 0, 0]), np.array([2, 0, 0])]
        d.current_waypoint_idx = 0
        d.advance_waypoint()
        assert d.current_waypoint_idx == 1
