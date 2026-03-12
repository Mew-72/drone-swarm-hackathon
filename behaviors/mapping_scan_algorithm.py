import numpy as np


class MappingScanAlgorithm:
    """
    Coordinates drones in a boustrophedon (lawnmower) sweep pattern to map an area.

    All drones start from a shared origin point and spread out to their assigned
    parallel lanes. Collision avoidance is handled by the separate
    CollisionAvoidanceAlgorithm that runs alongside this behavior.

    Phases:
        1. DISPERSE — drones move from origin to their lane starting positions
        2. SCAN     — drones sweep back and forth along their assigned lanes
        3. DONE     — drone has completed all waypoints
    """

    PHASE_DISPERSE = "disperse"
    PHASE_SCAN = "scan"
    PHASE_DONE = "done"

    def __init__(self, area_map, num_drones, origin=None, scan_radius=2,
                 speed=0.3, altitude=5.0):
        """
        Initialize the mapping scan algorithm.

        Parameters:
            area_map (AreaMap): The area map to scan.
            num_drones (int): Number of drones in the swarm.
            origin (array-like or None): Shared starting point for all drones.
            scan_radius (int): Number of cells scanned around the drone per step.
            speed (float): Step size factor when moving toward waypoints.
            altitude (float): Fixed altitude (z-coordinate) for mapping flight.
        """
        self.area_map = area_map
        self.num_drones = num_drones
        self.origin = np.array(origin) if origin is not None else np.array([0.0, 0.0, altitude])
        self.scan_radius = scan_radius
        self.speed = speed
        self.altitude = altitude

        # Per-drone state
        self.waypoints = {}        # drone_index -> list of 3D waypoints
        self.waypoint_idx = {}     # drone_index -> current waypoint index
        self.phases = {}           # drone_index -> current phase
        self.arrival_threshold = 1.0  # distance threshold to consider "arrived"

        # Generate waypoints for each drone
        self._generate_waypoints(num_drones)

    def _generate_waypoints(self, num_drones):
        """
        Generate boustrophedon sweep waypoints for each drone.

        The area is divided into parallel vertical lanes. Each drone gets one
        or more lanes. Waypoints start from the lane entry point and zigzag
        top-to-bottom, bottom-to-top across the lane.
        """
        area_w = self.area_map.width
        area_h = self.area_map.height
        z = self.altitude

        # Divide area into lanes, one per drone
        lane_width = area_w / max(num_drones, 1)

        for i in range(num_drones):
            # Lane center x-coordinate
            lane_x = (i + 0.5) * lane_width

            # Build waypoints: zigzag up and down the lane
            waypoints = []

            # Determine sweep step (how far drone moves per waypoint along y)
            sweep_step = self.scan_radius * self.area_map.resolution * 2

            if i % 2 == 0:
                # Even-indexed drones: sweep bottom to top first
                y_positions = np.arange(0, area_h, sweep_step)
            else:
                # Odd-indexed drones: sweep top to bottom first
                y_positions = np.arange(area_h, 0, -sweep_step)

            for y in y_positions:
                waypoints.append(np.array([lane_x, float(y), z]))

            # If the lane needs a return pass (boustrophedon), reverse and append
            if i % 2 == 0:
                return_y = np.arange(area_h, 0, -sweep_step)
            else:
                return_y = np.arange(0, area_h, sweep_step)

            # Offset by half a sweep step for the return pass
            offset_x = lane_width * 0.3
            for y in return_y:
                waypoints.append(np.array([lane_x + offset_x, float(y), z]))

            self.waypoints[i] = waypoints
            self.waypoint_idx[i] = 0
            self.phases[i] = self.PHASE_DISPERSE

    def get_origin(self):
        """Returns the shared origin point."""
        return self.origin.copy()

    def get_current_waypoint(self, drone_index):
        """
        Get the current target waypoint for a drone.

        Parameters:
            drone_index (int): Index of the drone.

        Returns:
            numpy array or None: Current waypoint, or None if done.
        """
        if drone_index not in self.waypoints:
            return None
        idx = self.waypoint_idx.get(drone_index, 0)
        wps = self.waypoints[drone_index]
        if idx < len(wps):
            return wps[idx].copy()
        return None

    def get_phase(self, drone_index):
        """Returns the current phase for a drone."""
        return self.phases.get(drone_index, self.PHASE_DONE)

    def apply(self, drone, neighbor_positions, current_position):
        """
        Apply the mapping scan behavior to move the drone toward its next waypoint.
        Also triggers area scanning when the drone moves.

        Parameters:
            drone (Drone): The current drone object.
            neighbor_positions (list): Positions of neighboring drones.
            current_position (numpy array): The drone's current position.

        Returns:
            numpy array: The updated position.
        """
        idx = drone.index

        # Initialize state for this drone if not yet tracked
        if idx not in self.phases:
            if idx < self.num_drones and idx in self.waypoints:
                self.phases[idx] = self.PHASE_DISPERSE
                self.waypoint_idx[idx] = 0
            else:
                return current_position

        phase = self.phases[idx]

        if phase == self.PHASE_DONE:
            # Drone has finished scanning; hold position
            return current_position

        # Get target waypoint
        target = self.get_current_waypoint(idx)
        if target is None:
            self.phases[idx] = self.PHASE_DONE
            return current_position

        # Move toward target
        direction = target - current_position
        distance = np.linalg.norm(direction)

        if distance < self.arrival_threshold:
            # Arrived at waypoint — advance
            self.waypoint_idx[idx] += 1

            # Transition from DISPERSE to SCAN after reaching first waypoint
            if phase == self.PHASE_DISPERSE:
                self.phases[idx] = self.PHASE_SCAN

            # Check if all waypoints are done
            if self.waypoint_idx[idx] >= len(self.waypoints[idx]):
                self.phases[idx] = self.PHASE_DONE
                return current_position

            # Get next waypoint
            target = self.get_current_waypoint(idx)
            if target is None:
                self.phases[idx] = self.PHASE_DONE
                return current_position
            direction = target - current_position
            distance = np.linalg.norm(direction)

        # Compute step
        if distance > 0:
            step = direction / distance * min(self.speed, distance)
            new_position = current_position + step
        else:
            new_position = current_position

        # Perform scanning at the new position
        self.area_map.scan_area(
            new_position[0], new_position[1], radius=self.scan_radius
        )

        return new_position

    def is_mapping_complete(self):
        """
        Check if all drones have finished their scan patterns.

        Returns:
            bool: True if all drones are in DONE phase.
        """
        if not self.phases:
            return False
        return all(p == self.PHASE_DONE for p in self.phases.values())

    def get_progress(self):
        """
        Get overall mapping progress.

        Returns:
            dict: Contains 'coverage_percent', 'drones_active', 'drones_done'.
        """
        active = sum(1 for p in self.phases.values() if p != self.PHASE_DONE)
        done = sum(1 for p in self.phases.values() if p == self.PHASE_DONE)
        return {
            "coverage_percent": self.area_map.get_coverage_percent(),
            "drones_active": active,
            "drones_done": done,
        }
