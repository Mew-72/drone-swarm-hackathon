import numpy as np


class MappingScanAlgorithm:
    """
    Coordinates drones in a boustrophedon (lawnmower) sweep pattern to map an area.

    All drones start from a shared origin point. They fan out to their assigned
    lane start positions while scanning, sweep their lane end-to-end, and then
    return to the origin once complete.

    Collision avoidance is handled externally by the CollisionAvoidanceAlgorithm.

    Phases:
        SCANNING  — drone is following sweep waypoints (including initial fan-out)
        RETURNING — drone has finished scanning and is flying back to origin
        DONE      — drone has returned to origin
    """

    PHASE_SCANNING = "scanning"
    PHASE_RETURNING = "returning"
    PHASE_DONE = "done"

    def __init__(self, area_map, num_drones, origin=None, scan_radius=2,
                 speed=0.5, altitude=5.0):
        """
        Initialize the mapping scan algorithm.

        Parameters:
            area_map (AreaMap): The area map to scan.
            num_drones (int): Number of drones in the swarm.
            origin (array-like or None): Shared launch/return point for all drones.
            scan_radius (int): Number of cells scanned around the drone per step.
            speed (float): Movement step size per tick.
            altitude (float): Fixed z-coordinate for mapping flight.
        """
        self.area_map = area_map
        self.num_drones = num_drones
        self.origin = np.array(origin) if origin is not None else np.array([0.0, 0.0, altitude])
        self.scan_radius = scan_radius
        self.speed = speed
        self.altitude = altitude

        # Per-drone tracking
        self.waypoints = {}       # drone_index -> list of 3D waypoints
        self.waypoint_idx = {}    # drone_index -> current waypoint index
        self.phases = {}          # drone_index -> current phase string
        self.arrival_threshold = 1.5

        self._generate_waypoints(num_drones)

    # ------------------------------------------------------------------
    #  Waypoint generation
    # ------------------------------------------------------------------

    def _generate_waypoints(self, num_drones):
        """
        Generate efficient boustrophedon sweep waypoints.

        Strategy:
          - The area is split into vertical lanes (one per drone).
          - Lane width = area_width / num_drones.
          - Each drone's first waypoint is the START of its lane (y=0 edge),
            so it fans out from the origin while scanning along the way.
          - It then sweeps straight up (or down) the lane in steps equal
            to 2 × scan_radius, ensuring full coverage with no gaps.
          - When the drone reaches the far edge, the final waypoint is
            the origin — so it flies home.

        This means:
          • No wasted "disperse" phase — the drone scans as it moves to its lane.
          • No redundant return pass — one pass covers the full lane width.
          • Drones return to base automatically.
        """
        area_w = self.area_map.width
        area_h = self.area_map.height
        z = self.altitude

        lane_width = area_w / max(num_drones, 1)
        sweep_step = max(self.scan_radius * self.area_map.resolution * 2, 1.0)

        for i in range(num_drones):
            lane_x = (i + 0.5) * lane_width  # center of this drone's lane
            waypoints = []

            # --- 1. Sweep the lane bottom-to-top or top-to-bottom ---
            # Alternate direction per drone so neighbours move in opposite
            # directions, reducing collision pressure.
            if i % 2 == 0:
                # Even drones: bottom → top
                y_vals = np.arange(0, area_h + sweep_step, sweep_step)
                y_vals = np.clip(y_vals, 0, area_h)
            else:
                # Odd drones: top → bottom
                y_vals = np.arange(area_h, -sweep_step, -sweep_step)
                y_vals = np.clip(y_vals, 0, area_h)

            for y in y_vals:
                waypoints.append(np.array([lane_x, float(y), z]))

            # --- 2. Return to origin ---
            waypoints.append(self.origin.copy())

            self.waypoints[i] = waypoints
            self.waypoint_idx[i] = 0
            self.phases[i] = self.PHASE_SCANNING

    # ------------------------------------------------------------------
    #  Accessors
    # ------------------------------------------------------------------

    def get_origin(self):
        """Returns the shared origin point."""
        return self.origin.copy()

    def get_current_waypoint(self, drone_index):
        """Returns the current target waypoint for a drone, or None if done."""
        if drone_index not in self.waypoints:
            return None
        idx = self.waypoint_idx.get(drone_index, 0)
        wps = self.waypoints[drone_index]
        if idx < len(wps):
            return wps[idx].copy()
        return None

    def get_phase(self, drone_index):
        """Returns the current phase string for a given drone."""
        return self.phases.get(drone_index, self.PHASE_DONE)

    # ------------------------------------------------------------------
    #  Core behaviour — called once per tick per drone
    # ------------------------------------------------------------------

    def apply(self, drone, neighbor_positions, current_position):
        """
        Move the drone toward its next waypoint and scan the terrain below.

        Parameters:
            drone (Drone): The drone object.
            neighbor_positions (list): Positions of other drones.
            current_position (numpy array): The drone's current 3D position.

        Returns:
            numpy array: Updated 3D position.
        """
        idx = drone.index

        # Lazy-init for drones we haven't seen yet
        if idx not in self.phases:
            if idx < self.num_drones and idx in self.waypoints:
                self.phases[idx] = self.PHASE_SCANNING
                self.waypoint_idx[idx] = 0
            else:
                return current_position

        phase = self.phases[idx]

        if phase == self.PHASE_DONE:
            return current_position  # hold at origin

        # --- Get target waypoint ---
        target = self.get_current_waypoint(idx)
        if target is None:
            self.phases[idx] = self.PHASE_DONE
            return current_position

        # --- Move toward it ---
        direction = target - current_position
        distance = np.linalg.norm(direction)

        if distance < self.arrival_threshold:
            # Reached waypoint — advance index
            self.waypoint_idx[idx] += 1

            if self.waypoint_idx[idx] >= len(self.waypoints[idx]):
                # All waypoints exhausted (including return-to-origin)
                self.phases[idx] = self.PHASE_DONE
                return self.origin.copy()  # snap to origin

            # Is the NEXT waypoint the origin?  → switch to RETURNING
            next_wp = self.waypoints[idx][self.waypoint_idx[idx]]
            if np.allclose(next_wp, self.origin, atol=0.5):
                self.phases[idx] = self.PHASE_RETURNING

            target = next_wp
            direction = target - current_position
            distance = np.linalg.norm(direction)

        if distance > 0:
            step = direction / distance * min(self.speed, distance)
            new_position = current_position + step
        else:
            new_position = current_position

        # --- Scan terrain while moving (except while returning) ---
        if self.phases[idx] == self.PHASE_SCANNING:
            self.area_map.scan_area(
                new_position[0], new_position[1], radius=self.scan_radius
            )

        return new_position

    # ------------------------------------------------------------------
    #  Status helpers
    # ------------------------------------------------------------------

    def is_mapping_complete(self):
        """True when every drone is in DONE phase."""
        if not self.phases:
            return False
        return all(p == self.PHASE_DONE for p in self.phases.values())

    def get_progress(self):
        """Returns a dict with coverage %, active count, and done count."""
        active = sum(1 for p in self.phases.values() if p != self.PHASE_DONE)
        done = sum(1 for p in self.phases.values() if p == self.PHASE_DONE)
        return {
            "coverage_percent": self.area_map.get_coverage_percent(),
            "drones_active": active,
            "drones_done": done,
        }
