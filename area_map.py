import numpy as np


class AreaMap:
    """
    Represents a rectangular area to be mapped by drones.
    
    Contains both a 'ground truth' terrain (procedurally generated)
    and a 'discovered' map that starts empty and fills in as drones scan.
    
    Terrain values:
        -1  = unmapped (discovered map only)
         0  = clear / passable ground
         1-5 = elevation levels
         8  = obstacle (building, tree, rock, etc.)
    """

    UNMAPPED = -1
    CLEAR = 0
    OBSTACLE = 8

    def __init__(self, width=100, height=100, resolution=1.0):
        """
        Initialize the area map.

        Parameters:
            width (float): Width of the area in world units.
            height (float): Height of the area in world units.
            resolution (float): Size of each grid cell in world units.
        """
        self.width = width
        self.height = height
        self.resolution = resolution

        # Grid dimensions
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)

        # Ground truth terrain (generated once)
        self.ground_truth = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

        # Discovered map (starts as all unmapped)
        self.discovered_map = np.full(
            (self.grid_height, self.grid_width), self.UNMAPPED, dtype=np.float32
        )

        # Generate procedural terrain
        self.generate_terrain()

    def generate_terrain(self, obstacle_density=0.10, num_clusters=12, seed=None):
        """
        Procedurally generate terrain with elevation and obstacles.

        Uses Perlin-like smooth noise for elevation and clustered random
        placement for obstacles (buildings, trees, etc.).

        Parameters:
            obstacle_density (float): Fraction of cells that are obstacles (0.0 - 1.0).
            num_clusters (int): Number of obstacle clusters to place.
            seed (int or None): Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)

        h, w = self.grid_height, self.grid_width

        # --- Smooth elevation using layered random noise ---
        elevation = np.zeros((h, w), dtype=np.float32)
        # Sum of scaled random fields at different resolutions for smooth hills
        for scale in [4, 8, 16, 32]:
            small = np.random.rand(max(1, h // scale), max(1, w // scale)).astype(np.float32)
            # Resize to full grid using bilinear-like interpolation (numpy only)
            elevation += self._upsample(small, h, w) / scale

        # Normalize elevation to 0-5 range
        e_min, e_max = elevation.min(), elevation.max()
        if e_max > e_min:
            elevation = (elevation - e_min) / (e_max - e_min) * 5.0
        elevation = np.clip(np.round(elevation), 0, 5).astype(np.float32)

        self.ground_truth = elevation

        # --- Place obstacle clusters ---
        num_obstacle_cells = int(h * w * obstacle_density)
        placed = 0
        for _ in range(num_clusters):
            # Random cluster center
            cy, cx = np.random.randint(0, h), np.random.randint(0, w)
            cluster_size = num_obstacle_cells // num_clusters
            for __ in range(cluster_size):
                # Gaussian spread around center
                oy = int(cy + np.random.randn() * (h * 0.06))
                ox = int(cx + np.random.randn() * (w * 0.06))
                if 0 <= oy < h and 0 <= ox < w:
                    self.ground_truth[oy, ox] = self.OBSTACLE
                    placed += 1
                    if placed >= num_obstacle_cells:
                        break
            if placed >= num_obstacle_cells:
                break

    def _upsample(self, small, target_h, target_w):
        """
        Upsample a small 2D array to (target_h, target_w) using nearest-neighbor
        stretched indexing (numpy-only, no scipy/PIL).
        """
        sh, sw = small.shape
        row_idx = (np.arange(target_h) * sh // target_h).clip(0, sh - 1)
        col_idx = (np.arange(target_w) * sw // target_w).clip(0, sw - 1)
        return small[np.ix_(row_idx, col_idx)]

    def world_to_grid(self, x, y):
        """
        Convert world coordinates to grid indices.

        Parameters:
            x (float): World x-coordinate.
            y (float): World y-coordinate.

        Returns:
            (int, int): Grid column and row indices, or None if out of bounds.
        """
        if np.isnan(x) or np.isnan(y):
            return None
        col = int(x / self.resolution)
        row = int(y / self.resolution)
        if 0 <= row < self.grid_height and 0 <= col < self.grid_width:
            return col, row
        return None

    def grid_to_world(self, col, row):
        """
        Convert grid indices to world coordinates (center of cell).

        Parameters:
            col (int): Grid column index.
            row (int): Grid row index.

        Returns:
            (float, float): World x, y coordinates.
        """
        x = (col + 0.5) * self.resolution
        y = (row + 0.5) * self.resolution
        return x, y

    def scan_cell(self, col, row):
        """
        Read the ground truth terrain value at grid coordinates.

        Parameters:
            col (int): Grid column index.
            row (int): Grid row index.

        Returns:
            float: Terrain value at (col, row), or None if out of bounds.
        """
        if 0 <= row < self.grid_height and 0 <= col < self.grid_width:
            return self.ground_truth[row, col]
        return None

    def update_cell(self, col, row, value):
        """
        Mark a cell as scanned in the discovered map.

        Parameters:
            col (int): Grid column index.
            row (int): Grid row index.
            value (float): Terrain value to record.
        """
        if 0 <= row < self.grid_height and 0 <= col < self.grid_width:
            self.discovered_map[row, col] = value

    def scan_area(self, world_x, world_y, radius=1):
        """
        Scan all cells within a given radius around a world position.
        Updates the discovered map with ground truth values.

        Parameters:
            world_x (float): World x-coordinate (drone position).
            world_y (float): World y-coordinate (drone position).
            radius (int): Number of cells around center to scan.

        Returns:
            int: Number of newly discovered cells.
        """
        result = self.world_to_grid(world_x, world_y)
        if result is None:
            return 0

        center_col, center_row = result
        newly_discovered = 0

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = center_row + dr, center_col + dc
                if 0 <= r < self.grid_height and 0 <= c < self.grid_width:
                    if self.discovered_map[r, c] == self.UNMAPPED:
                        value = self.ground_truth[r, c]
                        self.discovered_map[r, c] = value
                        newly_discovered += 1

        return newly_discovered

    def get_coverage_percent(self):
        """
        Calculate the percentage of the area that has been scanned.

        Returns:
            float: Coverage percentage (0.0 - 100.0).
        """
        total_cells = self.grid_height * self.grid_width
        scanned_cells = np.sum(self.discovered_map != self.UNMAPPED)
        return (scanned_cells / total_cells) * 100.0

    def get_discovered_map(self):
        """Returns a copy of the discovered map array."""
        return self.discovered_map.copy()

    def get_ground_truth(self):
        """Returns a copy of the ground truth terrain array."""
        return self.ground_truth.copy()
