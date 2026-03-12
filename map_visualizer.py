import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image


class MapVisualizer:
    """
    Renders the discovered area map as a 2D tactical heatmap.

    Uses a military-style color scheme:
        - Dark gray   = unmapped
        - Green       = clear ground (elevation 0)
        - Yellow/tan  = low elevation (1-2)
        - Brown       = mid elevation (3-4)
        - Dark brown  = high elevation (5)
        - Red/black   = obstacle
    
    Drone positions are overlaid as blue triangles.
    """

    # Custom colormap boundaries: unmapped, clear, elev 1-5, obstacle
    TERRAIN_COLORS = [
        (0.25, 0.25, 0.25),  # -1 : unmapped (dark gray)
        (0.30, 0.69, 0.31),  #  0 : clear (green)
        (0.55, 0.76, 0.29),  #  1 : slight elevation (yellow-green)
        (0.80, 0.73, 0.36),  #  2 : low elevation (tan)
        (0.72, 0.53, 0.26),  #  3 : mid elevation (brown)
        (0.55, 0.36, 0.17),  #  4 : mid-high elevation (darker brown)
        (0.38, 0.24, 0.12),  #  5 : high elevation (dark brown)
        (0.80, 0.12, 0.12),  #  8 : obstacle (red)
    ]

    # Values that map to each color
    TERRAIN_VALUES = [-1, 0, 1, 2, 3, 4, 5, 8]

    def __init__(self, area_map, figsize=(8, 8)):
        """
        Initialize the map visualizer.

        Parameters:
            area_map (AreaMap): The area map to visualize.
            figsize (tuple): Figure size in inches.
        """
        self.area_map = area_map
        self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
        self.fig.patch.set_facecolor('#1a1a2e')

        # Build the custom colormap
        self.cmap, self.norm = self._build_colormap()

        # Initial map image
        self.map_image = self.ax.imshow(
            area_map.discovered_map,
            cmap=self.cmap,
            norm=self.norm,
            origin='lower',
            extent=[0, area_map.width, 0, area_map.height],
            interpolation='nearest'
        )

        # Drone position scatter (empty initially)
        self.drone_scatter = self.ax.scatter(
            [], [], marker='^', c='#00d4ff', s=30,
            edgecolors='white', linewidths=0.5, zorder=5,
            label='Drones'
        )

        # Coverage text
        self.coverage_text = self.ax.text(
            0.02, 0.98, 'Coverage: 0.0%',
            transform=self.ax.transAxes,
            color='#00ff88', fontsize=12, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d0d1a',
                      edgecolor='#00ff88', alpha=0.8)
        )

        # Style the axes
        self.ax.set_xlabel('X (meters)', color='white', fontsize=10)
        self.ax.set_ylabel('Y (meters)', color='white', fontsize=10)
        self.ax.set_title('DRONE SWARM — AREA MAPPING',
                          color='#00ff88', fontsize=14, fontweight='bold',
                          pad=15)
        self.ax.set_facecolor('#0d0d1a')
        self.ax.tick_params(colors='white', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color('#333355')

        # Add grid
        self.ax.grid(True, alpha=0.15, color='white', linestyle='--')

        self.fig.tight_layout()

    def _build_colormap(self):
        """
        Build a custom discrete colormap for terrain values.

        Returns:
            (colormap, norm): Matplotlib colormap and normalization.
        """
        # Map terrain values to sequential indices
        boundaries = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 8.5]
        norm = mcolors.BoundaryNorm(boundaries, len(self.TERRAIN_COLORS))
        cmap = mcolors.ListedColormap(self.TERRAIN_COLORS)
        return cmap, norm

    def update(self, drone_positions=None):
        """
        Refresh the map visualization.

        Parameters:
            drone_positions (list of array-like or None): 3D positions of drones.
        """
        # Update map data
        self.map_image.set_data(self.area_map.discovered_map)

        # Update drone positions
        if drone_positions is not None and len(drone_positions) > 0:
            positions = np.array(drone_positions)
            self.drone_scatter.set_offsets(positions[:, :2])  # x, y only
        else:
            self.drone_scatter.set_offsets(np.empty((0, 2)))

        # Update coverage text
        coverage = self.area_map.get_coverage_percent()
        self.coverage_text.set_text(f'Coverage: {coverage:.1f}%')

        self.fig.canvas.draw_idle()

    def export_map(self, filepath='mapped_area.png', dpi=150):
        """
        Export the current discovered map as a PNG image.

        Parameters:
            filepath (str): Output file path.
            dpi (int): Resolution of the exported image.
        """
        # Create a clean export figure (without drone markers)
        export_fig, export_ax = plt.subplots(1, 1, figsize=(10, 10))
        export_fig.patch.set_facecolor('#1a1a2e')

        export_ax.imshow(
            self.area_map.discovered_map,
            cmap=self.cmap,
            norm=self.norm,
            origin='lower',
            extent=[0, self.area_map.width, 0, self.area_map.height],
            interpolation='nearest'
        )

        coverage = self.area_map.get_coverage_percent()
        export_ax.set_title(
            f'AREA MAP — Coverage: {coverage:.1f}%',
            color='#00ff88', fontsize=16, fontweight='bold', pad=15
        )
        export_ax.set_xlabel('X (meters)', color='white', fontsize=11)
        export_ax.set_ylabel('Y (meters)', color='white', fontsize=11)
        export_ax.set_facecolor('#0d0d1a')
        export_ax.tick_params(colors='white', labelsize=9)
        for spine in export_ax.spines.values():
            spine.set_color('#333355')
        export_ax.grid(True, alpha=0.15, color='white', linestyle='--')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.TERRAIN_COLORS[0], label='Unmapped'),
            Patch(facecolor=self.TERRAIN_COLORS[1], label='Clear'),
            Patch(facecolor=self.TERRAIN_COLORS[3], label='Low Elevation'),
            Patch(facecolor=self.TERRAIN_COLORS[5], label='High Elevation'),
            Patch(facecolor=self.TERRAIN_COLORS[7], label='Obstacle'),
        ]
        legend = export_ax.legend(
            handles=legend_elements, loc='lower right',
            facecolor='#0d0d1a', edgecolor='#333355',
            fontsize=9, labelcolor='white'
        )

        export_fig.tight_layout()
        export_fig.savefig(filepath, dpi=dpi, facecolor=export_fig.get_facecolor())
        plt.close(export_fig)

    def export_raw(self, filepath='mapped_area.npy'):
        """
        Export the discovered map as a numpy array file.

        Parameters:
            filepath (str): Output file path (.npy).
        """
        np.save(filepath, self.area_map.get_discovered_map())
