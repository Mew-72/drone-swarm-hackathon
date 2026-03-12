import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import numpy as np
import os

from behaviors.consensus_algorithm import ConsensusAlgorithm
from behaviors.collision_avoidance_algorithm import CollisionAvoidanceAlgorithm
from behaviors.formation_control_algorithm import FormationControlAlgorithm
from behaviors.mapping_scan_algorithm import MappingScanAlgorithm
from visualizer import DroneSwarmVisualizer
from map_visualizer import MapVisualizer
from area_map import AreaMap
from drone import Drone


class DroneSwarmApp:
    def __init__(self, root):
        """
        Initialize the Drone Swarm Simulation application.
        """
        self.root = root
        self.root.title("Drone Swarm Simulation")

        self.target_point = np.array([0, 0, 0])  # Initial target point
        self.is_x_at_origin = True  # State to track if the target is at the origin

        # Simulation parameters
        self.num_drones = 100  # Number of drones in the swarm
        self.iterations = 100  # Number of iterations (not currently used)
        self.epsilon = 0.1  # Parameter for the consensus algorithm
        self.collision_threshold = 1.0  # Minimum distance to avoid collisions
        self.interval = 200  # Time interval between simulation updates (ms)

        # UI control variables
        self.formation_type = tk.StringVar(value="line")  # Formation type selection
        self.zoom_level = tk.DoubleVar(value=10.0)  # Zoom level for visualization

        # Define behavior algorithms
        self.behavior_algorithms = [
            ConsensusAlgorithm(self.epsilon),
            CollisionAvoidanceAlgorithm(self.collision_threshold),
            FormationControlAlgorithm(self.formation_type.get())
        ]

        # Initialize the swarm with 3D random positions
        self.drones = [Drone(np.random.rand(3) * 10, i) for i in range(self.num_drones)]

        # Initialize the visualizer
        self.visualizer = DroneSwarmVisualizer(self.drones, self.formation_type.get())

        # Mapping mode state
        self.mapping_mode = False
        self.area_map = None
        self.mapping_algorithm = None
        self.map_visualizer = None
        self.mapping_canvas = None
        self.mapping_num_drones = 20  # Use fewer drones for mapping

        # Set up the UI
        self.setup_ui()

        # Simulation state
        self.running = False

    def setup_ui(self):
        """
        Set up the graphical user interface.
        """
        # Create a frame for controls
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # --- Mode Selection ---
        ttk.Label(control_frame, text="Mode:", font=("", 10, "bold")).pack(anchor=tk.W, pady=(5, 2))
        self.mode_var = tk.StringVar(value="swarm")
        ttk.Radiobutton(control_frame, text="Swarm Simulation",
                        variable=self.mode_var, value="swarm",
                        command=self.switch_mode).pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Area Mapping",
                        variable=self.mode_var, value="mapping",
                        command=self.switch_mode).pack(anchor=tk.W)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # --- Swarm Controls Frame ---
        self.swarm_controls_frame = ttk.LabelFrame(control_frame, text="Swarm Controls")
        self.swarm_controls_frame.pack(fill=tk.X, pady=2)

        # Formation selection radio buttons
        ttk.Label(self.swarm_controls_frame, text="Formation:").pack(anchor=tk.W)
        ttk.Radiobutton(self.swarm_controls_frame, text="Line", variable=self.formation_type, value="line", command=self.update_formation).pack(anchor=tk.W)
        ttk.Radiobutton(self.swarm_controls_frame, text="Circle", variable=self.formation_type, value="circle", command=self.update_formation).pack(anchor=tk.W)
        ttk.Radiobutton(self.swarm_controls_frame, text="Square", variable=self.formation_type, value="square", command=self.update_formation).pack(anchor=tk.W)
        ttk.Radiobutton(self.swarm_controls_frame, text="Random", variable=self.formation_type, value="random", command=self.update_formation).pack(anchor=tk.W)

        ttk.Separator(self.swarm_controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Zoom level control
        ttk.Label(self.swarm_controls_frame, text="Zoom Level:").pack(anchor=tk.W)
        zoom_scale = ttk.Scale(self.swarm_controls_frame, from_=5.0, to=20.0, orient=tk.HORIZONTAL, variable=self.zoom_level, command=self.update_zoom)
        zoom_scale.pack(anchor=tk.W)

        ttk.Separator(self.swarm_controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Color mode control
        ttk.Label(self.swarm_controls_frame, text="Color Mode:").pack(anchor=tk.W)
        self.color_mode = tk.StringVar(value="by_index")
        ttk.Radiobutton(self.swarm_controls_frame, text="By Index", variable=self.color_mode, value="by_index", command=self.update_color_mode).pack(anchor=tk.W)
        ttk.Radiobutton(self.swarm_controls_frame, text="By Distance", variable=self.color_mode, value="by_distance", command=self.update_color_mode).pack(anchor=tk.W)

        ttk.Separator(self.swarm_controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Button to change the target position
        ttk.Button(self.swarm_controls_frame, text="Change X Position", command=self.change_x_position).pack(pady=5)

        # --- Mapping Controls Frame ---
        self.mapping_controls_frame = ttk.LabelFrame(control_frame, text="Mapping Controls")
        self.mapping_controls_frame.pack(fill=tk.X, pady=2)

        # Area dimensions
        ttk.Label(self.mapping_controls_frame, text="Area Width:").pack(anchor=tk.W)
        self.area_width_var = tk.IntVar(value=100)
        ttk.Entry(self.mapping_controls_frame, textvariable=self.area_width_var, width=8).pack(anchor=tk.W, padx=5)

        ttk.Label(self.mapping_controls_frame, text="Area Height:").pack(anchor=tk.W)
        self.area_height_var = tk.IntVar(value=100)
        ttk.Entry(self.mapping_controls_frame, textvariable=self.area_height_var, width=8).pack(anchor=tk.W, padx=5)

        # Number of mapping drones
        ttk.Label(self.mapping_controls_frame, text="Drones:").pack(anchor=tk.W)
        self.mapping_drones_var = tk.IntVar(value=20)
        ttk.Entry(self.mapping_controls_frame, textvariable=self.mapping_drones_var, width=8).pack(anchor=tk.W, padx=5)

        ttk.Separator(self.mapping_controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Speed control
        ttk.Label(self.mapping_controls_frame, text="Speed:").pack(anchor=tk.W)
        self.speed_var = tk.IntVar(value=1)
        speed_frame = ttk.Frame(self.mapping_controls_frame)
        speed_frame.pack(anchor=tk.W, fill=tk.X, padx=5)
        self.speed_scale = ttk.Scale(speed_frame, from_=1, to=20,
                                     orient=tk.HORIZONTAL, variable=self.speed_var)
        self.speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.speed_label = ttk.Label(speed_frame, text="1x", width=4)
        self.speed_label.pack(side=tk.RIGHT)
        self.speed_scale.configure(command=self._update_speed_label)

        ttk.Separator(self.mapping_controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Generate new terrain button
        ttk.Button(self.mapping_controls_frame, text="Generate Terrain",
                   command=self.generate_new_terrain).pack(pady=3, fill=tk.X, padx=5)

        # Skip to end button
        ttk.Button(self.mapping_controls_frame, text="Skip to End",
                   command=self.skip_to_end).pack(pady=3, fill=tk.X, padx=5)

        # Export buttons
        ttk.Button(self.mapping_controls_frame, text="Export Map (PNG)",
                   command=self.export_map_png).pack(pady=3, fill=tk.X, padx=5)
        ttk.Button(self.mapping_controls_frame, text="Export Raw (.npy)",
                   command=self.export_map_raw).pack(pady=3, fill=tk.X, padx=5)

        # Coverage progress
        ttk.Label(self.mapping_controls_frame, text="Coverage:").pack(anchor=tk.W, pady=(5, 0))
        self.coverage_var = tk.StringVar(value="0.0%")
        self.coverage_label = ttk.Label(self.mapping_controls_frame,
                                        textvariable=self.coverage_var,
                                        font=("", 14, "bold"))
        self.coverage_label.pack(anchor=tk.W, padx=5)

        # Initially hide mapping controls
        self.mapping_controls_frame.pack_forget()

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Start/Stop button (shared)
        self.start_button = ttk.Button(control_frame, text="Animate", command=self.toggle_simulation)
        self.start_button.pack(pady=10)

        # Canvas frame for visualization
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

        # Swarm canvas
        self.canvas = FigureCanvasTkAgg(self.visualizer.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def switch_mode(self):
        """
        Switch between swarm simulation and mapping mode.
        """
        # Stop any running simulation
        self.running = False
        self.start_button.config(text="Animate")

        mode = self.mode_var.get()

        if mode == "swarm":
            self._activate_swarm_mode()
        elif mode == "mapping":
            self._activate_mapping_mode()

    def _activate_swarm_mode(self):
        """Switch to the swarm simulation view."""
        self.mapping_mode = False

        # Show swarm controls, hide mapping controls
        self.mapping_controls_frame.pack_forget()
        self.swarm_controls_frame.pack(fill=tk.X, pady=2)

        # Remove mapping canvas if present
        if self.mapping_canvas is not None:
            self.mapping_canvas.get_tk_widget().pack_forget()

        # Show swarm canvas
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.canvas.draw()

    def _activate_mapping_mode(self):
        """Switch to the mapping view."""
        self.mapping_mode = True

        # Show mapping controls, hide swarm controls
        self.swarm_controls_frame.pack_forget()
        self.mapping_controls_frame.pack(fill=tk.X, pady=2)

        # Hide swarm canvas
        self.canvas.get_tk_widget().pack_forget()

        # Generate terrain and create mapping visualization
        self.generate_new_terrain()

    def generate_new_terrain(self):
        """
        Generate a new random terrain, reinitialize mapping drones
        and create the mapping visualization.
        """
        # Stop any running simulation
        self.running = False
        self.start_button.config(text="Animate")

        area_w = self.area_width_var.get()
        area_h = self.area_height_var.get()
        num_mapping_drones = self.mapping_drones_var.get()
        self.mapping_num_drones = num_mapping_drones

        # Create a new area map
        self.area_map = AreaMap(width=area_w, height=area_h, resolution=1.0)

        # Create mapping drones — all start from origin (0, 0, 5)
        origin = np.array([0.0, 0.0, 5.0])
        self.mapping_drones = [
            Drone(origin.copy(), i) for i in range(num_mapping_drones)
        ]

        # Create mapping scan algorithm
        self.mapping_algorithm = MappingScanAlgorithm(
            area_map=self.area_map,
            num_drones=num_mapping_drones,
            origin=origin,
            scan_radius=2,
            speed=0.5,
            altitude=5.0
        )

        # Collision avoidance for mapping drones
        self.mapping_collision = CollisionAvoidanceAlgorithm(
            collision_threshold=self.collision_threshold
        )

        # Remove old mapping canvas if present
        if self.mapping_canvas is not None:
            self.mapping_canvas.get_tk_widget().pack_forget()

        # Create map visualizer
        self.map_visualizer = MapVisualizer(self.area_map, figsize=(8, 8))

        # Create new canvas for map
        self.mapping_canvas = FigureCanvasTkAgg(
            self.map_visualizer.fig, master=self.canvas_frame
        )
        self.mapping_canvas.draw()
        self.mapping_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        # Reset coverage
        self.coverage_var.set("0.0%")

    def _update_speed_label(self, val):
        """Update the speed label text when slider moves."""
        speed = int(float(val))
        self.speed_label.config(text=f"{speed}x")

    def skip_to_end(self):
        """
        Instantly complete the mapping — scans the entire area,
        places all drones back at origin, and shows the final map.
        """
        if self.area_map is None:
            return

        # Stop any running animation
        self.running = False
        self.start_button.config(text="Animate")

        # Scan every cell in the area
        for r in range(self.area_map.grid_height):
            for c in range(self.area_map.grid_width):
                val = self.area_map.scan_cell(c, r)
                if val is not None:
                    self.area_map.update_cell(c, r, val)

        # Return all drones to origin
        if self.mapping_algorithm is not None:
            origin = self.mapping_algorithm.get_origin()
            for drone in self.mapping_drones:
                drone.position = origin.copy()
            # Mark all drones as done
            for i in range(len(self.mapping_drones)):
                self.mapping_algorithm.phases[i] = self.mapping_algorithm.PHASE_DONE

        # Update visualization
        positions = [drone.get_position() for drone in self.mapping_drones]
        self.map_visualizer.update(positions)
        self.mapping_canvas.draw()
        self.coverage_var.set("100.0%")

    def export_map_png(self):
        """Export the current map as a PNG image."""
        if self.map_visualizer is not None:
            filepath = os.path.join(os.path.dirname(__file__), "mapped_area.png")
            self.map_visualizer.export_map(filepath)

    def export_map_raw(self):
        """Export the current map as a numpy .npy file."""
        if self.map_visualizer is not None:
            filepath = os.path.join(os.path.dirname(__file__), "mapped_area.npy")
            self.map_visualizer.export_raw(filepath)

    def update_formation(self):
        """
        Update the formation control algorithm when the user selects a different formation.
        """
        self.behavior_algorithms[-1] = FormationControlAlgorithm(self.formation_type.get())
        self.visualizer.formation_type = self.formation_type.get()
        self.canvas.draw()

    def update_zoom(self, event):
        """
        Update the visualization zoom level when the user adjusts the zoom slider.
        """
        self.visualizer.update_zoom(self.zoom_level.get())
        self.canvas.draw()

    def update_color_mode(self):
        """
        Update the color mode of the drones in the visualization.
        """
        self.visualizer.color_mode = self.color_mode.get()
        self.visualizer.update_colors()
        self.canvas.draw()

    def toggle_simulation(self):
        """
        Start or stop the simulation when the button is clicked.
        """
        if self.running:
            self.running = False
            self.start_button.config(text="Animate")
        else:
            self.running = True
            self.start_button.config(text="Stop")
            if self.mapping_mode:
                threading.Thread(target=self.run_mapping_simulation, daemon=True).start()
            else:
                threading.Thread(target=self.run_simulation, daemon=True).start()

    def change_x_position(self):
        """
        Toggle the target position between [20, 0, 0] and [0, 0, 0].
        """
        if self.is_x_at_origin:
            self.target_point = np.array([20, 0, 0])
        else:
            self.target_point = np.array([0, 0, 0])

        self.is_x_at_origin = not self.is_x_at_origin
        self.update_target_positions()

    def update_target_positions(self):
        """
        Update the target positions of the drones based on the current formation.
        """
        formation = self.behavior_algorithms[-1].get_formation(self.drones)
        for drone, target in zip(self.drones, formation):
            drone.target_position = self.target_point + target

        # Update the target point in the formation control algorithm
        self.behavior_algorithms[-1].set_target_point(self.target_point)

    def run_simulation(self):
        """
        Run the swarm simulation loop.
        """
        while self.running:
            # Update each drone's position based on behavior algorithms
            for drone in self.drones:
                neighbor_positions = [other_drone.communicate() for other_drone in self.drones if other_drone != drone]
                drone.update_position(neighbor_positions, self.behavior_algorithms)

            # Update the view to follow the drones
            self.visualizer.update_view(self.drones)

            # Refresh visualization
            self.visualizer.update()
            self.canvas.draw()

    def run_mapping_simulation(self):
        """
        Run the mapping simulation loop. Drones start from the origin,
        spread out, and sweep the area while avoiding collisions.
        Speed multiplier runs multiple ticks per visual frame.
        """
        if self.mapping_algorithm is None or self.map_visualizer is None:
            return

        while self.running:
            # Run multiple simulation ticks per frame based on speed setting
            ticks = max(1, self.speed_var.get())
            for _ in range(ticks):
                if not self.running:
                    break
                # Update each mapping drone
                for drone in self.mapping_drones:
                    neighbor_positions = [
                        other.communicate() for other in self.mapping_drones
                        if other != drone
                    ]

                    # Apply mapping scan behavior (steers toward waypoints + scans)
                    new_pos = self.mapping_algorithm.apply(
                        drone, neighbor_positions, drone.position.copy()
                    )

                    # Apply collision avoidance on top
                    new_pos = self.mapping_collision.apply(
                        drone, neighbor_positions, new_pos
                    )

                    drone.position = new_pos

            # Get current drone positions for visualization
            positions = [drone.get_position() for drone in self.mapping_drones]

            # Update map visualization
            self.map_visualizer.update(positions)
            self.mapping_canvas.draw()

            # Update coverage readout
            coverage = self.area_map.get_coverage_percent()
            self.coverage_var.set(f"{coverage:.1f}%")

            # Stop if mapping is complete
            if self.mapping_algorithm.is_mapping_complete():
                self.running = False
                self.root.after(0, lambda: self.start_button.config(text="Animate"))
                break


# Main entry point for the application
def main():
    root = tk.Tk()
    app = DroneSwarmApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
