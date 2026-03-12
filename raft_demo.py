# -*- coding: utf-8 -*-
"""
Raft Consensus — Drone Swarm Demo
==================================
A standalone Tkinter demo that visualises the Raft leader-election
protocol across a small swarm of drones.

Colours:
  - Follower  -> Green
  - Candidate -> Yellow
  - Leader    -> Red
  - Dead      -> Grey

Run:
    python raft_demo.py
"""

import tkinter as tk
import threading
import time
import random
import math

# ─── Constants ────────────────────────────────────────────────────────────────

FOLLOWER  = "Follower"
CANDIDATE = "Candidate"
LEADER    = "Leader"
DEAD      = "Dead"

COLOURS = {
    FOLLOWER:  "#2ecc71",   # green
    CANDIDATE: "#f1c40f",   # yellow
    LEADER:    "#e74c3c",   # red
    DEAD:      "#95a5a6",   # grey
}

NUM_DRONES          = 5
HEARTBEAT_INTERVAL  = 0.5        # seconds — leader sends a ping every 500 ms
ELECTION_TIMEOUT    = (1.5, 3.0) # seconds — randomised per drone
UI_REFRESH_MS       = 100        # milliseconds — canvas redraw interval
DRONE_RADIUS        = 30         # pixels
CANVAS_W            = 700
CANVAS_H            = 500
LOG_WIDTH            = 45         # characters


# ─── DroneNode ────────────────────────────────────────────────────────────────

class DroneNode:
    """Represents a single drone participating in the Raft protocol."""

    def __init__(self, node_id: int):
        self.node_id        = node_id
        self.state          = FOLLOWER
        self.term           = 0
        self.voted_for      = None      # id of candidate this node voted for in current term
        self.is_alive       = True
        self.last_heartbeat = time.time()
        self.election_timeout = random.uniform(*ELECTION_TIMEOUT)
        self.votes_received = 0
        self.lock           = threading.Lock()

    # ── Raft RPCs ─────────────────────────────────────────────────────────

    def request_vote(self, candidate_id: int, candidate_term: int) -> bool:
        """Called by a Candidate to ask this node for its vote."""
        with self.lock:
            if not self.is_alive:
                return False
            if candidate_term < self.term:
                return False
            if candidate_term > self.term:
                # Higher term — update and grant vote
                self.term = candidate_term
                self.voted_for = None
                self.state = FOLLOWER
            if self.voted_for is None or self.voted_for == candidate_id:
                self.voted_for = candidate_id
                return True
            return False

    def receive_heartbeat(self, leader_term: int):
        """Called by the Leader to reset this node's election timer."""
        with self.lock:
            if not self.is_alive:
                return
            if leader_term >= self.term:
                self.term = leader_term
                self.state = FOLLOWER
                self.voted_for = None
                self.last_heartbeat = time.time()

    def reset_election_timeout(self):
        """Pick a new random election timeout."""
        self.election_timeout = random.uniform(*ELECTION_TIMEOUT)

    def kill(self):
        """User clicked on this drone — mark it dead."""
        with self.lock:
            self.is_alive = False
            self.state = DEAD


# ─── RaftSwarmApp (Tkinter) ───────────────────────────────────────────────────

class RaftSwarmApp:
    """Tkinter application that runs and visualises the Raft protocol."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Raft Consensus - Drone Swarm")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        # ── Nodes ─────────────────────────────────────────────────────────
        self.nodes: list[DroneNode] = [DroneNode(i) for i in range(NUM_DRONES)]

        # ── Canvas ────────────────────────────────────────────────────────
        self.canvas = tk.Canvas(
            root, width=CANVAS_W, height=CANVAS_H,
            bg="#16213e", highlightthickness=0,
        )
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        # ── Log panel ────────────────────────────────────────────────────
        log_frame = tk.Frame(root, bg="#1a1a2e")
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10)

        tk.Label(log_frame, text="Event Log", fg="#e0e0e0", bg="#1a1a2e",
                 font=("Consolas", 11, "bold")).pack(anchor=tk.W)

        self.log_text = tk.Text(
            log_frame, width=LOG_WIDTH, height=30,
            bg="#0f3460", fg="#e0e0e0", font=("Consolas", 9),
            state=tk.DISABLED, wrap=tk.WORD, bd=0,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # ── Status bar ───────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Initialising...")
        tk.Label(
            root, textvariable=self.status_var,
            bg="#0f3460", fg="#e0e0e0", font=("Consolas", 10),
            anchor=tk.W, padx=8, pady=4,
        ).pack(side=tk.BOTTOM, fill=tk.X)

        # ── Draw drones ──────────────────────────────────────────────────
        self.oval_ids: dict[int, int] = {}   # node_id → canvas oval id
        self.text_ids: dict[int, int] = {}   # node_id → canvas text id
        self._layout_drones()

        # ── Title label on canvas ────────────────────────────────────────
        self.canvas.create_text(
            CANVAS_W // 2, 25, text="Click a drone to kill it",
            fill="#7f8c8d", font=("Consolas", 10, "italic"),
        )

        # ── Start threads ────────────────────────────────────────────────
        self._stop_event = threading.Event()
        for node in self.nodes:
            t = threading.Thread(target=self._node_loop, args=(node,), daemon=True)
            t.start()

        # ── UI refresh loop ──────────────────────────────────────────────
        self._refresh_ui()

        # ── Graceful shutdown ────────────────────────────────────────────
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Layout helpers ────────────────────────────────────────────────────

    def _layout_drones(self):
        """Arrange drones in a circle on the canvas."""
        cx, cy = CANVAS_W // 2, CANVAS_H // 2 + 10
        radius = min(CANVAS_W, CANVAS_H) // 2 - 60
        n = len(self.nodes)

        for i, node in enumerate(self.nodes):
            angle = 2 * math.pi * i / n - math.pi / 2
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)

            r = DRONE_RADIUS
            oval = self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=COLOURS[node.state],
                outline="#ecf0f1", width=2,
            )
            text = self.canvas.create_text(
                x, y, text=f"D{node.node_id}\n{node.state}",
                fill="white", font=("Consolas", 9, "bold"),
            )

            self.oval_ids[node.node_id] = oval
            self.text_ids[node.node_id] = text

            # Bind click-to-kill on both the oval and the text
            self.canvas.tag_bind(oval, "<Button-1>",
                                 lambda e, n=node: self._kill_drone(n))
            self.canvas.tag_bind(text, "<Button-1>",
                                 lambda e, n=node: self._kill_drone(n))

    # ── Kill ──────────────────────────────────────────────────────────────

    def _kill_drone(self, node: DroneNode):
        """Handle user click — kill the drone."""
        if not node.is_alive:
            return
        was_leader = (node.state == LEADER)
        node.kill()
        self._log(f"[KILL] Drone-{node.node_id} KILLED"
                  + (" (was Leader - election will trigger)" if was_leader else ""))

    # ── Per-node thread loop ──────────────────────────────────────────────

    def _node_loop(self, node: DroneNode):
        """
        Main loop for each drone — runs in its own daemon thread.
        • If Leader: send heartbeats.
        • If Follower: monitor for election timeout.
        • If Candidate: run an election.
        """
        while not self._stop_event.is_set():
            if not node.is_alive:
                time.sleep(0.2)
                continue

            with node.lock:
                state = node.state

            if state == LEADER:
                self._send_heartbeats(node)
                time.sleep(HEARTBEAT_INTERVAL)

            elif state == FOLLOWER:
                elapsed = time.time() - node.last_heartbeat
                if elapsed > node.election_timeout:
                    self._start_election(node)
                else:
                    time.sleep(0.05)

            elif state == CANDIDATE:
                # Already mid-election in _start_election; just wait
                time.sleep(0.05)

            else:
                time.sleep(0.2)

    # ── Heartbeat ─────────────────────────────────────────────────────────

    def _send_heartbeats(self, leader: DroneNode):
        """Leader broadcasts a heartbeat to all other alive nodes."""
        with leader.lock:
            if leader.state != LEADER or not leader.is_alive:
                return
            term = leader.term

        for node in self.nodes:
            if node.node_id != leader.node_id and node.is_alive:
                node.receive_heartbeat(term)

    # ── Election ──────────────────────────────────────────────────────────

    def _start_election(self, node: DroneNode):
        """Node becomes a Candidate and requests votes from the swarm."""
        with node.lock:
            if not node.is_alive:
                return
            node.term += 1
            node.state = CANDIDATE
            node.voted_for = node.node_id
            node.votes_received = 1          # vote for itself
            term = node.term

        self._log(f"[VOTE] Drone-{node.node_id} starts ELECTION (term {term})")

        # Small random delay to simulate network
        time.sleep(random.uniform(0.05, 0.15))

        # Request votes from all other alive nodes
        for other in self.nodes:
            if other.node_id == node.node_id:
                continue
            if other.is_alive:
                granted = other.request_vote(node.node_id, term)
                if granted:
                    with node.lock:
                        node.votes_received += 1
                    self._log(f"   [OK] Drone-{other.node_id} voted for Drone-{node.node_id}")

        # Check majority
        alive_count = sum(1 for n in self.nodes if n.is_alive)
        with node.lock:
            if not node.is_alive:
                return
            votes = node.votes_received
            if votes > alive_count // 2 and node.state == CANDIDATE:
                node.state = LEADER
                node.last_heartbeat = time.time()
                self._log(f"[LEADER] Drone-{node.node_id} elected LEADER "
                          f"({votes}/{alive_count} votes, term {node.term})")
            else:
                # Lost — revert to follower and wait
                node.state = FOLLOWER
                node.last_heartbeat = time.time()
                node.reset_election_timeout()
                self._log(f"   [LOST] Drone-{node.node_id} lost election "
                          f"({votes}/{alive_count} votes)")

    # ── UI refresh ────────────────────────────────────────────────────────

    def _refresh_ui(self):
        """Periodically redraw colours, labels, and status bar."""
        leader_id = None
        leader_term = 0

        for node in self.nodes:
            colour = COLOURS.get(node.state, COLOURS[DEAD])
            self.canvas.itemconfig(self.oval_ids[node.node_id], fill=colour)
            self.canvas.itemconfig(
                self.text_ids[node.node_id],
                text=f"D{node.node_id}\n{node.state}",
            )
            if node.state == LEADER and node.is_alive:
                leader_id = node.node_id
                leader_term = node.term

        if leader_id is not None:
            self.status_var.set(f"Leader: Drone-{leader_id}  |  Term: {leader_term}")
        else:
            self.status_var.set("No Leader - election in progress...")

        if not self._stop_event.is_set():
            self.root.after(UI_REFRESH_MS, self._refresh_ui)

    # ── Logging ───────────────────────────────────────────────────────────

    def _log(self, message: str):
        """Append a timestamped message to the log panel (thread-safe)."""
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {message}\n"

        def _append():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, line)
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)

        # Schedule on the main thread
        self.root.after(0, _append)

    # ── Shutdown ──────────────────────────────────────────────────────────

    def _on_close(self):
        """Stop all threads and destroy the window."""
        self._stop_event.set()
        self.root.destroy()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = RaftSwarmApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
