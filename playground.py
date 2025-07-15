"""
Drone Playground
================

A minimalist 2‑D simulation for experimenting with autonomous exploration
algorithms.  The program opens a Matplotlib window with two live views:

* **Left panel** – ground‑truth world.  Squares are drones, circles are
  tree trunks (obstacles).  Drones have limited vision (faded circle).
* **Right panel** – what the drones have mapped so far (their internal
  occupancy grid).

You can plug in different *DroneLogic* strategies (e.g. random walk,
frontier exploration, goal seeking) and quickly compare their
performance under various scenarios.

Run the module directly (``python drone_playground.py``) to launch a demo
with one random‑walk drone.  Scroll to **👟 EXAMPLE USAGE** at the bottom
for ready‑to‑run snippets that register additional logics or objectives.

Dependencies
------------
* Python ≥3.9
* numpy
* matplotlib

Install with::

    pip install numpy matplotlib
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Protocol, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ──────────────────────────────────────────────────────────────────────────────
# World Model
# ──────────────────────────────────────────────────────────────────────────────

Cell = Tuple[int, int]  # (row, col)

class Environment:
    """2‑D grid world with obstacles (tree trunks)."""

    def __init__(self, rows: int, cols: int, tree_density: float = 0.05,
                 seed: Optional[int] = None) -> None:
        self.rows = rows
        self.cols = cols
        rng = random.Random(seed)
        self.trees: set[Cell] = {
            (rng.randrange(rows), rng.randrange(cols))
            for _ in range(int(rows * cols * tree_density))
        }

    # ---------------------------------------------------------------------
    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and cell not in self.trees

# ──────────────────────────────────────────────────────────────────────────────
# Drone Logic Interface
# ──────────────────────────────────────────────────────────────────────────────

class DroneLogic(Protocol):
    """Decides where the drone should move next."""

    def next_move(self, drone: "Drone", env: Environment) -> Cell: ...

# -----------------------------------------------------------------------------
class RandomWalkLogic:
    """Moves to a random adjacent free cell."""

    def next_move(self, drone: "Drone", env: Environment) -> Cell:
        nbrs = drone.neighbors(env)
        return random.choice(nbrs) if nbrs else drone.pos

# -----------------------------------------------------------------------------
class FrontierExplorationLogic:
    """Greedy exploration: always head toward closest unknown cell."""

    def next_move(self, drone: "Drone", env: Environment) -> Cell:
        frontier: List[Cell] = [cell for cell, v in np.ndenumerate(drone.map)
                                if v == -1]  # ‑1 == unknown
        if not frontier:
            return drone.pos  # done!
        # Pick frontier cell closest to drone
        target = min(frontier, key=lambda c: manhattan(c, drone.pos))
        # Step one cell toward target along shortest Manhattan path
        r, c = drone.pos
        tr, tc = target
        step = (r + int(np.sign(tr - r)), c + int(np.sign(tc - c)))
        return step if env.is_free(step) else drone.pos

# ──────────────────────────────────────────────────────────────────────────────
# Drone
# ──────────────────────────────────────────────────────────────────────────────

def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

@dataclass
class Drone:
    pos: Cell
    sensing: int = 3  # radius
    logic: DroneLogic = field(default_factory=RandomWalkLogic)
    color: str = "tab:red"
    map: np.ndarray = field(init=False)  # -1 unknown, 0 free, 1 obstacle

    def __post_init__(self) -> None:
        # Map created lazily when added to simulation (needs env size)
        pass

    # ---------------------------------------------------------------------
    def attach_env(self, env: Environment) -> None:
        self.map = np.full((env.rows, env.cols), -1, dtype=np.int8)
        self.update_map(env)  # initial sensing

    # ---------------------------------------------------------------------
    def neighbors(self, env: Environment) -> List[Cell]:
        r, c = self.pos
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return [(r + dr, c + dc) for dr, dc in deltas if env.is_free((r + dr, c + dc))]

    # ---------------------------------------------------------------------
    def update_map(self, env: Environment) -> None:
        """Reveal cells within sensing radius."""
        r0, c0 = self.pos
        for dr in range(-self.sensing, self.sensing + 1):
            for dc in range(-self.sensing, self.sensing + 1):
                if dr * dr + dc * dc <= self.sensing * self.sensing:
                    cell = (r0 + dr, c0 + dc)
                    if env.in_bounds(cell):
                        self.map[cell] = 1 if cell in env.trees else 0

    # ---------------------------------------------------------------------
    def step(self, env: Environment) -> None:
        self.pos = self.logic.next_move(self, env)
        self.update_map(env)

# ──────────────────────────────────────────────────────────────────────────────
# Simulation & Visualisation
# ──────────────────────────────────────────────────────────────────────────────

class Simulation:
    def __init__(self, env: Environment, drones: List[Drone], fps: int = 10):
        self.env = env
        self.drones = drones
        for d in self.drones:
            d.attach_env(env)
        self.fps = fps

        # Matplotlib setup
        self.fig, (self.ax_world, self.ax_map) = plt.subplots(1, 2, figsize=(10, 5))
        self.fig.canvas.manager.set_window_title("Drone Playground")
        self._init_axes()

        self.ani = FuncAnimation(self.fig, self._update, interval=1000 // fps)

    # ---------------------------------------------------------------------
    def _init_axes(self):
        for ax in (self.ax_world, self.ax_map):
            ax.set_xlim(-0.5, self.env.cols - 0.5)
            ax.set_ylim(-0.5, self.env.rows - 0.5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        self.ax_world.set_title("Ground truth")
        self.ax_map.set_title("Drone map")

    # ---------------------------------------------------------------------
    def _draw_world(self):
        self.ax_world.collections.clear()
        # Draw trees
        tr, tc = zip(*self.env.trees) if self.env.trees else ([], [])
        self.ax_world.scatter(tc, tr, marker="o", c="forestgreen", s=30, alpha=0.9)
        # Draw drones & sensing radius
        for d in self.drones:
            r, c = d.pos
            circ = plt.Circle((c, r), d.sensing, color=d.color, alpha=0.15)
            self.ax_world.add_patch(circ)
            self.ax_world.scatter([c], [r], marker="s", c=d.color, s=50, edgecolors="k")

    # ---------------------------------------------------------------------
    def _draw_map(self):
        self.ax_map.collections.clear()
        # Assemble union of drone maps (cell‑wise OR of knowledge)
        aggregated = np.full_like(self.drones[0].map, -1)
        for d in self.drones:
            mask_unknown = aggregated == -1
            aggregated[mask_unknown] = d.map[mask_unknown]
        # Plot
        colors = { -1: "lightgray", 0: "white", 1: "dimgray" }
        for cell, val in np.ndenumerate(aggregated):
            if val == -1:
                continue  # keep light background
            r, c = cell
            self.ax_map.scatter([c], [r], marker="s", c=colors[val], s=20)
        # Drone positions
        for d in self.drones:
            r, c = d.pos
            self.ax_map.scatter([c], [r], marker="s", c=d.color, s=50, edgecolors="k")

    # ---------------------------------------------------------------------
    def _update(self, _frame):
        for d in self.drones:
            d.step(self.env)
        self._draw_world()
        self._draw_map()

    # ---------------------------------------------------------------------
    def run(self):
        plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 👟 EXAMPLE USAGE – tweak here then run the file
# ──────────────────────────────────────────────────────────────────────────────

def demo() -> None:
    env = Environment(rows=40, cols=40, tree_density=0.07, seed=0)
    drones = [
        Drone(pos=(0, 0), sensing=3, color="tab:red", logic=RandomWalkLogic()),
        # Add another drone with smarter exploration
        Drone(pos=(39, 39), sensing=4, color="tab:blue", logic=FrontierExplorationLogic()),
    ]
    Simulation(env, drones, fps=10).run()

if __name__ == "__main__":
    demo()
