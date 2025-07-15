# HoverQuest

*A minimalist sandbox for testing autonomous aerial exploration logic on a 2‑D grid.*

![screenshot](assets/demo.gif)

---

## What It Is

`Drone Playground` simulates one or more drones (UAVs) flying above a forested map.
Each drone has limited vision, an internal occupancy grid, and a swappable **logic module** that decides where to move next. Two synchronized panels appear when you run the demo:

| Panel                         | Shows                                                      |
| ----------------------------- | ---------------------------------------------------------- |
| **Ground‑Truth World (left)** | Drone positions, sensing radius, and actual tree locations |
| **Drone Map (right)**         | What the drones have discovered so far                     |

Use this environment to prototype exploration algorithms without wrangling physics engines or ROS.

---

## Downstream Uses

| Scenario                     | How the playground helps                                                                                  |
| ---------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Canopy Mapping**           | Test strategies for fully scanning dense tree canopies before flying a real multispectral camera mission. |
| **Search & Rescue**          | Benchmark logic that minimizes time to locate hidden targets.                                             |
| **Wildlife Monitoring**      | Evaluate patrol patterns for spotting tagged animals.                                                     |
| **Precision Agriculture**    | Prototype field‑scouting paths that maximise crop‑health coverage.                                        |
| **Multi‑Robot Coordination** | Experiment with swarming behaviours and division‑of‑labour heuristics.                                    |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your‑org/drone‑playground.git
cd drone‑playground
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # numpy, matplotlib

# 2. Launch the demo
python drone_playground.py
```

A window pops up with two live views. Watch a red *random‑walk* drone and a blue *frontier‑explorer* drone build their map.

---

## Project Structure

```
.
├── drone_playground.py  # core simulation (world, drones, logics, GUI)
├── README.md           # this file
└── requirements.txt    # bare‑bones deps (numpy, matplotlib)
```

Add more modules as the project grows—e.g. `logics/` for sophisticated planners.

---

## Customising Experiments

1. **Change Map Size / Density**

   ```python
   env = Environment(rows=100, cols=100, tree_density=0.10)
   ```
2. **Plug in New Logic**

   ```python
   class SpiralLogic(DroneLogic):
       def next_move(self, drone, env):
           # your rule here
           return next_cell
   ...
   Drone(pos=(0,0), sensing=5, color="purple", logic=SpiralLogic())
   ```
3. **Multiple Objectives** – track performance by writing tiny evaluators (coverage %, time‑to‑target, etc.) and logging per frame.

---

## Contributing

Pull requests welcome. Keep code readable, dependency‑light, and license‑compatible.

---

## License

MIT
