# Travelling Salesman Problem – Heuristic Toolkit (English)

This project collects several heuristics commonly used for the Travelling Salesman Problem (TSP). The code is written in Python and can be extended with new instances or optimization methods.

## Proposed Work Plan

1. **Data definition** – Select a set of cities with planar coordinates (CSV in `data/cities_50.csv`) and build utilities to load them quickly.
2. **Problem modeling** – Implement functions to compute distances, evaluate a tour, and manipulate permutations of cities.
3. **Constructive heuristics** – Provide fast methods that build an initial tour: Nearest Neighbor and a Greedy (Kruskal-like) approach.
4. **Local improvement heuristics** – Apply 2-opt on top of the initial tours to eliminate crossings and shorten the route.
5. **Metaheuristics** – Add stochastic methods (Simulated Annealing, simplified Genetic Algorithm) that start from constructive solutions and continue exploring the permutation space.
6. **Tooling and execution** – Build a small CLI (`python main.py --method=nearest_neighbor`, etc.) that compares the distances produced by each method and prints the results.

Additional methods (e.g., Tabu Search, Ant Colony) can be layered on later while keeping the same infrastructure.

## Data File

`data/cities_50.csv` stores 50 fictitious cities in the 2D plane. The columns are `name,x,y`, and the distances are computed with the Euclidean (Pythagorean) formula.

### Visualizing the Cities

To inspect the positions quickly, run:

```bash
python app/plot_cities.py --data data/cities_50.csv --show-labels
```

The script relies on matplotlib to draw the points (install it with `pip install matplotlib` if needed).

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # (only if external deps are added later)
python main.py --method nearest_neighbor --apply-2opt
```

The `--method` flag selects the base heuristic, while `--apply-2opt` and `--annealing`/`--ga` enable additional optimizations.

### CLI Options (`main.py`)

| Flag         | Values / Type                         | Description                                                  | Default                 |
|--------------|----------------------------------------|--------------------------------------------------------------|-------------------------|
| `--data`     | Path to CSV                            | Dataset of cities (`name,x,y`).                              | `data/cities_50.csv`    |
| `--method`   | `nearest_neighbor`, `greedy`, `random` | Constructive heuristic for the initial tour.                 | `nearest_neighbor`      |
| `--start`    | Int                                    | Starting city index for Nearest Neighbor.                   | `0`                     |
| `--apply-2opt` | Flag                                 | Enables 2-opt local search on the current tour.              | disabled                |
| `--annealing` | Flag                                  | Runs Simulated Annealing starting from the current solution. | disabled                |
| `--ga`       | Flag                                   | Runs a genetic algorithm initialized with the current tour.  | disabled                |
| `--seed`     | Int                                    | Random seed for reproducible runs.                           | `None` (random)         |

### Example Runs

```bash
# NN + 2-opt
python main.py --method nearest_neighbor --apply-2opt

# Greedy + annealing
python main.py --method greedy --annealing --seed 42

# Random tour improved with GA
python main.py --method random --ga --apply-2opt
```

## Roadmap: Heuristics + ML Selector

1. **Data Setup (already done)**  
   - a) create a CSV with at least 30 cities (`name,x,y`).  
   - b) add a loader (`City`, `load_cities`) and a function that builds the distance matrix (`build_distance_matrix`).  
   - c) sanity-check that `load_cities` + `build_distance_matrix` work via a small Python snippet.

2. **Heuristic Implementations (already done)**  
   - a) implement the basics: `tour_length`, `nearest_neighbor`, `greedy`, `random_tour`.  
   - b) add `two_opt` for local optimization.  
   - c) implement the metaheuristics `simulated_annealing` and `genetic_algorithm` to keep exploring after the solution gets stuck.

3. **TSPSolver & CLI (already done)**  
   - a) `TSPSolver` takes the city list, precomputes the distance matrix, and exposes `solve(method, apply_two_opt_flag, annealing, genetic, seed, start)`.  
   - b) `main.py` provides CLI arguments (`--method`, `--apply-2opt`, `--annealing`, `--ga`, `--seed`, `--data`, `--start`) and prints the resulting route plus distance.

4. **Benchmark Dataset for Auto-Selection**  
   - a) generate multiple TSP instances (random CSVs or imported from other sources).  
   - b) for each instance, run every strategy (NN, Greedy, combinations with 2-opt, SA, GA).  
   - c) store, e.g., in a CSV, each strategy’s distance/time and mark the winner (minimum distance).

5. **Feature Engineering & Model Training**  
   - a) compute simple features per instance: number of cities, mean/median/max/min distance, standard deviation, bounding box area, average nearest neighbor distance, etc.  
   - b) use the table (“features + winning strategy”) as a dataset and train a lightweight model (centroid classifier, random forest, etc.).  
   - c) serialize the model (JSON/pickle) so it can be loaded later.

6. **Auto-Method Integration**  
   - a) extend the CLI with `--auto-method` and `--selector-model`.  
   - b) when the flag is active, load the model, compute the features of the current instance, and pick the predicted strategy.  
   - c) run the solver with the recommended parameters; if the model is missing, fall back to manual options.

7. **Validation & Documentation**  
   - a) manually and programmatically verify that each strategy (and the auto mode) produces valid routes on multiple instances.  
   - b) update the README (or wiki) with run examples, how to generate benchmarks, and how to train/use the ML selector.
