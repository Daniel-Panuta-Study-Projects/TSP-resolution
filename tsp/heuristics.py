from __future__ import annotations  # Enable postponed annotations for typing clarity.

import math  # math is needed for exponential acceptance in simulated annealing.
import random  # random supplies stochastic behavior for heuristics/metaheuristics.
from typing import Iterable, List, Sequence  # Typing hints for documentation clarity.

Matrix = Sequence[Sequence[float]]  # Alias for distance matrix structure.
Tour = List[int]  # Alias for a tour represented as ordered city indices.


def tour_length(tour: Sequence[int], dist_matrix: Matrix) -> float:
    """Compute the round-trip distance for the provided permutation of cities."""
    total = 0.0  # Accumulate total distance traveled.
    for i in range(len(tour)):
        a = tour[i]  # Current city index.
        b = tour[(i + 1) % len(tour)]  # Next city (wrap around to start).
        total += dist_matrix[a][b]  # Add hop distance to aggregate.
    return total  # Return final circuit length.


def nearest_neighbor(dist_matrix: Matrix, start: int = 0) -> Tour:
    """Build a tour by repeatedly choosing the closest unvisited city."""
    n = len(dist_matrix)  # Number of cities encoded in the matrix.
    unvisited = set(range(n))  # Track which cities remain to be explored.
    tour: Tour = [start]  # Initialize tour with chosen starting city.
    unvisited.remove(start)  # Mark start as visited to avoid revisiting.
    current = start  # Keep pointer to the most recently added city.

    while unvisited:
        # Select the city with minimal distance to the current city.
        next_city = min(unvisited, key=lambda idx: dist_matrix[current][idx])
        tour.append(next_city)  # Append the best candidate to the route.
        unvisited.remove(next_city)  # Remove from unvisited set.
        current = next_city  # Update current pointer and continue.
    return tour  # Return completed tour (closure handled via tour_length).


def greedy(dist_matrix: Matrix) -> Tour:
    """Select edges greedily while preventing premature cycles and high degrees."""
    n = len(dist_matrix)  # Number of nodes in the graph.
    edges = [
        (dist_matrix[i][j], i, j)
        for i in range(n)
        for j in range(i + 1, n)
    ]  # Create list of all undirected edges with weights.
    edges.sort()  # Sort edges ascending by distance.

    parent = list(range(n))  # Disjoint-set arrays for cycle detection.
    degree = [0] * n  # Track degree per vertex to enforce TSP constraints.
    selected: List[tuple[int, int]] = []  # Accumulate edges that form the tour.

    def find(x: int) -> int:
        """Find with path compression for union-find structure."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # Path compression step.
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        """Merge sets to reflect newly connected components."""
        parent[find(a)] = find(b)

    for dist, i, j in edges:
        if degree[i] >= 2 or degree[j] >= 2:
            continue  # Skip edges that would exceed two edges per node.
        same_component = find(i) == find(j)  # Check if edge forms a cycle.
        if same_component and len(selected) < n - 1:
            continue  # Avoid early cycle unless closing final tour.
        selected.append((i, j))  # Accept the edge into the partial tour.
        degree[i] += 1
        degree[j] += 1
        union(i, j)  # Merge connected components.
        if len(selected) == n:
            break  # Stop when full tour has n edges (including closing edge).

    return _edges_to_tour(selected, n)  # Convert edge list into ordered route.


def random_tour(n: int) -> Tour:
    """Return a random permutation of city indices (baseline for stochastic methods)."""
    tour = list(range(n))  # Start with identity ordering.
    random.shuffle(tour)  # Shuffle in-place to produce a random route.
    return tour


def two_opt(
    tour: Tour,
    dist_matrix: Matrix,
    max_iterations: int = 2000,
) -> Tour:
    """Iteratively remove edge crossings by reversing segments (2-opt heuristic)."""
    best = list(tour)  # Copy so original tour remains untouched.
    best_distance = tour_length(best, dist_matrix)  # Track current best distance.
    improvement = True  # Flag to know when to stop iterating.
    iterations = 0  # Count iterations to cap runtime.

    while improvement and iterations < max_iterations:
        improvement = False
        for i in range(1, len(best) - 2):
            for k in range(i + 1, len(best) - 1):
                # Reverse the segment between i and k to test a new layout.
                new_tour = (
                    best[:i] + list(reversed(best[i : k + 1])) + best[k + 1 :]
                )
                new_distance = tour_length(new_tour, dist_matrix)
                if new_distance + 1e-9 < best_distance:
                    best = new_tour  # Accept improving move.
                    best_distance = new_distance
                    improvement = True
                    break  # Restart search from scratch with new best route.
            if improvement:
                break
        iterations += 1
    return best  # Final locally optimized tour.


def simulated_annealing(
    initial_tour: Tour,
    dist_matrix: Matrix,
    initial_temperature: float = 1000.0,
    cooling_rate: float = 0.995,
    iterations: int = 10000,
) -> Tour:
    """Explore neighbors probabilistically to escape local minima."""
    current = list(initial_tour)  # Work on a mutable copy.
    current_distance = tour_length(current, dist_matrix)  # Evaluate baseline.
    best = list(current)  # Track the best route observed so far.
    best_distance = current_distance
    temperature = initial_temperature  # Starting "heat" controlling acceptance.

    for _ in range(iterations):
        i, j = sorted(random.sample(range(len(current)), 2))  # Random segment.
        candidate = current[:]  # Copy present route.
        candidate[i:j] = reversed(candidate[i:j])  # Reverse segment as move.
        candidate_distance = tour_length(candidate, dist_matrix)
        delta = candidate_distance - current_distance  # Difference vs. current.

        if delta < 0 or _accept(delta, temperature):
            current = candidate  # Accept new state (better or probabilistic).
            current_distance = candidate_distance
            if candidate_distance < best_distance:
                best = candidate  # Update global best, if improved.
                best_distance = candidate_distance

        temperature *= cooling_rate  # Cool system gradually.
        if temperature < 1e-4:
            break  # Stop when "cold" to avoid unnecessary iterations.

    return best  # Return best route discovered.


def genetic_algorithm(
    dist_matrix: Matrix,
    base_tour: Tour | None = None,
    population_size: int = 40,
    generations: int = 250,
    mutation_rate: float = 0.2,
    elite_size: int = 4,
) -> Tour:
    """Run a simple GA using tournament selection, ordered crossover, and swap mutation."""
    n = len(dist_matrix)  # Determine chromosome length (number of cities).
    population: List[Tour] = []  # Store candidate solutions.
    if base_tour:
        population.append(list(base_tour))  # Seed population with provided route.
    while len(population) < population_size:
        population.append(random_tour(n))  # Fill remaining slots randomly.

    for _ in range(generations):
        population.sort(key=lambda t: tour_length(t, dist_matrix))  # Sort by fitness.
        new_population = population[:elite_size]  # Preserve elites to maintain good genes.

        while len(new_population) < population_size:
            parent1 = _tournament(population, dist_matrix)  # Select via tournament.
            parent2 = _tournament(population, dist_matrix)
            child = _ordered_crossover(parent1, parent2)  # Recombine parent tours.
            if random.random() < mutation_rate:
                _swap_mutation(child)  # Inject diversity through mutation.
            new_population.append(child)
        population = new_population  # Move to the next generation.

    population.sort(key=lambda t: tour_length(t, dist_matrix))  # Final fitness ranking.
    return population[0]  # Return best individual.


def _edges_to_tour(edges: Iterable[tuple[int, int]], n: int) -> Tour:
    """Convert a set of degree-two edges into an ordered tour."""
    adjacency: List[List[int]] = [[] for _ in range(n)]  # Build adjacency lists.
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    start = 0  # Begin traversal at node zero (arbitrary).
    tour = [start]
    current = start
    previous = None  # Track previous node to avoid backtracking.

    for _ in range(n - 1):
        neighbors = adjacency[current]
        next_city_candidates = [nbr for nbr in neighbors if nbr != previous]
        if not next_city_candidates:
            next_city_candidates = neighbors  # Fallback if we must backtrack.
        next_city = next_city_candidates[0]
        tour.append(next_city)
        previous, current = current, next_city  # Advance pointers along cycle.
    return tour


def _accept(delta: float, temperature: float) -> bool:
    """Decide if a worse candidate should be accepted in simulated annealing."""
    if temperature <= 0:
        return False  # Cannot accept when temperature is zero.
    return math.exp(-delta / temperature) > random.random()  # Boltzmann acceptance.


def _tournament(population: Sequence[Tour], dist_matrix: Matrix, k: int = 4) -> Tour:
    """Select an individual by sampling k candidates and picking the fittest."""
    competitors = random.sample(population, k)
    competitors.sort(key=lambda t: tour_length(t, dist_matrix))
    return competitors[0][:]  # Return a copy of the winning chromosome.


def _ordered_crossover(parent1: Tour, parent2: Tour) -> Tour:
    """Perform ordered crossover (OX1) to preserve subsequences from parents."""
    n = len(parent1)
    start, end = sorted(random.sample(range(n), 2))  # Choose crossover slice.
    child = [None] * n  # type: ignore  # Placeholder for OX1 offspring.
    child[start:end] = parent1[start:end]  # Copy segment from first parent.

    fill_values = [city for city in parent2 if city not in child]  # Preserve order.
    fill_index = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill_values[fill_index]
            fill_index += 1
    return child  # type: ignore


def _swap_mutation(route: Tour) -> None:
    """Swap two randomly chosen genes in-place to introduce variation."""
    i, j = random.sample(range(len(route)), 2)
    route[i], route[j] = route[j], route[i]
