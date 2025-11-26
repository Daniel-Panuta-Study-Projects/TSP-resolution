from __future__ import annotations

import csv
import io
import sys
import time
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.features import extract_features
from ml.model import load_model
from ml.strategies import STRATEGIES
from tsp.data import City, load_cities
from tsp.solver import TSPSolver

DEFAULT_DATA = Path("data/cities_50.csv")
DEFAULT_MODEL_PATH = Path("models/method_selector.pkl")


def main() -> None:
    st.set_page_config(page_title="TSP Heuristic Explorer", layout="wide")
    st.title("Travelling Salesman Problem – Heuristic Explorer")

    if "suggested_strategy" not in st.session_state:
        st.session_state["suggested_strategy"] = None

    with st.sidebar:
        st.header("Input data")
        uploaded = st.file_uploader(
            "Upload CSV (name,x,y)",
            type=["csv"],
            help="If empty, the default data/cities_50.csv will be used.",
        )
        data_preview = st.checkbox("Show raw city data", value=False)
        use_auto = st.checkbox("Auto-select strategy via ML model", value=False)
        model_path = st.text_input(
            "Model path",
            value=str(DEFAULT_MODEL_PATH),
            help="Used only when auto selection is enabled.",
        )

        st.markdown("---")
        st.subheader("Manual strategy")
        method = st.selectbox(
            "Constructive heuristic",
            options=["nearest_neighbor", "greedy", "random"],
            index=0,
        )
        apply_2opt = st.checkbox("Apply 2-opt", value=True)
        use_annealing = st.checkbox("Run Simulated Annealing", value=False)
        use_ga = st.checkbox("Run Genetic Algorithm", value=False)

        suggest_button = st.button("Suggest strategy", use_container_width=True)
        run_button = st.button("Solve TSP", type="primary")

    cities = load_cities_from_input(uploaded) if uploaded else load_cities(DEFAULT_DATA)

    if suggest_button:
        suggestion = suggest_strategy(
            cities=cities,
            model_path=Path(model_path),
        )
        st.session_state["suggested_strategy"] = suggestion

    if st.session_state.get("suggested_strategy"):
        st.info(
            f"Suggested strategy: **{st.session_state['suggested_strategy']}**. "
            "Poți rula solver-ul cu recomandarea sau să alegi altă metodă manual."
        )

    if data_preview:
        st.subheader("City table")
        st.dataframe(
            [{"name": city.name, "x": city.x, "y": city.y} for city in cities],
            use_container_width=True,
        )

    if run_button:
        with st.spinner("Solving..."):
            result = solve_instance(
                cities=cities,
                use_auto=use_auto,
                model_path=Path(model_path),
                manual_params={
                    "method": method,
                    "apply_two_opt_flag": apply_2opt,
                    "annealing": use_annealing,
                    "genetic": use_ga,
                    "start": 0,
                },
            )
        display_results(cities, result)
    else:
        st.info("Set your options in the sidebar and click **Solve TSP**.")


def load_cities_from_input(uploaded_file) -> List[City]:
    """Parse uploaded CSV content into City objects."""
    text = uploaded_file.getvalue().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    cities = [
        City(row["name"], float(row["x"]), float(row["y"]))
        for row in reader
    ]
    if not cities:
        raise ValueError("Uploaded CSV contains no cities.")
    return cities


def solve_instance(
    cities: Sequence[City],
    use_auto: bool,
    model_path: Path,
    manual_params: dict,
):
    """Solve the TSP instance using manual or auto-selected arguments."""
    solver = TSPSolver(cities)
    strategy_label = None
    params = manual_params

    if use_auto:
        if not model_path.exists():
            st.warning(
                f"Model file {model_path} not found. Falling back to manual settings."
            )
        else:
            model = load_model(model_path)
            features = extract_features(cities, solver.distance_matrix)
            strategy_label = model.predict(features)
            params = dict(STRATEGIES[strategy_label])
            params.setdefault("start", 0)

    start_time = time.perf_counter()
    solution = solver.solve(**params)
    elapsed = time.perf_counter() - start_time

    return {
        "solution": solution,
        "strategy_label": strategy_label,
        "params": params,
        "elapsed": elapsed,
    }


def display_results(cities: Sequence[City], result: dict) -> None:
    solution = result["solution"]
    params = result["params"]
    elapsed = result["elapsed"]
    strategy_label = result["strategy_label"]

    st.subheader("Results")
    cols = st.columns(4)
    cols[0].metric("Cities", f"{len(cities)}")
    cols[1].metric("Distance", f"{solution.distance:.2f}")
    cols[2].metric("Elapsed (s)", f"{elapsed:.4f}")
    cols[3].metric(
        "Method",
        strategy_label or params["method"],
    )

    if strategy_label:
        st.success(f"Auto-selected strategy: **{strategy_label}**")

    fig = plot_route(cities, solution.route_indices)
    st.pyplot(fig, clear_figure=True)

    with st.expander("Route order"):
        st.write(solution.as_city_names(cities))

    with st.expander("Parameters used"):
        st.json(params)


def plot_route(cities: Sequence[City], route: Sequence[int]):
    xs = [city.x for city in cities]
    ys = [city.y for city in cities]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, color="tab:blue", s=40, zorder=2)

    for idx, city in enumerate(cities):
        ax.annotate(city.name, (city.x, city.y), xytext=(3, 3), textcoords="offset points", fontsize=8)

    for i in range(len(route)):
        a = cities[route[i]]
        b = cities[route[(i + 1) % len(route)]]
        ax.plot([a.x, b.x], [a.y, b.y], color="tab:red", linewidth=1.5, zorder=1)

    ax.set_title("Computed Tour")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig


def suggest_strategy(cities: Sequence[City], model_path: Path) -> str | None:
    """Use the ML model to recommend a strategy without enforcing it."""
    if not model_path.exists():
        st.warning(
            f"Model file {model_path} missing; nu pot genera o recomandare."
        )
        return None
    solver = TSPSolver(cities)
    model = load_model(model_path)
    features = extract_features(cities, solver.distance_matrix)
    label = model.predict(features)
    st.success(f"Modelul recomandă strategia **{label}**.")
    return label


if __name__ == "__main__":
    main()
