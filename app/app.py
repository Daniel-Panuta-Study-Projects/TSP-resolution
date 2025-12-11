from __future__ import annotations

import csv
import io
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from ml.features import extract_features
from ml.model import load_model
from ml.strategies import STRATEGIES
from tsp.data import City, load_cities
from tsp.solver import TSPSolver

DEFAULT_DATA = ROOT / "data" / "cities_50.csv"
INSTANCES_DIR = ROOT / "data" / "instances"
DEFAULT_MODEL = ROOT / "models" / "method_selector.pkl"
FRIENDLY_STRATEGIES = {
    "nn": "Nearest Neighbor",
    "nn_2opt": "Nearest Neighbor",
    "nn_anneal": "Nearest Neighbor",
    "greedy_2opt": "Greedy",
    "greedy_anneal": "Greedy",
    "random_ga": "Random",
}
MANUAL_METHOD_NAMES = {
    "nearest_neighbor": "Nearest Neighbor",
    "greedy": "Greedy",
    "random": "Random",
}


def main() -> None:
    st.set_page_config(page_title="TSP Solver Control Panel", layout="wide")
    st.title("TSP Solver – ML Assisted")

    dataset_choices = discover_datasets()
    last_result = st.session_state.get("last_result")

    with st.sidebar:
        st.header("Date de intrare")
        dataset_path = st.selectbox(
            "CSV existent",
            options=dataset_choices,
            format_func=lambda p: _format_dataset_label(p),
        )
        uploaded = st.file_uploader(
            "sau încarcă un CSV nou (name,x,y)",
            type=["csv"],
            help="Dacă încarci un fișier, selecția din listă este ignorată.",
        )
        st.markdown("---")
        st.header("Selecție strategie")
        use_auto = st.checkbox("Folosește modelul ML pentru recomandare", value=True)
        model_path_str = st.text_input(
            "Fișier model (.pkl)",
            value=str(DEFAULT_MODEL),
            help="Folosit doar când este bifată recomandarea ML.",
        )
        manual_method = st.selectbox(
            "Metodă manuală",
            options=["nearest_neighbor", "greedy", "random"],
            index=0,
        )
        apply_two_opt = st.checkbox(
            "Aplică 2-opt",
            value=True,
            help="Heuristică locală care elimină intersecțiile din traseu prin inversarea segmentelor.",
        )
        use_annealing = st.checkbox(
            "Rulează Simulated Annealing",
            value=True,
            help="Metaeuristică stocastică ce explorează vecini ai soluției pentru a scăpa de minime locale.",
        )
        use_ga = st.checkbox(
            "Rulează Genetic Algorithm",
            value=True,
            help="Algoritm evolutiv care combină rute candidate și aplică mutații pentru a îmbunătăți turul.",
        )
        seed_text = st.text_input(
            "Seed (opțional)",
            value="42",
            help="Lasă gol pentru comportament aleator.",
        )
        manual_start = st.number_input(
            "Oraș de start (index)",
            min_value=0,
            value=0,
        )

        run_button = st.button("Rulează solver", type="primary", use_container_width=True)

    cities, source_label = load_dataset(dataset_path, uploaded)

    st.subheader(f"Date ({source_label})")
    st.dataframe([{"name": c.name, "x": c.x, "y": c.y} for c in cities])

    start_index = int(manual_start)
    start_index = max(0, min(start_index, len(cities) - 1))

    manual_params = {
        "method": manual_method,
        "apply_two_opt_flag": apply_two_opt,
        "annealing": use_annealing,
        "genetic": use_ga,
        "start": start_index,
    }
    manual_params["seed"] = parse_seed(seed_text)

    manual_signature = (
        manual_method,
        apply_two_opt,
        use_annealing,
        use_ga,
        seed_text.strip(),
        start_index,
    )
    prev_signature = st.session_state.get("manual_signature")
    manual_changed = prev_signature is not None and manual_signature != prev_signature
    st.session_state["manual_signature"] = manual_signature
    if manual_changed:
        st.session_state["manual_pending"] = True
    manual_pending = st.session_state.get("manual_pending", False)

    suggested_label = suggest_strategy(cities, Path(model_path_str)) if use_auto else None

    auto_preview_shown = False
    if use_auto and suggested_label and not run_button and not manual_pending:
        execute_solver(
            cities=cities,
            source_label=source_label,
            use_auto=True,
            model_path=Path(model_path_str),
            manual_params=manual_params,
            suggested_label=suggested_label,
            auto_preview=True,
            cache_key="auto_result",
        )
        auto_preview_shown = True

    if run_button:
        st.session_state["manual_pending"] = False
        execute_solver(
            cities=cities,
            source_label=source_label,
            use_auto=False,
            model_path=Path(model_path_str),
            manual_params=manual_params,
            suggested_label=None,
            cache_key="last_result",
        )
    elif auto_preview_shown:
        pass
    elif last_result:
        payload = dict(last_result)
        payload.setdefault("distance_matrix", last_result.get("distance_matrix"))
        show_results(**payload)
    else:
        st.info("Configurează opțiunile în bara laterală și apasă **Rulează solver**.")


def discover_datasets() -> List[Path]:
    files: List[Path] = []
    if DEFAULT_DATA.exists():
        files.append(DEFAULT_DATA)
    if INSTANCES_DIR.exists():
        files.extend(sorted(INSTANCES_DIR.glob("*.csv")))
    return files or [DEFAULT_DATA]


def load_dataset(selected: Path, uploaded) -> tuple[List[City], str]:
    if uploaded:
        content = uploaded.getvalue().decode("utf-8")
        cities = []
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            cities.append(City(row["name"], float(row["x"]), float(row["y"])))
        if not cities:
            st.error("Fișierul încărcat nu conține orașe.")
            st.stop()
        return cities, f"upload ({uploaded.name})"
    try:
        cities = load_cities(selected)
    except FileNotFoundError:
        st.error(f"Nu am găsit fișierul {selected}.")
        st.stop()
    return cities, selected.name


def _format_dataset_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def friendly_strategy_label(label: str | None) -> str:
    if not label:
        return ""
    return FRIENDLY_STRATEGIES.get(label, label)


def describe_manual_strategy(params: Dict) -> str:
    method_key = params.get("method", "")
    return MANUAL_METHOD_NAMES.get(method_key, method_key)


def parse_seed(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        st.warning("Seed-ul trebuie să fie un număr întreg. Ignor valoarea introdusă.")
        return None


def suggest_strategy(cities: Sequence[City], model_path: Path) -> str | None:
    if not model_path.exists():
        st.warning(f"Fișierul de model {model_path} lipsește. Selectează manual metoda dorită.")
        return None
    solver = TSPSolver(cities)
    model = load_model(model_path)
    features = extract_features(cities, solver.distance_matrix)
    label = model.predict(features)
    st.success(f"Strategia recomandată automat: **{friendly_strategy_label(label)}**")
    return label


def execute_solver(
    cities: Sequence[City],
    source_label: str,
    use_auto: bool,
    model_path: Path,
    manual_params: Dict,
    suggested_label: str | None,
    auto_preview: bool = False,
    cache_key: str | None = "last_result",
) -> None:
    solver = TSPSolver(cities)
    params = dict(manual_params)
    strategy_label = None

    if use_auto:
        if suggested_label and suggested_label in STRATEGIES:
            strategy_label = suggested_label
            params = dict(STRATEGIES[strategy_label])
            params.setdefault("start", manual_params.get("start", 0))
            if manual_params.get("seed") is not None:
                params["seed"] = manual_params["seed"]
        else:
            st.warning("Nu există o recomandare ML disponibilă. Folosesc setările manuale.")

    start_time = time.perf_counter()
    solution = solver.solve(**params)
    elapsed = time.perf_counter() - start_time

    payload = {
        "cities": cities,
        "solution": solution,
        "elapsed": elapsed,
        "source_label": source_label,
        "auto_label": strategy_label,
        "params": params,
        "distance_matrix": solver.distance_matrix,
    }
    show_results(**payload)

    if cache_key:
        st.session_state[cache_key] = payload

    if auto_preview:
        st.caption("Rezultatul a fost generat automat folosind recomandarea ML.")


def show_results(
    cities: Sequence[City],
    solution,
    elapsed: float,
    source_label: str,
    auto_label: str | None,
    params: Dict,
    distance_matrix=None,
) -> None:
    st.subheader("Rezultate")
    method_display = (
        friendly_strategy_label(auto_label) if auto_label else describe_manual_strategy(params)
    )
    cols = st.columns(4)
    cols[0].metric("Orașe", len(cities))
    cols[1].metric("Distanță", f"{solution.distance:.2f}", help="Lungimea totală a turului (suma distanțelor dintre orașe).")
    cols[2].metric("Durată (s)", f"{elapsed:.4f}", help="Timpul necesar pentru a calcula soluția curentă.")
    cols[3].metric("Metodă", method_display)

    if auto_label:
        st.success(f"Modelul ML a ales strategia **{friendly_strategy_label(auto_label)}**.")

    tab_labels = ["Turul TSP"]
    if distance_matrix is not None:
        tab_labels.append("Matrice distanțe")
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        fig = plot_route(cities, solution.route_indices)
        st.pyplot(fig, clear_figure=True)

    if distance_matrix is not None and len(tabs) > 1:
        with tabs[1]:
            st.caption("Matricea distanțelor (Euclidiene) pentru orașele încărcate.")
            names = [city.name for city in cities]
            df = pd.DataFrame(distance_matrix, columns=names, index=names)
            st.dataframe(df, use_container_width=True)

    with st.expander("Ordinea orașelor"):
        st.write(solution.as_city_names(cities))

    with st.expander("Parametri utilizați"):
        st.json(
            {
                "dataset": source_label,
                "auto_strategy": auto_label,
                **params,
                "elapsed_seconds": elapsed,
            }
        )


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

    ax.set_title("Turul TSP")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    main()
