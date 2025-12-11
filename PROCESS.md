# Ghid rapid pentru Travelling Salesman Problem – Heuristic Toolkit

Documentul acesta explică, pas cu pas, ce face proiectul și cum îl rulezi din terminal dacă îl vezi pentru prima dată.

## 1. Pentru ce este proiectul?
- **Obiectiv**: oferă un toolkit în Python pentru problema comis-voiajer (Traveling Salesman Problem – TSP) care combină euristici constructive, optimizare locală și metaeuristici.
- **Ce include**:
  - încărcare de instanțe TSP din CSV (`data/cities_50.csv` sau `data/instances/*.csv`);
  - euristici: Nearest Neighbor, Greedy, Random + 2-opt;
  - metaeuristici: Simulated Annealing, Genetic Algorithm;
  - `TSPSolver` + CLI (`main.py`) + aplicație Streamlit (`app/app.py`);
  - pipeline de benchmark + selector ML (centroid) care prezice ce strategie să alegi.

## 2. Structura minimă
```
.
├── data/                  # Instanțe CSV (name,x,y)
├── tsp/                   # Cod principal (data loader, heuristici, solver)
├── benchmarks/            # Pipeline instanțe -> benchmark -> dataset
├── ml/                    # Feature engineering + model selector
├── models/                # Modelul ML salvat (method_selector.pkl)
├── requirements.txt
├── main.py                # CLI oficial descris în README
└── app/app.py             # Interfață Streamlit
```

## 3. Setup rapid
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Rulare CLI (`main.py`)
CLI-ul acoperă exact argumentele din README (metoda, 2-opt, SA, GA, seed etc.).

```bash
# Soluție de bază NN + 2-opt
python main.py --method nearest_neighbor --apply-2opt

# Greedy + Simulated Annealing determinist
python main.py --method greedy --annealing --seed 7

# Tour aleator îmbunătățit cu GA și 2-opt final
python main.py --method random --ga --apply-2opt --seed 123

# Specifici alt CSV și oraș de start
python main.py --data data/instances/cities_042.csv --method nearest_neighbor --start 3
```

Ce face scriptul:
1. citește CSV-ul (`--data`);
2. validează `--start`;
3. creează `TSPSolver`;
4. rulează `solve()` cu opțiunile selectate;
5. afișează distanța totală și tour-ul (index și nume).

## 5. Aplicația Streamlit (`app/app.py`)
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Funcționalități:
- listă cu toate fișierele CSV existente (inclusiv `data/instances`) + posibilitatea de upload;
- încărcarea modelului ML (`models/method_selector.pkl`), iar când auto-mode este activ rezultatul recomandat se rulează instant și apare pe grafic;
- zonele **Distanță** și **Durată** includ tooltip-uri explicative; seed-ul implicit este 42, dar poate fi golit pentru comportament aleator;
- casetele „Aplică 2-opt”, „Rulează Simulated Annealing” și „Rulează Genetic Algorithm” sunt bifate implicit (au descrieri care explică rolul fiecăreia); orice modificare manuală suspendă auto-run-ul până când apeși **Rulează solver**;
- după fiecare rulare apare automat un tab secundar „Matrice distanțe” cu matricea completă a distanțelor pentru dataset-ul curent.
- vizualizare grafică 2D (puncte + muchii) + tabelul cu orașe, parametrii și ordinea turului.

## 6. Vizualizări rapide
Nu mai există un script separat; vizualizarea este integrată direct în aplicația Streamlit.

## 7. Pipeline Benchmark + ML
### 7.1 Benchmarkarea tuturor instanțelor existente
```bash
python -m benchmarks.benchmark_strategies \
  --instances-dir data/instances \
  --output benchmarks/raw_results.csv \
  --seed 42
```
- rulează toate strategiile din `ml/strategies.py` pe fiecare CSV, salvează distanța și timpul.

### 7.2 Construirea dataset-ului pentru ML
```bash
python -m benchmarks.build_training_table \
  --instances-dir data/instances \
  --raw-results benchmarks/raw_results.csv \
  --output benchmarks/training_dataset.csv
```
- pentru fiecare instanță identifică strategia câștigătoare (distanță minimă) și adaugă feature-uri extrase din date.

### 7.3 Antrenarea modelului selector
```bash
python -m ml.train_model \
  --dataset benchmarks/training_dataset.csv \
  --output models/method_selector.pkl
```
- încarcă dataset-ul, normalizează feature-urile, calculează centroidul per strategie și salvează modelul.

### 7.4 Pipeline complet (generare + benchmark + dataset)
```bash
python -m benchmarks.pipeline \
  --output-dir data/instances \
  --samples 30 \
  --min-cities 15 \
  --max-cities 50 \
  --seed 7
```
- generează instanțe noi și rulează automat pașii 7.1–7.3.

## 8. Integrarea modelului ML
- Aplicația Streamlit încarcă `models/method_selector.pkl`, calculează feature-urile instanței curente și alege automat un label din `ml/strategies.STRATEGIES`.  
- CLI-ul încă nu are flag `--auto-method`; dacă vrei comportament identic în terminal, rulează manual pipeline-ul și invocă `TSPSolver` cu parametrii recomandați de model.

## 9. Referințe utile
- `tsp/data.py`: citire CSV + matrice distanțe.
- `tsp/heuristics.py`: implementările tuturor euristicilor.
- `tsp/solver.py`: wrapper peste heuristici + orchestrare.
- `README.md`: descriere detaliată a proiectului, plan + roadmap.

Cu pașii de mai sus poți rula solver-ul (CLI sau Streamlit) și întreg pipeline-ul fără informații suplimentare.
