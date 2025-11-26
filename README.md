# Travelling Salesman Problem – Heuristic Toolkit

Acest proiect adună mai multe euristici folosite frecvent pentru problema comis-voiajer (Travelling Salesman Problem – TSP). Codul este scris în Python și poate fi extins cu noi instanțe sau metode de optimizare.

## Plan de lucru propus

1. **Definirea datelor** – Alegem un set de orașe cu coordonate planare (CSV în `data/cities_50.csv`) și construim utilitare pentru a le încărca rapid.
2. **Modelarea problemei** – Implementăm funcții pentru calculul distanțelor, evaluarea unui tur și operațiile de bază asupra permutărilor de orașe.
3. **Euristici constructive** – Implementăm metode rapide care generează un tur inițial: Nearest Neighbor și Greedy (Kruskal-like).
4. **Euristici de îmbunătățire locală** – Aplicăm 2-opt peste tururile inițiale pentru a elimina intersecțiile și a scurta traseul.
5. **Metaeuristici** – Adăugăm metode stocastice (Simulated Annealing, Genetic Algorithm simplificat) care pornesc cu soluțiile constructive și continuă căutarea în spațiul permutărilor.
6. **Instrumentare și rulare** – Construim un mic CLI (`python main.py --method=nearest_neighbor` etc.) care compară distanțele obținute și afișează rezultatele.

Urmează să adăugăm și alte metode (de ex. Tabu Search, Ant Colony) dacă este nevoie, păstrând aceeași infrastructură.

## Fișierul de date

`data/cities_50.csv` conține 50 de orașe fictive în planul 2D. Coloanele sunt `name,x,y`, iar distanțele sunt calculate cu formula lui Pitagora (distanță Euclidiană).

### Vizualizarea orașelor

Pentru o vedere rapidă a pozițiilor poți rula:

```bash
python app/plot_cities.py --data data/cities_50.csv --show-labels
```

Scriptul folosește matplotlib pentru a desena punctele (instalează-l cu `pip install matplotlib` dacă lipsește).

## Rulare rapidă

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # (dacă vom adăuga dependențe externe)
python main.py --method nearest_neighbor --apply-2opt
```

Flag-ul `--method` controlează euristica de bază, iar `--apply-2opt` și `--annealing`/`--ga` activează optimizările suplimentare.

### Opțiuni CLI (`main.py`)

| Flag               | Valori/Tip                       | Descriere                                                                                  | Implicit                |
|--------------------|----------------------------------|---------------------------------------------------------------------------------------------|-------------------------|
| `--data`           | Path către CSV                   | Setul de orașe folosit în test (`name,x,y`).                                                | `data/cities_50.csv`    |
| `--method`         | `nearest_neighbor`, `greedy`, `random` | Euristica constructivă folosită pentru ruta inițială.                                       | `nearest_neighbor`      |
| `--start`          | Int                              | Indexul orașului de start pentru Nearest Neighbor.                                          | `0`                     |
| `--apply-2opt`     | Flag                             | Activează local search 2-opt peste traseul curent.                                          | dezactivat              |
| `--annealing`      | Flag                             | Rulează Simulated Annealing pornind din soluția curentă.                                    | dezactivat              |
| `--ga`             | Flag                             | Rulează un algoritm genetic inițializat cu turul curent.                                    | dezactivat              |
| `--seed`           | Int                              | Sămânță pentru random pentru a obține rezultate reproductibile.                            | `None` (aleator)        |

### Exemple de rulare

```bash
# NN + 2-opt
python main.py --method nearest_neighbor --apply-2opt

# Greedy + annealing
python main.py --method greedy --annealing --seed 42

# Random tour îmbunătățit cu GA
python main.py --method random --ga --apply-2opt
```

## Roadmap: Heuristics + ML Selector

1. **Data Setup**  
   - Creează un CSV cu ≥30 orașe (`name,x,y`) și utilitare Python pentru încărcare (`City`, `load_cities`, `build_distance_matrix`).  
   - Asigură-te că matricea distanțelor este reutilizabilă de toate metodele.

2. **Heuristic Implementations**  
   - Adaugă metode constructive (`nearest_neighbor`, `greedy`, `random`) și funcția `tour_length`.  
   - Implementează optimizări locale (`two_opt`) și metaeuristici (`simulated_annealing`, `genetic_algorithm`).

3. **TSPSolver & CLI**  
   - Compune un `TSPSolver` care acceptă lista de orașe și expune `solve(method, apply_2opt, annealing, ga, seed)`.  
   - Construiește `main.py` cu argumente CLI pentru configurarea rapidă a combinațiilor de euristici.

4. **Benchmark Dataset for Auto-Selection**  
   - Generează sau colectează multiple instanțe TSP pentru a măsura performanța fiecărei strategii.  
   - Rulează toate strategiile pe fiecare instanță și memorează distanța/ timpul pentru a identifica câștigătorul.

5. **Feature Engineering & Model Training**  
   - Extrage caracteristici precum număr de orașe, statisticile distanțelor, aria bounding box, medie nearest neighbor etc.  
   - Antrenează un model simplu (centroid classifier, random forest etc.) care mapează feature-urile la strategia câștigătoare și salvează-l (JSON/pickle).

6. **Auto-Method Integration**  
   - Adaugă opțiuni CLI (`--auto-method`, `--selector-model`) care încarcă modelul, prezic strategia și rulează solver-ul cu parametrii recomandați.  
   - Păstrează fallback-ul manual: dacă modelul lipsește sau `--auto-method` e dezactivat, se folosesc opțiunile setate explicit.

7. **Validation & Documentation**  
   - Testează manual/automat că fiecare strategie și modul automat produc soluții valide pe dataset-ul principal.  
   - Actualizează README cu pașii de rulare, exemple pentru fiecare mod și instrucțiunile despre antrenarea/folosirea selectorului ML.
