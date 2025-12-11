# Travelling Salesman Problem – Heuristic Toolkit

Acest proiect adună mai multe euristici folosite frecvent pentru problema comis-voiajer (Travelling Salesman Problem – TSP). Codul este scris în Python și poate fi extins cu noi instanțe sau metode de optimizare.

> Pentru un ghid pas cu pas (setup, comenzi CLI/GUI și pipeline complet) consultă și `PROCESS.md`.

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

### Aplicație Streamlit

Noua interfață grafică oferă:
- listă cu toate CSV-urile din `data/` și `data/instances/`, plus upload de fișiere noi;
- încărcarea modelului ML (`models/method_selector.pkl`) și rularea automată a strategiei recomandate imediat ce este găsită (rezultatul apare fără acțiuni suplimentare);
- posibilitatea de a alege manual altă combinație de heuristici (2-opt, Simulated Annealing și Genetic Algorithm sunt bifate implicit) și de a re-rula solver-ul doar când apeși **Rulează solver**;
- vizualizarea rutei pe un grafic 2D (puncte și muchii), plus tabelul cu orașe și parametrii detaliați; după fiecare rulare apare automat un al doilea tab unde poți vedea matricea completă a distanțelor euclidiene.

Rulare:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Aplicația afișează distanța (lungimea totală a turului), timpul de execuție (în secunde), numele metodei (ML sau manual), ordinea orașelor și parametrii folosiți. Când modifici manual setările, rularea automată se suspendă până la apăsarea butonului.

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

# Alte dataset-uri + start custom
python main.py --data data/instances/cities_042.csv --method nearest_neighbor --start 3
```

### Benchmark Pipeline

```bash
python -m benchmarks.pipeline --samples 30 --min-cities 15 --max-cities 50 --seed 7
```

Comanda de mai sus:
1. generează instanțe random în `data/instances/`,
2. rulează toate strategiile definite pentru fiecare instanță și salvează rezultatele brute în `benchmarks/raw_results.csv`,
3. construiește tabelul final cu feature-uri + strategia câștigătoare (`benchmarks/training_dataset.csv`), care va fi folosit la antrenarea modelului ML.

### Antrenarea modelului ML

```bash
python -m ml.train_model --dataset benchmarks/training_dataset.csv --output models/method_selector.pkl
```

Acest script citește tabelul cu feature-uri, antrenează un clasificator simplu (centroid) și salvează modelul într-un fișier pickle, gata de folosit pentru selecție automată.

## Roadmap: Heuristics + ML Selector

1. **Data Setup (deja acoperit)**  
   - a) creezi un CSV cu cel puțin 30 de orașe (`name,x,y`).  
   - b) adaugi un loader (`City`, `load_cities`) și o funcție care pregătește matricea distanțelor (`build_distance_matrix`).  
   - c) testezi rapid că `load_cities`+`build_distance_matrix` funcționează cu un `python3 - <<'PY' ...`.

2. **Heuristic Implementations (deja acoperit)**  
   - a) scrii funcțiile de bază: `tour_length`, `nearest_neighbor`, `greedy`, `random_tour`.  
   - b) adaugi `two_opt` pentru optimizare locală.  
   - c) implementezi metaeuristicile `simulated_annealing` și `genetic_algorithm` pentru a continua căutarea când soluția e blocată.

3. **TSPSolver & CLI (deja acoperit)**  
   - a) `TSPSolver` primește lista de orașe, calculează matricea distanțelor și expune `solve(method, apply_two_opt_flag, annealing, genetic, seed, start)`.  
   - b) `main.py` oferă argumente CLI (`--method`, `--apply-2opt`, `--annealing`, `--ga`, `--seed`, `--data`, `--start`) și afișează ruta plus distanța.

4. **Benchmark Dataset for Auto-Selection**  
   - a) generezi mai multe instanțe TSP (CSV-uri random sau extrase din alte surse).  
   - b) pentru fiecare instanță rulezi toate strategiile (NN, Greedy, combinații cu 2-opt, SA, GA).  
   - c) salvezi într-un fișier (ex. CSV) distanța și timpul fiecărei strategii, plus strategia câștigătoare (cea cu distanță minimă).

5. **Feature Engineering & Model Training**  
   - a) pentru fiecare instanță calculezi feature-uri simple: număr de orașe, media/mediana/max/min distanțelor, deviația standard, aria bounding box, media celui mai apropiat vecin.  
   - b) folosești tabela cu „feature-uri + strategie câștigătoare” ca dataset și antrenezi un model de clasificare (în prezent un clasificator pe centroid).  
   - c) salvezi modelul într-un fișier (JSON/pickle) pentru a-l încărca ulterior.

6. **GUI pentru vizualizare și execuție rapidă (acoperit cu Streamlit)**  
   - a) aplicația `streamlit run app/app.py` permite selectarea/încărcarea instanțelor și rularea solver-ului cu sau fără model ML.  
   - b) UI-ul afișează orașele și traseul pe grafic, plus statisticile relevante.  
   - c) utilizatorii pot refuza sugestia ML și pot alege manual o altă metodă.

7. **Auto-Method Integration (CLI)**  
   - a) următorul pas este expunerea flag-urilor `--auto-method`/`--selector-model` și în CLI.  
   - b) când va fi disponibil, CLI-ul va reutiliza aceleași strategii și model ca aplicația Streamlit.  
   - c) configurațiile implicite vor fi armonizate între CLI și UI.

8. **Validation & Documentation**  
   - a) testezi manual și cu script că fiecare strategie și modul automat produc rute valide pentru diverse instanțe.  
   - b) actualizezi README (sau wiki) cu exemple de rulare, cum se generează benchmark-urile și cum se antrenează / folosește selectorul ML.
