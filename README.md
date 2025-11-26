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

### Interfață GUI (Streamlit)

Poți testa rapid soluțiile și vizualizările dintr-o interfață web simplă folosind Streamlit:

```bash
pip install streamlit matplotlib
streamlit run app/gui.py
```

GUI-ul îți permite să încarci un CSV nou, să rulezi solver-ul (manual sau folosind modelul ML pentru auto-select) și să vezi ruta afișată pe „hartă” împreună cu distanța și parametrii utilizați.

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

6. **GUI pentru vizualizare și execuție rapidă**  
   - a) construiești o interfață (PyQt, Tkinter, Streamlit etc.) care îți permite să selectezi/încarci un nou CSV cu orașe.  
   - b) GUI-ul afișează orașele pe o „hartă” 2D și trasează linia rutei găsite, astfel încât să vezi vizual calea.  
   - c) după rularea solver-ului (manual sau auto), interfața prezintă detalii precum distanța totală, numărul de muchii, timpul de execuție și strategia folosită.

7. **Auto-Method Integration**  
   - a) extinzi CLI-ul cu `--auto-method` și `--selector-model`.  
   - b) când flag-ul e activ, încarci modelul, calculezi feature-urile instanței curente și alegi strategia prezisă.  
   - c) rulezi solver-ul cu parametrii recomandați; dacă modelul lipsește, revii la metoda aleasă manual.

8. **Validation & Documentation**  
   - a) testezi manual și cu script că fiecare strategie și modul automat produc rute valide pentru diverse instanțe.  
   - b) actualizezi README (sau wiki) cu exemple de rulare, cum se generează benchmark-urile și cum se antrenează / folosește selectorul ML.
