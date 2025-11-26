# ML Selection Notes

## Why record every strategy result?

- The CSV (`instance_name, strategy, distance, time`) is the raw dataset for ML. Without it, we have no labels indicating which strategy truly performed best on each instance.
- Having all strategies benchmarked per instance lets us objectively decide the winner (minimum distance). That winner becomes the “ground truth” label for training.
- Aggregating measurements also reveals patterns—e.g., where certain heuristics dominate—which guides feature engineering and model selection.

## What does the ML model train on?

1. **Features** extracted from each instance (e.g., number of cities, mean distance, bounding-box size, average nearest-neighbor distance, variance).  
2. **Label** pulled from the benchmark table: the strategy with the smallest distance for that instance.  
3. The model learns a mapping `features → best_strategy`, so for a new instance it predicts which heuristic combo to run without benchmarking all of them again.

## How good will the model be?

- Quality depends heavily on the diversity and quantity of benchmarked instances plus the relevance of the chosen features.  
- If the dataset covers varied scenarios (small vs. large graphs, dense vs. sparse distributions) and features capture those differences, the classifier will generalize better.  
- Conversely, if all benchmarks look similar or are too few, the model may overfit and mis-predict on unseen instances.
