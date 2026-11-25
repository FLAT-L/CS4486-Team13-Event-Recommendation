# CS4486 Team13: Event Recommendation

Source notebooks and fixed experiment settings for the Kaggle Event Recommendation Engine Challenge. The project first reconstructs the historical seventh-place method, then studies direct XGBoost features, masked event pretraining, dense compression, and positive/negative friend histories.

## Results

All values below are local user-grouped five-fold development MAP@200, not official Kaggle leaderboard placements.

| Model | Method | OOF MAP@200 |
|---|---|---:|
| 01 | Seventh-place RF/LR reconstruction | 0.704493 |
| 02 | Direct tabular XGBoost | 0.721340 |
| 03 | Masked event residuals + XGBoost, 3:1 sampling | 0.717325 |
| 04 | Supervised FT-Transformer | 0.692191 |
| 05 | Cleaned and tuned XGBoost | 0.729627 |
| 06 | 64-dimensional dense event compression + XGBoost | 0.717045 |
| 09 | Positive friend-history profiles + XGBoost | 0.727866 |
| 15 | Explicit non-attendance friend-history profiles + XGBoost | 0.731427 |

## Files and execution

- `models/`: the eight model notebooks, support notebooks, fixed parameters, aggregate metrics and verification summaries.
- `models/environment/shared.ipynb`: feature construction and scoring used by the tree models.
- `models/environment/requirements.txt`: exact environment inventory from the tested macOS Apple Silicon runtime.
- `readme/data_processing.html`: cleaning, field definitions and matrix construction.
- `readme/model_specifications.html`: architecture, objectives, fixed settings and missing-value behavior.
- `readme/experiment_dag.html`: experimental relationships.
- `benchmark.html`: score visualization.

**This is the public source edition.** The complete offline course handoff is `cs4486_eight_models.zip`, supplied separately to course reviewers. It contains the raw tables, stored folds, candidate feature caches, checkpoints, interpreter and offline wheels used by notebook verification sections. These data and binary artifacts are not included in this repository. Cloning this repository alone is therefore insufficient to execute the checkpoint-verification cells.

For the complete handoff, extract the archive and follow `models/README.html`. Run the verification sections first, then the optional feature rebuilding and retraining sections. Preserve the directory structure. The source notebooks here mirror that handoff, with saved cell outputs cleared for publication; model training code is unchanged. Kaggle data must be obtained through the [competition data page](https://www.kaggle.com/competitions/event-recommendation-engine-challenge/data).

The archived environment uses Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.9.0, XGBoost 3.4.1, PyTorch 2.14.0 and rtdl_revisiting_models 0.0.2. Use the handoff's bundled wheels for the tested offline setup; the inventory is not a claim that every package is currently available for every platform.

## Evaluation details

The same stored user groups are used throughout. Each supervised fold excludes its validation users. Tuned configurations were selected once using development splits and then fixed for all five folds, so the reported scores are development OOF evaluation rather than nested CV. Model 09 uses explicit `event_attendees.yes` histories; Model 15 replaces these with explicit `no` histories. An unrecorded response is not a negative response. Both first average events separately for each friend, then average the available friends equally, field by field.

For 03 and 06, cached neural coordinates reproduce the listed scores. CPU-regenerated coordinates with the saved trees yield 0.717494 and 0.716736 respectively. The detailed verification JSON files record both paths.

## Sources

- [Competition](https://www.kaggle.com/competitions/event-recommendation-engine-challenge)
- [Andrei Olariu's historical source](https://github.com/andreiolariu/kaggle-event-recommendation): Model 01 reconstructs this approach and supplies missing preprocessing. The historical feature logic is attributed to its original author.
- [XGBoost paper](https://doi.org/10.1145/2939672.2939785)
- [FT-Transformer paper](https://arxiv.org/abs/2106.11959) and [implementation](https://github.com/yandex-research/rtdl-revisiting-models)

Original upstream works retain their respective rights and licenses. No new blanket license is applied to third-party source.
