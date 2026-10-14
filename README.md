# Daily irrigation scheduling with an LSTM

**A completed, reproducible synthetic experiment, not a field-validated irrigation product.**

This repository supports a five-part question about training an LSTM for a corn
farm's April 1–May 30 irrigation schedule, data requirements, transfer learning,
an unexpected May 1 storm, and training for environmental changes.

## Start here

1. Open **REPORT.html** in any browser. It is self-contained and includes all six
   figures and full result tables; no web connection is needed.
2. Read **docs/PROFESSOR_RESPONSE.md** for the proposed response to parts (a)–(e).
3. Read **docs/METHODS.md** for exact inputs, equations, assumptions, event timing,
   training setup, splits and sources.
4. Inspect **results/** for raw metrics, every test prediction, daily control
   trajectories, training histories, validation results and run metadata.

## What was actually executed

- 39 trained neural models across seeds 11, 22, 33.
- Eight source farms and two unseen target farms; all are synthetic.
- Six training seasons per source farm (2,880 daily decisions), one validation
  season, and two separate test seasons. No randomly split overlapping windows.
- LSTM versus MLP; 1/3/6 training seasons; 7/14/30 history days.
- Zero-shot, head-only adaptation, full adaptation and target-only training.
- Ordinary versus equally sized storm-augmented training datasets.
- Closed-loop simulation with teacher, learned, no-irrigation, guard-ablation and
  intentionally stale-schedule controls.
- A before-decision 80 mm rainfall event on May 1, with fixed learned weights.

Results are deliberately not all favorable to the LSTM. The MLP has lower
source imitation error; adaptation hurts one target farm; extreme augmentation
does not improve imitation accuracy. The report explains these outcomes rather
than hiding them. No parameter search was performed on test outcomes.

## Install and reproduce

Python 3.10+; executed with Python 3.12.14 on CPU. No GPU or external dataset is
needed. In a fresh virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiments.py --output rerun
python make_report.py --root rerun
```

The complete training/evaluation run took approximately one to two minutes on
the execution machine with two PyTorch threads. Other machines may be slower.
`requirements-executed.txt` records the exact top-level package versions. Small
numeric differences across PyTorch versions/platforms are possible. The archived
three-seed rerun was compared with the first execution; see
`results/reproducibility_check.json`.

For the included checkpoint and input example:

```bash
python predict.py \
  --checkpoint checkpoints/lstm_seed11.pt \
  --input data/example_input.json
```

This returns event probability, raw and guarded gross depth, and operating hours
when gross application rate is supplied. The checkpoint is trained on synthetic
labels and is **not suitable for operating a real irrigation system as supplied**.
The example uses a simulated target-farm record; it is not a real observation.

To regenerate the included outputs in place, omit `--output rerun` and use
`python make_report.py`. Only do this if you intend to replace the saved results.

## Files

| Path | Contents |
|---|---|
| `src/simulator.py` | Farms, generated weather, soil bucket, teacher and sequences |
| `src/model.py` | LSTM/MLP, normalization, training, inference and checkpoints |
| `run_experiments.py` | All experimental comparisons and closed-loop evaluations |
| `predict.py` | JSON input interface for an existing checkpoint |
| `make_report.py` | Six PNG/SVG figures, summaries and offline HTML report |
| `tests/test_core.py` | Water balance, calendar/causality, forecast, guard and checkpoint tests |
| `data/*.csv.gz` | Full generated daily datasets, including warmup |
| `data/farms.csv` | Ten synthetic farm parameter sets |
| `data/example_input.json` | Runnable ordered-feature inference example |
| `checkpoints/*.pt` | All 39 trained models with source normalization and metadata |
| `results/prediction_metrics.csv` | All one-step evaluation summaries |
| `results/all_test_predictions.csv.gz` | Every test label, probability and depth prediction |
| `results/rollout_metrics.csv` | Per-controller, per-farm-season control outcomes |
| `results/all_rollout_daily.csv.gz` | All daily controller/environment trajectories |
| `results/example_daily.csv` | Prespecified illustrative case used in the storm figure |
| `results/training_history.csv` | Training and validation loss at each epoch |
| `results/run_metadata.json` | Splits, versions, architecture, runtime and seed settings |
| `docs/PROFESSOR_RESPONSE.md` | Proposed natural-language answer with honest result interpretation |
| `docs/METHODS.md` | Detailed method, limitations and references |
| `docs/FIELD_DATA_PLAN.md` | What would be needed to turn this into a field study |

## Interpretation rules

- An MAE against teacher labels measures imitation, not optimal irrigation.
- Low mean error can hide missed irrigation days: positive events are rare.
- Water use must be interpreted with stress and ET shortfall, not in isolation.
- Seed variability reflects neural training variability on one fixed generated
  dataset, not uncertainty over real farms or weather distributions.
- The storm is measured before the decision. Same-day unseen storms require
  an intraday stop mechanism outside this daily model.
- Instantaneous drainage means prolonged waterlogging and crop injury are not
  represented. There are no yield claims.
- These experiments do not prove that six years, 14 days or an LSTM is required.
- Source and target closed-loop tables use separate cohorts; transfer-only
  methods must not be compared with an all-farms average.

