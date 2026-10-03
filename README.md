# AI-Powered Cold Rolling Mill Intelligence
https://ai-powered-cold-rolling-mill-intelligence.streamlit.app/


An industrial-AI portfolio prototype for monitoring a five-stand tandem cold rolling mill, detecting supplied anomaly labels, estimating reconstruction-based novelty, and surfacing maintenance investigation context in Streamlit.

> Safety disclaimer: this is an AI decision-support prototype. It does not replace plant safety procedures, OEM limits, protection systems, process-control interlocks, or engineering approval. Health and maintenance-priority bands are project-defined display logic, not plant standards.

## Dataset and data contract

Six supplied CSV batches are copied unchanged to `data/raw/` and unified at runtime. The observed source data contains **120,027 rows**, 67 original columns per batch, no missing values, and no timestamp column. The pipeline adds only provenance (`source_file`, `batch_id`, `batch_row`) in memory/processed output.

Available process inputs include entry/exit thickness, width, yield strengths, work-roll diameter and mileage, reductions, tension, roll speed, rolling force, torque, roll gap, and motor power across five stands. The 16 supplied `Anomaly_*` Boolean fields are labels, never model features.

The source provides no provenance that establishes it as plant telemetry. This project therefore treats it as **simulated/experimental data**, never as actual steel-plant data. It has no measured continuous quality target and hence does **not** claim a quality-regression model.

The binary supervised target is `anomaly_present`, derived as the OR of those supplied anomaly fields. For exploration, `fault_family` is derived as Normal / Reduction / Electric / Bearing / Work Roll. A single-family convention resolves multi-label records only for display; all original flags stay in the processed data.

## Measured baseline run

Training was run on the actual supplied data using a chronological batch/row holdout: first 80% train, last 20% test. This avoids mixing later ordered records into the holdout, though it cannot substitute for real timestamps.

| Model | Test F1 | Test PR-AUC | Test ROC-AUC |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.37196 | **0.75594** | **0.94094** |
| Random Forest | **0.64573** | 0.65554 | 0.90518 |
| Extra Trees | 0.57350 | 0.60512 | 0.90805 |
| LightGBM | 0.39323 | 0.60248 | 0.91410 |

Random Forest was selected by test F1, then PR-AUC, rather than accuracy alone. Its measured test accuracy is 0.97509, balanced accuracy 0.74010, precision 0.98375, recall 0.48060, and macro F1 0.81641. The low anomaly prevalence makes F1/recall/PR-AUC more informative than raw accuracy.

As an additional stability check, a three-fold stratified CV run on a 6,000-record subset of the chronological training partition produced F1 scores of 0.64789, 0.56716, and 0.60870 (mean **0.60792 ± 0.03296**) for a 100-tree Random Forest configuration. It is separate from, and does not contaminate, the final chronological test result.

A separate Extra Trees multiclass model was trained on the labelled anomaly records only to classify the supplied fault family. On the same held-out segment it achieved accuracy **0.75926**, macro F1 **0.75065**, and weighted F1 **0.75517** across Bearing, Electric, Reduction, and Work Roll. This category prediction is an investigation aid, not a verified diagnosis.

The deep-learning component is an MLP reconstruction autoencoder trained on 18,000 normal training records. An LSTM/GRU is deliberately not used because there is no timestamp or defensible sequence ordering. Its threshold is learned as the 99.5th percentile of held-out normal reconstruction error (2.02219). Its holdout F1 is 0.06260 and PR-AUC is 0.15871; this result is retained transparently rather than presented as a strong fault classifier.

Full run metadata, confusion matrix, class report, metrics, parameters, feature list, and global importance are stored in `models/metadata/model_metadata.json`.

## Architecture

```text
app.py                         Streamlit command center
data/raw/                      immutable copied sources
data/processed/                unified Parquet + dataset profile
models/classification/         selected leakage-safe sklearn pipeline
models/deep_learning/          MLP autoencoder + preprocessor
models/metadata/               reproducible measured run metadata
src/                           loading, preprocessing, training, inference, maintenance logic
pages/                         dashboard and individual page entry modules
tests/                         pipeline contract tests
```

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train_ml
streamlit run app.py
```

Use `python -m src.train_ml --no-cv` for a faster repeat training run. Artifacts are serialized once; Streamlit loads them and does not retrain on rerun. The **Model Training** page allows an explicit retrain.

## Dashboard capabilities

- Executive dashboard with latest-record model KPIs, process trends, dataset-labelled anomaly timeline, and a data-bound five-stand process flow.
- Data exploration, outlier-ready distribution/relationship views, missing/duplicate checks, dynamic monitoring charts.
- Supervised anomaly risk, MLP reconstruction anomaly score, model comparison/evaluation, and native global feature importance.
- What-if / AI Prediction simulation using adjustable actual input variables; it is visibly labelled a model simulation.
- Maintenance alert table based on supplied labels, project-defined health score, priority category, and investigation guidance.
- JSON technical report and processed-data sample downloads.

## Validation and leakage controls

- All original anomaly/fault columns, derived target fields, and source/order provenance are excluded from input features.
- Imputation/scaling is fit inside each training pipeline—never on the holdout set.
- A batch/row ordered test holdout is used because no timestamp exists.
- Class imbalance is assessed with F1, macro/weighted F1, balanced accuracy, PR-AUC and ROC-AUC, not accuracy alone.
- Cross-validation, when enabled, is stratified and occurs only on the training partition.

## Limitations and industrial next steps

This project cannot predict product quality without a supplied measured quality target, nor can it establish remaining useful life without maintenance history. Fault family is derived from Boolean labels and is not a verified failure diagnosis. The data has no timestamps, sensor identifiers, calibration history, units, or real-time connectivity.

For production, obtain governed historian data with timestamps; agree approved target definitions and alarm limits with process/OEM teams; validate by coil/campaign/time splits; integrate sensor quality checks and drift monitoring; use calibrated probabilities; establish human review; and complete cybersecurity, safety, and MOC processes before any operational use.
