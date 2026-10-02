# SIT720 11.1 HD Machine Learning Research

**Recovery update:** Start with `START_HERE.md`. The original project was recovered
and is being supplied with a fresh rerun verification.
The remaining-actions section below describes the original package state.

## What is complete

All six classifiers and stacking have been implemented and executed. The package contains an assumption-based reproduction, a proposed method, controlled ablations, three repeated five-fold evaluations, all predictions, five required metrics plus Brier/log loss, plots, an executed analysis notebook and a nine-page report using the supplied formatting.

The research is exploratory. It is not an exact replication of unavailable author code or a clinically validated predictor. The video, independently accessible code link and student review remain outstanding.

## Main finding

The 1,025-row CSV has 302 distinct records and 723 extra duplicates. The random-row reproduction has 202 of 205 test rows matching a training profile. Its 100% stacking accuracy is not a credible estimate for new profiles.

On identical group-separated outer folds, scoring each unique profile equally:

- Baseline stacking: mean accuracy 81.92%, AUC 0.8960, Brier 0.1310.
- Proposed pool: mean accuracy 84.88%, AUC 0.9150, Brier 0.1138.

The exact accuracy gain rounds to 2.97 percentage points using unrounded values. Most of the probability-score change arises before replacing the meta-learner: unique-record training and explicit categorical representation contribute more than the final pooling step. Full tables are in the report and CSVs.

## Install and run

Tested with Python 3.12.14 on Linux x86_64. Use an isolated environment. On Ubuntu/Linux:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python run_research.py
    python analyse_results.py
    python make_notebook.py
    python build_report.py

On Windows, activate with `.venv\Scripts\activate`. If the CPU-only XGBoost wheel is unavailable for your platform, use the same version of the full `xgboost` package instead and record the environment change. No external data download is needed: data/heart.csv is included unchanged. The run validates its SHA-256 before fitting. Training took roughly one minute in the original environment; other machines will differ.

`run_research.py` fits models and overwrites the experiment outputs. `analyse_results.py` verifies metrics from predictions and generates plots and report facts. `make_notebook.py` actually executes its analysis cells in order and stores stdout in research_analysis.ipynb; it does not retrain the models. `build_report.py` generates the editable report from those facts. The PDF is an export of the DOCX, not a separate hand-written numerical report.

To open the notebook interactively, install JupyterLab (`python -m pip install jupyterlab`) or use a notebook-capable editor. Open research_analysis.ipynb from the project directory and run all cells. The model-training code is in run_research.py, with clear functions for base models, out-of-fold stacking and constrained probability pooling.

To regenerate the PDF, open reports/SIT720_11_1HD_Research_Report.docx in Word or LibreOffice and export to PDF. With LibreOffice installed, this command also works:

    libreoffice --headless --convert-to pdf --outdir reports reports/SIT720_11_1HD_Research_Report.docx

Review the exported pages because different fonts or office versions may change pagination.

## Data and methodology

Dataset source: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
Downloaded 21 September 2026 via the Kaggle download API.
CSV SHA-256: ddb2996b2f4db2e00aad13f4518200179ff69f79093838e3c21ffa672ebec0f1

The selected paper is included under data/selected_paper.pdf for this assignment's review. Its citation and the literature/documentation references are in the report.

All 13 predictors are retained. The reproduction treats provided category codes numerically, with training-only standard scaling. The proposed typed version uses continuous scaling for age, trestbps, chol, thalach and oldpeak and one-hot encoding for the remaining eight features. No data were imputed because there are no missing cells. Unexpected ca=4 and thal=0 codes were retained rather than silently recoded.

`protocol.json` was fixed before the full experiments, after the earlier two-model pilot. It records seeds, folds, features, objective and regularisation. The split fraction, meta-learner, exact scaling/encoding and classifier settings are declared assumptions where the paper omits details. No seed search or outer-test hyperparameter tuning was performed.

The proposed pool fits non-negative weights summing to one using inner out-of-fold probabilities and Brier loss plus a 0.01 squared-distance penalty toward equal weights. SLSQP is used. This is a probability ensemble, not a post-hoc calibrated model. Base predictors are refitted on the full outer training split before inference.

Primary evaluation uses three seeds × five outer folds over unique profiles. Baseline repeated rows are mapped to those folds, and all inner baseline folds also respect profile grouping. The single random-row reproduction is separate from these robust comparisons. A grouped profile is NOT a known patient ID.

Ablations: repeated-row numeric-code stacking; unique-row numeric-code stacking; unique-row typed stacking; unique-row typed probability pooling. The first ablation also changes inner-fold construction, a limitation disclosed in the report. Typed stacking and proposed pooling share identical base probabilities.

## What the saved results mean

- published_metrics.csv: transcribed reference values, not new experiments.
- reproduction_metrics.csv and reproduction_predictions.csv: original-style random-row diagnostic run, 205 test rows.
- fold_metrics.csv: 300 records = 15 folds × 10 models/variants × 2 scoring weights.
- outer_predictions.csv: 9,060 predictions = 302 profiles × 3 repeats × 10 models/variants.
- outer_splits.json: exact profile and original-row indices; zero overlap for robust outer splits.
- pool_weights.csv: learned convex weights per fold.
- summary.csv / unique_summary.csv: fold means and sample SDs; no independent-fold confidence claim.
- uncertainty.json: bootstrap of paired per-profile loss differences with repeats kept together. Conditional on predictions, not a refit bootstrap or a clinical significance claim.
- error_analysis_repeat1.csv: baseline and proposed errors for the same repeat-1 profiles.
- verification.txt and clean_rerun_verification.json: validation and exact full-rerun evidence.
- report_facts.json: values consumed by the report builder.

There is no separate final deployment model: the deliverable is a reproducible evaluation study. Each outer model is trained for evaluation and its predictions are saved. Do not represent the mean metrics as an externally validated deployed model.

## Remaining student actions

1. Review the full paper, methods, code, outputs and assumptions. Independently explain the analysis and revise the writing in your own voice.
2. Record the mandatory video, no longer than six minutes, showing your face, screen and narration. Use VIDEO_OUTLINE.md to prepare. Do not invent lessons learned or claim actions you have not personally performed.
3. Upload the code/data ZIP to an accessible OneDrive/Dropbox location or an accessible GitHub project, and upload the video. Add the actual links in report Section 3.2. The local file link is not an external submission URL.
4. Open both links in a separate browser session, rerun the code on your machine, review the final PDF and submit via OnTrack. The assignment brief's feedback deadline is 28 September 2026.
