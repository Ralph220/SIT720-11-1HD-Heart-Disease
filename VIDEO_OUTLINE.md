# Six-minute presentation outline

Keep face video visible, share the screen and use clear voice narration. This is an outline to explain in your own words after understanding and running the work.

| Time | What to explain | What to show |
|---|---|---|
| 0:00–0:40 | Identify the paper and binary heart-disease task; distinguish it from predicting a future heart attack. | Report title and abstract |
| 0:40–1:20 | Explain the dataset audit: 1,025 rows, 302 distinct profiles, 723 duplicates. Explain why duplicates crossing splits matter. | Notebook data-audit cell; execute it |
| 1:20–2:05 | Identify the six classifiers and stacking; explain assumptions and the 202/205 overlapping reproduction test rows. | run_research.py model definitions, then the measured reproduction table |
| 2:05–3:00 | Explain the proposal: train on distinct records, encode categorical variables and learn non-negative probability-pooling weights. Describe what the objective minimises. | fit_level0 and pool functions; do not read every line |
| 3:00–4:10 | Explain shared outer test folds and the ablation results: 81.92% versus 84.88%; Brier 0.1310 versus 0.1138. Show code actually executing. | Execute analyse_results.py, then show result table and figures. A full training run can be started earlier and its completion shown if time permits. |
| 4:10–5:00 | Explain ROC, reliability and error trade-offs. The first repeat corrects 13 errors and introduces five. | Figures 2 and 3, or the notebook |
| 5:00–5:40 | Discuss small distinct sample, unknown patient IDs, original-paper assumptions, no external test, and the bootstrap limitation. | Report limitations |
| 5:40–6:00 | State what you personally learned and what external validation would add. | Conclusion and references |

Practice answering:
- Why is 100% random-row accuracy not stronger evidence than 84.88% grouped accuracy?
- How are out-of-fold stacking features constructed?
- Why can the test set not be used to learn scaling, categories or weights?
- Why do we retain the unusual category codes rather than silently delete them?
- What does each ablation isolate, and which comparison still bundles two changes?
- How does the probability pool differ from logistic stacking?
- Why does lower Brier score not necessarily mean better calibration alone?
- Why is an exact profile not necessarily a unique patient?
- Why is the bootstrap interval conditional rather than definitive?

Do not claim the paper's authors used this implementation or split. Do not say the code is clinically validated or that the HD grade is guaranteed.
