# Recovered project — SIT720 11.1HD

This is the original implementation recovered from the earlier saved
SIT720_11_1HD_Research_Starter.zip. Despite its filename, that archive already
contained the completed research pipeline, dataset, notebook and saved results.
The model implementation has not been rewritten to fit the report's numbers.
See results/recovery_verification.json for the fresh rerun comparison.

## Run it on Ubuntu

Extract this ZIP. Open a terminal in HD_Research_Project, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_research.py
python analyse_results.py
python make_notebook.py
```

Python 3.12 is recommended. If venv is unavailable, install Ubuntu's python3-venv
package first. All data are included; no Kaggle login is needed.
The notebook displays and explains saved experiment results; run_research.py
contains the model training. The README explains the full methodology.

## Complete the submission

1. Review the code, assumptions, results and notebook so you can explain them.
2. Upload this ZIP to OneDrive or Dropbox and give the tutor download access.
   Alternatively upload the project to an accessible GitHub repository.
3. In your latest report's Section 3.2, replace the complete source-link placeholder
   with the real link. Keep the video link: https://youtu.be/uSAETOpmBs0
4. Confirm the video meets the task requirements: at most six minutes, face,
   screen sharing, clear narration, selected code execution and results.
   Its contents and availability have not been checked during this recovery.
5. Export the corrected report PDF and resubmit it through OnTrack. Direct the
   tutor to the source link in Section 3.2.

The reports/ directory contains the historical report and formatting template.
The historical report still has placeholder links. Use your latest submitted
report for the link correction; do not upload the historical PDF unchanged.
Running build_report.py also regenerates the historical draft and therefore
requires updating the links and reviewing the narrative before submission.


## Fresh verification
All 15 folds completed and all 300 fold/weighting metric records passed checks.
Reported classification metrics match the original; probability differences
were at most 4.45e-16. This fresh recovery run is numerically equivalent,
not byte-identical across every output. The older clean_rerun_verification.json
is historical evidence; recovery_verification.json describes this run.
