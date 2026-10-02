"""Build and execute simple analysis cells, storing their real stdout in a notebook.
The notebook reads verified result files. It does not silently retrain models.
"""
from pathlib import Path
import json, io, contextlib, os, platform
ROOT=Path(__file__).resolve().parent
cells=[];scope={};count=0

def md(s):cells.append({'cell_type':'markdown','id':f'm{len(cells)}','metadata':{},'source':s.splitlines(True)})
def code(s):
 global count
 count+=1;buffer=io.StringIO()
 with contextlib.redirect_stdout(buffer):exec(compile(s,'notebook-cell','exec'),scope)
 cells.append({'cell_type':'code','id':f'c{len(cells)}','metadata':{},'execution_count':count,'source':s.splitlines(True),'outputs':[{'output_type':'stream','name':'stdout','text':buffer.getvalue().splitlines(True)}]})
os.chdir(ROOT)
md('# SIT720 11.1 HD research analysis\n\nThis executed notebook examines saved results from `run_research.py`. The full model implementations are in that script.')
code("from pathlib import Path\nimport json\nimport pandas as pd\nROOT = Path.cwd()\nassert (ROOT / 'data/heart.csv').exists(), 'Open the notebook from the project folder'\nprint('Project data and results located.')")
md('## Re-running the experiments\nRun `python run_research.py` followed by `python analyse_results.py` in the project terminal. These commands fit all models and regenerate tables, predictions and figures. The cells below display those output files; they do not retrain.')
code("df = pd.read_csv(ROOT / 'data/heart.csv')\nprint('Rows:', len(df))\nprint('Distinct records:', len(df.drop_duplicates()))\nprint('Extra duplicates:', int(df.duplicated().sum()))\nprint('Missing cells:', int(df.isna().sum().sum()))\nprint('Class counts:')\nprint(df.target.value_counts().sort_index().to_string())")
md('## Published and reproduced results\nThe random-row diagnostic reproduction allows repeated profiles to cross the split. It is an assumption-based reproduction, not a reliable new-patient estimate.')
code("print('Published results:')\nprint(pd.read_csv(ROOT / 'results/published_metrics.csv').round(4).to_string(index=False))\nprint('Measured reproduction:')\nprint(pd.read_csv(ROOT / 'results/reproduction_metrics.csv').round(4).to_string(index=False))\nsplit = json.loads((ROOT / 'results/reproduction_split.json').read_text())\nprint('Test rows with matching training profile:', split['overlapping_test_rows'], '/', split['test_rows'])")
md('## Evaluation on identical unseen-profile folds\nPrimary results give one vote per distinct profile. The three repeats have overlapping training sets; fold variation is not an independent-sample confidence interval.')
code("scores = pd.read_csv(ROOT / 'results/fold_metrics.csv')\nprimary = scores[scores.weighting == 'unique']\nprint(primary.groupby('model')[['accuracy','precision','recall','f1','auc','brier']].mean().round(4).to_string())")
md('![Mean accuracy and fold standard deviation](figures/accuracy_comparison.png)\n\n![Brier score and fold standard deviation](figures/brier_comparison.png)')
code("print('Row-weighted sensitivity:')\nprint(scores[scores.weighting == 'row'].groupby('model')[['accuracy','brier']].mean().round(4).to_string())\nprint('Learned pool weights averaged across folds:')\nprint(pd.read_csv(ROOT / 'results/pool_weights.csv')[['LR','DT','RF','XGB','NB','KNN']].mean().round(4).to_string())")
md('## Probability and error analysis\nThese plots use repeat 1 only, with one cross-fitted prediction per profile. Lower Brier loss does not alone prove better calibration.\n\n![ROC and reliability](figures/roc_reliability.png)\n\n![Confusion matrices](figures/confusion_matrices.png)')
code("print((ROOT / 'results/uncertainty.json').read_text())\nprint((ROOT / 'results/verification.txt').read_text())")
md('## Explain these points in your own words\n1. Why is the 100% random-row score misleading?\n2. Why must preprocessing and stacking folds also respect training boundaries?\n3. Which ablation made the largest difference?\n4. What does Brier score measure, and why is it not a pure calibration metric?\n5. Why does the bootstrap interval not prove performance on a new population?\n6. What assumptions prevent this from being an exact replication of the paper?\n\nRecord a face-and-screen video after reviewing and understanding the work. Add the actual public-accessible links before submission.')
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':platform.python_version()},'execution_note':'Code cells were executed in order in a shared Python namespace with captured stdout. These cells analyse saved experiment outputs.'},'nbformat':4,'nbformat_minor':5}
(ROOT/'research_analysis.ipynb').write_text(json.dumps(nb,indent=1))
print('Executed',count,'analysis cells and saved notebook.')
if not all(c['execution_count'] for c in cells if c['cell_type']=='code'):raise RuntimeError('Missing execution count')
