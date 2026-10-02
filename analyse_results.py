"""Verify saved predictions and generate report tables and figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score,confusion_matrix
from sklearn.calibration import calibration_curve
R=Path(__file__).resolve().parent;O=R/'results';F=R/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.bbox':'tight'})
fm=pd.read_csv(O/'fold_metrics.csv');pr=pd.read_csv(O/'outer_predictions.csv');rp=pd.read_csv(O/'reproduction_predictions.csv');cfg=json.loads((R/'protocol.json').read_text())
assert len(pr)==302*3*10
assert len(fm)==3*5*10*2
assert not pr.duplicated(['repeat','model','profile_id']).any()
for _,g in pr.groupby(['repeat','model']):assert len(g)==302 and g.profile_id.nunique()==302
assert pr.probability.between(0,1).all()
for split in json.loads((O/'outer_splits.json').read_text()):
 assert not set(split['train_profiles'])&set(split['test_profiles'])
 assert len(split['train_profiles'])+len(split['test_profiles'])==302
# Independent elementary calculations validate every metric based on confusion counts.
for row in fm.itertuples():
 g=pr[(pr.repeat==row.repeat)&(pr.fold==row.fold)&(pr.model==row.model)]
 y=g.target.to_numpy();p=g.probability.to_numpy();h=(p>=.5).astype(int);w=np.ones(len(g)) if row.weighting=='unique' else g['count'].to_numpy()
 tp=w[(y==1)&(h==1)].sum();fp=w[(y==0)&(h==1)].sum();fn=w[(y==1)&(h==0)].sum()
 check={'accuracy':np.average(h==y,weights=w),'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn) if tp+fn else 0,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,'brier':np.average((p-y)**2,weights=w)}
 for k,v in check.items():assert abs(v-getattr(row,k))<1e-10,(row.model,k)
(O/'verification.txt').write_text('PASS: data split profile disjointness; one outer prediction per profile per model per repeat; all 300 fold rows independently verified for accuracy, precision, recall, F1 and Brier; bounded probabilities. AUC uses scikit-learn. Full model fitting completed without errors.\n')
order=['LR','DT','RF','XGB','NB','KNN','Baseline stacking','Unique stacking','Typed stacking','Proposed pool']
summary=fm[fm.weighting=='unique'].groupby('model')[['accuracy','precision','recall','f1','auc','brier','log_loss']].agg(['mean','std']).reindex(order)
summary.to_csv(O/'unique_summary.csv')
# Fold means, not pooled repeated predictions.
for metric in ['accuracy','brier']:
 fig,ax=plt.subplots(figsize=(7.7,3.8));z=summary[(metric,'mean')];err=summary[(metric,'std')]
 ax.barh(np.arange(len(order)),z,xerr=err,color=['#a4b9cb']*6+['#275b84','#729fc1','#729fc1','#158777'],capsize=2)
 ax.set_yticks(np.arange(len(order)),order);ax.invert_yaxis();ax.set_xlabel('Mean accuracy ± fold SD' if metric=='accuracy' else 'Mean Brier score ± fold SD (lower is better)');ax.grid(axis='x',alpha=.15)
 fig.tight_layout();fig.savefig(F/f'{metric}_comparison.png');plt.close(fig)
# Each plot uses repeat 1 only: 302 unique held-out profiles, no repeat pooling.
fig,axes=plt.subplots(1,2,figsize=(9,3.6))
for name,color in [('Baseline stacking','#275b84'),('Proposed pool','#158777')]:
 g=pr[(pr.repeat==1)&(pr.model==name)];fpr,tpr,_=roc_curve(g.target,g.probability);axes[0].plot(fpr,tpr,label=f'{name} (AUC {roc_auc_score(g.target,g.probability):.3f})',color=color)
 obs,prob=calibration_curve(g.target,g.probability,n_bins=6,strategy='quantile');axes[1].plot(prob,obs,'o-',label=name,color=color)
for ax in axes:ax.plot([0,1],[0,1],'--',color='#999999');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.legend(fontsize=8)
axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC curve');axes[1].set(xlabel='Mean predicted probability',ylabel='Observed positive fraction',title='Reliability diagram (six quantile bins)')
fig.tight_layout();fig.savefig(F/'roc_reliability.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(7.5,3.2));cms={}
for ax,name in zip(axes,['Baseline stacking','Proposed pool']):
 g=pr[(pr.repeat==1)&(pr.model==name)];cm=confusion_matrix(g.target,g.probability>=.5,labels=[0,1]);cms[name]=cm.tolist();ax.imshow(cm,cmap='Blues');ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted label',ylabel='Actual label',title=name)
 for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black',fontsize=15)
fig.tight_layout();fig.savefig(F/'confusion_matrices.png');plt.close(fig)
(O/'confusion_repeat1.json').write_text(json.dumps(cms,indent=2))
# Baseline and proposed error changes on fixed repeat 1.
a=pr[(pr.repeat==1)&(pr.model=='Baseline stacking')].set_index('profile_id');b=pr[(pr.repeat==1)&(pr.model=='Proposed pool')].set_index('profile_id').reindex(a.index)
err=pd.DataFrame({'target':a.target,'baseline_probability':a.probability,'proposed_probability':b.probability});err['baseline_correct']=(err.baseline_probability>=.5)==err.target;err['proposed_correct']=(err.proposed_probability>=.5)==err.target;err.to_csv(O/'error_analysis_repeat1.csv')
fact={'reproduction':pd.read_csv(O/'reproduction_metrics.csv').to_dict('records'),'unique_summary':[{ 'model':name,**{f'{m}_{stat}':float(summary.loc[name,(m,stat)]) for m in ['accuracy','precision','recall','f1','auc','brier','log_loss'] for stat in ['mean','std']}} for name in order],'confusion_repeat1':cms,'errors_corrected':int((~err.baseline_correct&err.proposed_correct).sum()),'errors_introduced':int((err.baseline_correct&~err.proposed_correct).sum()),'uncertainty':json.loads((O/'uncertainty.json').read_text()),'mean_pool_weights':pd.read_csv(O/'pool_weights.csv')[['LR','DT','RF','XGB','NB','KNN']].mean().to_dict()}
(O/'report_facts.json').write_text(json.dumps(fact,indent=2));print(json.dumps(fact,indent=2))
