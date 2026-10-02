"""Full assumption-based reproduction and controlled exploratory extension.
Run from any directory: python run_research.py
All numerical report evidence is written beneath results/.
"""
from pathlib import Path
import json, hashlib, time, platform
import numpy as np
import pandas as pd
import scipy, sklearn, xgboost
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, log_loss
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'protocol.json').read_text())
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
NUM=CFG['numerical_features'];CAT=CFG['categorical_features'];NAMES=CFG['models']
METRICS=['accuracy','precision','recall','f1','auc','brier','log_loss']

def metrics(y,p,w=None):
    label=(p>=CFG['threshold']).astype(int)
    return dict(accuracy=accuracy_score(y,label,sample_weight=w),precision=precision_score(y,label,sample_weight=w,zero_division=0),recall=recall_score(y,label,sample_weight=w,zero_division=0),f1=f1_score(y,label,sample_weight=w,zero_division=0),auc=roc_auc_score(y,p,sample_weight=w),brier=brier_score_loss(y,p,sample_weight=w),log_loss=log_loss(y,p,labels=[0,1],sample_weight=w))

def models(seed,typed=False):
    estimators=[LogisticRegression(C=1,max_iter=2000,random_state=seed),DecisionTreeClassifier(random_state=seed),RandomForestClassifier(n_estimators=200,random_state=seed,n_jobs=1),XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.05,subsample=1,colsample_bytree=1,reg_lambda=1,objective='binary:logistic',eval_metric='logloss',tree_method='hist',n_jobs=1,random_state=seed),GaussianNB(),KNeighborsClassifier(n_neighbors=5)]
    result={}
    for name,model in zip(NAMES,estimators):
        if typed:
            prep=ColumnTransformer([('continuous',StandardScaler(),NUM),('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)],sparse_threshold=0)
        else:
            # Original CSV categorical codes retained; scale is a declared assumption.
            prep=StandardScaler()
        result[name]=make_pipeline(prep,model)
    return result

def fit_level0(X,y,Xtest,seed,typed=False,groups=None):
    if groups is None:
        cv=StratifiedKFold(CFG['inner_folds'],shuffle=True,random_state=seed)
        splits=list(cv.split(X,y))
    else:
        cv=StratifiedGroupKFold(CFG['inner_folds'],shuffle=True,random_state=seed)
        splits=list(cv.split(X,y,groups))
        for tr,va in splits:assert not set(groups[tr]) & set(groups[va])
    Z=np.full((len(y),len(NAMES)),np.nan);T=np.zeros((len(Xtest),len(NAMES)))
    for j,(name,prototype) in enumerate(models(seed,typed).items()):
        for tr,va in splits:
            m=clone(prototype).fit(X.iloc[tr],y[tr]);Z[va,j]=m.predict_proba(X.iloc[va])[:,1]
        m=clone(prototype).fit(X,y);T[:,j]=m.predict_proba(Xtest)[:,1]
    assert np.isfinite(Z).all() and np.isfinite(T).all()
    return Z,T

def logistic_stack(Z,y,T,seed):
    m=LogisticRegression(C=1,max_iter=2000,random_state=seed).fit(Z,y)
    return m.predict_proba(T)[:,1],m.coef_[0].tolist()

def pool(Z,y,T):
    k=Z.shape[1];lam=CFG['pool_regularization'];u=np.ones(k)/k
    def objective(w):return np.mean((Z@w-y)**2)+lam*np.sum((w-u)**2)
    def jac(w):return 2*Z.T@(Z@w-y)/len(y)+2*lam*(w-u)
    opt=minimize(objective,u,jac=jac,bounds=[(0,1)]*k,constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(k)},method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
    assert opt.success,opt.message
    w=opt.x;assert abs(w.sum()-1)<1e-7 and w.min()>-1e-7
    return T@w,w.tolist(),float(opt.fun)

def main():
    start=time.time();data_path=ROOT/'data/heart.csv'
    assert hashlib.sha256(data_path.read_bytes()).hexdigest()==CFG['data_sha256']
    df=pd.read_csv(data_path);X=df.drop(columns='target');y=df.target.to_numpy()
    assert not df.isna().any().any() and set(y)=={0,1}
    groups=pd.factorize(pd.MultiIndex.from_frame(X))[0]
    assert df.groupby(list(X)).target.nunique().max()==1
    # Factorization follows first occurrence, as does drop_duplicates.
    unique=df.drop_duplicates(list(X)).reset_index(drop=True);Xu=unique.drop(columns='target');yu=unique.target.to_numpy();counts=np.bincount(groups)
    assert len(unique)==302
    audit=json.loads((ROOT/'data/audit.json').read_text());audit['duplicate_fraction']=float(df.duplicated().mean());audit['code_counts']={k:{str(a):int(b) for a,b in df[k].value_counts().items()} for k in CAT};(OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    published=pd.DataFrame([['LR',.8439,.8196,.9090,.8620,.9204],['DT',.9268,.9702,.8909,.9289,.9702],['RF',.9268,.9130,.9545,.9333,.9734],['XGB',.9073,.9174,.9090,.9132,.9830],['NB',.8439,.8250,.9000,.8608,.9185],['KNN',.8585,.8648,.8727,.8687,.9304],['Stacking',.9853,1,.9727,.9861,.988]],columns=['model','accuracy','precision','recall','f1','auc']);published.to_csv(OUT/'published_metrics.csv',index=False)
    tr,te=train_test_split(np.arange(len(df)),test_size=CFG['reproduction_test_fraction'],random_state=CFG['reproduction_seed'],stratify=y)
    Z,T=fit_level0(X.iloc[tr],y[tr],X.iloc[te],42)
    ps,_=logistic_stack(Z,y[tr],T,42)
    reproduction=[];rp=[]
    for name,p in list(zip(NAMES,T.T))+[('Stacking',ps)]:
        reproduction.append(dict(model=name,**metrics(y[te],p)))
        rp.extend(dict(model=name,row_index=int(i),profile_id=int(groups[i]),target=int(y[i]),probability=float(v)) for i,v in zip(te,p))
    pd.DataFrame(reproduction).to_csv(OUT/'reproduction_metrics.csv',index=False);pd.DataFrame(rp).to_csv(OUT/'reproduction_predictions.csv',index=False)
    repinfo={'train_indices':tr.tolist(),'test_indices':te.tolist(),'test_rows':len(te),'overlapping_test_rows':int(np.isin(groups[te],groups[tr]).sum()),'unique_test_profiles':int(len(set(groups[te])))}
    (OUT/'reproduction_split.json').write_text(json.dumps(repinfo,indent=2))
    print('Reproduction complete',pd.DataFrame(reproduction)[['model','accuracy']].to_dict('records'),flush=True)
    records=[];pred=[];folds=[];weights=[]
    for repeat,seed in enumerate(CFG['outer_seeds'],1):
        cv=StratifiedKFold(CFG['outer_folds'],shuffle=True,random_state=seed)
        for fold,(utr,ute) in enumerate(cv.split(Xu,yu),1):
            seed_model=seed+fold
            rawtr=np.flatnonzero(np.isin(groups,utr));rawte=np.flatnonzero(np.isin(groups,ute))
            assert not set(groups[rawtr]) & set(groups[rawte])
            folds.append(dict(repeat=repeat,fold=fold,seed=seed,train_profiles=utr.tolist(),test_profiles=ute.tolist(),train_rows=rawtr.tolist(),test_rows=rawte.tolist(),overlap=0))
            # Baseline keeps repeated training rows but groups every inner fold.
            Z0,T0=fit_level0(X.iloc[rawtr],y[rawtr],Xu.iloc[ute],seed_model,groups=groups[rawtr])
            p0,c0=logistic_stack(Z0,y[rawtr],T0,seed_model)
            # Ablation removes repeat weighting while keeping numeric codes.
            Z1,T1=fit_level0(Xu.iloc[utr],yu[utr],Xu.iloc[ute],seed_model)
            p1,c1=logistic_stack(Z1,yu[utr],T1,seed_model)
            # Ablation changes feature representation with the same learner families.
            Z2,T2=fit_level0(Xu.iloc[utr],yu[utr],Xu.iloc[ute],seed_model,typed=True)
            p2,c2=logistic_stack(Z2,yu[utr],T2,seed_model)
            pp,w,obj=pool(Z2,yu[utr],T2)
            candidates=dict(zip(NAMES,T0.T));candidates.update({'Baseline stacking':p0,'Unique stacking':p1,'Typed stacking':p2,'Proposed pool':pp})
            weights.append(dict(repeat=repeat,fold=fold,objective=obj,**dict(zip(NAMES,w))))
            for name,p in candidates.items():
                assert np.isfinite(p).all() and p.min()>=-1e-9 and p.max()<=1+1e-9
                p=np.clip(p,0,1)
                for weighting,wscore in [('unique',None),('row',counts[ute])]:records.append(dict(repeat=repeat,fold=fold,model=name,weighting=weighting,**metrics(yu[ute],p,wscore)))
                pred.extend(dict(repeat=repeat,fold=fold,model=name,profile_id=int(i),count=int(counts[i]),target=int(yu[i]),probability=float(v)) for i,v in zip(ute,p))
            pd.DataFrame(records).to_csv(OUT/'fold_metrics.csv',index=False);pd.DataFrame(pred).to_csv(OUT/'outer_predictions.csv',index=False)
            (OUT/'outer_splits.json').write_text(json.dumps(folds));pd.DataFrame(weights).to_csv(OUT/'pool_weights.csv',index=False)
            print(f'Repeat {repeat}/3 fold {fold}/5 completed ({time.time()-start:.1f}s)',flush=True)
    results=pd.DataFrame(records)
    results.groupby(['weighting','model'])[METRICS].agg(['mean','std']).to_csv(OUT/'summary.csv')
    # Exploratory uncertainty on fixed cross-fitted predictions. Resample profiles,
    # keeping all repeats together; not a refit/bootstrap CI for training uncertainty.
    pred_df=pd.DataFrame(pred);base=pred_df[pred_df.model=='Baseline stacking'].sort_values(['profile_id','repeat']);prop=pred_df[pred_df.model=='Proposed pool'].sort_values(['profile_id','repeat'])
    assert np.array_equal(base[['profile_id','repeat']].values,prop[['profile_id','repeat']].values)
    losses=(prop.probability.to_numpy()-prop.target.to_numpy())**2-(base.probability.to_numpy()-base.target.to_numpy())**2
    per_profile=losses.reshape(302,3).mean(axis=1);rng=np.random.default_rng(42)
    boot=np.mean(per_profile[rng.integers(0,302,size=(5000,302))],axis=1)
    uncertainty={'comparison':'proposed minus baseline Brier loss; negative favors proposed','mean_profile_loss_difference':float(per_profile.mean()),'conditional_bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist(),'resamples':5000,'unit':'distinct profile, repeats kept together','limitation':'Conditional on fitted cross-validated predictions; overlapping training sets and retraining uncertainty are not captured. Exploratory, not an independent-test significance claim.'}
    (OUT/'uncertainty.json').write_text(json.dumps(uncertainty,indent=2))
    versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'xgboost-cpu':xgboost.__version__}
    (OUT/'environment.json').write_text(json.dumps(versions,indent=2));(OUT/'run_info.json').write_text(json.dumps({'elapsed_seconds':time.time()-start,'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    print(results[results.weighting=='unique'].groupby('model')[['accuracy','auc','brier']].mean().to_string(),flush=True)
    print('All experiments complete.',flush=True)
if __name__=='__main__':main()
