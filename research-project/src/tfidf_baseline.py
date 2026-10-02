"""CPU-only chronological pilot. No target/structured features enter vectorizer."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,re,json,time,warnings,platform
from pathlib import Path
import numpy as np,pandas as pd,yaml,joblib,sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT,config,dump_json
from src.evaluate import evaluate

def text_inputs(frame):
    return frame['narrative'].astype(str).tolist()

def main():
    assert (ROOT/'outputs/audits/audit_complete.json').exists(),'Complete audit before training'
    start=time.time();cfg=config();df=pd.read_pickle(ROOT/'data/processed/pilot.pkl')
    classes=sorted(df[df.train_keep].issue.unique());classmap={c:i for i,c in enumerate(classes)}
    train=df[df.train_keep];val=df[df.period=='validation'];y=train.issue.map(classmap).to_numpy();yv=val.issue.map(classmap).to_numpy()
    av=json.loads((ROOT/'outputs/audits/availability_decision.json').read_text())
    windows=['validation','2024Q4']+cfg['primary_future_quarters']
    if av['q2_eligible_exploratory']:windows.append('2026Q2')
    if av['july_eligible_exploratory']:windows.append('2026-07')
    for folder in ['metrics','predictions','tables','models']:(ROOT/'outputs'/folder).mkdir(parents=True,exist_ok=True)
    def vectorizer():return TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=100000,sublinear_tf=True,dtype=np.float64)
    vec=vectorizer();xt=vec.fit_transform(text_inputs(train));xv=vec.transform(text_inputs(val))
    print('MATRIX',xt.shape,'nnz',xt.nnz,flush=True)
    tuning=[];best=None;bestscore=-1
    for c in cfg['tfidf']['C_candidates']:
        clf=LogisticRegression(C=c,solver='lbfgs',max_iter=500,tol=1e-4,random_state=cfg['seed'])
        with warnings.catch_warnings(record=True) as ws,threadpool_limits(limits=4):clf.fit(xt,y)
        score=float(f1_score(yv,clf.predict(xv),average='macro',zero_division=0))
        tuning.append({'C':c,'validation_macro_f1':score,'iterations':int(clf.n_iter_.max()),'warnings':' | '.join(str(w.message) for w in ws)})
        print('TUNE',tuning[-1],flush=True)
        if score>bestscore:bestscore=score;best=(c,clf)
    pd.DataFrame(tuning).to_csv(ROOT/'outputs/tables/historical_tuning.csv',index=False)
    c,clf=best;joblib.dump({'vectorizer':vec,'classifier':clf,'classes':classes},ROOT/'outputs/models/tfidf_normal.joblib')
    terms=vec.get_feature_names_out();features=[]
    for k,label in enumerate(classes):
        for j in np.argsort(clf.coef_[k])[-25:][::-1]:features.append({'issue':label,'feature':terms[j],'coefficient':float(clf.coef_[k,j])})
    pd.DataFrame(features).to_csv(ROOT/'outputs/tables/top_features.csv',index=False)
    phrases=yaml.safe_load((ROOT/'configs/keyword_ablation.yaml').read_text())['phrases']
    pattern=re.compile(r'\b(?:'+'|'.join(re.escape(p) for p in sorted(phrases,key=len,reverse=True))+r')\b',re.I)
    mask=lambda t:pattern.sub(' ',t)
    abvec=vectorizer();xa=abvec.fit_transform([mask(t) for t in text_inputs(train)])
    abclf=LogisticRegression(C=c,solver='lbfgs',max_iter=500,tol=1e-4,random_state=cfg['seed'])
    with warnings.catch_warnings(record=True) as ws,threadpool_limits(limits=4):abclf.fit(xa,y)
    dump_json(ROOT/'outputs/tables/ablation_fit.json',{'C':c,'iterations':int(abclf.n_iter_.max()),'warnings':[str(w.message) for w in ws]})
    joblib.dump({'vectorizer':abvec,'classifier':abclf,'classes':classes},ROOT/'outputs/models/tfidf_ablated.joblib')
    prior=np.bincount(y,minlength=len(classes))/len(y)
    # Literal rule uses only issue-name fragments; fallback is historical majority.
    rulephrases={
      'Attempts to collect debt not owed':['debt not owed'],
      'Communication tactics':['communication tactics'],
      'Electronic communications':['electronic communications'],
      'False statements or representation':['false statements','false representation'],
      'Threatened to contact someone or share information improperly':['threatened to contact','share information improperly'],
      'Took or threatened to take negative or legal action':['negative or legal action','threatened legal action'],
      'Written notification about debt':['written notification']}
    metrics=[];perclass=[];selects=[];curves=[];binsall=[];masks=[];ruleout=[]
    for window in windows:
        g=df[(df.period==window)&df.included].copy();texts=text_inputs(g);yy=g.issue.map(classmap).to_numpy();masked=[mask(t) for t in texts]
        probs={'normal':clf.predict_proba(vec.transform(texts)),
               'keyword_mask_refit':abclf.predict_proba(abvec.transform(masked)),
               'keyword_mask_frozen':clf.predict_proba(vec.transform(masked)),
               'majority_prior':np.tile(prior,(len(g),1))}
        rule=[];hits=0
        for t in texts:
            matched=[classmap[k] for k,v in rulephrases.items() if k in classmap and any(p in t.lower() for p in v)]
            hits+=bool(matched);rule.append(matched[0] if len(matched)==1 else int(prior.argmax()))
        ruleout.append({'period':window,'n':len(g),'any_phrase_fraction':hits/len(g),'accuracy':float((np.array(rule)==yy).mean()),'macro_f1':float(f1_score(yy,rule,labels=np.arange(len(classes)),average='macro',zero_division=0))})
        masks.append({'period':window,'n':len(g),'n_affected':sum(a!=b for a,b in zip(texts,masked))})
        for variant,p in probs.items():
            pred=g[['complaint_id','date','period','issue','family','novel_family','family_representative']].copy()
            pred['predicted_issue']=[classes[i] for i in p.argmax(1)];pred['confidence']=p.max(1);pred['correct']=p.argmax(1)==yy
            for k in range(len(classes)):pred[f'p_{k}']=p[:,k]
            pred.to_csv(ROOT/f'outputs/predictions/{variant}_{window}.csv.gz',index=False)
            slices={'operational':np.ones(len(g),dtype=bool)}
            if variant=='normal':
                slices['novel_family']=g.novel_family.to_numpy()&g.family_representative.to_numpy()
                slices['unique_family']=g.family_representative.to_numpy()
            for subset,m in slices.items():
                if not m.any():continue
                met,pc,cm,bins,sel,curve=evaluate(yy[m],p[m],classes,cfg['thresholds'])
                keys={'model':variant,'period':window,'subset':subset}
                metrics.append({**keys,**met});perclass.extend({**keys,**r} for r in pc)
                selects.extend({**keys,**r} for r in sel);curves.extend({**keys,**r} for r in curve);binsall.extend({**keys,**r} for r in bins)
                pd.DataFrame(cm,index=classes,columns=classes).to_csv(ROOT/f'outputs/metrics/confusion_{variant}_{window}_{subset}.csv')
                print('EVAL',variant,window,subset,{k:round(v,4) if isinstance(v,float) else v for k,v in met.items()},flush=True)
    for name,rows in [('classification_probability',metrics),('per_class',perclass),('selective',selects),('risk_coverage',curves),('reliability_bins',binsall)]:pd.DataFrame(rows).to_csv(ROOT/f'outputs/metrics/{name}.csv',index=False)
    pd.DataFrame(masks).to_csv(ROOT/'outputs/tables/keyword_mask_counts.csv',index=False);pd.DataFrame(ruleout).to_csv(ROOT/'outputs/tables/literal_rule.csv',index=False)
    dump_json(ROOT/'outputs/models/run_manifest.json',{'selected_C':c,'classes':classes,'training_rows':len(train),'training_date_max':train.date.max(),
        'feature_columns':['narrative'],'validation_period':'2024-07-01 through 2024-09-30','windows':windows,
        'vocabulary_size':len(terms),'elapsed_seconds':time.time()-start,'python':platform.python_version(),'sklearn':sklearn.__version__,
        'numpy':np.__version__,'pandas':pd.__version__,'gpu_used':False,'seed':cfg['seed'],'ece':'15 fixed equal-width bins','brier':'sum across classes, range 0 to 2',
        'future_selection':'No future performance used for C, vocabulary, labels or phrases.'})
if __name__=='__main__':main()
