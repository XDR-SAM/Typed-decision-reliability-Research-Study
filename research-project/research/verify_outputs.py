"""Independent numerical checks against saved predictions; output inventory."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT,dump_json
def main():
    m=pd.read_csv(ROOT/'outputs/metrics/classification_probability.csv');run=json.loads((ROOT/'outputs/models/run_manifest.json').read_text());classes=run['classes']
    rows=[];classselect=[]
    for p in run['windows']:
        d=pd.read_csv(ROOT/f'outputs/predictions/normal_{p}.csv.gz');pr=d[[f'p_{i}' for i in range(len(classes))]].to_numpy();y=d.issue.map({x:i for i,x in enumerate(classes)}).to_numpy()
        ref=m[(m.model=='normal')&(m.period==p)&(m.subset=='operational')].iloc[0]
        assert np.allclose(pr.sum(1),1)
        assert abs((pr.argmax(1)==y).mean()-ref.accuracy)<1e-10
        assert abs(-np.log(pr[np.arange(len(y)),y]).mean()-ref.nll)<1e-10
        assert len(d)==ref.n
        rows.append({'period':p,'n':len(d),'probability_sum_check':True,'accuracy_recomputed':True,'nll_recomputed':True})
        for t in [.5,.6,.7,.8,.9,.95]:
            for issue,g in d.groupby('issue'):
                a=g[g.confidence>=t]
                classselect.append({'period':p,'threshold':t,'issue':issue,'support':len(g),'accepted':len(a),'coverage':len(a)/len(g),'accepted_accuracy':float(a.correct.mean()) if len(a) else None})
    pd.DataFrame(classselect).to_csv(ROOT/'outputs/metrics/selective_by_class.csv',index=False)
    manifest=json.loads((ROOT/'data/source_manifest.json').read_text())
    for r in manifest:
        path=ROOT/r['local_path'];assert path.stat().st_size==r['bytes']
    dump_json(ROOT/'outputs/audits/output_verification.json',{'prediction_checks':rows,'manifest_files':len(manifest),'source_sizes_verified':True,'all_checks_passed':True})
    print('Verified',len(rows),'prediction cohorts;',len(manifest),'source sizes; per-class selective table written.')
if __name__=='__main__':main()
