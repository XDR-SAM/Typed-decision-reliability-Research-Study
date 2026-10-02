"""Write audit plots and availability decisions BEFORE any classifier fitting."""
import sys,json
from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT,config,dump_json

def main():
    f=ROOT/'outputs/figures';f.mkdir(parents=True,exist_ok=True)
    df=pd.read_pickle(ROOT/'data/processed/pilot.pkl');a=pd.read_csv(ROOT/'outputs/audits/availability_monthly.csv')
    ref=a[a.month.between('2025-01','2026-03')].debt_narrative_fraction.median()
    q2=a[a.month.between('2026-04','2026-06')]
    counts=pd.read_csv(ROOT/'outputs/audits/independent_counts_by_class.csv')
    q2ok=bool((q2.debt_narrative_fraction>=.5*ref).all() and counts[counts.period=='2026Q2'].families.min()>=200)
    julyok=bool(a.loc[a.month=='2026-07','debt_narrative_fraction'].iloc[0]>=.5*ref and counts[counts.period=='2026-07'].families.min()>=200)
    status={'reference_median_fraction':float(ref),'q2_eligible_exploratory':q2ok,'july_eligible_exploratory':julyok,'august_eligible':False,
            'note':'Availability heuristics are not proof of missing-at-random publication. Earlier cohorts also show selection shifts.'}
    dump_json(ROOT/'outputs/audits/availability_decision.json',status)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    x=np.arange(len(a));axes[0].plot(x,a.debt_records,label='All Debt Collection complaints');axes[0].plot(x,a.debt_narratives,label='Published narratives')
    axes[0].set_ylabel('Records');axes[0].legend()
    axes[1].plot(x,a.debt_narrative_fraction*100,color='#b24b35',label='Debt Collection')
    axes[1].plot(x,a.narrative_fraction*100,color='#607d8b',label='All products');axes[1].set_ylabel('Narrative availability (%)');axes[1].legend()
    axes[1].set_xticks(x[::2],a.month.iloc[::2],rotation=45,ha='right');fig.suptitle('Official CFPB archive: complaint counts and narrative selection');fig.tight_layout();fig.savefig(f/'availability.png',dpi=160);plt.close(fig)
    table=df.groupby(['period','issue'],sort=False).size().unstack(fill_value=0);props=table.div(table.sum(1),axis=0)
    ax=props.plot.bar(stacked=True,figsize=(12,6),colormap='tab20');ax.set_ylabel('Class proportion');ax.set_xlabel('Period');ax.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=8);plt.title('Debt Collection: original Issue labels');plt.tight_layout();plt.savefig(f/'class_distribution.png',dpi=160);plt.close()
    d=pd.read_csv(ROOT/'outputs/audits/duplicates_by_period.csv');d=d[~d.period.isin(['train','validation'])]
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(d.period,d.seen_historical_fraction*100,color='#756bb1');ax.axhline(50,ls='--',color='black',label='Prespecified dominance warning');ax.set_ylabel('Seen historical template family (%)');ax.tick_params(axis='x',rotation=30);ax.legend();fig.tight_layout();fig.savefig(f/'duplicate_exposure.png',dpi=160);plt.close(fig)
    # Descriptive label-prior JS divergence from training, not a causal attribution.
    from scipy.spatial.distance import jensenshannon
    drift=[{'period':p,'label_prior_js_bits':float(jensenshannon(props.loc['train'],row,base=2)**2)} for p,row in props.iterrows()]
    pd.DataFrame(drift).to_csv(ROOT/'outputs/audits/label_prior_drift.csv',index=False)
    if not (ROOT/'outputs/audits/audit_complete.json').exists():
        dump_json(ROOT/'outputs/audits/audit_complete.json',{'complete':True,'stage':'before model fitting','records':len(df),'availability':status,
            'required_audits':['monthly/quarterly availability','labels and counts','lengths','exact duplicates','MinHash verified families','chronological splits']})
    print(status)
if __name__=='__main__':main()
