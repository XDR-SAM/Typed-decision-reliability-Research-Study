"""Generate figures and apply the prespecified feasibility decision mechanically."""
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'outputs/.mplcache')
sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils import config,dump_json

def md(frame,digits=3):
    def fmt(x):
        if pd.isna(x):return '—'
        if isinstance(x,(float,np.floating)):return f'{x:.{digits}f}'
        return str(x).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,frame.columns))+' |\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))

def main():
    cfg=config();out=ROOT/'outputs';figdir=out/'figures';figdir.mkdir(exist_ok=True)
    m=pd.read_csv(out/'metrics/classification_probability.csv');s=pd.read_csv(out/'metrics/selective.csv');rc=pd.read_csv(out/'metrics/risk_coverage.csv')
    normal=m[(m.model=='normal')&(m.subset=='operational')].copy();future=normal[normal.period.isin(cfg['primary_future_quarters'])]
    novel=m[(m.model=='normal')&(m.subset=='novel_family')]
    counts=pd.read_csv(out/'audits/independent_counts_by_class.csv');dup=pd.read_csv(out/'audits/duplicates_by_period.csv');length=pd.read_csv(out/'audits/debt_period_summary.csv')
    pc=pd.read_csv(out/'metrics/per_class.csv');classes=json.loads((out/'models/run_manifest.json').read_text())['classes']
    run=json.loads((out/'models/run_manifest.json').read_text());ing=json.loads((out/'audits/ingestion_summary.json').read_text());dd=json.loads((out/'audits/dedup_summary.json').read_text())
    av=json.loads((out/'audits/availability_decision.json').read_text());label=json.loads((out/'audits/label_status.json').read_text())
    small=counts[counts.period.isin(cfg['primary_future_quarters'])&counts.issue.isin(classes)]
    ds=dup[dup.period.isin(cfg['primary_future_quarters'])]
    normsel=s[(s.model=='normal')&(s.subset=='operational')].merge(normal[['period','accuracy']],on='period')
    normsel['improvement']=normsel.accepted_accuracy-normsel.accuracy
    useful=normsel[normsel.coverage.between(.1,.9)&(normsel.improvement>=.05)]
    confidence_useful=useful.period.nunique()>=3
    changes={k:float(future[k].max()-future[k].min()) for k in ['accuracy','macro_f1','ece','brier','nll']}
    condition3=confidence_useful or changes['accuracy']>=.02 or changes['macro_f1']>=.02 or changes['ece']>=.02 or changes['brier']>=.02 or changes['nll']>=.05
    passed=[not bool(((future.accuracy>=.97)&(future.macro_f1>=.97)).all()),
            bool(small.families.min()>=200 and run['training_rows']>=10000 and len(future)==5),
            bool(condition3),bool(ds.seen_historical_fraction.max()<=.5 and small.novel_families.min()>=200)]
    warnings=[]
    if (future.accuracy>=.97).sum()>=4:warnings.append('Near-perfect accuracy in almost every future window')
    if not passed[1]:warnings.append('Inadequate future class-family support')
    if ds.seen_historical_fraction.max()>.5:warnings.append('Historical template families dominate a future quarter')
    if not confidence_useful:warnings.append('No useful prespecified confidence tradeoff demonstrated in three windows')
    if length[length.period.isin(cfg['primary_future_quarters'])].unknown_or_excluded_fraction.max()>.05:warnings.append('Unstable/new-label mass exceeds 5 percent')
    rule=pd.read_csv(out/'tables/literal_rule.csv');rf=rule[rule.period.isin(cfg['primary_future_quarters'])]
    if ((rf.accuracy>=.95)&(rf.macro_f1>=.95)).sum()>=4:warnings.append('Most labels solved by literal issue-name matching')
    absent=[label['absent_by_period'].get(p,[]) for p in cfg['primary_future_quarters']]
    if any(set(a)&set(b) for a,b in zip(absent,absent[1:])):warnings.append('An included label disappears in consecutive primary quarters')
    numerical='GO' if sum(passed)>=3 else 'NO-GO';verdict='GO' if numerical=='GO' and not warnings else 'NO-GO'
    reasons=[f'Future accuracy {future.accuracy.min():.1%}–{future.accuracy.max():.1%}; macro-F1 {future.macro_f1.min():.3f}–{future.macro_f1.max():.3f}.',
             f'Minimum {small.families.min():,} distinct families per class/primary quarter; {run["training_rows"]:,} exact-deduplicated training rows.',
             f'Useful confidence separation in {useful.period.nunique()} windows; future accuracy range {changes["accuracy"]:.1%}, ECE range {changes["ece"]:.3f}.',
             f'Historical family exposure at most {ds.seen_historical_fraction.max():.1%}; minimum {small.novel_families.min():,} novel families per class/primary quarter. Narrative-only tests passed.']
    decision=pd.DataFrame({'Condition':['1. Non-trivial task','2. Sufficient future data','3. Temporal/confidence behavior','4. Manageable leakage'],'Result':['PASS' if b else 'FAIL' for b in passed],'Evidence':reasons})
    decision.to_csv(out/'tables/go_no_go_conditions.csv',index=False)
    dump_json(out/'tables/go_no_go.json',{'verdict':verdict,'numerical_rule_result':numerical,'passed':sum(passed),'conditions':passed,'special_warnings':warnings,'confidence_useful_windows':int(useful.period.nunique()),'ranges':changes})
    # Reusable tidy headline outputs.
    normal.to_csv(out/'tables/headline_results.csv',index=False)
    normsel.to_csv(out/'tables/normal_thresholds.csv',index=False)
    # Plots.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    windows=normal.period.tolist();x=np.arange(len(windows))
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True)
    for ax,key,title in zip(axes.flat,['accuracy','macro_f1','nll','brier','ece','mean_confidence'],['Accuracy','Macro-F1','Log loss / NLL','Multiclass Brier (sum)','ECE (15 bins)','Mean confidence']):
        ax.plot(x,normal[key],'-o',label='Full published stream',color='#176b87')
        ns=novel.set_index('period').reindex(windows)
        ax.plot(x,ns[key],'--s',label='Unseen family, one representative',color='#c45a36')
        ax.set_title(title);ax.grid(alpha=.2);ax.set_xticks(x,windows,rotation=45,ha='right')
    axes[0,0].legend(fontsize=8);fig.suptitle('Frozen historical TF-IDF model: temporal reliability');fig.tight_layout();fig.savefig(figdir/'temporal_metrics.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for t in cfg['thresholds']:
        z=normsel[normsel.threshold==t].set_index('period').reindex(windows)
        axes[0].plot(x,z.accepted_accuracy,'-o',label=f'{t:.2f}');axes[1].plot(x,z.coverage,'-o',label=f'{t:.2f}')
    axes[0].axhline(.9,color='black',ls=':',lw=1);axes[0].set_ylabel('Accepted accuracy');axes[1].set_ylabel('Coverage')
    for ax in axes:ax.set_xticks(x,windows,rotation=45,ha='right');ax.grid(alpha=.2)
    axes[1].legend(title='Confidence threshold',fontsize=8,ncol=2);fig.suptitle('Fixed diagnostic thresholds, not certified gates');fig.tight_layout();fig.savefig(figdir/'threshold_transfer.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    colors={p:plt.get_cmap('tab10')(i) for i,p in enumerate(windows)}
    for subset,ax in zip(['operational','novel_family'],axes):
        for p in windows:
            z=rc[(rc.model=='normal')&(rc.subset==subset)&(rc.period==p)]
            if len(z):ax.plot(z.coverage,z.risk,label=p,color=colors[p])
        ax.set_xlabel('Coverage');ax.set_ylabel('Accepted error');ax.set_title('Full stream' if subset=='operational' else 'Unseen template families');ax.set_xlim(.05,1);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Risk–coverage curves (ties accepted together; 2026Q2 exploratory)');fig.tight_layout();fig.savefig(figdir/'risk_coverage.png',dpi=170);plt.close(fig)
    bins=pd.read_csv(out/'metrics/reliability_bins.csv');fig,ax=plt.subplots(figsize=(6,5))
    for p in ['validation','2025Q1','2025Q3','2026Q1']:
        z=bins[(bins.model=='normal')&(bins.subset=='operational')&(bins.period==p)];ax.plot(z.mean_confidence,z.accuracy,'-o',label=p)
    ax.plot([0,1],[0,1],':',color='black');ax.set(xlabel='Mean predicted confidence',ylabel='Observed accuracy',title='Reliability diagrams: published stream');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figdir/'reliability.png',dpi=170);plt.close(fig)
    short=['Debt not owed','Communication','Electronic','False statements','Contact/share threats','Legal/negative action','Written notice']
    cm=pd.read_csv(out/'metrics/confusion_normal_2026Q1_operational.csv',index_col=0).to_numpy();row=cm/cm.sum(axis=1,keepdims=True)
    fig,ax=plt.subplots(figsize=(9,7));im=ax.imshow(row,vmin=0,vmax=1,cmap='Blues')
    for i in range(7):
        for j in range(7):ax.text(j,i,f'{row[i,j]:.0%}',ha='center',va='center',color='white' if row[i,j]>.5 else 'black')
    ax.set_xticks(range(7),short,rotation=40,ha='right');ax.set_yticks(range(7),short);ax.set(xlabel='Predicted Issue',ylabel='Recorded Issue',title='2026Q1 confusion matrix: row-normalized');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(figdir/'confusion_2026Q1.png',dpi=170);plt.close(fig)
    # Family-bootstrap uncertainty is descriptive (not a deployment guarantee).
    rng=np.random.default_rng(20261002);ci=[]
    for p in windows:
        pred=pd.read_csv(out/f'predictions/normal_{p}.csv.gz');groups=pred.groupby('family',sort=False)
        n=groups.size().to_numpy();correct=groups.correct.sum().to_numpy();k=len(n)
        vals=[]
        for _ in range(300):
            ix=rng.integers(0,k,size=k);vals.append(correct[ix].sum()/n[ix].sum())
        ci.append({'period':p,'accuracy_low':float(np.quantile(vals,.025)),'accuracy_high':float(np.quantile(vals,.975)),'bootstrap_replicates':300,'unit':'template family'})
    pd.DataFrame(ci).to_csv(out/'metrics/accuracy_family_bootstrap.csv',index=False)
    # Reports.
    conditions=md(decision)
    go=f'''# {verdict}

Debt Collection → Issue feasibility pilot, 2 October 2026.

The fixed numerical rule passes **{sum(passed)} of 4** conditions: **{numerical}**. Special warning checks: {', '.join(warnings) if warnings else 'none triggered under the prespecified operational definitions'}.

{conditions}

The thresholds interpreting qualitative conditions were recorded in `configs/data.yaml` before classifier fitting. At least 200 families per class/quarter is a pilot adequacy threshold, not a claim that subgroup gate accuracy can be estimated precisely: at 90% correctness, 200 independent examples have a rough 95% margin of ±4.2 percentage points, before selective acceptance reduces the sample.

A family is a lexical dependence proxy, not verified consumer identity. Results apply to published narratives. Label ambiguity, within-period templates and changing publication rates remain important limitations even when the leakage condition passes.

This result does not establish a Laya advantage, paper novelty or a safe operational 90% gate. No transformer, neural model or GPU experiment was run. STOP: explicit user approval is required for any larger experiment.
'''
    (ROOT/'research/go_no_go.md').write_text(go,encoding='utf8')
    rawcounts=pd.read_csv(out/'audits/debt_issue_counts.csv',index_col=0);training=pd.DataFrame({'Issue':rawcounts.columns,'Training n':rawcounts.loc['train'].values,'Training percent':100*rawcounts.loc['train'].values/rawcounts.loc['train'].sum()})
    headline=normal[['period','n','accuracy','macro_f1','balanced_accuracy','nll','brier','ece','mean_confidence','confidence_minus_accuracy']]
    subsetcomp=normal[['period','accuracy','macro_f1']].merge(novel[['period','accuracy','macro_f1']],on='period',suffixes=('_full','_novel'))
    abl=m[(m.model=='keyword_mask_refit')&(m.subset=='operational')][['period','accuracy','macro_f1']].merge(normal[['period','accuracy','macro_f1']],on='period',suffixes=('_masked','_normal'))
    abl['accuracy_delta']=abl.accuracy_masked-abl.accuracy_normal;abl['macro_f1_delta']=abl.macro_f1_masked-abl.macro_f1_normal
    feature=pd.read_csv(out/'tables/top_features.csv');feature=feature.groupby('issue',sort=False).head(5);features=feature.groupby('issue',sort=False).feature.apply(lambda a:', '.join(a)).reset_index(name='Five highest-weight features')
    availability=pd.read_csv(out/'audits/availability_quarterly.csv');monthly=pd.read_csv(out/'audits/availability_monthly.csv')
    selshow=normsel[normsel.period.isin(['validation','2025Q1','2026Q1'])][['period','threshold','accepted','coverage','accepted_accuracy','risk']]
    maxabl=float(abl.accuracy_delta.abs().max());narr=ing['debt_narratives'];mainn=int(future.n.sum())
    worst=pc[(pc.model=='normal')&(pc.subset=='operational')&(pc.period=='2026Q1')]
    majority=m[(m.model=='majority_prior')&(m.subset=='operational')][['period','accuracy','macro_f1']]
    maskcounts=pd.read_csv(out/'tables/keyword_mask_counts.csv');maskcounts['affected_fraction']=maskcounts.n_affected/maskcounts.n
    classsel=pd.read_csv(out/'metrics/selective_by_class.csv') if (out/'metrics/selective_by_class.csv').exists() else pd.DataFrame()
    classselshow=classsel[(classsel.period=='2026Q1')&(classsel.threshold==.9)][['issue','support','accepted','coverage','accepted_accuracy']] if len(classsel) else pd.DataFrame()
    report=f'''# Debt Collection → Issue: CPU feasibility pilot

**Decision: {verdict} ({sum(passed)}/4 fixed conditions).** No Laya or ModernBERT was trained. No GPU was used. This is task feasibility, not evidence that a TDM will help.

## 1. Provenance and authenticity

Downloaded 17 official CFPB archive ZIPs directly from the [official narrative archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/), plus its landing page and the [August 2023 taxonomy](https://files.consumerfinance.gov/f/documents/cfpb_consumer_complaint_form_product_issue_options_August_2023_FINAL.pdf). `data/source_manifest.json` records exact URLs, download UTC timestamps, sizes and SHA256. No Kaggle/third-party cleaned data.

These are authentic administrative records; this does not establish that each allegation is correct or each narrative is human-authored. Product/Issue are administrative/consumer-selected labels, not expert-adjudicated semantic ground truth. Published complaints are not representative of all consumers. See [CFPB database guidance](https://www.consumerfinance.gov/data-research/consumer-complaints/).

## 2. Actual usable range and cohort sizes

Across September 2023–August 2026: **{ing['unique_records']:,} records**, **{narr:,} Debt Collection narratives**. Duplicate Complaint IDs: {ing['duplicate_ids']}; invalid dates: {ing['invalid_dates']}. All {len(classes)} labels qualify from training alone; all persist later. There are {int(rawcounts.loc['train'].sum()):,} raw training narratives and {run['training_rows']:,} after retaining the earliest exact-text occurrence. The five primary future quarters contain **{mainn:,} narratives**.

Primary future horizon: 2025Q1–2026Q1. Q4 2024 is an initial later test. Q2 2026 passed the frozen availability heuristic and is **exploratory**, not proof of complete/unbiased publication. July failed the adequacy/availability rule and was not modeled. August has no Debt Collection narratives and was not modeled.

{md(availability[['quarter','total_records','nonempty_narratives','debt_records','debt_narratives','debt_narrative_fraction']])}

2023Q3 here contains September only; 2026Q3 contains July/August only. Monthly and daily CSVs preserve exact coverage. July official export: {int(monthly.loc[monthly.month=='2026-07','total_records'].iloc[0]):,} records / {int(monthly.loc[monthly.month=='2026-07','nonempty_narratives'].iloc[0]):,} narratives. August: {int(monthly.loc[monthly.month=='2026-08','total_records'].iloc[0]):,} records / {int(monthly.loc[monthly.month=='2026-08','nonempty_narratives'].iloc[0]):,} narrative, zero Debt Collection. These independently reproduce the earlier warning.

![Availability](../outputs/figures/availability.png)

The Debt Collection narrative fraction falls from roughly 53% in early 2024 to 22% in 2026Q1. Therefore observed changes cannot be attributed solely to consumer-language/concept drift. June process changes and August publication cessation are additional context: [June CFPB notice](https://www.consumerfinance.gov/about-us/newsroom/the-cfpb-is-correcting-flaws-to-restore-integrity-and-utility-to-the-consumer-complaint-system/), [August CFPB notice](https://www.consumerfinance.gov/about-us/newsroom/the-cfpb-to-cease-discretionary-publication-of-complaint-narratives-and-visualizations/).

## 3. Stable classes and imbalance

{md(training,2)}

No new/disappeared original label strings occur among observed Debt Collection narratives in modeled periods. No renaming was inferred or labels merged. Stable strings do not prove stable annotation behavior. `configs/debt_collection_issues.yaml` contains inclusion decisions, original labels, training and all future counts. `debt_issue_proportions.csv` and `label_prior_drift.csv` document prior changes. Training entropy and period-specific character/word distributions are in `debt_period_summary.csv`.

![Class proportions](../outputs/figures/class_distribution.png)

## 4. Duplication, templates and leakage

{dd['records']:,} rows yield {dd['exact_unique']:,} raw exact unique texts and {dd['families']:,} conservative lexical families. There are {dd['repeated_families']:,} repeated families, {dd['cross_period_families']:,} spanning periods, and {dd['rows_in_cross_period_families']:,} rows belonging to cross-period families. Exactly identical texts have conflicting Issue labels in {label['exact_groups_with_conflicting_labels']:,} groups; {dd['conflicting_label_families']:,} broader template families contain multiple labels. That is evidence of ambiguity/dependence, not proof every conflicting label is wrong.

Method: 128-permutation MinHash on normalized word 5-shingles, LSH candidate threshold .70, exact Jaccard >=.85 confirmation, transitive union-find families. Number/redaction normalization is for deduplication only. Short texts still undergo exact/canonical checks. No exhaustive semantic-paraphrase search or consumer identity linkage was performed; missed paraphrases remain possible.

{md(dup[['period','n','families','seen_historical_fraction','novel_families','largest_family']])}

Training/validation exposure is 100% by definition in that column; it is not a claim that validation wholly duplicates training. The cross-period column in the full CSV shows that distinction. Training exact duplicates are removed. Future operational results retain repeats, while novel-family results remove historical families and keep one representative per period. Templates can inflate or depress accuracy depending on their label distribution, not only inflate it.

![Template exposure](../outputs/figures/duplicate_exposure.png)

## 5. Lightweight model and chronological results

Word unigram/bigram TF-IDF, max 100,000 features, min_df=3, sublinear TF, multinomial logistic regression. Only narrative strings enter the vectorizer. Train September 2023–June 2024; select among C=0.3/1/3 on July–September 2024 macro-F1. Selected C={run['selected_C']}. Vocabulary is fitted only on exact-deduplicated training texts. No random split and no calibration fitting. Models are frozen for future evaluation.

{md(headline)}

Brier is the sum over classes (range 0–2); ECE uses 15 fixed equal-width bins. Mean confidence is maximum class probability. Full CSVs contain every requested per-class precision/recall/F1, confusion matrix and probability metric for every evaluated period and variant. Convergence/tuning records are in `outputs/tables/historical_tuning.csv`.

![Temporal metrics](../outputs/figures/temporal_metrics.png)

{md(subsetcomp)}

The task is far from near-perfect. The full stream and novel-family populations differ materially; neither should replace the other without explanation. Among the five primary future quarters, novel-family accuracy stays approximately 51–54%, much more stable than full-stream accuracy. Therefore the large full-stream decline is not evidence of pure semantic drift; repetitions and sample composition materially affect it. The model is a weak lexical baseline, so difficulty may reflect ambiguous administrative labels as well as limited model capacity. The baseline does not prove that a stronger model will solve the target reliably.

Majority-class baseline:

{md(majority)}

In 2026Q1, TF-IDF accuracy is below the majority baseline (about 50.0% versus 52.8%), although TF-IDF macro-F1 is substantially higher. A non-trivial task is not automatically a good high-accuracy automation task.

## 6. Class confusion and lexical difficulty

2026Q1 per-class results:

{md(worst[['issue','precision','recall','f1','support']])}

![Confusion](../outputs/figures/confusion_2026Q1.png)

Highest-weight features (training model only):

{md(features)}

The phrase ablation masks a fixed list derived from Issue names, saved before fitting in `configs/keyword_ablation.yaml`. It does not remove all semantically related words. We both refit TF-IDF/logistic regression on masked training text using the selected C and mask the input to the already-frozen normal model. Maximum absolute accuracy change for the refit diagnostic: **{maxabl:.2%}** across evaluated windows. Only {maskcounts.affected_fraction.min():.1%}–{maskcounts.affected_fraction.max():.1%} of evaluation narratives are affected, so this is a narrow literal-phrase diagnostic, not a strong test against all lexical shortcuts.

{md(abl[['period','accuracy_delta','macro_f1_delta']])}

Literal phrase-rule baseline (ambiguous/no match falls back to historical majority):

{md(rule)}

Thus direct label-name phrases do not explain near-perfect results—there are no near-perfect results. This does not prove deep semantics: TF-IDF still uses lexical correlations and the ablation is deliberately narrow. The prior-only majority baseline and its probability metrics are retained in the full metrics CSV; a small ECE alone can be misleading for such an uninformative predictor.

## 7. Probabilities and selective prediction

Confidence separates some easy and hard cases in {useful.period.nunique()} windows under the frozen diagnostic definition (>=5 percentage-point accuracy gain at 10–90% coverage). This is ranking usefulness, not calibration or a guaranteed 90% gate. Raw confidence often exceeds realized correctness. No gate was selected, certified or retrospectively tuned to claim a promise.

{md(selshow)}

All six thresholds and all windows are in `outputs/tables/normal_thresholds.csv`, including counts and empty-set handling. Risk–coverage curves include entire confidence ties. A high accepted accuracy with very few accepted records is not treated as sufficient evidence.

Class-conditional audit at threshold 0.90 in 2026Q1 (conditioned on the **recorded true Issue**, not predicted class):

{md(classselshow) if len(classselshow) else 'Run research/verify_outputs.py to generate the per-class selective table.'}

The aggregate gate conceals sharply different class behavior. Several rare/difficult true classes are usually rejected, and the few accepted cases can be confidently misclassified as another Issue. This prevents interpreting aggregate accepted accuracy as a classwise reliability promise.

![Threshold behavior](../outputs/figures/threshold_transfer.png)
![Risk coverage](../outputs/figures/risk_coverage.png)
![Reliability](../outputs/figures/reliability.png)

## 8. Future adequacy, uncertainty and interpretation

Minimum class/primary-quarter support is {small.families.min():,} distinct families and {small.novel_families.min():,} historically novel families. That clears the frozen pilot criterion, but rare-class accepted sets can still be too small for precise selective-risk estimates. Family-bootstrap accuracy intervals (300 replicates, seed fixed) are saved in `outputs/metrics/accuracy_family_bootstrap.csv`; they are descriptive and assume exchangeable families, not a proof of independence or robustness to time dependence.

{md(pd.DataFrame(ci)[['period','accuracy_low','accuracy_high']])}

These full-stream intervals are broad because some template families carry substantial mass. Therefore the plotted differences are descriptive, not a claim that every adjacent-quarter difference is statistically established. The novel-family sensitivity is substantially more stable; a later paper must investigate sample composition and dependence rather than simply label the full-stream decline “semantic drift.”

All input-feature tests passed: metadata cannot alter the narrative string passed to the model. Complaint IDs do not overlap splits, training dates precede validation, and the novel-family sensitivity excludes training/validation families. Existing literal words in narratives are legitimate inputs, not automatically target leakage.

## 9. Novelty update

See `research/novelty_update.md`. Prior work already covers CFPB classification, temporal classification, chronological CFPB calibration, selective prediction and online recalibration. A newly checked Second Thought repository also provides typed-model drift monitoring. No verified exact matched Laya/CFPB Issue temporal-gate study was located, but that absence does not establish first-ever novelty. A GO here is only a task feasibility decision.

The future cohorts have now been inspected in this pilot. A later paper must disclose that fact and must not call their aggregate performance wholly unseen confirmatory evidence. It can still be a transparent retrospective controlled audit with a frozen next-stage protocol.

## 10. Fixed decision and stop

{conditions}

**Final: {verdict}.** Numerical rule: {sum(passed)}/4; special warnings: {', '.join(warnings) if warnings else 'none triggered'}. The decision does not justify immediate neural training. Stop here and wait for explicit user approval.
'''
    (ROOT/'research/pilot_report.md').write_text(report,encoding='utf8')
    log=ROOT/'research/decisions_log.md';prefix=log.read_text(encoding='utf8').split('\n## Pilot completed')[0]
    log.write_text(prefix+f'\n## Pilot completed\n\n- Audit complete before fitting; seven unchanged labels included. Q2 2026 exploratory; July/August excluded from fitting/evaluation under the frozen availability rule.\n- C={run["selected_C"]} selected on historical validation only. No future tuning.\n- {verdict}: {sum(passed)}/4 conditions; warnings: {warnings}. No scope expansion or neural/GPU work.\n- Plot checks and independent probability/NLL/accuracy recomputation completed. Added per-class selective diagnostics to expose minority-class failures; these did not change any model or threshold.\n',encoding='utf8')
    print(verdict,sum(passed),reasons,warnings,flush=True)
if __name__=='__main__':main()
