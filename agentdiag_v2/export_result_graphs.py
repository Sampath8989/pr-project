"""Export saved experimental results as report figures, without rerunning training."""
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results_graphs'
OUT.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': 'white'})
COLORS = ['#64748b', '#0891b2', '#e89532', '#b45365']
figures, sources, values = [], {}, []

def load(relative):
    p = ROOT / relative
    sources[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text(encoding='utf-8'))

def finish(fig, name, caption):
    fig.text(.04, .015, caption, fontsize=9, color='#475569', wrap=True)
    fig.tight_layout(rect=(0, .08, 1, 1))
    p = OUT / (name + '.png')
    fig.savefig(p, dpi=220)
    plt.close(fig)
    with Image.open(p) as im:
        im.verify()
    figures.append({'file': p.name, 'caption': caption})

def bars(name, title, labels, series, caption, ylabel='Accuracy (%)', limit=100):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = .8 / len(series)
    for i, (legend, nums) in enumerate(series.items()):
        b = ax.bar(x + (i-(len(series)-1)/2)*width, nums, width,
                   label=legend, color=COLORS[i % len(COLORS)])
        ax.bar_label(b, labels=[f'{v:.1f}' for v in nums], padding=4, fontsize=10)
        for label, value in zip(labels, nums):
            values.append([name, legend, label, float(value), ylabel])
    ax.set(title=title, ylabel=ylabel, xticks=x, xticklabels=labels)
    if limit is not None:
        ax.set_ylim(0, limit)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right')
    finish(fig, name, caption)

m = load('contextual_upgrade/metrics.json')
selection = load('contextual_upgrade/selection.json')
groups = ['Algorithm-Generated', 'Hand-Crafted', 'Overall']
note = 'Saved original-corpus evaluation; exploratory. Overall combines different subsets.'
for metric, title in [('agent','Agent identification'), ('step','Exact failure-step identification'), ('joint','Both agent and step correct')]:
    bars('01_' + metric + '_accuracy', title, groups,
         {label: [100*m[key][g][metric+'_accuracy'] for g in groups]
          for key,label in [('old','Previous local ranker'),('new','Updated contextual model')]}, note)
bars('02_dataset_sizes', 'Evaluation dataset sizes', groups[:2],
     {'Conversations': [m['new'][g]['traces'] for g in groups[:2]]},
     '184 saved Who&When conversations; this figure is an inventory, not an accuracy result.', 'Number of conversations', 150)
for g,slug in zip(groups[:2], ['algorithm','handcrafted']):
    bars('03_'+slug+'_correct_counts', g + ': number of correct predictions',
         ['Agent','Exact step','Both'],
         {label: [m[key][g][field] for field in ['correct_agent','correct_step','correct_both']]
          for key,label in [('old','Previous'),('new','Updated')]},
         f"Same {m['new'][g]['traces']} conversations for both versions.", 'Correct conversations', m['new'][g]['traces']*1.15)
    pairs=m['paired_changes'][g]
    bars('04_'+slug+'_gains_losses', g + ': changed predictions', ['Agent','Exact step'],
         {'Wrong to correct':[pairs[k]['gains'] for k in ['agent','step']],
          'Correct to wrong':[pairs[k]['losses'] for k in ['agent','step']]},
         'Show gains and regressions together. Net improvement = gains minus losses.', 'Number of conversations', 22)
    fig, ax=plt.subplots(figsize=(10,6))
    for i,k in enumerate(['agent','step']):
        v=100*m['new'][g][k+'_accuracy']; lo,hi=np.array(m['new'][g][k+'_95pct_interval'])*100
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=7,color=COLORS[i+1])
        ax.text(v,i+.15,f'{v:.1f}% [{lo:.1f}, {hi:.1f}]',ha='center')
    ax.set(xlim=(0,100),ylim=(-.5,1.6),yticks=[0,1],yticklabels=['Agent','Exact step'],xlabel='Accuracy (%)',title=g+': uncertainty in updated results')
    finish(fig,'05_'+slug+'_uncertainty','Saved 95% trace-bootstrap intervals; they do not remove development-set reuse or question dependence.')

predfile=ROOT/'contextual_upgrade/benchmark_predictions.jsonl'
sources[str(predfile.relative_to(ROOT))]=hashlib.sha256(predfile.read_bytes()).hexdigest()
rows=[json.loads(x) for x in predfile.read_text().splitlines() if x]
assert len(rows)==184
for g,slug in zip(groups[:2],['algorithm','handcrafted']):
    rr=[r for r in rows if r['subset']==g]
    agent=sum(r['actual_agent']==r['predicted_agent'] for r in rr)
    step=sum(r['actual_step']==r['predicted_step'] for r in rr)
    both=sum(r['actual_agent']==r['predicted_agent'] and r['actual_step']==r['predicted_step'] for r in rr)
    assert (agent,step,both)==tuple(m['new'][g][k] for k in ['correct_agent','correct_step','correct_both'])
    bars('06_'+slug+'_outcomes',g+': prediction outcomes',
         ['Both correct','Agent only','Step only','Both wrong'],
         {'Updated':[both,agent-both,step-both,len(rr)-agent-step+both]},
         'Computed directly from individual saved predictions.', 'Number of conversations', len(rr)*.65)
    errors=[abs(r['predicted_step']-r['actual_step']) for r in rr]
    bins=[sum(e==0 for e in errors),sum(e==1 for e in errors),sum(2<=e<=3 for e in errors),sum(4<=e<=7 for e in errors),sum(e>=8 for e in errors)]
    bars('07_'+slug+'_step_distance',g+': distance from the true failure step',
         ['Exact','1 message','2–3 messages','4–7 messages','8+ messages'],{'Updated':bins},
         'Absolute raw-message index distance. Near misses do not count toward exact step accuracy.', 'Number of conversations', max(bins)*1.35)

cv=selection['selected_external_cv_metrics']
bars('08_validation_vs_original','External validation and original-corpus results',
     ['Algorithm agent','Algorithm step','Hand-Crafted agent','Hand-Crafted step'],
     {'External grouped validation':[100*cv[g][k+'_accuracy'] for g in groups[:2] for k in ['agent','step']],
      'Original corpus':[100*m['new'][g][k+'_accuracy'] for g in groups[:2] for k in ['agent','step']]},
     'Different datasets. Validation was used for selection; it is not an independent final benchmark.')
cal=load('artifacts/phase5_calibration_evaluation/calibration.json')
bars('09_bn_calibration_scores','Bayesian network: probability quality before and after calibration',
     ['Brier score','Log loss','Calibration error (ECE)'],
     {label:[cal[key][k] for k in ['brier','log_loss','ece']]
      for key,label in [('uncalibrated_test','Before calibration'),('calibrated_test','After calibration')]},
     'Separate frozen Bayesian model, n=13. Lower is better. Top-choice accuracy stays 5/13.', 'Error score (lower is better)', 1.8)
fig,ax=plt.subplots(figsize=(10,6))
ax.plot([0,1],[0,1],'--',color='#94a3b8',label='Ideal calibration')
for key,label,color in [('uncalibrated_test','Before',COLORS[0]),('calibrated_test','After',COLORS[1])]:
    bb=[b for b in cal[key]['bins'] if b['count']]
    ax.plot([b['avg_confidence'] for b in bb],[b['accuracy'] for b in bb],'o-',label=label,color=color)
    for b in bb: ax.annotate('n='+str(b['count']),(b['avg_confidence'],b['accuracy']),xytext=(3,6),textcoords='offset points',fontsize=8)
ax.set(xlim=(0,1),ylim=(0,1.08),xlabel='Average predicted confidence',ylabel='Observed accuracy',title='Bayesian network reliability diagram');ax.legend()
finish(fig,'10_bn_reliability','Separate frozen Bayesian model, n=13. Tiny bins make this estimate noisy; not the contextual model.')
missing=load('artifacts/phase4_missing_evidence/metrics.json')
bars('11_missing_evidence','Bayesian network: missing-evidence experiment',
     [k.replace('_',' ') for k in missing],{'Agent accuracy':[100*v['accuracy'] for v in missing.values()]},
     'Separate Bayesian model; 13 examples per condition. A higher small-sample score does not prove missing evidence helps.')
hmm=load('artifacts/phase6_temporal_hmm/metrics.json')
bars('12_early_warning','Early-warning results', ['Early recall (10 eligible)','Wrong-agent warnings (12 traces)'],
     {k.replace('_',' '):[100*v['early_recall_eligible'],100*v['false_agent_warning_rate']] for k,v in hmm.items()},
     'Higher early recall is better; lower wrong-agent warning rate is better. HMM remains unreliable.', 'Rate (%)', 110)
triage=load('artifacts/phase9_evidence_triage/metrics.json')
bars('13_shortlist','Bayesian network: first choice and two-agent shortlist', ['First choice','Actual agent in top two'],
     {'Correct coverage':[100*triage['top_one_correct']/triage['test_count'],100*triage['top_two_coverage']]},
     'Separate Bayesian model, n=13. Shortlist coverage is not top-choice accuracy.')
hybrid=load('hybrid_upgrade/metrics.json')
bars('14_hybrid_grouped_validation','Earlier probabilistic hybrid: grouped validation',
     ['Bayesian network','Probabilistic hybrid'],
     {'Agent accuracy':[100*hybrid['nested_group_cv'][k]['accuracy'] for k in ['bn','hybrid']]},
     'Separate 91-example grouped evaluation; paired p=0.0768. Improvement is not conclusive.')
base=load('artifacts/phase2_evidence_baselines/baselines.json')['metrics']
for split in ['validation_4agent','test_4agent']:
    entries={k:v[split] for k,v in base.items() if split in v}
    bars('15_baselines_'+split,'Original simple baselines: '+split.replace('_',' '),
         [k.replace('_',' ') for k in entries],{'Agent accuracy':[100*v['accuracy'] for v in entries.values()]},
         'Original four-agent split, n=13. Separate from the 184-example contextual evaluation.')
paper_values={'Algorithm-Generated':[[51.12,13.53],[63.73,37.30]],
              'Hand-Crafted':[[53.44,3.51],[63.82,20.68]]}
for g,slug in zip(groups[:2],['algorithm','handcrafted']):
    bars('16_paper_reference_'+slug,g+': published reference scores', ['Agent','Exact step'],
         {'Previous local':[100*m['old'][g][k+'_accuracy'] for k in ['agent','step']],
          'Updated local':[100*m['new'][g][k+'_accuracy'] for k in ['agent','step']],
          'Who&When (reported)':paper_values[g][0], 'AgenTracer (reported)':paper_values[g][1]},
         'No answer supplied. Paper numbers are references, not a matched head-to-head or a proven paper win.')
bars('17_algorithm_targets','Algorithm-Generated: current results and requested targets', ['Agent','Exact step'],
     {'Current':[100*m['new']['Algorithm-Generated'][k+'_accuracy'] for k in ['agent','step']],
      'Requested target':[60,35]},
     'Targets remain unmet. At n=126, reaching them requires 76 correct agents and 45 correct steps.')

# Preserve all scalar result values for phase coverage and audit, without graphing model weights.
scalar_rows=[]
def flatten(obj,path,source):
    if isinstance(obj,dict):
        for k,v in obj.items(): flatten(v,path+[k],source)
    elif isinstance(obj,(int,float)) and not isinstance(obj,bool):
        scalar_rows.append([source,'.'.join(path),obj])
for pattern in ['artifacts/**/metrics.json','artifacts/**/baselines.json','artifacts/**/summary.json','artifacts/**/calibration.json','hybrid_upgrade/metrics.json','benchmark_step/metrics.json','contextual_upgrade/metrics.json']:
    for p in ROOT.glob(pattern): flatten(load(str(p.relative_to(ROOT))),[],str(p.relative_to(ROOT)))
for name,header,data in [('graph_values.csv',['figure','series','category','value','unit'],values),('saved_phase_metrics.csv',['source','metric','value'],scalar_rows)]:
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(data)
with (OUT/'prediction_evidence.csv').open('w',newline='',encoding='utf-8') as f:
    fields=['id','subset','actual_agent','predicted_agent','actual_step','predicted_step']
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
(OUT/'source_manifest.json').write_text(json.dumps({'sources_sha256':sources,'figures':figures,'prediction_count':len(rows),'counts_recomputed':True},indent=2),encoding='utf-8')
readme='# Saved result graphs and evidence\n\nThese figures visualize saved experiments; no new model training was performed.\n\n## Use in your report\n\nGraphs support the reported results. The prediction CSV, source JSON files, source hashes, and reproduction checks provide the underlying evidence. A graph alone does not prove generalization.\n\nOriginal-corpus results are exploratory. The contextual model has not been evaluated on live or image/audio/video inputs. Bayesian and contextual results belong to different models and evaluation sets. Published numbers in reference charts are taken from the saved Contextual_Report.md: [Who&When](https://arxiv.org/html/2505.00212) and [AgenTracer](https://arxiv.org/html/2509.03312). They are no-answer-supplied reported results, not independently rerun or matched comparisons.\n\n## Figures\n\n'
for item in figures: readme+=f"- [{item['file']}]({item['file']}): {item['caption']}\n"
readme+='\n## Evidence files\n\n- prediction_evidence.csv: all 184 actual and predicted answers.\n- graph_values.csv: values plotted by the bar charts.\n- saved_phase_metrics.csv: numeric scalar metrics from saved phase reports.\n- source_manifest.json: source hashes, figure captions, and checked prediction count.\n\n## Reproduce\n\nRun `python agentdiag_v2/export_result_graphs.py` from the repository parent with matplotlib, numpy and Pillow installed. All PNGs are checked for valid image encoding. The generator asserts that prediction counts match the saved accuracy metrics.\n'
(OUT/'README.md').write_text(readme,encoding='utf-8')
print(json.dumps({'png_count':len(figures),'predictions_checked':len(rows),'folder':str(OUT)}))
