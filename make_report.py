#!/usr/bin/env python3
"""Generate figures, summary tables, input example and an offline HTML report."""
import argparse,base64,json,html
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from src.simulator import DYNAMIC,STATIC,sequences

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':180})
BLUE='#156082';TEAL='#168b83';ORANGE='#df8430';RED='#ba4d51';GRAY='#6c7885'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');args=ap.parse_args();root=Path(args.root)
    out=root/'results';figdir=out/'figures';figdir.mkdir(exist_ok=True)
    p=pd.read_csv(out/'prediction_metrics.csv');r=pd.read_csv(out/'rollout_metrics.csv')
    meta=json.loads((out/'run_metadata.json').read_text())
    def finish(fig,name):
        fig.savefig(figdir/(name+'.png'),bbox_inches='tight');fig.savefig(figdir/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    # Actual architecture: three inputs, explicit updated measurements and guard.
    fig,ax=plt.subplots(figsize=(11,6));ax.set(xlim=(0,11),ylim=(0,6));ax.axis('off')
    def box(x,y,w,h,text,color=BLUE):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',fc=color,ec='none'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',color='white',fontsize=10)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=GRAY,lw=1.6))
    box(.3,4.4,3.1,1,'Previous 14 days\n14 daily features / day')
    box(4.2,4.4,2.5,1,'LSTM\n24 hidden units');arrow((3.5,4.9),(4.05,4.9))
    box(.3,2.7,3.1,1,'Current measurements + forecasts\n14 dynamic + 6 static inputs',TEAL)
    box(4.2,2.7,2.5,1,'Concatenate: 44 features\nDense 32 + ReLU');arrow((5.45,4.25),(5.45,3.85));arrow((3.5,3.2),(4.05,3.2))
    box(7.5,2.7,3,1,'Two output heads\nEvent probability + gross mm');arrow((6.85,3.2),(7.35,3.2))
    box(7.5,.9,3,1,'Wet-soil / capacity guard\nFinal daily irrigation depth',TEAL);arrow((9,2.55),(9,2.05))
    box(3,.9,3.5,1,'Apply irrigation; observe next day\nNew soil and weather observations',GRAY);arrow((7.35,1.4),(6.65,1.4))
    ax.annotate('',xy=(1.8,2.55),xytext=(3,1.4),arrowprops=dict(arrowstyle='->',color=GRAY,lw=1.6,connectionstyle='angle3,angleA=180,angleB=90'))
    ax.set_title('Daily irrigation scheduler | 5,346 trainable parameters',loc='left',fontsize=15,pad=12)
    finish(fig,'01_architecture')
    # Data quantity and lookback length.
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for ax,names,xs,title,xlabel in [
        (axes[0],['seasons_1','seasons_3','seasons_6'],[480,1440,2880],'More training data reduced error','Training decisions across eight farms'),
        (axes[1],['history_7','history_14','history_30'],[7,14,30],'History length is a design choice','Past days supplied to LSTM')]:
        q=p[(p.test=='source_future_years')&p.experiment.isin(names)].groupby('experiment').mae_guarded_mm.agg(['mean','std']).reindex(names)
        ax.errorbar(xs,q['mean'],yerr=q['std'],fmt='o-',capsize=4,color=BLUE)
        ax.set(title=title,xlabel=xlabel,ylabel='Guarded irrigation MAE (mm/day)',xticks=xs,ylim=(0,1.3));ax.grid(axis='y',alpha=.18)
    fig.suptitle(f'Synthetic held-out seasons | mean ± SD over {len(meta["seeds"])} training seeds',fontsize=11)
    fig.tight_layout();finish(fig,'02_data_and_history')
    # Target adaptation, negative result explicitly visible.
    fig,axes=plt.subplots(1,2,figsize=(10,4.4));names=['zero_shot','head_tune','full_tune','scratch']
    labels=['Unchanged','Head only','All layers','From scratch']
    for ax,test,title in zip(axes,['target_8','target_9'],['Target farm 8','Target farm 9']):
        q=p[p.test==test].groupby('experiment').mae_guarded_mm.agg(['mean','std']).reindex(names)
        ax.bar(labels,q['mean'],yerr=q['std'],capsize=4,color=[BLUE,TEAL,ORANGE,GRAY]);ax.set(title=title,ylabel='Guarded MAE (mm/day)',ylim=(0,2));ax.tick_params(axis='x',rotation=15)
    fig.suptitle('Adapt with 120 local training days; test on 120 separate days',fontsize=11);fig.tight_layout();finish(fig,'03_transfer')
    # A prespecified farm/season example plus aggregate storm result.
    d=pd.read_csv(out/'example_daily.csv');d=d[d.day>=0]
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    rain=d[(d.method=='lstm')&(d.variant=='storm')];axes[0].bar(rain.day,rain.rain_mm,color='#8ab8cd');axes[0].set(ylabel='Rain (mm)',title=f'Prespecified example: target farm 8, season 7, seed {meta["seeds"][0]}')
    for name,label,color in [('lstm','Updated LSTM',BLUE),('stale_schedule','Stale schedule',ORANGE),('teacher','Water-balance teacher',TEAL)]:
        a=d[(d.method==name)&(d.variant=='storm')]
        axes[1].plot(a.day,a.depletion_fraction_obs,label=label,color=color)
        axes[2].step(a.day,a.irrigation_mm.cumsum(),where='post',label=label,color=color)
    axes[1].axhline(.55,color=GRAY,ls='--',lw=1,label='Stress threshold (illustrative)')
    axes[1].set(ylabel='Observed depletion / TAW');axes[1].legend(fontsize=8,ncol=2)
    axes[2].set(ylabel='Cumulative irrigation (mm)',xlabel='Date',xticks=[0,14,30,44,59],xticklabels=['Apr 1','Apr 15','May 1','May 15','May 30'])
    for ax in axes:ax.axvline(30,color=RED,ls=':',lw=1);ax.grid(axis='y',alpha=.15)
    fig.suptitle('An 80 mm storm is observed before the May 1 decision; weights stay fixed',fontsize=11)
    fig.tight_layout();finish(fig,'04_storm_response')
    # Closed-loop target shift; source and target cohorts must never be pooled with transfer-only methods.
    names=['teacher','lstm','mlp','head_tune','full_tune','scratch','no_irrigation']
    labels=['Teacher','LSTM','MLP','Head tune','Full tune','Scratch','No irrig.']
    target=r[(r.farm>=8)&(r.variant=='normal')]
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for ax,col,title in [(axes[0],'water_mm','Applied gross irrigation'),(axes[1],'stress_days','End-of-day threshold exceedances')]:
        # Aggregate within seed first; seeds are optimizer replicates, not weather replicates.
        q=target.groupby(['method','seed'])[col].mean().groupby('method').agg(['mean','std']).reindex(names)
        ax.bar(labels,q['mean'],yerr=q['std'],capsize=3,color=[GRAY,BLUE,ORANGE,TEAL,TEAL,GRAY,RED]);ax.tick_params(axis='x',rotation=35)
        ax.set(title=title,ylabel='mm / 60 days' if col=='water_mm' else 'days / 60 days')
    fig.suptitle('Closed-loop target farms | lower water use must be checked against stress',fontsize=11)
    fig.tight_layout();finish(fig,'05_closed_loop')
    fig,axes=plt.subplots(1,2,figsize=(10,4.3))
    for ax,test,title in zip(axes,['source_future_years','source_storm'],['Ordinary held-out seasons','May 1 storm seasons']):
        names=['lstm','augmented','mlp'];q=p[(p.test==test)&p.experiment.isin(names)].groupby('experiment').mae_guarded_mm.agg(['mean','std']).reindex(names)
        ax.bar(['LSTM','Storm-trained LSTM','MLP'],q['mean'],yerr=q['std'],capsize=4,color=[BLUE,TEAL,ORANGE]);ax.set(title=title,ylabel='Guarded MAE (mm/day)',ylim=(0,.9));ax.tick_params(axis='x',rotation=10)
    fig.suptitle('Ordinary training, extreme-event augmentation and a current-state baseline',fontsize=11)
    fig.tight_layout();finish(fig,'06_baselines_and_augmentation')
    # Save human-readable summary tables with cohort labels.
    ps=p.groupby(['test','experiment'])[['mae_guarded_mm','f1','brier']].agg(['mean','std'])
    ps.columns=['_'.join(c) for c in ps.columns];ps.to_csv(out/'prediction_summary.csv')
    rr=r.assign(cohort=np.where(r.farm<8,'source','target'))
    rs=rr.groupby(['cohort','variant','method','seed'])[['water_mm','stress_days','et_shortfall_mm','wet_actions','may_water_mm']].mean().groupby(['cohort','variant','method']).agg(['mean','std'])
    rs.columns=['_'.join(c) for c in rs.columns];rs.to_csv(out/'closed_loop_summary.csv')
    # A runnable example contains measured input fields only, no target/true state.
    frame=pd.read_csv(root/'data/synthetic_normal.csv.gz');g=frame[(frame.farm==8)&(frame.year==7)].sort_values('day')
    current_day=40
    example=dict(history=g[g.day.between(current_day-14,current_day-1)][DYNAMIC].to_dict('records'),
                 current=g[g.day==current_day][DYNAMIC].iloc[0].to_dict(),
                 farm={**g[STATIC].iloc[0].to_dict(),'gross_application_rate_mm_hour':8.})
    (root/'data/example_input.json').write_text(json.dumps(example,indent=2))
    def im(name,caption):
        b64=base64.b64encode((figdir/(name+'.png')).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    def table(df):return df.to_html(float_format=lambda x:f'{x:.3f}',border=0)
    body='''<h1>LSTM irrigation scheduling</h1><p class="subtitle">Executed synthetic experiments supporting a five-part technical response</p>
    <div class="notice"><b>Evidence boundary.</b> These are simulated farms and generated weather. The network learns a known water-balance scheduling rule. Results do not demonstrate field readiness, yield benefits, or superiority over expert irrigation.</div>
    <h2>What was actually run</h2><p>39 trained models, three initialization seeds, eight source farms, two unseen target farms, and separate training, validation and test seasons. Each controller is evaluated for all 60 days from April 1 through May 30. Checkpoints, input datasets, every test prediction and every closed-loop daily trajectory are included.</p>
    <h2>1. Training design</h2><p>The model sees past sequences and fresh morning observations. Labels come from an explicit teacher. See <code>docs/METHODS.md</code> for the feature dictionary, equations, event timing, split definitions and assumptions.</p>'''
    body+=im('01_architecture','Implemented architecture. Current observations reach the output independently of recurrent history.')
    body+='''<h2>2. How much data?</h2><p>Increasing the source training set from one to six synthetic seasons per farm reduced mean MAE from 1.056 to 0.548 mm/day. This does not establish a real-world minimum. Irrigation events are rare: only 55 of 960 ordinary source test days have positive teacher labels. F1 and closed-loop outcomes are essential alongside MAE.</p>'''
    body+=im('02_data_and_history','Error bars show sample SD across three optimizer seeds on the same fixed test dataset; they are not confidence intervals over farms.')
    body+='''<h2>3. Transfer is not automatically beneficial</h2><p>Head-only adaptation changed target-farm-8 MAE from 1.111 to 1.040 mm/day. On target farm 9 it worsened MAE from 0.504 to 1.272 mm/day. Local validation and an unchanged-model comparison are necessary. Sparse labels make learning from scratch difficult: the target-farm-9 scratch models predicted no positive events on the test teacher trajectories (F1 = 0).</p>'''
    body+=im('03_transfer','Both target farms are shown, including the negative-transfer result. Adaptation uses historical local seasons, not the final test period.')
    body+='''<h2>4. A storm changes inputs, not necessarily weights</h2><p>All ordinary-trained LSTM rollouts recommended zero irrigation on May 1 after the 80 mm overnight event was observed. No May retraining was performed. Across source farms, updating daily reduced May irrigation from 71.607 mm under the stale counterfactual schedule to 41.938 mm. This is an intentionally stale-plan comparison, not a claim of savings against a competent water-balance scheduler.</p><p>The simulator drains excess water immediately; it cannot establish safe behavior under persistent waterlogging. A storm arriving after the morning decision would require an intraday rain interlock.</p>'''
    body+=im('04_storm_response','A prespecified illustrative farm/season, not the case with the largest benefit. Aggregate outcomes are in the tables below.')
    body+='''<h2>5. Check control outcomes, not only prediction error</h2><p>The MLP has lower source one-step MAE (0.313 versus 0.548 mm/day), but on the shifted target farms its ordinary-weather closed-loop stress proxy averages 10.333 days versus 0.833 for the unchanged LSTM. These tests ask different questions. The no-irrigation baseline and scratch models show why small water totals alone are not evidence of a good scheduler.</p>'''
    body+=im('05_closed_loop','Matched target-farm cohort for every bar. Teacher and no-irrigation are deterministic and repeat identically across seeds.')
    body+='''<h2>6. Preparing for sudden changes</h2><p>We tested equal-size ordinary and storm-augmented training datasets. Augmentation did not improve held-out imitation MAE: storm-test MAE was 0.521 versus 0.370 mm/day. Ordinary observations plus measured storage were already sufficient for the simulated storm. This negative result argues for evaluating augmentation, not assuming it must help. The wet-state guard also showed no additional reduction in wet-state irrigation events in the learned rollouts, because neither guarded nor unguarded LSTM produced such events here.</p>'''
    body+=im('06_baselines_and_augmentation','No test-based hyperparameter retuning was used to remove unfavorable findings.')
    # Narrative interpretation below was reviewed against the archived default
    # experiment. For changed training settings, suppress that narrative rather
    # than presenting the archived interpretation as a new computed finding.
    if meta['seeds'] != [11,22,33] or meta['epochs_max'] != 65:
        body='<h1>LSTM irrigation: alternate configuration</h1><p>Synthetic results. Tables and figures below are regenerated; the archived narrative applies only to the default configuration. Inspect this run independently.</p>'
        for name in ['01_architecture','02_data_and_history','03_transfer','04_storm_response','05_closed_loop','06_baselines_and_augmentation']:
            body+=im(name,'Computed from this run. Variability is across optimizer seeds, not independent observed farms.')
    body+='<h2>Complete prediction summaries</h2>'+table(ps.reset_index())
    body+='<h2>Complete closed-loop summaries</h2><p>First averaged over farm-seasons within each seed, then summarized across seeds. Source and target cohorts remain separate. All water units are mm; stress units are days per 60-day season.</p>'+table(rs.reset_index())
    body+='''<h2>Limitations and next field step</h2><p>This is a small, single-bucket, early-season experiment with directly observed root-zone moisture and labels generated by a known rule. Training weather and test weather come from the same generator apart from the injected storm and two parameter-shifted farms. It does not test regional climate transfer, damaged drains, yield, prolonged flooding, or learning from imperfect human actions. Farm-level soil profiles, measured irrigation, historical forecasts, calibration and prospective monitored testing are needed before field use.</p><p>The proposed response is in <code>docs/PROFESSOR_RESPONSE.md</code>. Methods, assumptions and primary references are in <code>docs/METHODS.md</code>. Run instructions are in <code>README.md</code>.</p>'''
    body+='<h2>Run metadata</h2><pre>'+html.escape(json.dumps(meta,indent=2))+'</pre>'
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>LSTM Irrigation — Synthetic Experiment Report</title><style>
    body{font-family:system-ui,sans-serif;color:#203345;max-width:1080px;margin:42px auto;padding:0 24px;line-height:1.6}h1{font-size:38px;line-height:1.15;color:#156082}h2{margin-top:42px;color:#156082}.subtitle{font-size:18px;color:#637889}.notice{background:#fff1d9;border-left:5px solid #df8430;padding:18px}figure{margin:24px 0}img{max-width:100%;height:auto}figcaption{font-size:13px;color:#637889}table{border-collapse:collapse;font-size:11px;display:block;overflow:auto}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:right}th{background:#eef5f8}pre{white-space:pre-wrap;font-size:12px;background:#f3f6f8;padding:15px}@media print{body{max-width:none;margin:0;font-size:10pt}h2{break-after:avoid}figure{break-inside:avoid}.notice{break-inside:avoid}table{font-size:7pt}}
    </style></head><body>'''+body+'</body></html>'
    (root/'REPORT.html').write_text(page)
    print('Created 6 PNG/SVG figures, two summary tables, example_input.json and REPORT.html')

if __name__=='__main__':main()
