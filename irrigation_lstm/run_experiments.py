#!/usr/bin/env python3
"""Rebuild data, checkpoints and raw experimental results from fixed seeds."""
import argparse,copy,json,platform,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score,mean_absolute_error,mean_squared_error
from src.simulator import farms,simulate,sequences,guard,DYNAMIC,STATIC
from src.model import fit,predict,policy,save

def score(model,scaler,data,fs):
    pred,p=predict(model,scaler,data);h,c,y,ids=data
    safe=np.array([guard(a,x[:len(DYNAMIC)],fs[int(i[0])]) for a,x,i in zip(pred,c,ids)])
    return dict(mae_raw_mm=mean_absolute_error(y,pred),mae_guarded_mm=mean_absolute_error(y,safe),
                rmse_guarded_mm=float(np.sqrt(mean_squared_error(y,safe))),
                f1=float(f1_score(y>0,safe>0,zero_division=0)),
                brier=float(np.mean((p-(y>0))**2)),n=len(y),positive_days=int((y>0).sum()))

def rollout_metrics(frame):
    g=frame[frame.day>=0]
    return dict(water_mm=float(g.irrigation_mm.sum()),stress_days=int(g.stress_day.sum()),
                et_shortfall_mm=float(g.et_shortfall_mm.sum()),drainage_mm=float(g.drainage_mm.sum()),
                wet_actions=int(g.wet_action.sum()),
                may1_irrigation_mm=float(g.loc[g.date=='05-01','irrigation_mm'].iloc[0]),
                may_water_mm=float(g.loc[g.day>=30,'irrigation_mm'].sum()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='.');ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33]);ap.add_argument('--epochs',type=int,default=65)
    args=ap.parse_args();root=Path(args.output);start=time.time()
    for d in ['data','results','checkpoints']: (root/d).mkdir(parents=True,exist_ok=True)
    fs=farms();raw=[]
    for f in fs:
        for year in range(9):raw.append(simulate(f,year))
    raw=pd.concat(raw,ignore_index=True)
    augmented=pd.concat([simulate(f,y,'augmented') for f in fs[:8] for y in range(7)],ignore_index=True)
    storm=pd.concat([simulate(f,y,'storm') for f in fs for y in [7,8]],ignore_index=True)
    raw.to_csv(root/'data/synthetic_normal.csv.gz',index=False)
    augmented.to_csv(root/'data/synthetic_augmented.csv.gz',index=False)
    storm.to_csv(root/'data/synthetic_storm_teacher.csv.gz',index=False)
    pd.DataFrame([vars(f) for f in fs]).to_csv(root/'data/farms.csv',index=False)
    def source(years,history=14):return sequences(raw[(raw.farm<8)&raw.year.isin(years)],history)
    metrics=[];logs=[];rollouts=[];daily=[];all_daily=[];all_predictions=[];models={}
    def train_one(name,seed,tr,va,history=14,**kw):
        print(f'Training {name} seed={seed} n={len(tr[2])}',flush=True)
        m,s,log=fit(tr,va,seed=seed,epochs=args.epochs,**kw)
        logs.extend(dict(experiment=name,seed=seed,**r) for r in log)
        save(root/f'checkpoints/{name}_seed{seed}.pt',m,s,history,dict(experiment=name,seed=seed,synthetic=True))
        return m,s
    def assess(name,seed,m,s,data,test):
        metrics.append(dict(experiment=name,seed=seed,test=test,**score(m,s,data,fs)))
        pred,prob=predict(m,s,data);_,c,y,ids=data
        safe=[guard(a,x[:len(DYNAMIC)],fs[int(i[0])]) for a,x,i in zip(pred,c,ids)]
        all_predictions.append(pd.DataFrame(dict(experiment=name,seed=seed,test=test,
            farm=ids[:,0],year=ids[:,1],day=ids[:,2],target_mm=y,
            raw_prediction_mm=pred,guarded_prediction_mm=safe,probability=prob)))
    for seed in args.seeds:
        train=source(range(6));val=source([6]);test=source([7,8])
        base,sc=train_one('lstm',seed,train,val);models[(seed,'lstm')]=(base,sc)
        assess('lstm',seed,base,sc,test,'source_future_years')
        for n in [1,3,6]:
            m,s=(base,sc) if n==6 else train_one(f'seasons_{n}',seed,source(range(n)),val)
            assess(f'seasons_{n}',seed,m,s,test,'source_future_years')
        for h in [7,14,30]:
            m,s=(base,sc) if h==14 else train_one(f'history_{h}',seed,source(range(6),h),source([6],h),history=h)
            assess(f'history_{h}',seed,m,s,source([7,8],h),'source_future_years')
        mlp,ms=train_one('mlp',seed,train,val,kind='mlp');models[(seed,'mlp')]=(mlp,ms)
        assess('mlp',seed,mlp,ms,test,'source_future_years')
        # Replace half of source training trajectories with augmented trajectories:
        # same sample count as ordinary training, not a data-volume advantage.
        aug_train=pd.concat([raw[(raw.farm<8)&(raw.year<6)&(raw.year%2==0)],
                             augmented[(augmented.year<6)&(augmented.year%2==1)]])
        aug_val=pd.concat([raw[(raw.farm<4)&(raw.year==6)],
                           augmented[(augmented.farm>=4)&(augmented.year==6)]])
        am,asc=train_one('augmented',seed,sequences(aug_train),sequences(aug_val));models[(seed,'augmented')]=(am,asc)
        assess('augmented',seed,am,asc,test,'source_future_years')
        for f in fs[8:]:
            target=lambda years:sequences(raw[(raw.farm==f.farm_id)&raw.year.isin(years)])
            tt=target([7,8]);assess('zero_shot',seed,base,sc,tt,f'target_{f.farm_id}')
            for name,kwargs in [('head_tune',dict(model=base,scaler=sc,freeze=True,lr=.001)),
                                ('full_tune',dict(model=base,scaler=sc,lr=.0005)),
                                ('scratch',dict())]:
                m,s=train_one(f'{name}_farm{f.farm_id}',seed,target([0,1]),target([2]),**kwargs)
                assess(name,seed,m,s,tt,f'target_{f.farm_id}')
                models[(seed,f'{name}_farm{f.farm_id}')]=(m,s)
        for name in ['lstm','augmented','mlp']:
            m,s=models[(seed,name)]
            assess(name,seed,m,s,sequences(storm[storm.farm<8]),'source_storm')
        # Closed-loop control: each method gets its own resulting soil state.
        for f in fs:
            for year in [7,8]:
                for variant in ['normal','storm']:
                    for name in ['teacher','no_irrigation','lstm','augmented','mlp','lstm_unguarded']:
                        if name in ['teacher','no_irrigation']:
                            pol=None if name=='teacher' else lambda h,c,s:0.
                        else:
                            m,s=models[(seed,'lstm' if name=='lstm_unguarded' else name)];pol=policy(m,s)
                        frame=simulate(f,year,variant,pol,guarded=name!='lstm_unguarded')
                        all_daily.append(frame.assign(method=name,seed=seed))
                        rollouts.append(dict(seed=seed,method=name,farm=f.farm_id,year=year,variant=variant,**rollout_metrics(frame)))
                        if seed==args.seeds[0] and f.farm_id==8 and year==7:
                            daily.append(frame.assign(method=name,seed=seed))
                    if f.farm_id>=8:
                        for name in ['head_tune','full_tune','scratch']:
                            m,s=models[(seed,f'{name}_farm{f.farm_id}')]
                            frame=simulate(f,year,variant,policy(m,s))
                            all_daily.append(frame.assign(method=name,seed=seed))
                            rollouts.append(dict(seed=seed,method=name,farm=f.farm_id,year=year,variant=variant,**rollout_metrics(frame)))
                # Replay an ordinary-weather schedule without reacting to the storm.
                m,s=models[(seed,'lstm')]
                ordinary=simulate(f,year,'normal',policy(m,s))
                fixed=simulate(f,year,'storm',replay=ordinary.loc[ordinary.day>=0,'irrigation_mm'].to_numpy(),guarded=False)
                all_daily.append(fixed.assign(method='stale_schedule',seed=seed))
                rollouts.append(dict(seed=seed,method='stale_schedule',farm=f.farm_id,year=year,variant='storm',**rollout_metrics(fixed)))
                if seed==args.seeds[0] and f.farm_id==8 and year==7:daily.append(fixed.assign(method='stale_schedule',seed=seed))
        pd.DataFrame(metrics).to_csv(root/'results/prediction_metrics.csv',index=False)
        pd.DataFrame(rollouts).to_csv(root/'results/rollout_metrics.csv',index=False)
        pd.DataFrame(logs).to_csv(root/'results/training_history.csv',index=False)
        print(f'Finished seed {seed}; elapsed {(time.time()-start)/60:.1f} minutes',flush=True)
    pd.concat(daily).to_csv(root/'results/example_daily.csv',index=False)
    pd.concat(all_daily).to_csv(root/'results/all_rollout_daily.csv.gz',index=False)
    pd.concat(all_predictions).to_csv(root/'results/all_test_predictions.csv.gz',index=False)
    # Keep all actual predictions as auditable evidence, not only summary metrics.
    m,s=models[(args.seeds[0],'lstm')];dt=source([7,8]);pred,p=predict(m,s,dt)
    pd.DataFrame(dict(farm=dt[3][:,0],year=dt[3][:,1],day=dt[3][:,2],target_mm=dt[2],raw_prediction_mm=pred,probability=p)).to_csv(root/f'results/source_predictions_seed{args.seeds[0]}.csv',index=False)
    metadata=dict(synthetic=True,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,
                  seeds=args.seeds,epochs_max=args.epochs,elapsed_seconds=time.time()-start,
                  train_source_farms=list(range(8)),train_years=list(range(6)),validation_years=[6],test_years=[7,8],
                  target_farms=[8,9],adaptation_years=[0,1],adaptation_validation_years=[2],
                  train_days_per_seed=2880,history_days=14,decision_days=60,
                  architecture=str(models[(args.seeds[0],'lstm')][0]),
                  trainable_parameters=sum(p.numel() for p in models[(args.seeds[0],'lstm')][0].parameters()))
    (root/'results/run_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
