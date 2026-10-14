#!/usr/bin/env python3
"""Predict from explicit, ordered morning features. Demonstration use only."""
import argparse,json
from pathlib import Path
import numpy as np
from src.model import load,predict
from src.simulator import Farm,DYNAMIC,STATIC,guard

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--input',required=True)
    args=ap.parse_args();m,s,h,meta=load(args.checkpoint);d=json.loads(Path(args.input).read_text())
    history=np.array([[r[k] for k in DYNAMIC] for r in d['history']],np.float32)
    current=np.array([d['current'][k] for k in DYNAMIC],np.float32)
    static=np.array([d['farm'][k] for k in STATIC],np.float32)
    if history.shape!=(h,len(DYNAMIC)):raise ValueError(f'Require exactly {h} prior daily rows')
    if not all(np.isfinite(x).all() for x in [history,current,static]):raise ValueError('Missing or nonfinite features')
    fc,wp,infiltration,efficiency,capacity,runoff=static
    if not (0<=wp<fc<=1 and 0<efficiency<=1 and capacity>0 and infiltration>0 and 0<=runoff<1):
        raise ValueError('Invalid soil/equipment properties')
    if not (0<=current[2]<=1 and current[11]>0):raise ValueError('Invalid depletion/root depth')
    f=Farm(-1,*map(float,static),root_max=float(current[11]),climate=1.)
    raw,p=predict(m,s,(history[None],np.r_[current,static][None],None,None))
    amount=guard(float(raw[0]),current,f)
    output=dict(synthetic_checkpoint=meta.get('synthetic',True),probability_irrigation=float(p[0]),
                raw_gross_depth_mm=float(raw[0]),guarded_gross_depth_mm=amount,
                irrigation_today=amount>0,
                warning='Synthetic demonstration checkpoint; requires local field validation before operational use.')
    rate=d['farm'].get('gross_application_rate_mm_hour')
    if rate is not None:
        if rate<=0:raise ValueError('Application rate must be positive')
        output['runtime_hours']=amount/rate
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
