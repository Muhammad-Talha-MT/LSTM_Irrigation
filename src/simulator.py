"""Transparent, simplified root-zone bucket and synthetic weather generator.

Information timing: rainfall is observed overnight BEFORE the morning action.
ET occurs after irrigation. The next overnight rainfall is forecast with error.
This is NOT AquaCrop, a calibrated soil profile, or a yield model.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

DYNAMIC = ['rain_mm', 'theta_obs', 'depletion_fraction_obs', 'eto_forecast_mm',
           'rain_forecast_mm', 'tmin_c', 'tmax_c', 'rh_pct', 'wind_ms',
           'solar_mj_m2', 'days_after_planting', 'root_depth_m',
           'previous_irrigation_mm', 'crop_coefficient']
STATIC = ['field_capacity', 'wilting_point', 'infiltration_mm_day',
          'efficiency', 'max_gross_mm_day', 'runoff_fraction']
MAX_HISTORY = 30
N_DAYS = MAX_HISTORY + 60

@dataclass
class Farm:
    farm_id: int
    field_capacity: float
    wilting_point: float
    infiltration_mm_day: float
    efficiency: float
    max_gross_mm_day: float
    runoff_fraction: float
    root_max: float
    climate: float
    def vector(self):
        return np.array([getattr(self, k) for k in STATIC], dtype=np.float32)

def farms():
    rng = np.random.default_rng(932)
    result = []
    for i in range(10):
        fc = rng.uniform(.25,.36); wp = rng.uniform(.09,.16)
        result.append(Farm(i, fc, wp, rng.uniform(25,65), rng.uniform(.72,.92),
                           rng.uniform(18,30), rng.uniform(.03,.18),
                           rng.uniform(.75,1.05), rng.uniform(.85,1.15)))
    # Deliberately shifted soils/equipment at two unseen target farms.
    result[8] = Farm(8,.23,.09,70,.70,20,.04,.80,1.15)
    result[9] = Farm(9,.38,.19,18,.88,22,.20,.90,.90)
    return result

def weather(farm, year, variant='normal'):
    rng = np.random.default_rng(10000+farm.farm_id*100+year)
    n=N_DAYS; day=np.arange(n)-MAX_HISTORY
    tmean=14+.14*day+rng.normal(0,2.8,n)
    rain=np.where(rng.random(n)<.20, rng.gamma(1.5,6,n),0)
    rain=np.minimum(rain,35)
    # Synthetic ETo, not a Penman-Monteith estimate.
    eto=np.clip((2.5+.035*day+rng.normal(0,.5,n))*farm.climate,1,7)
    fc_eto=np.clip(eto+rng.normal(0,.5,n),.5,8)
    next_rain=np.r_[rain[1:],0]
    fc_rain=np.maximum(0,next_rain*rng.uniform(.4,1.3,n)+rng.normal(0,2,n))
    fc_rain[rng.random(n)<.15]=0
    if variant=='storm':
        rain[MAX_HISTORY+30]=80. # May 1; previous forecast intentionally unchanged
    elif variant=='augmented':
        # Two randomly timed, unforecast heavy events per training trajectory.
        for d in rng.choice(np.arange(MAX_HISTORY,N_DAYS),2,replace=False):
            rain[d]=rng.uniform(45,95)
    elif variant!='normal':
        raise ValueError(variant)
    return dict(rain=rain,eto=eto,fc_eto=fc_eto,fc_rain=fc_rain,
                tmin=tmean-5,tmax=tmean+6,rh=np.clip(65+rng.normal(0,12,n),25,95),
                wind=np.clip(rng.normal(2,.6,n),.2,5),
                solar=np.clip(15+.06*day+rng.normal(0,3,n),5,28),
                sensor_noise=rng.normal(0,.003,n))

def root_and_kc(day,farm):
    age=max(day,0)
    return min(farm.root_max,.20+.011*age), min(1.10,.35+.013*age)

def dawn(storage,old_capacity,day,rain,farm):
    root,kc=root_and_kc(day,farm)
    capacity=1000*(farm.field_capacity-farm.wilting_point)*root
    # Newly explored subsoil has 65% of plant-available capacity.
    growth_water=.65*max(0,capacity-old_capacity)
    infiltration=min(rain*(1-farm.runoff_fraction),farm.infiltration_mm_day)
    runoff=rain-infiltration
    drainage=max(0,storage+growth_water+infiltration-capacity)
    morning=storage+growth_water+infiltration-drainage
    return morning,capacity,root,kc,growth_water,runoff,drainage

def guard(amount,context,farm):
    """Use observed state, never hidden true simulator storage."""
    frac=float(context[2]); root=float(context[11])
    taw=1000*(farm.field_capacity-farm.wilting_point)*root
    if frac < .25:
        return 0.
    return float(np.clip(amount,0,min(farm.max_gross_mm_day,frac*taw/farm.efficiency)))

def teacher(context,farm):
    """Heuristic teacher available to a real controller with these observations."""
    taw=1000*(farm.field_capacity-farm.wilting_point)*context[11]
    forecast_d=context[2]*taw+context[3]*context[13]-.5*context[4]
    amount=(forecast_d-.15*taw)/farm.efficiency if forecast_d>.50*taw else 0.
    return guard(amount,context,farm)

def simulate(farm,year,variant='normal',policy=None,history=14,guarded=True,
             replay=None):
    """Return all 90 daily records. Policy receives history + current + static.

    Warmup uses the same teacher for every controller. Replay is a fixed
    April-May schedule, used to demonstrate why stale schedules are unsuitable.
    """
    if not 1<=history<=MAX_HISTORY: raise ValueError('history must be 1..30')
    w=weather(farm,year,variant)
    root,_=root_and_kc(-MAX_HISTORY,farm)
    capacity=1000*(farm.field_capacity-farm.wilting_point)*root
    storage=.65*capacity; previous=0.; contexts=[]; records=[]
    for t in range(N_DAYS):
        day=t-MAX_HISTORY; old_storage=storage
        morning,capacity,root,kc,growth,runoff,drainage=dawn(storage,capacity,day,w['rain'][t],farm)
        theta=farm.wilting_point+morning/(1000*root)
        theta_obs=np.clip(theta+w['sensor_noise'][t],farm.wilting_point,farm.field_capacity)
        frac=(farm.field_capacity-theta_obs)/(farm.field_capacity-farm.wilting_point)
        x=np.array([w['rain'][t],theta_obs,frac,w['fc_eto'][t],w['fc_rain'][t],
                    w['tmin'][t],w['tmax'][t],w['rh'][t],w['wind'][t],w['solar'][t],
                    max(day,0),root,previous,kc],dtype=np.float32)
        label=teacher(x,farm)
        if day<0 or (policy is None and replay is None):
            raw=label
        elif replay is not None:
            raw=float(replay[day])
        else:
            raw=float(policy(np.array(contexts[-history:]),x,farm.vector()))
        amount=guard(raw,x,farm) if guarded else max(0,min(raw,farm.max_gross_mm_day))
        # Applying irrigation above true available storage creates drainage.
        net=amount*farm.efficiency
        irrigation_drain=max(0,morning+net-capacity)
        after=min(capacity,morning+net)
        d_before_et=capacity-after
        ks=min(1.,after/(.45*capacity))  # RAW=0.55 TAW (fixed illustrative maize p)
        potential_et=kc*w['eto'][t]
        actual_et=min(after,ks*potential_et)
        storage=after-actual_et
        residual=storage-old_storage-growth-w['rain'][t]-net+runoff+drainage+irrigation_drain+actual_et
        row=dict(farm=farm.farm_id,year=year,variant=variant,day=day,
                 date=(pd.Timestamp('2001-04-01')+pd.Timedelta(days=day)).strftime('%m-%d'),
                 true_eto_mm=w['eto'][t],true_theta=theta,true_depletion=capacity-morning,
                 capacity_mm=capacity,storage_end_mm=storage,root_growth_water_mm=growth,
                 target_mm=label,raw_mm=raw,irrigation_mm=amount,runoff_mm=runoff,
                 drainage_mm=drainage+irrigation_drain,actual_et_mm=actual_et,
                 potential_et_mm=potential_et,stress_day=int(capacity-storage>.55*capacity),
                 et_shortfall_mm=potential_et-actual_et,balance_residual_mm=residual,
                 wet_action=int(frac<.25 and amount>1e-6),
                 **dict(zip(DYNAMIC,x)),**dict(zip(STATIC,farm.vector())))
        records.append(row);contexts.append(x);previous=amount
    return pd.DataFrame(records)

def sequences(frame,history=14):
    hs=[];cs=[];ys=[];ids=[]
    for _,g in frame.groupby(['farm','year','variant'],sort=True):
        g=g.sort_values('day'); dyn=g[DYNAMIC].to_numpy(np.float32)
        stat=g[STATIC].to_numpy(np.float32)
        for j in range(MAX_HISTORY,len(g)):
            hs.append(dyn[j-history:j]);cs.append(np.r_[dyn[j],stat[j]])
            ys.append(g.iloc[j].target_mm);ids.append((int(g.iloc[j].farm),int(g.iloc[j].year),int(g.iloc[j].day)))
    return np.stack(hs),np.stack(cs),np.asarray(ys,np.float32),np.asarray(ids)
