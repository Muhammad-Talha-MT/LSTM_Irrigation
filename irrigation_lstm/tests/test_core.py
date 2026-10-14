import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from src.simulator import farms,simulate,sequences,guard,weather,DYNAMIC,MAX_HISTORY
from src.model import Scheduler,Scaler,save,load,predict,fit

class Tests(unittest.TestCase):
    def test_balance_and_bounds(self):
        for f in farms():
            for scenario in ['normal','storm','augmented']:
                d=simulate(f,7,scenario)
                self.assertLess(d.balance_residual_mm.abs().max(),1e-10)
                self.assertTrue((d.storage_end_mm>=0).all())
                self.assertTrue((d.storage_end_mm<=d.capacity_mm+1e-8).all())
                self.assertTrue((d.irrigation_mm<=f.max_gross_mm_day+1e-6).all())
    def test_calendar_and_sequence_causality(self):
        d=simulate(farms()[0],0);h,c,y,ids=sequences(d,14)
        self.assertEqual(h.shape,(60,14,14));self.assertEqual(c.shape,(60,20))
        self.assertEqual(d[d.day>=0].iloc[-1].date,'05-30')
        np.testing.assert_allclose(h[0,-1],d[d.day==-1][DYNAMIC].iloc[0].to_numpy(np.float32))
        self.assertEqual(int(ids[0,2]),0)
    def test_storm_is_unforecast_and_observed(self):
        f=farms()[8];n=weather(f,7);s=weather(f,7,'storm')
        self.assertEqual(s['rain'][60],80.)
        np.testing.assert_array_equal(n['fc_rain'],s['fc_rain'])
        d=simulate(f,7,'storm');self.assertEqual(d.loc[d.date=='05-01','irrigation_mm'].iloc[0],0.)
    def test_wet_guard(self):
        f=farms()[0];x=np.zeros(14);x[11]=.3
        self.assertEqual(guard(30,x,f),0)
    def test_checkpoint_roundtrip_and_training(self):
        tr=sequences(simulate(farms()[0],0));va=sequences(simulate(farms()[0],1))
        m,s,logs=fit(tr,va,epochs=2)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'model.pt';save(p,m,s,14);m2,s2,h,meta=load(p)
            np.testing.assert_allclose(predict(m,s,va)[1],predict(m2,s2,va)[1])
        self.assertTrue(np.isfinite(logs[-1]['train_loss']))
    def test_scaler_uses_only_supplied_training(self):
        tr=sequences(simulate(farms()[0],0));s=Scaler().fit(*tr[:2])
        np.testing.assert_allclose(s.cm,tr[1].mean(0))
    def test_replay_really_overrides(self):
        f=farms()[0];d=simulate(f,0,replay=np.full(60,1.),guarded=False)
        np.testing.assert_allclose(d[d.day>=0].irrigation_mm,1.)

if __name__=='__main__':unittest.main()
