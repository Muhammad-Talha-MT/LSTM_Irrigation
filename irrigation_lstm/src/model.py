"""Compact PyTorch scheduler and source-only standardization."""
import copy
import numpy as np
import torch
from torch import nn
from .simulator import DYNAMIC,STATIC

torch.set_num_threads(2)

class Scheduler(nn.Module):
    def __init__(self,kind='lstm',hidden=24):
        super().__init__(); self.kind=kind;self.hidden=hidden
        self.lstm=nn.LSTM(len(DYNAMIC),hidden,batch_first=True) if kind=='lstm' else None
        n=len(DYNAMIC)+len(STATIC)+(hidden if self.lstm is not None else 0)
        self.head=nn.Sequential(nn.Linear(n,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self,h,c):
        if self.lstm is not None:
            z,_=self.lstm(h); c=torch.cat([z[:,-1],c],dim=1)
        z=self.head(c)
        return z[:,0],30*torch.sigmoid(z[:,1])

class Scaler:
    def fit(self,h,c):
        self.hm=h.reshape(-1,h.shape[-1]).mean(0);self.hs=h.reshape(-1,h.shape[-1]).std(0).clip(.01)
        self.cm=c.mean(0);self.cs=c.std(0).clip(.01)
        return self
    def transform(self,h,c):
        return ((h-self.hm)/self.hs).astype('float32'),((c-self.cm)/self.cs).astype('float32')
    def state(self): return {k:getattr(self,k).tolist() for k in ['hm','hs','cm','cs']}
    @classmethod
    def restore(cls,state):
        obj=cls()
        for k,v in state.items():setattr(obj,k,np.array(v,np.float32))
        return obj

def objective(logit,amount,y):
    event=(y>0).float()
    bce=nn.functional.binary_cross_entropy_with_logits(logit,event)
    positive=y>0
    reg=((amount[positive]-y[positive])/30).square().mean() if positive.any() else amount.sum()*0
    return bce+reg

def fit(train,val,seed=0,kind='lstm',epochs=65,model=None,scaler=None,freeze=False,lr=.003):
    torch.manual_seed(seed);np.random.seed(seed)
    if model is None:model=Scheduler(kind)
    else:model=copy.deepcopy(model)
    for p in model.parameters():p.requires_grad_(True)
    if freeze and model.lstm is not None:
        for p in model.lstm.parameters():p.requires_grad_(False)
    h,c,y,_=train; vh,vc,vy,_=val
    if scaler is None:scaler=Scaler().fit(h,c)
    h,c=scaler.transform(h,c);vh,vc=scaler.transform(vh,vc)
    h,c,y=map(torch.from_numpy,(h,c,y));vh,vc,vy=map(torch.from_numpy,(vh,vc,vy))
    opt=torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=lr)
    best=float('inf');state=None;stale=0;logs=[]
    gen=torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train();indices=torch.randperm(len(y),generator=gen);losses=[]
        for batch in indices.split(128):
            opt.zero_grad();logit,amount=model(h[batch],c[batch]);loss=objective(logit,amount,y[batch])
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():vl=float(objective(*model(vh,vc),vy))
        logs.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_loss=vl))
        if vl<best-1e-5:best=vl;state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=12:break
    model.load_state_dict(state);model.eval()
    return model,scaler,logs

def predict(model,scaler,data):
    h,c,_,_=data; h,c=scaler.transform(h,c)
    with torch.no_grad():
        logit,amount=model(torch.from_numpy(h),torch.from_numpy(c))
        prob=torch.sigmoid(logit).numpy();amount=amount.numpy()
    return np.where(prob>=.5,amount,0),prob

def policy(model,scaler):
    def call(history,current,static):
        data=(history[None].astype(np.float32),np.r_[current,static][None].astype(np.float32),None,None)
        return predict(model,scaler,data)[0][0]
    return call

def save(path,model,scaler,history,metadata=None):
    torch.save(dict(state_dict=model.state_dict(),kind=model.kind,hidden=model.hidden,
                    scaler=scaler.state(),history=history,metadata=metadata or {}),path)

def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    model=Scheduler(ck['kind'],ck['hidden']);model.load_state_dict(ck['state_dict']);model.eval()
    return model,Scaler.restore(ck['scaler']),ck['history'],ck['metadata']
