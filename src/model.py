import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim=dim
    def forward(self,t):
        half=self.dim//2
        emb=math.log(10000)/(half-1)
        emb=torch.exp(torch.arange(half,device=t.device)*-emb)
        emb=t[:,None]*emb[None,:]
        return torch.cat((emb.sin(),emb.cos()),dim=1)

class Block(nn.Module):
    def __init__(self, cin, cout, temb):
        super().__init__()
        self.c1=nn.Conv2d(cin,cout,3,padding=1)
        self.c2=nn.Conv2d(cout,cout,3,padding=1)
        self.t=nn.Linear(temb,cout)
        self.n1=nn.GroupNorm(8,cout); self.n2=nn.GroupNorm(8,cout)
    def forward(self,x,t):
        h=F.silu(self.n1(self.c1(x)))
        h=h+self.t(t)[:,:,None,None]
        h=F.silu(self.n2(self.c2(h)))
        return h

class SmallUNet(nn.Module):
    def __init__(self,base=32,temb=128):
        super().__init__()
        self.time=nn.Sequential(SinusoidalTimeEmbedding(temb),nn.Linear(temb,temb),nn.SiLU())
        self.b1=Block(1,base,temb)
        self.d1=nn.Conv2d(base,base*2,4,2,1)
        self.b2=Block(base*2,base*2,temb)
        self.d2=nn.Conv2d(base*2,base*4,4,2,1)
        self.b3=Block(base*4,base*4,temb)
        self.u1=nn.ConvTranspose2d(base*4,base*2,4,2,1)
        self.b4=Block(base*4,base*2,temb)
        self.u2=nn.ConvTranspose2d(base*2,base,4,2,1)
        self.b5=Block(base*2,base,temb)
        self.out=nn.Conv2d(base,1,3,padding=1)
    def forward(self,x,t):
        t=self.time(t)
        a=self.b1(x,t)
        b=self.b2(self.d1(a),t)
        c=self.b3(self.d2(b),t)
        d=self.b4(torch.cat([self.u1(c),b],1),t)
        e=self.b5(torch.cat([self.u2(d),a],1),t)
        return self.out(e)
