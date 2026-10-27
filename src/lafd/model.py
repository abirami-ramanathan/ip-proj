import math
import torch
from torch import nn

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim=dim
    def forward(self,t):
        half=self.dim//2
        freq=torch.exp(torch.arange(half,device=t.device)*(-math.log(10000.0)/max(half-1,1)))
        phase=t.reshape(-1,1)*freq.reshape(1,-1)
        emb=torch.cat((phase.sin(),phase.cos()),dim=1)
        return torch.nn.functional.pad(emb,(0,self.dim-emb.shape[1]))

class Block(nn.Module):
    def __init__(self,cin,cout,time_dim):
        super().__init__()
        self.conv1=nn.Conv2d(cin,cout,3,padding=1); self.norm1=nn.GroupNorm(8,cout)
        self.conv2=nn.Conv2d(cout,cout,3,padding=1); self.norm2=nn.GroupNorm(8,cout)
        self.time=nn.Linear(time_dim,cout); self.skip=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x,t):
        h=torch.nn.functional.silu(self.norm1(self.conv1(x)))
        h=h+self.time(t)[:,:,None,None]
        h=torch.nn.functional.silu(self.norm2(self.conv2(h)))
        return h+self.skip(x)

class ScoreUNet(nn.Module):
    """Four-resolution time-conditioned U-Net predicting clean x0 with an identity skip."""
    def __init__(self,base=32,time_dim=128,in_channels=3):
        super().__init__()
        self.time=nn.Sequential(TimeEmbedding(time_dim),nn.Linear(time_dim,time_dim),nn.SiLU(),nn.Linear(time_dim,time_dim))
        self.in_channels=in_channels
        self.e1=Block(2*in_channels+2,base,time_dim); self.d1=nn.Conv2d(base,base*2,4,2,1); self.e2=Block(base*2,base*2,time_dim)
        self.d2=nn.Conv2d(base*2,base*4,4,2,1); self.e3=Block(base*4,base*4,time_dim)
        self.d3=nn.Conv2d(base*4,base*8,4,2,1); self.mid=Block(base*8,base*8,time_dim)
        self.u3=nn.ConvTranspose2d(base*8,base*4,4,2,1); self.r3=Block(base*8,base*4,time_dim)
        self.u2=nn.ConvTranspose2d(base*4,base*2,4,2,1); self.r2=Block(base*4,base*2,time_dim)
        self.u1=nn.ConvTranspose2d(base*2,base,4,2,1); self.r1=Block(base*2,base,time_dim)
        self.out=nn.Conv2d(base,in_channels,3,padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self,x,measurement,descriptors,t):
        te=self.time(t); inp=torch.cat((x,measurement,descriptors),1)
        a=self.e1(inp,te); b=self.e2(self.d1(a),te); c=self.e3(self.d2(b),te); m=self.mid(self.d3(c),te)
        h=self.r3(torch.cat((self.u3(m),c),1),te); h=self.r2(torch.cat((self.u2(h),b),1),te)
        h=self.r1(torch.cat((self.u1(h),a),1),te)
        return measurement+self.out(h)
