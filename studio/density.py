"""Resource-bounded gsplat density control; keep optimizer moments during edits."""
import torch
from gsplat.strategy import DefaultStrategy


class BoundedStrategy(DefaultStrategy):
    def __init__(self,max_gaussians,growth_fraction=.1,**kwargs):
        super().__init__(**kwargs)
        self.max_gaussians=max_gaussians
        self.growth_fraction=growth_fraction

    @torch.no_grad()
    def _grow_gs(self,params,optimizers,state,step):
        # Pinned to gsplat 1.5.3: each split/duplicate adds exactly one Gaussian.
        budget=min(max(0,self.max_gaussians-len(params['means'])),max(64,int(len(params['means'])*self.growth_fraction)))
        grads=state['grad2d']/state['count'].clamp_min(1)
        candidates=torch.where(grads>self.grow_grad2d)[0]
        if len(candidates)>budget:
            keep=torch.zeros_like(grads,dtype=torch.bool)
            if budget:
                selected=candidates[grads[candidates].topk(budget).indices]
                keep[selected]=True
            state['grad2d'].masked_fill_(~keep,0)
        return super()._grow_gs(params,optimizers,state,step)
