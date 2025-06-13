import os

import torch 
import numpy as np 


# ------------------------------------------------------------------------
# ------------------ MOP with structure constraints ----------------------
# ------------------------------------------------------------------------
from problems.problem_f_re import get_problem_f_re

class mop_sc(): 
    def __init__(self, 
                 pname: str, 
                 share_comp: list, 
                ) -> None:
        self.share_comp = share_comp 
        self.p = get_problem_f_re(pname) 
        # ideal and nadir points for RE problems
        if pname.lower() in ['re21', 're24', 're32', 're33','re36','re37']:
            self.ideal_point = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'RE/ideal_nadir_points/ideal_point_{pname.upper()}.dat'))
            self.nadir_point = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'RE/ideal_nadir_points/nadir_point_{pname.upper()}.dat'))
        else: 
            self.ideal_point = self.p.ideal_point
            self.nadir_point = self.p.nadir_point
        # bounds for variables and parameters
        mask = torch.ones(self.p.n_dim, dtype=bool)
        mask[self.share_comp] = False
        self.xl = self.p.lbound[mask] 
        self.xu = self.p.ubound[mask] 
        self.pl = self.p.lbound[~mask] 
        self.pu = self.p.ubound[~mask] 
        
    @property
    def n_dim(self): 
        return self.p.n_dim - self.n_params
    
    @property
    def n_obj(self): 
        return self.p.n_obj
    
    @property
    def n_params(self): 
        return len(self.share_comp)
    
    def normalize_p(self, param): 
        if self.pl.device.type != param.device.type: 
            device = param.device.type
            self.pu = self.pu.to(device)
            self.pl = self.pl.to(device)
        return (param - self.pl) / (self.pu - self.pl)
    
    def unnormalize_p(self, param): 
        if self.pl.device.type != param.device.type: 
            device = param.device.type
            self.pu = self.pu.to(device)
            self.pl = self.pl.to(device)
        return param * (self.pu - self.pl) + self.pl

    def set_share_comp(self, share_comp: list) -> None: 
        self.share_comp = share_comp
        
        mask = torch.ones(self.p.n_dim, dtype=bool)
        mask[self.share_comp] = False
        self.xl = self.p.lbound[mask] 
        self.xu = self.p.ubound[mask] 
        self.pl = self.p.lbound[~mask] 
        self.pu = self.p.ubound[~mask] 

    def evaluate(self, x, param): 
        device = x.device.type
        if x.device.type != self.xl.device.type:        
            self.xl = self.xl.to(device)
            self.xu = self.xu.to(device)
        
        param = self.normalize_p(torch.atleast_1d(param.to(device)))
        x = torch.atleast_2d(x)
        nr = x.shape[0]
        
        for i, index in enumerate(self.share_comp): 
            x = torch.cat((x[:,:index].reshape(nr,-1), torch.ones((nr,1), device=device)*param[i], x[:,index:].reshape(nr,-1)), dim=1)

        return self.p.evaluate(x)


# ------------------------------------------------------------------------
# ------------------ Dynamic Multi-Objective Problems --------------------
# ------------------------------------------------------------------------
from .problem_dyn import get_problem_dyn

class mop_dyn(): 
    def __init__(self,
                 pname: str, 
                 n_dim: int, 
                 tau: int = 1,
                 nt: int = 10, 
                 taut: int = 20, 
                 ):
        # self.p = get_problem(pname, taut=1, nt=change_severity, n_var=n_dim)
        self.p = get_problem_dyn(pname, n_dim=n_dim)
        self.n_dim = n_dim
        self.xl, self.xu = torch.tensor(self.p.xl), torch.tensor(self.p.xu)
        self.pl, self.pu = torch.tensor([0.]), torch.tensor([1.])
        self.ideal_point = self.p.ideal_point
        self.nadir_point = self.p.nadir_point
        
        self.tau = tau
        self.nt = nt
        self.taut = taut
        
        self._time = None
    
    @property
    def n_obj(self): 
        return self.p.n_obj
    
    @property
    def n_params(self): 
        return 1
    
    @property
    def time(self):
        if self._time is not None:
            return self._time
        else:
            return 1 / self.nt * (self.tau // self.taut)

    @time.setter
    def time(self, value):
        self._time = value
    
    def normalize_p(self, param): 
        if self.pl.device.type != param.device.type: 
            device = param.device.type
            self.pu = self.pu.to(device)
            self.pl = self.pl.to(device)
        return (param - self.pl) / (self.pu - self.pl)
    
    def unnormalize_p(self, param): 
        if self.pl.device.type != param.device.type: 
            device = param.device.type
            self.pu = self.pu.to(device)
            self.pl = self.pl.to(device)
        return param * (self.pu - self.pl) + self.pl

    def tic(self, elapsed: int = 1): 
        # increase the time counter by one
        self.tau += elapsed
        
    def evaluate(self, x, param = None): 
        if self.xu.device.type != x.device.type: 
            device = x.device.type
            self.xu = self.xu.to(device)
            self.xl = self.xl.to(device)
        x = x * (self.xu - self.xl) + self.xl
        return self.p._evaluate(x, time=self.time)
    
    def _calc_pareto_front(self, n_pareto_points=100, **kwargs):
        return self.p._calc_pareto_front(time=self.time, n_pareto_points=n_pareto_points)
    


        
        
if __name__ == '__main__':
    print()