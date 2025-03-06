import torch
import torch.nn as nn

def EYE(r, x):
    '''
    expert yielded estimation
    r: risk factors indicator (d,)
    x: attribution (d,)
    
    '''
    assert r.shape[-1] == x.shape[-1] # can broadcast
    l1 = (x * (1-r)).abs().sum(-1)
    l2sq = ((r * x)**2).sum(-1)
    return  l1 + torch.sqrt(l1**2 + l2sq)


def wL1(r, x):
    assert r.shape[-1] == x.shape[-1] # can broadcast    
    return ((x * (1-r))).sum(-1)


def wL2(r, x):
    '''
    addtional penalty to (1-r)
    r: risk factors indicator (d,)
    x: attribution (d,)
    '''
    assert r.shape[-1] == x.shape[-1] # can broadcast    
    return ((x * (1-r))**2).sum(-1)


def causal(r, x):
    '''
    addtional penalty to (1-r)
    r: causal effect/probability of affecting y
    x: attribution (d,)
    '''
    assert r.shape[-1] == x.shape[-1] # can broadcast   
    return ((x**2)*(1/r)).sum(-1)


class loss_reg(nn.Module):
    def __init__(self, model, basic_loss, reg_name, reg_strength, r):
        super().__init__()
        self.model = model
        self.basic_loss = basic_loss
        self.reg_name = reg_name ### can be chosen from l1, l2, eye, causal
        self.reg_strength = reg_strength
        self.r = r

    def forward(self, y, y_pred):

        empirical_risk = self.basic_loss(y, y_pred)

        if self.reg_name != "causal":
            c_idx = (self.r == 1)
            us_idx = (self.r == 0)

        for param in self.model.parameters():
            if self.reg_name == "without_reg":
                reg_risk = torch.sum(torch.tensor([0]))
            if self.reg_name == "l1":
                reg_risk = torch.sum(param[:, us_idx].abs())
            if self.reg_name == "l2":
                reg_risk = torch.sum(param[:, us_idx]**2)
            if self.reg_name == "eye":
                c_risk = torch.sum(param[:, c_idx]**2)
                us_risk = torch.sum(param[:, us_idx].abs())
                reg_risk = us_risk + torch.sqrt(us_risk**2 + c_risk)
            if self.reg_name == "causal":
                reg_risk = torch.sum(param**2 * (1/self.r))
            break

        risk = empirical_risk + self.reg_strength * reg_risk

        return risk





