import torch
import torch.nn as nn
import copy
from utils.metrics import compute_js_divergence, compute_entropy

def configure_ln_parameters(model):
    """
    Parameter-Efficient Fine-Tuning (PEFT) Strategy:
    Exclusively isolates and optimizes the affine parameters of Layer Normalization.
    """
    model.requires_grad_(False)
    opt_params = []
    for name, param in model.named_parameters():
        if "visual" in name and "norm" in name.lower():
            param.requires_grad = True
            opt_params.append(param)
    return opt_params

class DualMinMaxAdaptation(nn.Module):
    """
    ECFR: Entropy-Consistency Flow Rectification
    Integrates a Teacher-Student EMA architecture, Stochastic Parameter Restoration, 
    and a Bidirectional Cross-Control Mechanism guided by Memory Bank coefficients.
    """
    def __init__(self, student_model, optimizer, ema_alpha=0.999, p_restore=0.01):
        super().__init__()
        self.student = student_model
        # Initialize an independent Teacher model via deepcopy and freeze its gradients
        self.teacher = copy.deepcopy(student_model)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        
        self.optimizer = optimizer
        self.ema_alpha = ema_alpha
        self.p_restore = p_restore

        # Archive initial optimized parameters for Stochastic Restoration
        self.initial_params = {
            name: param.clone().detach()
            for name, param in self.student.named_parameters()
            if param.requires_grad
        }

    def forward_and_adapt(self, x_clean, x_weak, x_strong, memory_bank):
        # ---------------------------------------------------------------------
        # 1. Stochastic Parameter Restoration (Mitigates Catastrophic Forgetting)
        # ---------------------------------------------------------------------
        with torch.no_grad():
            for name, param in self.student.named_parameters():
                if param.requires_grad:
                    init_param = self.initial_params[name]
                    # Restore a fraction (p_restore) of the parameters to their pre-trained state
                    mask = (torch.rand_like(param) < self.p_restore).float()
                    param.data.copy_(mask * init_param + (1 - mask) * param.data)

        self.optimizer.zero_grad()
        
        # ---------------------------------------------------------------------
        # 2. Forward Propagation (Teacher-Student Paradigm)
        # ---------------------------------------------------------------------
        # Teacher generates stable pseudo-targets using weakly augmented views
        with torch.no_grad():
            logits_weak = self.teacher(x_weak)
            logits_clean = self.teacher(x_clean) # Evaluated strictly for final metrics
            
        # Student learns from strongly augmented views
        logits_strong = self.student(x_strong)
        
        probs_weak = torch.softmax(logits_weak, dim=1)
        probs_strong = torch.softmax(logits_strong, dim=1)
        
        # ---------------------------------------------------------------------
        # 3. Memory Bank Dynamics & Coefficient Retrieval
        # ---------------------------------------------------------------------
        # Evaluate metrics strictly on the reliable Teacher outputs
        ent_val = compute_entropy(probs_weak)
        cons_val = compute_js_divergence(probs_weak, probs_strong.detach())
        
        # Retrieve optimization directions (-1.0 for Max, 1.0 for Min)
        coef_cons, coef_ent = memory_bank.get_coefficients(ent_val, cons_val)
        memory_bank.update(ent_val, cons_val)
        
        # ---------------------------------------------------------------------
        # 4. Bidirectional Cross-Control Optimization
        # ---------------------------------------------------------------------
        # Optimize Student's output entropy and its consistency with the Teacher
        loss_ent = torch.mean(coef_ent * compute_entropy(probs_strong))
        loss_cons = torch.mean(coef_cons * compute_js_divergence(probs_weak, probs_strong))
        
        total_loss = loss_ent + loss_cons
        
        total_loss.backward()
        self.optimizer.step()
        
        # ---------------------------------------------------------------------
        # 5. Exponential Moving Average (EMA) Update
        # ---------------------------------------------------------------------
        with torch.no_grad():
            for name, param_s in self.student.named_parameters():
                if param_s.requires_grad:
                    param_t = dict(self.teacher.named_parameters())[name]
                    param_t.data.mul_(self.ema_alpha).add_(param_s.data, alpha=1 - self.ema_alpha)
            
        return logits_clean, ent_val.detach(), cons_val.detach()
