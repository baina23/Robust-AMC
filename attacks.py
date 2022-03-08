import torch
import torch.nn as nn
import torch.nn.functional as F

# Compute the gradient of the loss w.r.t. the input data
def gradient_wrt_data(model, device, data, lbl):
    dat = data.clone().detach()
    dat.requires_grad = True
    out = model(dat)
    loss = F.cross_entropy(out,lbl)
    model.zero_grad()
    loss.backward()
    data_grad = dat.grad.data
    return data_grad.data.detach()

def FGSM_Linf_attack(model, device, dat, lbl, eps):
    x_nat    = dat.clone().detach()
    x_nat    = x_nat.to(device)
    gradient = gradient_wrt_data(model, device, x_nat, lbl).to(device)
    x_adv    = x_nat + eps * torch.sign(gradient).to(device)
    return x_adv

def FGSM_L2_attack(model, device, dat, lbl, eps) :
    x_nat    = dat.clone().detach()
    x_nat    = x_nat.to(device)
    gradient = gradient_wrt_data(model, device, x_nat, lbl).to(device)
    # Compute sample-wise L2 norm of gradient (L2 norm for each batch element)
    # HINT: Flatten gradient tensor first, then compute L2 norm
    gradient_flatten = torch.flatten(gradient, start_dim = 1)
    grad_distance = torch.sqrt(torch.sum(gradient_flatten * gradient_flatten, dim = 1))
    # Perturb the data using the gradient
    # HINT: Before normalizing the gradient by its L2 norm, use
    # torch.clamp(l2_of_grad, min=1e-12) to prevent division by 0
    grad_L2 = torch.clamp(grad_distance, min = 1e-12)
    # Add perturbation
    for i in range(x_nat.shape[0]) :
        gradient[i] = gradient[i] / grad_L2[i]
    x_adv = x_nat.clone().detach() + (eps * gradient)
    # Return the perturbed samples
    return x_adv
