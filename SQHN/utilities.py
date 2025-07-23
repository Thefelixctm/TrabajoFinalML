import torch
from torch import nn


def to_one_hot(y_onehot, y, dev='cpu'):
    y = y.view(y_onehot.size(0), -1).to(dev)
    y_onehot.zero_().to(dev)
    y_onehot.scatter_(1, y, 1).to(dev)

def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total

def compute_num_correct(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct

def sigmoid_d(x):
    return torch.sigmoid(x) * (1 - torch.sigmoid(x))

def tanh_d(x):
    a = torch.tanh(x)
    return 1.0 - a ** 2.0

def relu_d(x):
    return x > 0

def piecewise(x, min=0, max=1):
    return x * (x > min).float() * (x < max).float() + (x < min).float() * min + (x > max).float() * max

def boxcar(x, min=0, max=0, beta=1):
    return beta * ((x > min).float() * (x < max).float())

def softmax_d_error(input, error):
    sm = nn.Sequential(nn.Softmax(dim=1))
    return torch.autograd.functional.vjp(sm, input, error)

def poisson(x, beta=1):
    return 1 - torch.exp(-x * beta)

def test(model, iter, test_loader, dev):
    """
    Calcula el error de reconstrucción (MSE) en el conjunto de test.
    """
    # Esta función necesita manejar tanto los modelos SQHN (con infer_step)
    # como los autoencoders (que son nn.Module estándar).
    is_sqhn_model = hasattr(model, 'infer_step')
    
    with torch.no_grad():
        test_mse = 0
        for batch_idx, (images, y) in enumerate(test_loader):
            images = images.view(images.size(0), -1).to(dev)
            if is_sqhn_model:
                reconstructed = model.infer_step(images)
            else: # Es un autoencoder u otro nn.Module
                reconstructed = model(images)
            
            test_mse += torch.mean((reconstructed - images)**2)
    
    return test_mse.item() / len(test_loader)


def corrupt_test(noise_lvl, msk_lvl, model, mems):
    """
    Prueba la capacidad de reconstrucción del modelo bajo diferentes tipos de corrupción.
    """
    # Determinar si es un modelo SQHN o un callable (como nuestro wrapper de autoencoder)
    if hasattr(model, 'infer_step'):
        reconstruction_fn = model.infer_step
    else:
        reconstruction_fn = model

    with torch.no_grad():
        # 1. Test sin corrupción
        recalled = reconstruction_fn(mems)
        mse = torch.mean((recalled - mems)**2).item()
        recalled_bin = (recalled > .5).float()
        pct = torch.mean((recalled_bin == (mems > .5).float()).float()).item()

        # 2. Test con ruido Gaussiano
        noise = torch.randn(mems.size()).to(mems.device) * noise_lvl
        recalled_n = reconstruction_fn(torch.clamp(mems + noise, 0, 1))
        msen = torch.mean((recalled_n - mems)**2).item()
        recalled_bin_n = (recalled_n > .5).float()
        pctn = torch.mean((recalled_bin_n == (mems > .5).float()).float()).item()

        # 3. Test con máscara (oclusión)
        mask = (torch.rand(mems.size()) > msk_lvl).float().to(mems.device)
        recalled_m = reconstruction_fn(mems * mask)
        mse_msk = torch.mean((recalled_m - mems)**2).item()
        recalled_bin_m = (recalled_m > .5).float()
        pct_msk = torch.mean((recalled_bin_m == (mems > .5).float()).float()).item()

    return mse, pct, msen, pctn, mse_msk, pct_msk