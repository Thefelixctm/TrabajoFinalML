import torch
from torch import nn
import math
import torch.nn.functional as F

softmax = nn.Softmax(dim=1)
NLL = nn.NLLLoss(reduction='sum')
mse = torch.nn.MSELoss(reduction='sum')
bce = torch.nn.BCELoss(reduction='sum')
cos = torch.nn.CosineSimilarity(dim=0)
relu = nn.ReLU()


class MemUnit(nn.Module):
    def __init__(self, layer_szs=[3072, 100], n_iter=1, lr=1, actFunc=0, simFunc=0, beta=10000, alpha=1, wt_up=0,
                 det_type=0, gamma=.95):
        super().__init__()

        # --- LA MODIFICACIÓN CLAVE ---
        # 1. Detectar automáticamente el dispositivo disponible (GPU o CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Modelo MemUnit inicializado en el dispositivo: {self.device}") # Mensaje de depuración útil
        # -----------------------------

        self.layer_szs = layer_szs
        self.num_layers = len(layer_szs)
        self.n_iter = n_iter
        self.l_rate = lr
        self.wtUp = wt_up
        self.simFunc = simFunc
        self.actFunc = actFunc
        self.detect_type = det_type
        
        # El método create_wts ya mueve los pesos al dispositivo correcto
        self.wts = self.create_wts()
        
        self.beta = beta
        self.alpha = alpha
        self.age = 0
        
        # 2. Mover todos los tensores al dispositivo detectado
        self.counts = torch.zeros(self.layer_szs[1]).to(self.device)
        self.prior = torch.ones(self.layer_szs[1]).to(self.device)
        self.ages = torch.zeros(self.layer_szs[1]).to(self.device)
        
        self.avgmxlk = 0
        self.gamma = gamma
        self.err_avg = 0

    def create_wts(self):
        with torch.no_grad():
            # 3. Mover la capa de pesos al dispositivo detectado durante la creación
            wts = nn.Linear(self.layer_szs[1], self.layer_szs[0], bias=False).to(self.device)
            if self.wtUp != 2:
                wts.weight *= 0.0
            return wts
    
    def load_wts(self, data):
        with torch.no_grad():
            self.wts.weight *= 0.0
            self.wts.weight.data += data.t()

    def load_wts2(self, data):
        with torch.no_grad():
            self.wts.weight *= 0.0
            self.wts.weight.data += data[0:self.layer_szs[1], :].t()
            self.counts += 1
            for x in range(self.layer_szs[1], data.size(0)):
                lk = self.infer_step(data[x].view(1,-1))
                z = F.one_hot(torch.argmax(lk, dim=1), num_classes=self.layer_szs[1]).float()
                self.update_wts(lk, z, data[x].view(1,-1))

    def update_wts(self, lk, z, input):
        err = torch.mean(torch.square(input - self.wts(z)))
        self.err_avg += (err - self.err_avg) / (self.age + 1)
        self.rules(lk, z, input)

    def wt_update(self, targ, z, N):
        with torch.no_grad():
            p = self.wts(z)
            err_pred = targ - p
            dw = err_pred.t().matmul(z)
            if self.wtUp == 0 or self.wtUp == 2:
                self.wts.weight += dw / self.counts[N]
            elif self.wtUp == 1:
                self.wts.weight += dw
            else:
                self.wts.weight += dw * self.l_rate

    def detect(self, mxlk, mxC):
        if self.detect_type == 0:
            dir_prior = (self.l_rate / (self.age * (1 / self.alpha) + 1))
            new_detect = ((mxlk.item() < dir_prior) and (torch.min(self.counts) == 0))
        elif self.detect_type == 1:
            new_detect = (mxlk < self.alpha)
        elif self.detect_type == 2:
            thr = self.prior[mxC] * self.alpha
            new_detect = (mxlk.item() < thr.item())
        else:
            dir_prior = (self.l_rate / (self.age * (1 / self.alpha) + 1))
            new_detect = (mxlk.item() < dir_prior)
        return new_detect

    def rules(self, lk, z, targ):
        with torch.no_grad():
            mxlk = torch.max(lk)
            mxC = torch.argmax(lk)
            new_detect = self.detect(mxlk, mxC)
            if new_detect and self.wtUp != 2:
                z *= 0
                N = torch.argmin(self.counts)
                z[0, N] = 1
                self.counts[N] = 1
                self.prior[N] = 1
                self.wt_update(targ, z, N)
            elif self.wtUp != 1:
                self.counts[mxC] += 1
                self.update_prior(mxlk, mxC)
                self.wt_update(targ, z, mxC)
            self.age += 1

    def rule2(self, z, targ):
        with torch.no_grad():
            N = torch.argmax(z)
            self.wt_update(targ, z, N)
            self.age += 1

    def update_prior(self, mxlk, mxC):
        self.prior[mxC] = self.prior[mxC] + (mxlk - self.prior[mxC]) / (self.counts[mxC] + 1)


    def infer_step(self, input):
        with torch.no_grad():
            if self.simFunc == 0:
                return input.matmul(self.wts.weight)

            # Cosine sim
            elif self.simFunc == 1:
                nm = torch.outer(torch.linalg.norm(input, dim=1), torch.linalg.norm(self.wts.weight, dim=0))
                return input.matmul(self.wts.weight) / (nm + .0000001)

            # Mean-shifted cosine sim
            elif self.simFunc == 2:
                nm = torch.outer(torch.linalg.norm(input - .5, dim=1), torch.linalg.norm(self.wts.weight - .5, dim=0))
                out = (input - .5).matmul(self.wts.weight - .5) / (nm + .0000001)
                return .5 * (out + 1)

            # Hetero-Cosine
            elif self.simFunc == 3:
                msk = (input > 0)[0, :, None]
                nm = torch.outer(torch.linalg.norm(input, dim=1), torch.linalg.norm(self.wts.weight * msk, dim=0))
                out = input.matmul(self.wts.weight * msk) / (nm + .0000001)
                return out

            # Hetero-Mean-shift cosine
            elif self.simFunc == 4:
                msk = (input > 0)[0, :, None]
                msk2 = (input > 0)
                nm = torch.outer(torch.linalg.norm((input - .5) * msk2, dim=1), torch.linalg.norm((self.wts.weight-.5) * msk, dim=0))
                out = ((input - .5)).matmul((self.wts.weight-.5) * msk) / (nm + .0000001)
                return out


            # Manhattan (this is for MHN-manh)
            else:
                out = torch.zeros(input.size(0), self.wts.weight.size(1), device=self.device) # Mover out al dispositivo
                for i in range(input.size(0)):
                    t = input[i].repeat(self.wts.weight.size(1), 1).t()
                    out[i] = - torch.abs(t - self.wts.weight.data).sum(0)
                return out

    def predict(self, z):
        return self.wts(z)

    def recall(self, targ):
        with torch.no_grad():
            t = targ.clone().view(targ.size(0), -1)
            a = self.infer_step(t)
            if self.actFunc == 0:
                # 4. Mover el tensor one_hot al dispositivo correcto
                z = F.one_hot(torch.argmax(a, dim=1), num_classes=self.layer_szs[1]).float().to(self.device)
            else:
                z = softmax(a * self.beta)
            return self.predict(z)

    def recognize(self, targ):
        with torch.no_grad():
            t = targ.clone().view(targ.size(0), -1)
            a = self.infer_step(t)
            mxlk = torch.max(a, dim=1)[0]
            mxC = torch.argmax(a, dim=1)
            return mxlk >= (self.prior[mxC] * self.gamma)

    def recognize2(self, targ):
        with torch.no_grad():
            t = targ.clone().view(targ.size(0), -1)
            p = self.recall(t)
            err = torch.mean(torch.square(t - p), dim=1)
            return err <= (self.err_avg * self.gamma)
