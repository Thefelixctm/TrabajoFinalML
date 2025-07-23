import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from autoencoder_model import Autoencoder
from data_loader import get_data
from utilities import corrupt_test # Importamos la función de testeo

def train_autoencoder(in_sz, hid_sz, data, dev='cuda', max_iter=3000, num_seeds=7, lr=1e-3, t_fq=100, **kwargs):
    """
    Entrena un autoencoder estándar y guarda los resultados de reconstrucción.
    'hid_sz' se interpreta como el tamaño del espacio latente (el cuello de botella).
    **kwargs absorbe cualquier parámetro extra no utilizado (como 'opt', 'beta', etc.).
    """
    # Creamos tensores para todas las métricas para mantener la consistencia con otros archivos
    recall_mse = torch.zeros(num_seeds, int(max_iter / t_fq) + 1)
    recall_pcnt = torch.zeros(num_seeds, int(max_iter / t_fq) + 1)
    recall_mse_n = torch.zeros(num_seeds, int(max_iter / t_fq) + 1)
    recall_pcnt_n = torch.zeros(num_seeds, int(max_iter / t_fq) + 1)
    recall_mse_msk = torch.zeros(num_seeds, int(max_iter / t_fq) + 1)
    recall_pcnt_msk = torch.zeros(num_seeds, int(max_iter / t_fq) + 1)
    test_mse_results = torch.zeros(num_seeds, int(max_iter / t_fq) + 1)

    for s in range(num_seeds):
        print(f"  -> Autoencoder Seed {s+1}/{num_seeds}")
        model = Autoencoder(input_size=in_sz, latent_size=hid_sz).to(dev)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Usamos b_sz=64 para un entrenamiento más estable y rápido. 
        # max_iter ahora se refiere al número de batches.
        train_loader, test_loader = get_data(shuf=True, data=data, max_iter=max_iter * 64, cont=False, b_sz=64)
        
        # Preparamos un batch fijo de imágenes para una evaluación consistente
        eval_images, _ = next(iter(train_loader))
        eval_images = eval_images.to(dev)

        for batch_idx, (images, _) in enumerate(train_loader):
            if batch_idx >= max_iter: break # Detenerse después de max_iter batches

            images = images.view(images.size(0), -1).to(dev)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % t_fq == 0:
                # Creamos un 'callable' para que sea compatible con corrupt_test
                def ae_reconstruct(img_batch):
                    img_flat = img_batch.view(img_batch.size(0), -1)
                    reconstructed_flat = model(img_flat)
                    return reconstructed_flat.view(img_batch.size())

                with torch.no_grad():
                    # Usamos el batch de evaluación fijo para consistencia
                    mse, pct, msen, pctn, mse_msk, pct_msk = corrupt_test(0.2, 0.5, ae_reconstruct, eval_images)
                    
                    idx = int(batch_idx / t_fq)
                    recall_mse[s, idx] = mse; recall_pcnt[s, idx] = pct
                    recall_mse_n[s, idx] = msen; recall_pcnt_n[s, idx] = pctn
                    recall_mse_msk[s, idx] = mse_msk; recall_pcnt_msk[s, idx] = pct_msk

    # --- GUARDADO DE RESULTADOS (LA PARTE QUE FALTABA) ---
    # Este nombre de archivo DEBE COINCIDIR con el que espera 'checkpoint_utils.py'
    os.makedirs('data', exist_ok=True)
    filename = f'data/AE_Online_data{data}_hdsz{hid_sz}.data'
    
    print(f"Autoencoder training complete. Final MSE: {torch.mean(recall_mse[:, -1]):.4f}")
    print(f"Guardando resultados en: '{filename}'")
    
    with open(filename, 'wb') as filehandle:
        pickle.dump([recall_mse, recall_pcnt, recall_mse_n, recall_pcnt_n, 
                     recall_mse_msk, recall_pcnt_msk, test_mse_results], filehandle)