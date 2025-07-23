import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from data_loader import get_data
import Unit
from autoencoder_model import Autoencoder

def run_batch_visualization(num_images=8, dataset_id=0, epochs_ae=200):
    """
    Entrena los modelos con un lote completo de imágenes para comparar la 
    generalización (Autoencoder) vs. la interferencia de memoria (SQHN).
    """
    print(f"Cargando {num_images} imágenes del dataset ID: {dataset_id}...")
    train_loader, _ = get_data(shuf=True, data=dataset_id, max_iter=num_images, cont=False, b_sz=num_images)
    images, _ = next(iter(train_loader))
    images = images.to('cuda')
    
    # --- 1. Entrenamiento SQHN (Todos a la vez, causando interferencia) ---
    print("Entrenando SQHN con todo el lote (demostrando interferencia)...")
    sqhn_model = Unit.MemUnit(layer_szs=[784, 2300], simFunc=2, wt_up=0, alpha=50000, det_type=0).to('cuda')
    # Se le enseña cada imagen una tras otra, sobrescribiendo la memoria
    for i in range(num_images):
        img_flat = images[i].view(1, -1)
        lk = sqhn_model.infer_step(img_flat)
        z = F.one_hot(torch.argmax(lk, dim=1), num_classes=sqhn_model.layer_szs[1]).float()
        sqhn_model.update_wts(lk, z, img_flat)
    
    # Reconstruye usando la memoria final "confundida"
    with torch.no_grad():
        hidden_activations = sqhn_model.infer_step(images.view(num_images, -1))
        reconstructed_flat = torch.sigmoid(hidden_activations @ sqhn_model.wts.weight.T)
        sqhn_reconstructions = reconstructed_flat.view_as(images)

    # --- 2. Entrenamiento Autoencoder (Todos a la vez, aprendiendo a generalizar) ---
    print(f"Entrenando Autoencoder por {epochs_ae} épocas...")
    ae_model = Autoencoder(input_size=784, latent_size=2300).to('cuda')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)
    images_flat = images.view(num_images, -1)
    for epoch in range(epochs_ae):
        outputs = ae_model(images_flat); loss = criterion(outputs, images_flat)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        ae_reconstructions = ae_model(images_flat).view_as(images)

    # --- 3. Generar y Guardar el Gráfico ---
    print("Generando visualización de Lote...")
    fig, axs = plt.subplots(num_images, 3, figsize=(6, num_images * 2))
    fig.suptitle('Comparación en Lote (Generalización)', fontsize=16)
    
    for i in range(num_images):
        axs[i, 0].imshow(images[i].cpu().numpy().squeeze(), cmap='gray'); axs[i, 0].set_title('Original'); axs[i, 0].axis('off')
        axs[i, 1].imshow(sqhn_reconstructions[i].cpu().numpy().squeeze(), cmap='gray'); axs[i, 1].set_title('SQHN (Batch)'); axs[i, 1].axis('off')
        axs[i, 2].imshow(ae_reconstructions[i].cpu().numpy().squeeze(), cmap='gray'); axs[i, 2].set_title('Autoencoder'); axs[i, 2].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"reconstruction_batch_comparison_data{dataset_id}_{num_images}imgs.png")
    plt.savefig(filename, bbox_inches='tight')
    print(f"✅ Visualización guardada localmente en: {filename}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualizar reconstrucciones en modo Lote.")
    parser.add_argument('--num_images', type=int, default=8, help='Número de imágenes a visualizar.')
    parser.add_argument('--dataset_id', type=int, default=0, help='ID del dataset a usar (0=MNIST).')
    parser.add_argument('--epochs', type=int, default=200, help='Épocas para entrenar el autoencoder.')
    args = parser.parse_args()
    run_batch_visualization(num_images=args.num_images, dataset_id=args.dataset_id, epochs_ae=args.epochs)