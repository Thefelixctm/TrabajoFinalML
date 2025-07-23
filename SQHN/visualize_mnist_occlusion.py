import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

import Unit
from autoencoder_model import Autoencoder
from data_loader import get_data

def run_mnist_occlusion_test(latent_size=512, epochs_ae=200):
    # --- 1. Cargar una Imagen de MNIST ---
    # Usamos el data_loader para obtener una sola imagen de MNIST (dataset_id=0)
    train_loader, _ = get_data(shuf=True, data=0, max_iter=1, cont=False, b_sz=1)
    original_image, _ = next(iter(train_loader))
    original_image = original_image.to('cuda')

    input_size = 784  # MNIST es 28x28 = 784
    print(f"Imagen de MNIST cargada. Tamaño del vector: {input_size}")

    # --- 2. Crear una Versión con Oclusión Severa ---
    print("Ocultando la mitad derecha de la imagen...")
    occluded_image = torch.clone(original_image)
    h, w = occluded_image.shape[2], occluded_image.shape[3]
    occluded_image[:, :, :, w//2:] = 0 # Poner a negro la mitad derecha

    original_flat = original_image.view(1, -1)
    occluded_flat = occluded_image.view(1, -1)

    # --- 3. "Memoria Perfecta" con SQHN ---
    print("Enseñando el dígito ORIGINAL al SQHN...")
    sqhn_model = Unit.MemUnit(layer_szs=[input_size, latent_size * 4], simFunc=2, wt_up=0, alpha=50000, det_type=0).to('cuda')
    lk = sqhn_model.infer_step(original_flat)
    z = F.one_hot(torch.argmax(lk, dim=1), num_classes=sqhn_model.layer_szs[1]).float()
    sqhn_model.update_wts(lk, z, original_flat)

    print("Restaurando el dígito OCULTO con SQHN...")
    with torch.no_grad():
        hidden = sqhn_model.infer_step(occluded_flat)
        sqhn_restored_flat = torch.sigmoid(hidden @ sqhn_model.wts.weight.T)
        sqhn_restoration = sqhn_restored_flat.view_as(original_image)

    # --- 4. "Generalización" con Autoencoder ---
    print(f"Entrenando Autoencoder con el dígito ORIGINAL por {epochs_ae} épocas...")
    ae_model = Autoencoder(input_size=input_size, latent_size=latent_size).to('cuda')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)
    for epoch in range(epochs_ae):
        outputs = ae_model(original_flat); loss = criterion(outputs, original_flat)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    print("Restaurando el dígito OCULTO con Autoencoder...")
    with torch.no_grad():
        ae_restored_flat = ae_model(occluded_flat)
        ae_restoration = ae_restored_flat.view_as(original_image)

    # --- 5. Generar y Guardar la Visualización Final ---
    print("Generando comparación final...")
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    fig.suptitle('Test de Oclusión en MNIST (La Pelea Justa)', fontsize=16)

    def imshow(ax, img, title):
        ax.imshow(img.cpu().squeeze(), cmap='gray'); ax.set_title(title); ax.axis('off')

    imshow(axs[0], original_image, 'Original')
    imshow(axs[1], occluded_image, 'Oculta')
    imshow(axs[2], sqhn_restoration, 'SQHN (Restaurada)')
    imshow(axs[3], ae_restoration, 'Autoencoder (Restaurada)')
    
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "mnist_occlusion_test.png")
    plt.savefig(filename, bbox_inches='tight')
    print(f"✅ Visualización guardada localmente en: {filename}")
    plt.show()

if __name__ == '__main__':
    run_mnist_occlusion_test()