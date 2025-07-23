import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

# Importamos los modelos y el cargador de datos del proyecto
import Unit
from autoencoder_model import Autoencoder

def preprocess_image(image_path, size=(32, 32)):
    """
    Carga una imagen desde una ruta, la convierte a RGB, la redimensiona y la
    transforma en un tensor de PyTorch listo para ser usado por los modelos.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),  # Convierte la imagen a tensor y normaliza los píxeles a [0, 1]
        ])
        # Añade una dimensión de 'batch' (lote) para que tenga la forma [1, C, H, W]
        return transform(image).unsqueeze(0).to('cuda')
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró la imagen en la ruta: '{image_path}'")
        print("   Asegúrate de que el archivo está en la carpeta /content/SQHN/ y el nombre es correcto.")
        return None

def run_custom_visualization(original_path, corrupted_path, latent_size=512, epochs_ae=300):
    """
    Realiza una demostración de restauración de imágenes comparando SQHN y un Autoencoder.
    """
    # --- 1. Cargar las dos versiones de la imagen ---
    original_image = preprocess_image(original_path)
    corrupted_image = preprocess_image(corrupted_path)
    if original_image is None or corrupted_image is None:
        return

    input_size = original_image.nelement() // original_image.shape[0]  # -> 32*32*3 = 3072
    print(f"Imágenes cargadas. Tamaño del vector de entrada: {input_size}")

    original_flat = original_image.view(1, -1)
    corrupted_flat = corrupted_image.view(1, -1)

    # --- 2. "Memoria Perfecta" con SQHN ---
    print("Enseñando la imagen ORIGINAL al SQHN (memoria one-shot)...")
    sqhn_model = Unit.MemUnit(layer_szs=[input_size, latent_size * 4], simFunc=2, wt_up=0, alpha=50000, det_type=0).to('cuda')
    
    # Memorizar la imagen original limpia
    lk = sqhn_model.infer_step(original_flat)
    z = F.one_hot(torch.argmax(lk, dim=1), num_classes=sqhn_model.layer_szs[1]).float()
    sqhn_model.update_wts(lk, z, original_flat)

    print("Restaurando la imagen CORRUPTA con SQHN...")
    with torch.no_grad():
        hidden = sqhn_model.infer_step(corrupted_flat)
        # Accedemos a la capa .wts, luego a su matriz .weight y la transponemos
        sqhn_restored_flat = torch.sigmoid(hidden @ sqhn_model.wts.weight.T)
        sqhn_restoration = sqhn_restored_flat.view_as(original_image)

    # --- 3. "Generalización" con Autoencoder ---
    print(f"Entrenando Autoencoder con la imagen ORIGINAL por {epochs_ae} épocas...")
    ae_model = Autoencoder(input_size=input_size, latent_size=latent_size).to('cuda')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)
    for epoch in range(epochs_ae):
        outputs = ae_model(original_flat); loss = criterion(outputs, original_flat)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    print("Restaurando la imagen CORRUPTA con Autoencoder...")
    with torch.no_grad():
        ae_restored_flat = ae_model(corrupted_flat)
        ae_restoration = ae_restored_flat.view_as(original_image)

    # --- 4. Generar y Guardar la Visualización Final ---
    print("Generando comparación final...")
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    fig.suptitle('Herramienta de Restauración de Imágenes', fontsize=16)

    def imshow(ax, img, title):
        # Mueve el tensor a la CPU, quita la dimensión de batch, y cambia el orden
        # de los ejes de (Canales, Alto, Ancho) a (Alto, Ancho, Canales) para matplotlib.
        img = img.cpu().squeeze().permute(1, 2, 0)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    imshow(axs[0], original_image, 'Original')
    imshow(axs[1], corrupted_image, 'Corrupta')
    imshow(axs[2], sqhn_restoration, 'SQHN (Restaurada)')
    imshow(axs[3], ae_restoration, 'Autoencoder (Restaurada)')
    
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "custom_image_restoration.png")
    plt.savefig(filename, bbox_inches='tight')
    print(f"✅ Visualización guardada localmente en: {filename}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Restaurar una imagen corrupta usando SQHN y un Autoencoder.")
    parser.add_argument('original_path', type=str, help='Ruta a la imagen original y limpia (ej. logo_original.png).')
    parser.add_argument('corrupted_path', type=str, help='Ruta a la imagen dañada que se quiere restaurar (ej. logo_corrupto.png).')
    args = parser.parse_args()
    
    run_custom_visualization(original_path=args.original_path, corrupted_path=args.corrupted_path)