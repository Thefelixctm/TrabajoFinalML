import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import pickle

# Importamos los módulos locales. Asegúrate de que estos archivos .py están en la misma carpeta.
import Unit
from autoencoder_model import Autoencoder

# --- LÓGICA DE RUTAS INFALIBLES (La Clave para Streamlit) ---
# 1. Obtener la ruta del directorio donde se encuentra ESTE script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. Construir las rutas absolutas a las carpetas y archivos que usaremos.
PLOTS_PATH = os.path.join(SCRIPT_DIR, 'plots')
LIBRARY_FILE_PATH = os.path.join(SCRIPT_DIR, "qr_model_library.pkl")
# --- FIN DE LA LÓGICA DE RUTAS ---

def preprocess_image(image_path, resolution=96):
    """
    Carga, aplica padding, redimensiona (sin distorsión) y binariza una imagen.
    """
    try:
        image = Image.open(image_path).convert('L')
        w, h = image.size
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        
        transform = T.Compose([
            T.Pad((pad_w, pad_h), fill=255),
            T.Resize((resolution, resolution), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
            lambda x: (x < 0.5).float()
        ])
        
        # En entornos sin GPU, cambia 'cuda' por 'cpu'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"ERROR al cargar la imagen {image_path}: {e}")
        return None

def build_model_library(folder_path, resolution=96):
    """
    Recorre una carpeta de imágenes limpias, entrena los modelos para cada una,
    y guarda la biblioteca en una ruta absoluta.
    """
    print(f"Construyendo biblioteca de modelos ({resolution}x{resolution}px) desde: {folder_path}")
    model_library = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files: 
        print("ADVERTENCIA: No se encontraron imágenes en la carpeta.")
        return

    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        image_tensor = preprocess_image(image_path, resolution)
        if image_tensor is None: continue

        print(f"  -> Entrenando modelos para '{filename}'...")
        img_flat = image_tensor.view(1, -1)
        input_size = resolution * resolution

        # Entrenar SQHN
        sqhn = Unit.MemUnit(layer_szs=[input_size, input_size * 2], simFunc=2, wt_up=0, alpha=50000, det_type=0).to(device)
        lk = sqhn.infer_step(img_flat); z = F.one_hot(torch.argmax(lk, dim=1), num_classes=sqhn.layer_szs[1]).float()
        sqhn.update_wts(lk, z, img_flat)
        
        # Entrenar Autoencoder
        ae = Autoencoder(input_size=input_size, latent_size=input_size // 4).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
        for _ in range(200):
            outputs = ae(img_flat); loss = criterion(outputs, img_flat)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        model_library[filename] = {
            'sqhn_state': sqhn.state_dict(),
            'ae_state': ae.state_dict()
        }
    
    with open(LIBRARY_FILE_PATH, 'wb') as f: pickle.dump(model_library, f)
    print(f"\n[OK] Biblioteca con {len(model_library)} modelos guardada en: '{LIBRARY_FILE_PATH}'")

def restore_image(corrupted_path, resolution=96):
    """
    Restaura una imagen corrupta usando la biblioteca de modelos guardada en una ruta absoluta.
    """
    print(f"Intentando restaurar ({resolution}x{resolution}px): {corrupted_path}")
    if not os.path.exists(LIBRARY_FILE_PATH): 
        print(f"ERROR: Biblioteca no encontrada en '{LIBRARY_FILE_PATH}'. Ejecuta el modo 'learn' primero."); return

    with open(LIBRARY_FILE_PATH, 'rb') as f: model_library = pickle.load(f)

    corrupted_image = preprocess_image(corrupted_path, resolution)
    if corrupted_image is None: return
    corrupted_flat = corrupted_image.view(1, -1)
    input_size = resolution * resolution
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Lógica de SQHN ---
    best_sqhn_match = None; highest_activation = -float('inf')
    for filename, models in model_library.items():
        model = Unit.MemUnit(layer_szs=[input_size, input_size*2], simFunc=2, wt_up=0).to(device)
        model.load_state_dict(models['sqhn_state'])
        with torch.no_grad():
            activation = model.infer_step(corrupted_flat).sum()
            if activation > highest_activation: highest_activation = activation; best_sqhn_match = filename
    print(f"  -> SQHN recordó el patrón de: '{best_sqhn_match}'")
    sqhn = Unit.MemUnit(layer_szs=[input_size, input_size*2], simFunc=2, wt_up=0).to(device)
    sqhn.load_state_dict(model_library[best_sqhn_match]['sqhn_state'])
    with torch.no_grad():
        hidden = sqhn.infer_step(corrupted_flat)
        sqhn_restored_flat = torch.sigmoid(hidden @ sqhn.wts.weight.T)
        sqhn_restoration = sqhn_restored_flat.view_as(corrupted_image)

    # --- Lógica de Autoencoder (Pelea Justa) ---
    best_ae_match = None; lowest_error = float('inf')
    for filename, models in model_library.items():
        model = Autoencoder(input_size=input_size, latent_size=input_size//4).to(device)
        model.load_state_dict(models['ae_state'])
        with torch.no_grad():
            reconstruction = model(corrupted_flat)
            error = F.mse_loss(reconstruction, corrupted_flat)
            if error < lowest_error: lowest_error = error; best_ae_match = filename
    print(f"  -> Autoencoder (pelea justa) encontró la mejor coincidencia con: '{best_ae_match}'")
    ae = Autoencoder(input_size=input_size, latent_size=input_size//4).to(device)
    ae.load_state_dict(model_library[best_ae_match]['ae_state'])
    with torch.no_grad():
        ae_restored_flat = ae(corrupted_flat)
        ae_restoration = ae_restored_flat.view_as(corrupted_image)

    # --- Visualización ---
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    fig.suptitle(f'Restaurador de QR a {resolution}x{resolution}px (La Pelea Justa)', fontsize=16)
    
    clean_folder = os.path.dirname(corrupted_path).replace("corrupted", "clean")
    original_image_path = os.path.join(clean_folder, best_sqhn_match)
    original_image = preprocess_image(original_image_path, resolution)
    def imshow(ax, img, title):
        ax.imshow(img.cpu().squeeze(), cmap='gray'); ax.set_title(title); ax.axis('off')
    
    imshow(axs[0], original_image, f'Original ({best_sqhn_match})')
    imshow(axs[1], corrupted_image, 'Corrupta')
    imshow(axs[2], sqhn_restoration, 'SQHN (Restaurada)')
    imshow(axs[3], ae_restoration, 'Autoencoder (Restaurada)')
    
    # Guarda el gráfico en la ruta absoluta correcta
    os.makedirs(PLOTS_PATH, exist_ok=True)
    filename = os.path.join(PLOTS_PATH, f"qr_restoration_{resolution}px_result.png")
    plt.savefig(filename, bbox_inches='tight')
    print(f"\n[OK] Visualizacion guardada localmente en: {filename}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Restaurador de Códigos QR a mayor resolución.")
    parser.add_argument('mode', choices=['learn', 'restore'])
    parser.add_argument('path', type=str, help="Ruta a la carpeta de QRs limpios (learn) o a un QR corrupto (restore).")
    parser.add_argument('--resolution', type=int, default=96, help="Resolución para procesar las imágenes (ej. 64, 96, 128).")
    args = parser.parse_args()

    if args.mode == 'learn':
        build_model_library(args.path, args.resolution)
    elif args.mode == 'restore':
        restore_image(args.path, args.resolution)