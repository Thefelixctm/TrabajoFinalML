import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import pickle

import Unit
from autoencoder_model import Autoencoder

# --- FUNCIÓN CORREGIDA ---
def preprocess_image(image_path, size=(32, 32)):
    """
    Carga una imagen, la convierte en un cuadrado perfecto añadiendo márgenes blancos
    (padding), y luego la redimensiona y binariza.
    """
    try:
        image = Image.open(image_path).convert('L') # Convertir a blanco y negro
        
        # --- LÓGICA DE PADDING INTELIGENTE ---
        # 1. Obtener las dimensiones originales
        w, h = image.size
        # 2. Encontrar la dimensión más grande para crear un lienzo cuadrado
        max_dim = max(w, h)
        # 3. Calcular el padding necesario para cada lado
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        # 4. Crear la secuencia de transformaciones
        transform = T.Compose([
            # Añadir márgenes blancos (fill=255) para hacerlo cuadrado
            T.Pad((pad_w, pad_h), fill=255), 
            # Ahora que es cuadrado, podemos redimensionar sin distorsión
            T.Resize(size),
            T.ToTensor(),
            # Binarizar la imagen a 0s y 1s puros
            lambda x: (x < 0.5).float() # Invertimos la lógica para que el negro sea 1
        ])
        # --- FIN DE LA LÓGICA INTELIGENTE ---
        
        return transform(image).unsqueeze(0).to('cuda')
    except Exception as e:
        print(f"❌ Error al cargar la imagen {image_path}: {e}")
        return None

# ... (EL RESTO DEL SCRIPT SE QUEDA EXACTAMENTE IGUAL) ...

def build_model_library(folder_path, library_file="qr_model_library.pkl"):
    print(f"Construyendo biblioteca de modelos desde: {folder_path}")
    model_library = {}
    
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))]
    if not image_files:
        print("⚠️ No se encontraron imágenes en la carpeta.")
        return

    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        image_tensor = preprocess_image(image_path)
        if image_tensor is None: continue

        print(f"  -> Entrenando modelos para '{filename}'...")
        img_flat = image_tensor.view(1, -1)
        input_size = img_flat.shape[1]

        # Entrenar SQHN
        sqhn = Unit.MemUnit(layer_szs=[input_size, 2048], simFunc=2, wt_up=0, alpha=50000, det_type=0).to('cuda')
        lk = sqhn.infer_step(img_flat); z = F.one_hot(torch.argmax(lk, dim=1), num_classes=sqhn.layer_szs[1]).float()
        sqhn.update_wts(lk, z, img_flat)
        
        # Entrenar Autoencoder
        ae = Autoencoder(input_size=input_size, latent_size=512).to('cuda')
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
        for _ in range(200):
            outputs = ae(img_flat); loss = criterion(outputs, img_flat)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        model_library[filename] = {
            'sqhn_state': sqhn.state_dict(),
            'ae_state': ae.state_dict()
        }
    
    with open(library_file, 'wb') as f:
        pickle.dump(model_library, f)
    print(f"\n✅ Biblioteca con {len(model_library)} modelos guardada en '{library_file}'")

def restore_image(corrupted_path, library_file="qr_model_library.pkl"):
    print(f"Intentando restaurar: {corrupted_path}")
    if not os.path.exists(library_file):
        print(f"❌ Error: Biblioteca '{library_file}' no encontrada."); return

    with open(library_file, 'rb') as f:
        model_library = pickle.load(f)

    corrupted_image = preprocess_image(corrupted_path)
    if corrupted_image is None: return
    corrupted_flat = corrupted_image.view(1, -1)
    input_size = corrupted_flat.shape[1]

    # --- Encontrar la mejor memoria coincidente para SQHN ---
    best_sqhn_match = None
    highest_activation = -float('inf')
    for filename, models in model_library.items():
        model = Unit.MemUnit(layer_szs=[input_size, 2048], simFunc=2, wt_up=0).to('cuda')
        model.load_state_dict(models['sqhn_state'])
        with torch.no_grad():
            activation = model.infer_step(corrupted_flat).sum()
            if activation > highest_activation:
                highest_activation = activation
                best_sqhn_match = filename
    print(f"  -> SQHN recordó el patrón de: '{best_sqhn_match}'")

    # --- Cargar el modelo SQHN correspondiente y restaurar ---
    sqhn = Unit.MemUnit(layer_szs=[input_size, 2048], simFunc=2, wt_up=0).to('cuda')
    sqhn.load_state_dict(model_library[best_sqhn_match]['sqhn_state'])
    with torch.no_grad():
        hidden = sqhn.infer_step(corrupted_flat)
        sqhn_restored_flat = torch.sigmoid(hidden @ sqhn.wts.weight.T)
        sqhn_restoration = sqhn_restored_flat.view_as(corrupted_image)

    # --- Encontrar el mejor modelo AE ---
    best_ae_match = None
    lowest_error = float('inf')
    for filename, models in model_library.items():
        model = Autoencoder(input_size=input_size, latent_size=512).to('cuda')
        model.load_state_dict(models['ae_state'])
        with torch.no_grad():
            reconstruction = model(corrupted_flat)
            # Comparamos la reconstrucción con la versión "limpia" de esa memoria
            clean_version_path = os.path.join("qr_codes_clean", filename)
            clean_tensor = preprocess_image(clean_version_path).view(1, -1)
            error = F.mse_loss(reconstruction, clean_tensor)
            if error < lowest_error:
                lowest_error = error
                best_ae_match = filename
    print(f"  -> Autoencoder encontró la mejor coincidencia con: '{best_ae_match}'")

    # --- Cargar el modelo AE y restaurar ---
    ae = Autoencoder(input_size=input_size, latent_size=512).to('cuda')
    ae.load_state_dict(model_library[best_ae_match]['ae_state'])
    with torch.no_grad():
        ae_restored_flat = ae(corrupted_flat)
        ae_restoration = ae_restored_flat.view_as(corrupted_image)

    # --- Generar Visualización ---
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    fig.suptitle('Restaurador de Códigos QR (Mundo Real)', fontsize=16)
    original_image_path = os.path.join(os.path.dirname(corrupted_path).replace("corrupted", "clean"), best_sqhn_match)
    original_image = preprocess_image(original_image_path)
    def imshow(ax, img, title):
        ax.imshow(img.cpu().squeeze(), cmap='gray'); ax.set_title(title); ax.axis('off')
    
    imshow(axs[0], original_image, f'Original ({best_sqhn_match})')
    imshow(axs[1], corrupted_image, 'Corrupta')
    imshow(axs[2], sqhn_restoration, 'SQHN (Restaurada)')
    imshow(axs[3], ae_restoration, 'Autoencoder (Restaurada)')
    
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "qr_restoration_result.png")
    plt.savefig(filename, bbox_inches='tight')
    print(f"✅ Visualización guardada en: {filename}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Restaurador de Códigos QR.")
    parser.add_argument('mode', choices=['learn', 'restore'])
    parser.add_argument('path', type=str, help="Ruta a la carpeta de QRs limpios (learn) o a un QR corrupto (restore).")
    args = parser.parse_args()

    if args.mode == 'learn':
        build_model_library(args.path)
    elif args.mode == 'restore':
        restore_image(args.path)