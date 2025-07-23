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

# --- FUNCIÓN CORREGIDA Y MEJORADA ---
def preprocess_image(image_path, size=(32, 32)):
    """
    Carga una imagen, la convierte en un cuadrado perfecto con padding, y luego la
    redimensiona usando el método 'NEAREST' para preservar los bordes duros.
    """
    try:
        image = Image.open(image_path).convert('L')
        
        w, h = image.size
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        
        transform = T.Compose([
            T.Pad((pad_w, pad_h), fill=255),
            # --- ¡LA CORRECCIÓN CLAVE! ---
            # Usamos NEAREST para que no se creen grises ni se suavicen los bordes.
            T.Resize(size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
            lambda x: (x < 0.5).float()
        ])
        
        return transform(image).unsqueeze(0).to('cuda')
    except Exception as e:
        print(f"❌ Error al cargar la imagen {image_path}: {e}")
        return None

# ... (EL RESTO DEL SCRIPT SE QUEDA EXACTAMENTE IGUAL) ...
# ... (build_model_library, restore_image, etc.)
def build_model_library(folder_path, library_file="qr_model_library.pkl"):
    print(f"Construyendo biblioteca de modelos desde: {folder_path}")
    model_library = {}
    
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
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

        # Entrenar SQHN (memoria one-shot)
        sqhn = Unit.MemUnit(layer_szs=[input_size, 2048], simFunc=2, wt_up=0, alpha=50000, det_type=0).to('cuda')
        lk = sqhn.infer_step(img_flat); z = F.one_hot(torch.argmax(lk, dim=1), num_classes=sqhn.layer_szs[1]).float()
        sqhn.update_wts(lk, z, img_flat)
        
        # Entrenar Autoencoder (sobreajustándolo para que actúe como una memoria)
        ae = Autoencoder(input_size=input_size, latent_size=512).to('cuda')
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
        for _ in range(200): # Entrenar para memorizar un solo patrón
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
        print(f"❌ Error: Biblioteca '{library_file}' no encontrada. Ejecuta el modo 'learn' primero."); return

    with open(library_file, 'rb') as f:
        model_library = pickle.load(f)

    corrupted_image = preprocess_image(corrupted_path)
    if corrupted_image is None: return
    corrupted_flat = corrupted_image.view(1, -1)
    input_size = corrupted_flat.shape[1]

    # --- SQHN: Encontrar la mejor memoria por "resonancia" (método honesto) ---
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

    # Cargar el modelo SQHN ganador y restaurar la imagen
    sqhn = Unit.MemUnit(layer_szs=[input_size, 2048], simFunc=2, wt_up=0).to('cuda')
    sqhn.load_state_dict(model_library[best_sqhn_match]['sqhn_state'])
    with torch.no_grad():
        hidden = sqhn.infer_step(corrupted_flat)
        sqhn_restored_flat = torch.sigmoid(hidden @ sqhn.wts.weight.T)
        sqhn_restoration = sqhn_restored_flat.view_as(corrupted_image)

    # --- Autoencoder: Encontrar el mejor modelo por error de reconstrucción (LA PELEA JUSTA) ---
    best_ae_match = None
    lowest_error = float('inf')
    for filename, models in model_library.items():
        model = Autoencoder(input_size=input_size, latent_size=512).to('cuda')
        model.load_state_dict(models['ae_state'])
        with torch.no_grad():
            reconstruction = model(corrupted_flat)
            # --- ¡LA LÍNEA CORREGIDA Y JUSTA! ---
            # Comparamos la reconstrucción contra la ENTRADA CORRUPTA.
            # El modelo que menos altera la entrada corrupta (mientras la "limpia") es el mejor.
            error = F.mse_loss(reconstruction, corrupted_flat)
            if error < lowest_error:
                lowest_error = error
                best_ae_match = filename
    print(f"  -> Autoencoder (pelea justa) encontró la mejor coincidencia con: '{best_ae_match}'")

    # Cargar el modelo AE ganador y restaurar la imagen
    ae = Autoencoder(input_size=input_size, latent_size=512).to('cuda')
    ae.load_state_dict(model_library[best_ae_match]['ae_state'])
    with torch.no_grad():
        ae_restored_flat = ae(corrupted_flat)
        ae_restoration = ae_restored_flat.view_as(corrupted_image)

    # --- Generar Visualización Final ---
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    fig.suptitle('Restaurador de Códigos QR (La Pelea Justa)', fontsize=16)
    
    # Suponemos que la imagen original se encuentra en una carpeta paralela
    clean_folder = os.path.dirname(corrupted_path).replace("corrupted", "clean")
    original_image_path = os.path.join(clean_folder, best_sqhn_match)
    original_image = preprocess_image(original_image_path)
    
    def imshow(ax, img, title):
        ax.imshow(img.cpu().squeeze(), cmap='gray'); ax.set_title(title); ax.axis('off')
    
    imshow(axs[0], original_image, f'Original ({best_sqhn_match})')
    imshow(axs[1], corrupted_image, 'Corrupta')
    imshow(axs[2], sqhn_restoration, 'SQHN (Restaurada)')
    imshow(axs[3], ae_restoration, 'Autoencoder (Restaurada)')
    
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "qr_restoration_fair_fight_result.png")
    plt.savefig(filename, bbox_inches='tight')
    print(f"\n✅ Visualización guardada localmente en: {filename}")
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