import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import pickle
from torch.utils.data import Dataset, DataLoader

import Unit
from autoencoder_model import Autoencoder

# --- Clases y Funciones de Ayuda ---
class QRDataset(Dataset):
    def __init__(self, folder_path, resolution):
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))]
        self.resolution = resolution
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        return preprocess_image(self.image_paths[idx], self.resolution).squeeze(0)

def preprocess_image(image_path, resolution=96):
    try:
        image = Image.open(image_path).convert('L')
        w, h = image.size; max_dim = max(w, h)
        pad_w, pad_h = (max_dim - w) // 2, (max_dim - h) // 2
        transform = T.Compose([
            T.Pad((pad_w, pad_h), fill=255),
            T.Resize((resolution, resolution), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
            lambda x: (x < 0.5).float()
        ])
        return transform(image).unsqueeze(0).to('cuda')
    except Exception as e:
        print(f"❌ Error al cargar {image_path}: {e}"); return None

def build_sqhn_library(folder_path, resolution=96, library_file="sqhn_specialist_library.pkl"):
    print(f"Construyendo biblioteca de SQHN especialistas ({resolution}x{resolution}px)...")
    model_library = {}
    for filename in os.listdir(folder_path):
        if not filename.endswith(('.png', '.jpg')): continue
        image_path = os.path.join(folder_path, filename)
        image_tensor = preprocess_image(image_path, resolution)
        if image_tensor is None: continue
        print(f"  -> Memorizando '{filename}'...")
        img_flat = image_tensor.view(1, -1)
        input_size = resolution * resolution
        sqhn = Unit.MemUnit(layer_szs=[input_size, input_size * 2], simFunc=2, wt_up=0, alpha=50000, det_type=0).to('cuda')
        lk = sqhn.infer_step(img_flat); z = F.one_hot(torch.argmax(lk, dim=1), num_classes=sqhn.layer_szs[1]).float()
        sqhn.update_wts(lk, z, img_flat)
        model_library[filename] = sqhn.state_dict()
    with open(library_file, 'wb') as f: pickle.dump(model_library, f)
    print(f"\n✅ Biblioteca SQHN con {len(model_library)} modelos guardada en '{library_file}'")

def train_generalist_ae(folder_path, resolution=96, model_file="generalist_ae.pth", epochs=500):
    print(f"Entrenando Autoencoder Generalista ({resolution}x{resolution}px)...")
    dataset = QRDataset(folder_path, resolution)
    if len(dataset) == 0: print("⚠️ Carpeta vacía."); return
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    input_size = resolution * resolution
    ae = Autoencoder(input_size=input_size, latent_size=input_size // 4).to('cuda')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        for images in dataloader:
            img_flat = images.view(images.size(0), -1)
            outputs = ae(img_flat); loss = criterion(outputs, img_flat)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (epoch + 1) % 100 == 0: print(f"  -> Época {epoch+1}/{epochs}, Pérdida: {loss.item():.6f}")
            
    torch.save(ae.state_dict(), model_file)
    print(f"\n✅ Autoencoder Generalista guardado en '{model_file}'")

def restore_image(corrupted_path, resolution=96, sqhn_library="sqhn_specialist_library.pkl", ae_model_file="generalist_ae.pth"):
    print(f"Restaurando ({resolution}x{resolution}px): {corrupted_path}")
    if not os.path.exists(sqhn_library) or not os.path.exists(ae_model_file):
        print("❌ Error: Faltan archivos de modelos. Ejecuta los modos 'learn' primero."); return

    with open(sqhn_library, 'rb') as f: sqhn_library_data = pickle.load(f)
    
    corrupted_image = preprocess_image(corrupted_path, resolution)
    if corrupted_image is None: return
    corrupted_flat = corrupted_image.view(1, -1)
    input_size = resolution * resolution

    # --- Restauración con SQHN (El Especialista) ---
    best_sqhn_match = None; highest_activation = -float('inf')
    for filename, state_dict in sqhn_library_data.items():
        model = Unit.MemUnit(layer_szs=[input_size, input_size*2], simFunc=2, wt_up=0).to('cuda')
        model.load_state_dict(state_dict)
        with torch.no_grad():
            activation = model.infer_step(corrupted_flat).sum()
            if activation > highest_activation: highest_activation = activation; best_sqhn_match = filename
    print(f"  -> SQHN recordó el patrón de: '{best_sqhn_match}'")
    sqhn = Unit.MemUnit(layer_szs=[input_size, input_size*2], simFunc=2, wt_up=0).to('cuda')
    sqhn.load_state_dict(sqhn_library_data[best_sqhn_match])
    with torch.no_grad():
        hidden = sqhn.infer_step(corrupted_flat)
        sqhn_restored_flat = torch.sigmoid(hidden @ sqhn.wts.weight.T)
        sqhn_restoration = sqhn_restored_flat.view_as(corrupted_image)

    # --- Restauración con Autoencoder (El Generalista) ---
    print(f"  -> Autoencoder Generalista está reconstruyendo...")
    ae = Autoencoder(input_size=input_size, latent_size=input_size//4).to('cuda')
    ae.load_state_dict(torch.load(ae_model_file))
    ae.eval()
    with torch.no_grad():
        ae_restored_flat = ae(corrupted_flat)
        ae_restoration = ae_restored_flat.view_as(corrupted_image)

    # --- Visualización ---
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    fig.suptitle('Especialista (SQHN) vs. Generalista (Autoencoder)', fontsize=16)
    clean_folder = os.path.dirname(corrupted_path).replace("corrupted", "clean")
    original_image = preprocess_image(os.path.join(clean_folder, best_sqhn_match), resolution)
    def imshow(ax, img, title):
        ax.imshow(img.cpu().squeeze(), cmap='gray'); ax.set_title(title); ax.axis('off')
    
    imshow(axs[0], original_image, f'Original ({best_sqhn_match})')
    imshow(axs[1], corrupted_image, 'Corrupta')
    imshow(axs[2], sqhn_restoration, 'SQHN (Restaura Info)')
    imshow(axs[3], ae_restoration, 'AE (Restaura Apariencia)')
    
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"final_comparison_{resolution}px.png")
    plt.savefig(filename, bbox_inches='tight')
    print(f"\n✅ Visualización guardada en: {filename}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Comparación Final: Especialista vs. Generalista.")
    parser.add_argument('mode', choices=['learn_sqhn', 'learn_ae', 'restore'])
    parser.add_argument('path', type=str, help="Ruta a la carpeta de QRs limpios o a un QR corrupto.")
    parser.add_argument('--resolution', type=int, default=96, help="Resolución para procesar las imágenes.")
    args = parser.parse_args()

    if args.mode == 'learn_sqhn':
        build_sqhn_library(args.path, args.resolution)
    elif args.mode == 'learn_ae':
        train_generalist_ae(args.path, args.resolution)
    elif args.mode == 'restore':
        restore_image(args.path, args.resolution)