import os
import shutil
import traceback

# Directorio local en Colab para guardar los checkpoints.
# Esto ayuda a que la comprobación de si existe sea muy rápida.
RESULTS_DIR = 'results'

def get_expected_filename(func_name, params):
    """
    Construye el nombre de archivo de resultado esperado basado en el nombre de la
    función de entrenamiento y sus parámetros. Es el "traductor" central.
    """
    try:
        # Lógica para funciones en el módulo aa_online
        if func_name == 'train_online':
            return f"AA_Online_Simf{params['simf']}_numN{params['hid_sz']}_data{params['data']}_numData{params['max_iter']}_upType{params['wtupType']}_det{params['det_type']}.data"
        
        elif func_name == 'train_onCont':
            return f"AA_OnlineCont_Simf{params['simf']}_numN{params['hid_sz']}_data{params['data']}_numData{params['max_iter']}_upType{params['wtupType']}_det{params['det_type']}.data"
        
        elif func_name == 'train_onContDom':
            cont_nm = '' if params.get('cont', True) else 'online'
            return f"AA_OnlineContDom_Simf{params['simf']}_numN{params['hid_sz']}_numData{params['max_iter']}_upType{params['wtupType']}_det{params['det_type']}{cont_nm}.data"
        
        # Lógica para funciones en el módulo aa_onlineBP
        elif func_name == 'bp_train_online':
            return f"AA_OnlineBP_data{params['data']}_opt{params['opt']}_hdsz{params['hid_sz']}.data"
        
        elif func_name == 'bp_train_onCont':
            return f"AA_OnlineContBP_data{params['data']}_opt{params['opt']}_hdsz{params['hid_sz']}.data"
        
        elif func_name == 'bp_train_onContDom':
            cont_nm = '' if params.get('cont', True) else 'online'
            return f"AA_OnlineContDomBP_opt{params['opt']}_hdsz{params['hid_sz']}{cont_nm}.data"

        # --- AÑADIR ESTA NUEVA CONDICIÓN ---
        elif func_name == 'train_autoencoder':
            return f"AE_Online_data{params['data']}_hdsz{params['hid_sz']}.data"
        # ----------------------------------- 

        # Si la función no está en la lista, lanza un error para avisarnos.
        raise ValueError(f"Nombre de función desconocido para checkpointing: {func_name}")

    except KeyError as e:
        print(f"❌ Error de Parámetro: No se pudo encontrar la clave {e} para construir el nombre de archivo.")
        raise
        

def run_if_not_exists(func, func_name_key, params, drive_save_path=None):
    """
    Wrapper inteligente que ejecuta una función de entrenamiento solo si su resultado
    no existe, y luego guarda inmediatamente el resultado en Google Drive.
    """
    # 1. Construir el nombre y la ruta del archivo de resultado esperado en Colab.
    try:
        filename = get_expected_filename(func_name_key, params)
        local_filepath = os.path.join(RESULTS_DIR, filename)
    except KeyError:
        # El error ya fue impreso por get_expected_filename
        return

    # 2. Comprobar si el checkpoint ya existe LOCALMENTE (esto es muy rápido).
    if os.path.exists(local_filepath):
        print(f"✔️ SALTANDO: El archivo ya existe en Colab en '{local_filepath}'")
        return

    # 3. Si no existe, ejecutar el entrenamiento.
    param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
    print(f"🚀 EJECUTANDO: {func.__module__}.{func.__name__} con {param_str}")
    
    # Asegurarse de que los directorios necesarios existan antes de la ejecución.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Ejecutar la función de entrenamiento original.
    func(**params)
    
    # 4. Mover el resultado desde 'data/' a la carpeta de checkpoints locales 'results/'.
    original_source_path = os.path.join('data', filename)
    # --- AÑADE ESTA LÍNEA DE DEPURACIÓN ---
    print(f"DEBUG (checkpoint_utils): Buscando el archivo en: '{original_source_path}'")
    # -------------------------------------
    if os.path.exists(original_source_path):
        # Mover a la carpeta de checkpoints de Colab.
        os.rename(original_source_path, local_filepath)
        print(f"📦 Resultado movido a '{local_filepath}'")
        
        # 5. GUARDADO INMEDIATO EN GOOGLE DRIVE.
        if drive_save_path:
            try:
                # Asegurarse de que la carpeta de destino en Drive existe.
                os.makedirs(drive_save_path, exist_ok=True)
                # Copiar el archivo desde la carpeta de Colab a Drive.
                shutil.copy(local_filepath, drive_save_path)
                print(f"✅ ¡ÉXITO! Resultado copiado a Google Drive.")
            except Exception as e:
                print(f"🚨 ¡ERROR AL COPIAR A DRIVE! El entrenamiento se completó, pero el guardado falló.")
                print(traceback.format_exc())
        else:
            print("❔ No se especificó 'drive_save_path', el resultado solo se guardó localmente en Colab.")
            
    else:
        print(f"⚠️ ¡ADVERTENCIA! La función se ejecutó pero no se encontró el archivo de resultado esperado en '{original_source_path}'.")
        print("   Verifica que la función de entrenamiento está guardando el archivo con el nombre correcto.")