import os
import pickle
import torch
import pylab
import matplotlib

# --- Configuración de Matplotlib (se mantiene tu estilo) ---
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['savefig.dpi'] = 400.
matplotlib.rcParams['font.size'] = 9.0
matplotlib.rcParams['figure.figsize'] = (5.0, 3.5)
# ... (el resto de tus rcParams)

# --- LÓGICA DE RUTAS INFALIBLES (para Colab y Streamlit) ---
# 1. Obtener la ruta del directorio donde se encuentra este script (plot.py).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. Construir la ruta a la carpeta 'results' que está en el mismo directorio.
RESULTS_PATH = os.path.join(SCRIPT_DIR, 'results')
# 3. Construir la ruta a la carpeta 'plots' para guardar los gráficos.
PLOTS_PATH = os.path.join(SCRIPT_DIR, 'plots')
# -----------------------------------------------------------------

def plot_sensit():
    """Genera el gráfico de Sensibilidad al Orden."""
    with torch.no_grad():
        fig, axs = pylab.subplots(1, 3, figsize=(6, 1.5))
        names = ['MHN-SGD', 'MHN-Adam', 'MHN-EWC++', 'MHN-ER']
        clrs = ['#1f77b4', 'black', 'orange', '#2ca02c']
        lst = [':', ':', ':', ':']
        num_hd = [[300, 1300, 2300], [700, 2000, 3300]]
        num_d = [3000, 5000]
        dt = [6, 4]

        # Bloque 1: SQHN (OCI)
        for d in range(2):
            sens_mse = []
            for hdz in range(3):
                path_cont = os.path.join(RESULTS_PATH, f'AA_OnlineCont_Simf2_numN{num_hd[d][hdz]}_data{dt[d]}_numData{num_d[d]}_upType0_det0.data')
                path_on = os.path.join(RESULTS_PATH, f'AA_Online_Simf2_numN{num_hd[d][hdz]}_data{dt[d]}_numData{num_d[d]}_upType0_det0.data')
                with open(path_cont, 'rb') as f_cont, open(path_on, 'rb') as f_on:
                    dta_cont, dta_on = pickle.load(f_cont), pickle.load(f_on)
                cuml_cont = torch.mean(dta_cont[2]); cuml_on = torch.mean(dta_on[2])
                sens_mse.append(torch.abs(torch.mean(cuml_cont - cuml_on)))
            axs[d].plot(num_hd[d], sens_mse, marker='s', alpha=.6, ls='-', label='SQHN', markersize=4.5, color='red')

        # Bloque 2: SQHN (ODI)
        sens_mse = []
        for hdz in range(3):
            path_cont = os.path.join(RESULTS_PATH, f'AA_OnlineContDom_Simf2_numN{num_hd[0][hdz]}_numData{num_d[0]}_upType0_det0.data')
            path_on = os.path.join(RESULTS_PATH, f'AA_OnlineContDom_Simf1_numN{num_hd[0][hdz]}_numData{num_d[0]}_upType0_det0online.data')
            with open(path_cont, 'rb') as f_cont, open(path_on, 'rb') as f_on:
                dta_cont, dta_on = pickle.load(f_cont), pickle.load(f_on)
            cuml_cont = torch.mean(dta_cont[2]); cuml_on = torch.mean(dta_on[2])
            sens_mse.append(torch.abs(torch.mean(cuml_cont - cuml_on)))
        axs[2].plot(num_hd[0], sens_mse, marker='s', alpha=.6, ls='-', label='SQHN', markersize=4.5, color='red')

        # Bloque 3: MHN (OCI)
        for d in range(2):
            for o in range(4):
                sens_mse = []
                for hdz in range(3):
                    path_cont = os.path.join(RESULTS_PATH, f'AA_OnlineContBP_data{dt[d]}_opt{o}_hdsz{num_hd[d][hdz]}.data')
                    path_on = os.path.join(RESULTS_PATH, f'AA_OnlineBP_data{dt[d]}_opt{o}_hdsz{num_hd[d][hdz]}.data')
                    with open(path_cont, 'rb') as f_cont, open(path_on, 'rb') as f_on:
                        dta_cont, dta_on = pickle.load(f_cont), pickle.load(f_on)
                    cuml_cont = torch.mean(dta_cont[2]); cuml_on = torch.mean(dta_on[2])
                    sens_mse.append(torch.abs(torch.mean(cuml_cont - cuml_on)))
                axs[d].plot(num_hd[d], sens_mse, marker='s', alpha=.6, ls=lst[o], label=names[o], markersize=4.5, color=clrs[o])

        # Bloque 4: MHN (ODI)
        for o in range(4):
            sens_mse = []
            for hdz in range(3):
                path_cont = os.path.join(RESULTS_PATH, f'AA_OnlineContDomBP_opt{o}_hdsz{num_hd[0][hdz]}.data')
                path_on = os.path.join(RESULTS_PATH, f'AA_OnlineContDomBP_opt{o}_hdsz{num_hd[0][hdz]}online.data')
                with open(path_cont, 'rb') as f_cont, open(path_on, 'rb') as f_on:
                    dta_cont, dta_on = pickle.load(f_cont), pickle.load(f_on)
                cuml_cont = torch.mean(dta_cont[2]); cuml_on = torch.mean(dta_on[2])
                sens_mse.append(torch.abs(torch.mean(cuml_cont - cuml_on)))
            axs[2].plot(num_hd[0], sens_mse, marker='s', alpha=.6, ls=lst[o], label=names[o], markersize=4.5, color=clrs[o])
        
        # Configuración del gráfico
        for x_ax in range(3): axs[x_ax].set(xlabel='# Hid. Neurons')
        axs[0].set(title='EMNIST (OCI)'); axs[1].set(title='CIFAR-100 (OCI)'); axs[2].set(title='MNIST (ODI)')
        axs[0].set(ylabel='Order\nSensitivity')
        for x_ax in range(3): axs[x_ax].set(ylim=(-0.01, .068))
        for x_ax in range(1, 3): axs[x_ax].yaxis.set_ticklabels([])
        pylab.tight_layout()

        # Lógica de guardado
        os.makedirs(PLOTS_PATH, exist_ok=True)
        filename = os.path.join(PLOTS_PATH, 'OnCont_L1_Sensitivity.png')
        pylab.savefig(filename)
        print(f" Gráfico de Sensibilidad guardado en: {filename}")
        pylab.show()

def plot_cont(simf=2):
    """Genera el gráfico de Aprendizaje Continuo (rendimiento a lo largo del tiempo)."""
    with torch.no_grad():
        fig, axs = pylab.subplots(2, 3, figsize=(8, 3))
        names = ['MHN-SGD', 'MHN-Adam', 'MHN-EWC++', 'MHN-ER']
        hid_sz = [1300, 2000]
        max_iter = [3000, 5000]
        data = [6,4]
        clrs = ['#1f77b4', 'black', 'orange', '#2ca02c']

        for d in range(2):
            filepath = os.path.join(RESULTS_PATH, f'AA_OnlineCont_Simf{simf}_numN{hid_sz[d]}_data{data[d]}_numData{max_iter[d]}_upType0_det0.data')
            with open(filepath, 'rb') as filehandle: dta = pickle.load(filehandle)
            x = torch.linspace(1, max_iter[d], dta[0].size(1))
            axs[0,d].errorbar(x, torch.mean(dta[3], dim=0), yerr=torch.std(dta[3], dim=0), fmt='o', alpha=.6, ls='-', label='SQHN', markersize=2.5, color='red')
            axs[1,d].errorbar(x, torch.mean(dta[2], dim=0), yerr=torch.std(dta[2], dim=0), fmt='o', alpha=.6, ls='-', label='SQHN', markersize=2.5, color='red')

            for o in range(4):
                filepath = os.path.join(RESULTS_PATH, f'AA_OnlineContBP_data{data[d]}_opt{o}_hdsz{hid_sz[d]}.data')
                with open(filepath, 'rb') as filehandle: dta = pickle.load(filehandle)
                x = torch.linspace(1, max_iter[d], dta[0].size(1))
                axs[0,d].errorbar(x, torch.mean(dta[3], dim=0), yerr=torch.std(dta[3], dim=0), fmt='o', alpha=.6, ls='-', label=names[o], markersize=2.5, color=clrs[o], marker='s')
                axs[1,d].errorbar(x, torch.mean(dta[2], dim=0), yerr=torch.std(dta[2], dim=0), fmt='o', alpha=.6, ls='-', label=names[o], markersize=2.5, color=clrs[o], marker='s')
        
        filepath = os.path.join(RESULTS_PATH, 'AA_OnlineContDom_Simf2_numN1300_numData3000_upType0_det0.data')
        with open(filepath,'rb') as filehandle: dta = pickle.load(filehandle)
        x = torch.linspace(1, 3000, dta[0].size(1))
        axs[0,2].errorbar(x, torch.mean(dta[3], dim=0), yerr=torch.std(dta[3], dim=0), fmt='o', alpha=.6, ls='-', label='SQHN', markersize=2.5, color='red')
        axs[1,2].errorbar(x, torch.mean(dta[2], dim=0), yerr=torch.std(dta[2], dim=0), fmt='o', alpha=.6, ls='-', label='SQHN', markersize=2.5, color='red')

        for o in range(4):
            filepath = os.path.join(RESULTS_PATH, f'AA_OnlineContDomBP_opt{o}_hdsz1300.data')
            with open(filepath, 'rb') as filehandle: dta = pickle.load(filehandle)
            x = torch.linspace(1, 3000, dta[0].size(1))
            axs[0,2].errorbar(x, torch.mean(dta[3], dim=0), yerr=torch.std(dta[3], dim=0), fmt='o', alpha=.6, ls='-', label=names[o], markersize=2.5, color=clrs[o], marker='s')
            axs[1,2].errorbar(x, torch.mean(dta[2], dim=0), yerr=torch.std(dta[2], dim=0), fmt='o', alpha=.6, ls='-', label=names[o], markersize=2.5, color=clrs[o], marker='s')
        
        # Configuración del gráfico
        for x_ax in range(3): axs[1, x_ax].set(xlabel='Training Iteration')
        axs[0,0].set(ylabel='Recall Acc'); axs[1,0].set(ylabel='Recall MSE')
        axs[0,0].set(title='EMNIST (OCI)'); axs[0,1].set(title='CIFAR-100 (OCI)'); axs[0,2].set(title='MNIST (ODI)')
        for x_ax in range(3): axs[1,x_ax].set(ylim=(-0.01, .3)); axs[0,x_ax].set(ylim=(-0.1, 1.1))
        
        # Lógica de guardado
        os.makedirs(PLOTS_PATH, exist_ok=True)
        filename = os.path.join(PLOTS_PATH, 'OnCont_L1_Continual.png')
        pylab.savefig(filename)
        print(f" Gráfico de Aprendizaje Continuo guardado en: {filename}")
        pylab.show()

def plot_cont_cumul():
    """Genera el gráfico de Rendimiento Final, comparando todos los modelos."""
    with torch.no_grad():
        fig, axs = pylab.subplots(2, 3, figsize=(7, 4))
        names = ['MHN-SGD', 'MHN-Adam', 'MHN-EWC++', 'MHN-ER']
        clrs = ['#1f77b4', 'black', 'orange', '#2ca02c']
        lst = [':', ':', ':', ':']; mrks = ['^', 'o', 's', 'p']
        num_hd = [[300, 1300, 2300], [700, 2000, 3300]]
        num_d = [3000, 5000]
        dt = [6, 4]

        # --- Bloque 1: SQHN ---
        for d in range(2):
            final_acc, final_mse, final_acc_std, final_mse_std = [], [], [], []
            for hdz in range(3):
                filepath = os.path.join(RESULTS_PATH, f'AA_OnlineCont_Simf2_numN{num_hd[d][hdz]}_data{dt[d]}_numData{num_d[d]}_upType0_det0.data')
                with open(filepath, 'rb') as filehandle: dta = pickle.load(filehandle)
                final_mse.append(torch.mean(dta[2][:, -1])); final_mse_std.append(torch.std(dta[2][:, -1]))
                final_acc.append(torch.mean(dta[3][:, -1])); final_acc_std.append(torch.std(dta[3][:, -1]))
            axs[0, d].errorbar(num_hd[d], final_acc, yerr=final_acc_std, fmt='s', alpha=.7, ls='-', label='SQHN', markersize=5, color='red')
            axs[1, d].errorbar(num_hd[d], final_mse, yerr=final_mse_std, fmt='s', alpha=.7, ls='-', label='SQHN', markersize=5, color='red')

        final_acc, final_mse, final_acc_std, final_mse_std = [], [], [], []
        for hdz in range(3):
            filepath = os.path.join(RESULTS_PATH, f'AA_OnlineContDom_Simf2_numN{num_hd[0][hdz]}_numData{num_d[0]}_upType0_det0.data')
            with open(filepath, 'rb') as filehandle: dta = pickle.load(filehandle)
            final_mse.append(torch.mean(dta[2][:, -1])); final_mse_std.append(torch.std(dta[2][:, -1]))
            final_acc.append(torch.mean(dta[3][:, -1])); final_acc_std.append(torch.std(dta[3][:, -1]))
        axs[0, 2].errorbar(num_hd[0], final_acc, yerr=final_acc_std, fmt='s', alpha=.7, ls='-', label='SQHN', markersize=5, color='red')
        axs[1, 2].errorbar(num_hd[0], final_mse, yerr=final_mse_std, fmt='s', alpha=.7, ls='-', label='SQHN', markersize=5, color='red')

        # --- Bloque 2: MHN ---
        for d in range(2):
            for o in range(4):
                final_acc, final_mse, final_acc_std, final_mse_std = [], [], [], []
                for hdz in range(3):
                    filepath = os.path.join(RESULTS_PATH, f'AA_OnlineContBP_data{dt[d]}_opt{o}_hdsz{num_hd[d][hdz]}.data')
                    with open(filepath,'rb') as filehandle: dta = pickle.load(filehandle)
                    final_mse.append(torch.mean(dta[2][:, -1])); final_mse_std.append(torch.std(dta[2][:, -1]))
                    final_acc.append(torch.mean(dta[3][:, -1])); final_acc_std.append(torch.std(dta[3][:, -1]))
                axs[0, d].errorbar(num_hd[d], final_acc, yerr=final_acc_std, fmt=mrks[o], alpha=.6, ls=lst[o], label=names[o], markersize=4.5, color=clrs[o])
                axs[1, d].errorbar(num_hd[d], final_mse, yerr=final_mse_std, fmt=mrks[o], alpha=.6, ls=lst[o], label=names[o], markersize=4.5, color=clrs[o])

        for o in range(4):
            final_acc, final_mse, final_acc_std, final_mse_std = [], [], [], []
            for hdz in range(3):
                filepath = os.path.join(RESULTS_PATH, f'AA_OnlineContDomBP_opt{o}_hdsz{num_hd[0][hdz]}.data')
                with open(filepath, 'rb') as filehandle: dta = pickle.load(filehandle)
                final_mse.append(torch.mean(dta[2][:, -1])); final_mse_std.append(torch.std(dta[2][:, -1]))
                final_acc.append(torch.mean(dta[3][:, -1])); final_acc_std.append(torch.std(dta[3][:, -1]))
            axs[0, 2].errorbar(num_hd[0], final_acc, yerr=final_acc_std, fmt=mrks[o], alpha=.6, ls=lst[o], label=names[o], markersize=4.5, color=clrs[o])
            axs[1, 2].errorbar(num_hd[0], final_mse, yerr=final_mse_std, fmt=mrks[o], alpha=.6, ls=lst[o], label=names[o], markersize=4.5, color=clrs[o])

        # --- Bloque 3: Autoencoder ---
        for d in range(2):
            ae_acc, ae_mse, ae_acc_std, ae_mse_std = [], [], [], []
            for hdz in range(3):
                filepath = os.path.join(RESULTS_PATH, f'AE_Online_data{dt[d]}_hdsz{num_hd[d][hdz]}.data')
                with open(filepath,'rb') as filehandle: dta = pickle.load(filehandle)
                ae_mse.append(torch.mean(dta[2][:, -1])); ae_mse_std.append(torch.std(dta[2][:, -1]))
                ae_acc.append(torch.mean(dta[3][:, -1])); ae_acc_std.append(torch.std(dta[3][:, -1]))
            axs[0, d].errorbar(num_hd[d], ae_acc, yerr=ae_acc_std, fmt='D', alpha=.7, ls=':', label='Autoencoder', markersize=4.5, color='purple')
            axs[1, d].errorbar(num_hd[d], ae_mse, yerr=ae_mse_std, fmt='D', alpha=.7, ls=':', label='Autoencoder', markersize=4.5, color='purple')
        
        ae_acc, ae_mse, ae_acc_std, ae_mse_std = [], [], [], []
        for hdz in range(3):
            filepath = os.path.join(RESULTS_PATH, f'AE_Online_data6_hdsz{num_hd[0][hdz]}.data')
            with open(filepath, 'rb') as filehandle: dta = pickle.load(filehandle)
            ae_mse.append(torch.mean(dta[2][:, -1])); ae_mse_std.append(torch.std(dta[2][:, -1]))
            ae_acc.append(torch.mean(dta[3][:, -1])); ae_acc_std.append(torch.std(dta[3][:, -1]))
        axs[0, 2].errorbar(num_hd[0], ae_acc, yerr=ae_acc_std, fmt='D', alpha=.7, ls=':', label='Autoencoder', markersize=4.5, color='purple')
        axs[1, 2].errorbar(num_hd[0], ae_mse, yerr=ae_mse_std, fmt='D', alpha=.7, ls=':', label='Autoencoder', markersize=4.5, color='purple')

        # Configuración final del gráfico
        for x in range(3): axs[1,x].set(xlabel='# Hid. Neurons')
        axs[0,0].set(title='EMNIST (OCI)'); axs[0,1].set(title='CIFAR-100 (OCI)'); axs[0,2].set(title='MNIST (ODI)')
        axs[0,2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        axs[0,0].set(ylabel='Final Acc.'); axs[1,0].set(ylabel='Final MSE')
        for x in range(3):
            axs[0,x].set(ylim=(-0.08, 1.05)); axs[1,x].set(ylim=(-0.01, .17))
            axs[0, x].xaxis.set_ticklabels([])
        for x in range(1,3):
            axs[0,x].yaxis.set_ticklabels([]); axs[1,x].yaxis.set_ticklabels([])
        pylab.tight_layout()
        
        # Lógica de guardado
        os.makedirs(PLOTS_PATH, exist_ok=True)
        filename = os.path.join(PLOTS_PATH, 'OnCont_L1_Final_Performance.png')
        pylab.savefig(filename, bbox_inches='tight')
        print(f" Gráfico de Rendimiento Final guardado en: {filename}")
        pylab.show()