import argparse
import trainer
import plot

# Esta función 'get_parser' no se usa, pero la dejamos para no alterar la estructura original.
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--', type=int, default=0)

def main():
    parser = argparse.ArgumentParser()

    # Argumentos originales
    parser.add_argument('--test', type=str, required=False)
    parser.add_argument('--plot', type=str, required=False)

    # --- NUEVO ARGUMENTO ---
    # Añadimos un argumento para que el script pueda recibir la ruta de guardado
    # desde la línea de comandos de nuestro notebook.
    parser.add_argument('--drive_path', type=str, default=None, 
                        help='Ruta en Google Drive para guardar resultados inmediatamente.')
    # -----------------------

    args = parser.parse_args()

    # Ejecutar los Tests
    if args.test == 'assoc_comp':
        trainer.modCompareAA()
    
    # --- CAMBIO ---
    # Modificamos esta llamada para que pase la ruta de Drive al trainer.
    elif args.test == 'OnCont-L1':
        trainer.run_onCont_L1(drive_save_path=args.drive_path)
    # --------------

    elif args.test == 'OnCont-L3':
        trainer.run_onCont_L3() # Esta llamada permanece original (a menos que la modifiques en trainer.py)
    elif args.test == 'nsEncode-L1':
        trainer.nsEncode_L1()
    elif args.test == 'nsEncode-L3':
        trainer.nsEncode_L3()
    elif args.test == 'recog':
        trainer.recog()
    elif args.test == 'arch_compare':
        trainer.run_arch_compare()
    elif args.test is not None:
        assert False, 'Invalid test argument. Argument must be from list [assoc_comp, OnCont-L1, OnCont-L3, nsEncode-L1, ' \
                      'nsEncode-L3, recog, arch_compare]'


    # Select and Generate Plots
    if args.plot == 'OnCont-L1':
        plot.plot_sensit()
        plot.plot_cont()
        plot.plot_cont_cumul()
    elif args.plot == 'OnCont-L3':
        plot.plot_sensit_tree()
        plot.plot_cont_tree()
        plot.plot_cont_cumul_tree()
    elif args.plot == 'nsEncode-L1':
        plot.plot_noisy_online()
    elif args.plot == 'nsEncode-L3':
        plot.plot_noisyTree_online()
    elif args.plot == 'recog':
        plot.plot_recog_all()
    elif args.plot == 'arch_compare':
        plot.plot_tree_aa()
        plot.plot_accTest_tree()
        plot.plot_recognition_tree()
    elif args.plot is not None:
        assert False, 'Invalid test argument. Argument must be from list [assoc_comp, OnCont-L1, OnCont-L3, nsEncode-L1, ' \
                      'nsEncode-L3, recog, arch_compare]'


    if args.plot is None and args.test is None:
        assert False, 'No Arguments Inputted. Must input a test and/or a plot argument'

    return




if __name__ == "__main__":
    main()