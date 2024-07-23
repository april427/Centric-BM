#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pickle


def plot_fig(filenames, ebno_dbs,list_markers, list_label, list_colour):
    n_files = len(filenames)
    n_f = 0
    while n_f !=n_files:
        for i in range(n_files):
            with open(filenames[i], 'rb') as f:
                f1 = pickle.load(f)
            plt.semilogy(ebno_dbs, f1['neural-receiver'], marker = list_markers[i],label=list_label[i], c=list_colour, linewidth = 2.0)
            n_f = n_f+1                     
            plt.xlabel(r"$E_b/N_0$ (dB)")
            plt.ylabel("BLER")           
            plt.ylim((1e-3, 1.0))
            plt.xlim((-10.0, 20.0))
            plt.legend()
            plt.tight_layout()
            plt.grid(which="both")
        
