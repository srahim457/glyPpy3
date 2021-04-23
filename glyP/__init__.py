from .conformer import *
#from conformer import *
from .conf_space import *
#from conf_space import *
from .utilities import *
#from utilities import *
from .display_utilities import *
import numpy as np
import networkx as nx


def _main():
    
    print ("package initialized")
    A154 = Space('Tri_A154')
    A154.gaussian_broadening(broaden=5)
    A154.reference_to_zero(energy_function='F')
    print (A154)
    for conf in A154:  conf.plot_ir(plot_exp = True, exp_data = A154.expIR)
    A154.create_connectivity_matrix()
    A154.assign_pyranose_atoms()
    A154.assign_ring_puckers()

    #testing to see the ring conformations here
    #print(A154[0].ring)
    #print(A154[1].ring)
    
    #pendry factor tests
    #print(A154.expIR)
    #rfac(A154[1].IR,A154[0].IR)


    #If you want to test out the plotting stuff uncomment these line
    #return a 2D array of rmsd, rmsd no hydrogen and pendry
    #rmsd_all_atoms, rmsd_no_hydrogen, pendry_all = return_2d_lists(A154)
    #make_plots(A154)
    #generate_heatmap(rmsd_no_hydrogen)
    #generate_heatmap(pendry_all)

if __name__ == '__main__':

    _main()
