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
    A154 = Space()
    A154.load_dir('Tri_A154')
    A154.gaussian_broadening(broaden=5)
    A154.reference_to_zero(energy_function='F')
    #print (A154)
    #for conf in A154:  conf.plot_ir2(plot_exp = False, exp_data = A154.expIR)
    A154.create_connectivity_matrix()
    A154.assign_pyranose_atoms()
    A154.assign_ring_puckers()

    #TEST: Dihedral calculation

    for i in range(3):
        print(A154[i]._id)
        print(A154[i].dih_atoms)

        for j in A154[i].dih_atoms:
            at1 = A154[i].xyz[j[0][0]]
            at2 = A154[i].xyz[j[0][1]]
            at3 = A154[i].xyz[j[0][2]]
            at4 = A154[i].xyz[j[0][3]]
            at5 = A154[i].xyz[j[0][4]]

            phi = True
            psi = True
            calculate_dihedral(at1,at2,at3,at4,at5,phi,psi)

if __name__ == '__main__':

    _main()
