from .conformer import *
from .conf_space import *
from .utilities import *
from .display_utilities import *
from .ga_operations import *
import numpy as np
import networkx as nx
import copy


def _main():
    
    print ("package initialized")
    #A154 = Space()
    #A154.load_dir('Tri_A154')
    #A154.gaussian_broadening(broaden=5)
    #A154.reference_to_zero(energy_function='F')
    #print (A154)
    #for conf in A154:  conf.plot_ir2(plot_exp = False, exp_data = A154.expIR)
    #A154.create_connectivity_matrix()
    #A154.assign_pyranose_atoms()
    #A154.assign_ring_puckers()


    n=0 ; initial_pool  = 10 

    GArun = Space('GA-test')
    GArun.load_models('models')
    GArun.set_theory(nprocs=12, mem='16GB', charge=1, basis_set='STO-3G')

    while n < initial_pool: 
        print("generate structure {0:5d}".format(n))

        m = draw_random_int(len(GArun.models))
        GArun.append(copy.deepcopy(GArun.models[m]))
        GArun[n]._id = "initial-{0:2s}".format(n)

        for d in range(len(GArun[n].dih_atoms)):
             phi, psi = ((draw_random()*360)-180.0, (draw_random()*360)-180.0)
             GArun[n].set_glycosidic(d, phi, psi)

        GArun[n].measure_glycosidic()
        GArun[n].update_vector()
        print( GArun[n].ga_vector ) 
        GArun[n].create_input(GArun.theory, GArun.path)
        GArun[n].run_gaussian()
        n += 1 

if __name__ == '__main__':

    _main()
