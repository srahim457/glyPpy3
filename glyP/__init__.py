from .conformer import *
#from conformer import *
from .conf_space import *
#from conf_space import *
from .utilities import *
#from utilities import *
from .display_utilities import *
import numpy as np


def _main():
    
    print ("package initialized")
    A154 = Space('Tri_A154')
    A154.gaussian_broadening(broaden=5)
    A154.reference_to_zero(energy_function='F')
    #print (A154)
    #for conf in A154:  conf.plot_ir(plot_exp = True, exp_data = A154.expIR)
    A154.create_connectivity_matrix()
    A154.assign_pyranose_atoms()
    A154.assign_ring_puckers()

    #testing to see the ring conformations here
    #print(A154[0].ring)
    #print(A154[1].ring)

    #If you want to test out the plotting stuff uncomment these line
    #return a 2D array of rmsd, rmsd no hydrogen and pendry
    #rmsd_all_atoms, rmsd_no_hydrogen, pendry_all = return_2d_lists(A154)
    #make_plots(A154)
    #generate_heatmap(rmsd_no_hydrogen)
    #generate_heatmap(pendry_all)

    print(len(A154[1].IR))

    #split the array in half 0-1200 and then 1200-4000, the rest are 0's
    first_half=1201
    latter_half=len(A154[0].IR)-first_half

    
    #the first 1200 data points
    lower = A154[0].IR[:first_half]
    #print(rfac(A154[1].IR[:first_half],lower,start=900,stop=1200))
    #print(rfac(A154[1].IR, A154[0].IR,start=900,stop=1200))

    upper = A154[0].IR[first_half:]
    #print(rfac(A154[1].IR[first_half:],upper,start=1201,stop=1800))
    #print(rfac(A154[1].IR, A154[0].IR,start=1201,stop=1800))

    print(A154[1].IR[:first_half])
    print(A154[1].IR[first_half:])
    
    
    pendry_first_half=[]
    for d1 in range(len(A154)):
        pendry_first_half.append([])
        for d2 in range(len(A154)):
            pendry_first_half[d1].append(rfac(A154[d1].IR[:first_half],A154[d2].IR[:first_half],start=1000,stop=1200)) 
    print(pendry_first_half)
    generate_heatmap(pendry_first_half)

    pendry_other_half=[]
    for d1 in range(len(A154)):
        pendry_other_half.append([])
        for d2 in range(len(A154)):
            pendry_other_half[d1].append(rfac(A154[d1].IR,A154[d2].IR,start=1201,stop=1800)) 
    print(pendry_other_half)
    generate_heatmap(pendry_other_half)
    
    
    A154[0].plot_ir(plot_exp = True, exp_data = lower, scaling_factor = 1)
    A154[0].plot_ir(plot_exp = True, exp_data = upper, scaling_factor = 1)
    A154[0].plot_ir(plot_exp = True, exp_data = A154[0].IR, scaling_factor = 1)
    
    #I guess I dont really need this
    '''
    z = []
    for i in A154[0].IR[first_half:]:
        temp=[]
        temp.append(i[0])
        temp.append(0)
        z.append(temp)

    lower = np.concatenate([lower,z])
    print(A154[0].IR[:1500],len(A154[0].IR))
    print(lower[:1500],len(lower))
    A154[0].plot_ir(plot_exp = True, exp_data = A154[0].IR)
    '''
    



if __name__ == '__main__':

    _main()
