
#  10/2018 - CP and MM / HC-CUNY
#  A class that creates an instance of a molecule defined as conformer.
#  It parses gaussian output file (optimization + freq at this moment to
#  set proper flags, to be fixed for only freq calcs) and creates an object with
#  following attibutes:
#  - self.Geom -  np.array with xyz of all atoms
#  - self.atoms - list of atomic numbers/int sorted as xyz
#  - self.[EHG/Ezpe] - float, respective energy function value
#  - self.Freq + self.Ints - np.array consisting either freqs or ints
#  - self.Vibs - 3D np.array with normal modes of all vibrations
#  - self.NAtoms - int with number of atoms
#  - self._ir    - identification/directory name


import numpy as np
import re
#from utilities import * 
from .utilities import *
import networkx as nx

class Conformer():

    def __init__(self, file_path):

        normal_mode_flag=False
        freq_flag = False
        read_geom = False

        #temprorary variables to hold the data
        freq = [] ; ints = [] ; vibs = [] ; geom = [] ; atoms = []

        self.NAtoms = None
        self._id    = str(file_path).split('/')[1]

        for line in open(file_path, 'r').readlines():                     

                if self.NAtoms == None and re.search('^ NAtoms=', line): 
                    self.NAtoms = int(line.split()[1])
                    self.NVibs  = self.NAtoms*3-6

                if re.search('^ Frequencies', line):        
                    freq_line = line.strip() 
                    for f in freq_line.split()[2:5]: freq.append(float(f))
                    normal_mode_flag = False

                elif re.search('^ IR Inten', line):        
                    ir_line = line.strip()                 
                    for i in ir_line.split()[3:6]: ints.append(float(i))

                elif re.search('^  Atom  AN', line): 
                     normal_mode_flag = True          #locating normal modes of a frequency
                     mode_1 = []; mode_2 = []; mode_3 = [];
                     continue

                elif normal_mode_flag == True and re.search('^\s*\d*\s*.\d*', line) and len(line.split()) > 3:
                     #mode_1.append(map(float, line.split()[2:5]))
                     #mode_2.append(map(float, line.split()[5:8]))
                     #mode_3.append(map(float, line.split()[8:11]))
                     #replaced the maps with list parseing
                     mode_1.append([float(x) for x in line.split()[2:5]])
                     mode_2.append([float(x) for x in line.split()[5:8]])
                     mode_3.append([float(x) for x in line.split()[8:11]])
                     

                elif normal_mode_flag == True: 
                     normal_mode_flag = False 
                     for m in [mode_1, mode_2, mode_3]: vibs.append(np.array(m))

                elif freq_flag == False and re.search('Normal termination', line): freq_flag = True

                elif freq_flag == True and re.search('SCF Done',   line): self.E = float(line.split()[4])
                elif freq_flag == True and re.search('Sum of electronic and zero-point Energies',   line): self.Ezpe = float(line.split()[6])
                elif freq_flag == True and re.search('Sum of electronic and thermal Enthalpies' ,   line): self.H    = float(line.split()[6])                    
                elif freq_flag == True and re.search('Sum of electronic and thermal Free Energies', line): self.F    = float(line.split()[7])

                elif freq_flag == True and re.search('Coordinates', line) : read_geom = True
                elif freq_flag == True and read_geom == True and re.search('^\s*.\d', line):
                     #geom.append(map(float, line.split()[3:6])) 
                     #convert to a parse directly into list rather than map
                     geom.append([float(x) for x in line.split()[3:6]])
                     atoms.append(element_symbol(line.split()[1]))
                     if int(line.split()[0]) == self.NAtoms:
                       read_geom = False
     
        self.Freq = np.array( freq ) 
        self.Ints = np.array( ints )
        self.Vibs=np.zeros((self.NVibs, self.NAtoms, 3))
        for i in range(self.NVibs): self.Vibs[i,:,:] = vibs[i]
        self.xyz = np.array(geom)
        self.atoms = atoms

        #making a tuple: EXPERIMENTAL
        #I'm combining the atoms and xyz into a tuple because each atom has a respective xyz coordinate. idk might be a more useful container than 2 separate lists
        list_combining_atoms_and_xyz =[]
        for i in range(len(self.xyz)):
            temp_tuple = (self.atoms[i],self.xyz[i])
            list_combining_atoms_and_xyz.append(temp_tuple)
        self.atoms_and_xyz = list_combining_atoms_and_xyz

    def __str__(self): 

       '''Prints a some molecular properties'''

       print ("%30s            NAtoms=%5d" %(self._id, self.NAtoms))
       print ("E=%20.4f H=%20.4f F=%20.4f" %( self.E, self.H, self.F))
       if hasattr(self, 'rings'):
           for n in range(len(self.ring)):
                print ("Ring%3d:%5s phi/psi%6.1f/%6.1f" %(n, self.ring[n], self.ring_angle[n][0], self.ring_angle[n][1]), end='')
                for at in ['C0', 'C1', 'C2', 'C3', 'C4', 'O' ]: 
                    print ("%3s:%3s" %(at, self.ring_atoms[n][at]), end='')
                print()

       return ' '

    def gaussian_broadening(self, broaden, resolution=1):
 
        ''' Performs gaussian broadening on IR spectrum:
        Args:
            broaden - gaussian broadening in wn-1
            resolution - resolution of the spectrum (number of points for 1 wn)
                         defaults is 1, needs to be fixed in plotting
        Returns:
            self.IR - np.array with dimmension 4000/resolution consisting
                      gaussian-boraden spectrum
        '''

        IR = np.zeros((int(4000/resolution) + 1,))
        X = np.linspace(0,4000, int(4000/resolution)+1)
        for f, i in zip(self.Freq, self.Ints):  IR += i*np.exp(-0.5*((X-f)/int(broaden))**2)
        self.IR=np.vstack((X, IR)).T #tspec
 

    def connectivity_matrix(self, distXX, distXH):

        Nat = self.NAtoms
        self.conn_mat = np.zeros((Nat, Nat))
        for at1 in range(Nat):
            for at2 in range(Nat):
                dist = get_distance(self.xyz[at1], self.xyz[at2])
                if at1 == at2: pass
                elif (self.atoms[at1] == 'H' or self.atoms[at2] == 'H') and dist < distXH: self.conn_mat[at1,at2] = 1; self.conn_mat[at2,at1] = 1
                elif dist < distXX: self.conn_mat[at1,at2] = 1; self.conn_mat[at2,at1] = 1

    def assign_ring_atoms(self):


        cm = nx.graph.Graph(self.conn_mat)
        cycles_in_graph = nx.cycle_basis(cm) #a cycle in the conn_mat would be a ring
        atom_names = self.atoms
        self.ring_atoms = []
        n = 0
        for r in cycles_in_graph:
            self.ring_atoms.append({}) #dictionary, probably atom desc
            # C4 and O
            rd = self.ring_atoms[n] # rd = ring dicitionary
            for at in r:
                if atom_names[at] == 'O':
                    rd['O'] = at #atom one of the rings
                else:
                    for at2 in np.where(self.conn_mat[at] == 1)[0]:
                        if atom_names[at2] == 'C' and at2 not in r:
                            rd['C4'] = at
            #
            for at in rd.values(): r.remove(at)
            for at in r:
                if self.conn_mat[at][rd['O']] == 1: rd['C0'] = at
                elif self.conn_mat[at][rd['C4']] == 1: rd['C3'] = at
            for at in [rd['C3'], rd['C0']]:  r.remove(at)
            for at in r:
                if self.conn_mat[at][rd['C0']] == 1: rd['C1'] = at
                elif self.conn_mat[at][rd['C3']] == 1: rd['C2'] = at
            for at in [rd['C2'], rd['C1']]:  r.remove(at)
            n += 1

    def create_ga_vector(self ):

        self.ga_vectorR = []
        for ring in self.ring_angle:
            self.ga_vectorR.append([ring[0], ring[1]])
        self.ga_vectorD = []
        

    def plot_ir(self, xmin = 800, xmax = 1800, scaling_factor = 0.965,  plot_exp = False, exp_data = None):

        ''' Plots the IR spectrum in xmin -- xmax range,
        x-axis is multiplied by scaling factor, everything
        is normalized to 1. If exp_data is specified, 
        then the top panel is getting plotted too. 
        Need to add output directory. Default name is self._id'''

        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter

        fig, ax = plt.subplots(1, figsize=(16, 4))

        ax.tick_params(axis='both', which='both', bottom=True, top=False, labelbottom=True, right=False, left=False, labelleft=False)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False);  ax.spines['left'].set_visible(False)
        ax.xaxis.set_tick_params(direction='out')
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.set_ylim(0,2) ; ax.set_xlim(xmin, xmax)

        xticks = np.linspace(xmin,xmax,int((xmax-xmin)/100)+1)
        ax.set_xticks(xticks[:-1])
        ax.set_xticklabels([int(x) for x in xticks[:-1]], fontsize=12)
        for t in xticks:
            ax.plot([t,t],[0,3], 'k--')

        shift=1
        incr = (self.IR[-1,0] - self.IR[0,0])/(len(self.IR)-1)

        scale_t  =  1/np.amax(self.IR[int(xmin/incr):int(xmax/incr)+100,1])
        Xsc = self.IR[:,0]* scaling_factor ; IRsc = self.IR[:,1]*scale_t

        ir_theo = ax.plot(Xsc, -IRsc+shift, color='0.25', linewidth=2)
        ax.fill_between(Xsc, np.linspace(shift, shift, len(IRsc)), -IRsc+1, color='0.5', alpha=0.5)
        ax.plot([xmin,xmax], [shift, shift], 'k', lw=2)
        for l in range(len(self.Freq)):
             ax.plot([scaling_factor*self.Freq[l], scaling_factor*self.Freq[l]], [shift, -self.Ints[l]*scale_t+shift], linewidth=2, color='0.25')

        if plot_exp == True:
            scale_exp=  1/np.amax(exp_data[:,1]) 
            ax.plot(exp_data[:,0], exp_data[:,1]*scale_exp+shift, color='r', alpha=0.5, linewidth=2)
            ax.fill_between(exp_data[:,0], exp_data[:,1]*scale_exp+shift, np.linspace(shift,shift, len(exp_data[:,1])), color='r', alpha=0.5)

        fig.tight_layout() 
        plt.savefig(self._id+'.png', dpi=200)




