
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
import re, os
from subprocess import Popen, PIPE
from .utilities import *
import networkx as nx
from operator import itemgetter, attrgetter

class Conformer():

    def __init__(self, _id):

        self._id = _id 

    def load_model(self, file_path):

        self.NAtoms = None
        self._id    = str(file_path).split('/')[1]
        geom = [] ; atoms = []

        for n, line in enumerate(open(file_path, 'r').readlines()):

            if n == 0 and self.NAtoms == None: self.NAtoms = int(line)
            if n > 1:
                 if len(line.split()) == 0: break 
                 geom.append([float(x) for x in line.split()[1:4]])
                 #atoms.append(element_symbol(line.split()[0]))
                 atoms.append(line.split()[0])

        self.xyz = np.array(geom)
        self.atoms = atoms

    def create_input(self, theory, output):

        if theory['disp'] == True or theory['disp'] == 'EmpiricalDispersion=GD3':
            theory['disp'] = 'EmpiricalDispersion=GD3'
        else: 
            theory['disp'] = ' '

        outdir = '/'.join([output, self._id])
        input_file = outdir + '/input.com'
        self.outdir = outdir
        try:
            os.makedirs(outdir)
        except: 
            os.remove(input_file) 
            os.remove(outdir+'/input.log') 

        f = open(input_file, 'w')
        f.write('%nproc=' + str(theory['nprocs'])+'\n')
        f.write('%mem='+theory['mem']+'\n')
        f.write(' '.join(['#P', theory['method'], theory['basis_set'],  theory['jobtype'], theory['other_options'], theory['disp'], '\n']))
        f.write('\n')
        f.write(self._id + '\n')
        f.write('\n ')
        f.write(str(theory['charge']) + ' ' + str(theory['multiplicity']) + '\n')
        for at, xyz in zip(self.atoms, self.xyz):
            line = '{0:5s} {1:10.3f} {2:10.3f} {3:10.3f}\n'.format(at, xyz[0], xyz[1], xyz[2])
            f.write(line)
        f.write(' ')
        f.close()

    def run_gaussian(self, mpi=False):

        try: hasattr(self, 'outdir')
        except:
            print("Create input first")
            sys.exit(1)

        cwd=os.getcwd(); os.chdir(self.outdir)

        with open('input.log', 'w') as out:

            if mpi == True: 
                gauss_job = Popen("mpiexec -n " + str(theory['nprocs']) + "g16 input.com ", shell=True, stdout=out, stderr=out)
            elif mpi == False: 
                gauss_job = Popen("g16 input.com ", shell=True, stdout=out, stderr=out)

            gauss_job.wait()

        os.chdir(cwd)
        return gauss_job.returncode

    def load_log(self, file_path):

        #try:
        #    logfile = open(file_path, 'r')
        #except IOError: 
        #    print("%30s not accessible", file_path)
        #    return 1 
            
        normal_mode_flag=False
        freq_flag = False
        read_geom = False

        #temprorary variables to hold the data
        freq = [] ; ints = [] ; vibs = [] ; geom = [] ; atoms = []

        job_opt = False ; job_freq = False ; job_optfreq = False ; job_sp = False ; job = 0

        self.NAtoms = None
        self._id    = str(file_path).split('/')[1]

        for line in open(file_path, 'r').readlines():

                if re.search('^ #', line) and job == 0:
                    if "opt" in line:

                        if "freq" in line: 
                            job_optfreq = True ; job += 1 
                            #print("Reading optfreq")
                        else: 
                            job_opt = True ; job += 1 
                            #print("Reading opt")
                    elif "freq" in line: 
                            job_optfreq = True ; freq_flag = True ;  job += 1 
                            #print("Reading freq")
                    else: job_sp = True ; job += 1 

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

                     mode_1.append([float(x) for x in line.split()[2:5]])
                     mode_2.append([float(x) for x in line.split()[5:8]])
                     mode_3.append([float(x) for x in line.split()[8:11]])

                elif normal_mode_flag == True: 

                     normal_mode_flag = False 
                     for m in [mode_1, mode_2, mode_3]: vibs.append(np.array(m))


                if job_optfreq == True:

                    if freq_flag == False and re.search('Normal termination', line): freq_flag = True

                    elif freq_flag == True and re.search('SCF Done',   line): self.E = float(line.split()[4]) 
                    elif freq_flag == True and re.search('Sum of electronic and zero-point Energies',   line): self.Ezpe = float(line.split()[6])
                    elif freq_flag == True and re.search('Sum of electronic and thermal Enthalpies' ,   line): self.H    = float(line.split()[6])                    
                    elif freq_flag == True and re.search('Sum of electronic and thermal Free Energies', line): self.F    = float(line.split()[7])

                    elif freq_flag == True and re.search('Coordinates', line): 
                        if len(geom) == 0: read_geom = True
                    elif freq_flag == True and read_geom == True and re.search('^\s*.\d', line):
                        #geom.append(map(float, line.split()[3:6])) 
                        #convert to a parse directly into list rather than map
                        geom.append([float(x) for x in line.split()[3:6]])
                        atoms.append(element_symbol(line.split()[1]))
                        if int(line.split()[0]) == self.NAtoms:
                           read_geom = False

                elif job_opt == True: 

                    if re.search('SCF Done',   line): E = float(line.split()[4])
                    if re.search('Optimization completed.', line): 
                        self.E = E ; freq_Flag = True
                    elif freq_flag == True and re.search('Coordinates', line) : read_geom = True
                    elif freq_flag == True and read_geom == True and re.search('^\s*.\d', line):
                         #geom.append(map(float, line.split()[3:6])) 
                         #convert to a parse directly into list rather than map
                         geom.append([float(x) for x in line.split()[3:6]])
                         atoms.append(element_symbol(line.split()[1]))
                         if int(line.split()[0]) == self.NAtoms:
                           read_geom = False

                elif job_sp == True:

                    print("No idea what you're dong")

        self.Freq = np.array( freq ) 
        self.Ints = np.array( ints )
        self.Vibs=np.zeros((self.NVibs, self.NAtoms, 3))
        for i in range(self.NVibs): self.Vibs[i,:,:] = vibs[i]
        self.xyz = np.array(geom)
        self.atoms = atoms

    def __str__(self): 

        '''Prints a some molecular properties'''

        print ("%20s    NAtoms=%5d" %(self._id, self.NAtoms))
        if hasattr(self, 'E'):  print ("E=%20.4f H=%20.4f F=%20.4f" %( self.E, self.H, self.F))
        for n  in self.graph.nodes:
            ring = self.graph.nodes[n]
            print ("Ring    {0:3d}:  {1:6s} {2:6.1f} {3:6.1f}".format(n, ring['ring'], ring['CP'][0], ring['CP'][1]), end='\n')
        for e in self.graph.edges:
            edge = self.graph.edges[e]

            if len(edge['dihedral']) == 2: 
                print ("Link {0}:  {1:6s} {2:6.1f} {3:6.1f}".format(e, edge['linker_type'], edge['dihedral'][0], edge['dihedral'][1]), end='\n' )

            elif len(edge['dihedral']) == 3: 
                print ("Link {0}:  {1:6s} {2:6.1f} {3:6.1f} {4:6.1f}".format(e, edge['linker_type'], edge['dihedral'][0], edge['dihedral'][1], edge['dihedral'][2]), end='\n')

            elif len(edge['dihedral']) == 4: 
                print ("Link {0}:  {1:6s} {2:6.1f} {3:6.1f} {4:6.1f} {5:6.1f}".format(e, edge['linker_type'], edge['dihedral'][0], edge['dihedral'][1], edge['dihedral'][2], edge['dihedral'][3]), end='\n')

                #for at in ['C1', 'C2', 'C3', 'C4', 'C5', 'O' ]: 
                #    print ("%3s:%3s" %(at, self.ring_atoms[n][at]), end='')
                #print()

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
                elif (self.atoms[at1] == 'H' or self.atoms[at2] == 'H'):
                    if dist < distXH: self.conn_mat[at1,at2] = 1; self.conn_mat[at2,at1] = 1
                elif dist < distXX: self.conn_mat[at1,at2] = 1; self.conn_mat[at2,at1] = 1

    def assign_atoms(self):

        self.graph = nx.DiGraph()

        #Assign ring atoms: 
        cm = nx.graph.Graph(self.conn_mat)
        cycles_in_graph = nx.cycle_basis(cm) #a cycle in the conn_mat would be a ring
        atom_names = self.atoms
        ring_atoms = []
        n = 0
        for r in cycles_in_graph:
            ring_atoms.append({}) #dictionary, probably atom desc
            # C5 and O
            rd = ring_atoms[n] # rd = ring dicitionary
            for at in r:
                if atom_names[at] == 'O':
                    rd['O'] = at #atom one of the rings
                else:
                    for at2 in np.where(self.conn_mat[at] == 1)[0]:
                        if atom_names[at2] == 'C' and at2 not in r:
                            rd['C5'] = at
            #
            for at in rd.values(): r.remove(at)
            for at in r:
                if self.conn_mat[at][rd['O']] == 1: rd['C1'] = at
                elif self.conn_mat[at][rd['C5']] == 1: rd['C4'] = at
            for at in [rd['C4'], rd['C1']]:  r.remove(at)
            for at in r:
                if self.conn_mat[at][rd['C1']] == 1: rd['C2'] = at
                elif self.conn_mat[at][rd['C4']] == 1: rd['C3'] = at
            for at in [rd['C3'], rd['C2']]:  r.remove(at)
            n += 1

        #Find reduncing end and sort the list:

        C1s = [ x['C1'] for x in ring_atoms]; C1pos = []
        for n, C1 in enumerate(C1s): 
            NRed=0; NNon=0 #NRed = reducing end NNon = non reducing end
            for C12 in C1s:
                path = nx.shortest_path(cm, C1, C12)
                if len(path) == 1: continue
                elif path[1] in ring_atoms[n].values(): NNon+=1
                else: NRed += 1
            C1pos.append(NRed)
        ring_atoms = [ i[0] for i in sorted(zip(ring_atoms, C1pos), key=itemgetter(1)) ]

        for n, i in enumerate(ring_atoms): 
            self.graph.add_node(n, ring_atoms = i)

        #print(self.graph.nodes)
        #print(self.graph.number_of_nodes())
        #identify bonds:
        #1. Create shortest paths between anomeric carbons to get O's and bond types.
        
        #C1s = [ x['C1'] for x in ring_atoms] #Sorted list of C1s, first C1 is reducing end. 
        #self.dih_atoms = [] ; self.dih = [] 

        #for at in range(len(C1s)-1):
        #
        #    self.dih_atoms.append({})
        #    at1 = at+1 
        #    path = nx.shortest_path(cm, C1s[at], C1s[at1])
        #    n=2 ; 
        #    self.dih_atoms[at]['C1l'] = C1s[at1]
        #    self.dih_atoms[at]['O']  = self.ring_atoms[at1]['O']
        #    #print(path)
        #    while n < len(path):
        #        if path[-n] in self.ring_atoms[at].values(): 
        #            linker_type = (list(self.ring_atoms[at].keys())[list(self.ring_atoms[at].values()).index(path[-n])])[-1]
        #            #if linker_type == '5': linker_type = '6' 
        #            self.dih_atoms[at]['C'+linker_type+'l'] = path[-n]
        #            C_phi = 'C'+str(int(linker_type)-1)
        #            #if linker_type == 5: linker_type += 1 
        #            self.dih_atoms[at][C_phi] = self.ring_atoms[at][C_phi]
        #            break
        #        else: 
        #            if n == 2: 
        #                self.dih_atoms[at]['Ol']  = path[-n]
        #            elif n == 3: 
        #                self.dih_atoms[at]['C6']  = path[-n]
        #        n=n+1

        #    dih = self.dih_atoms[at] 
        #    adj = adjacent_atoms(self.conn_mat, dih['C1l']) 
        #    for at in adj:
        #        if self.atoms[at] == 'H': 
        #            list_of_atoms = [ dih['O'], dih['C1l'], dih['Ol'], at ] 
        #    idih = measure_dihedral( self, list_of_atoms )[0] 
        #    if linker_type == '5': linker_type = '6'
        #    if idih < 0.0: self.dih.append('b1'+linker_type)
        #    else: self.dih.append('a1'+linker_type)

        C1s = [ x['C1'] for x in ring_atoms] #Sorted list of C1s, first C1 is reducing end. 
        cycles_in_graph = nx.cycle_basis(cm) #a cycle in the conn_mat would be a ring
        for r1 in range(self.graph.number_of_nodes()):
            for r2 in range(self.graph.number_of_nodes()):
                linker_atoms = [] ; linked = False
                if r1 == r2 : pass
                else:
                    path = nx.shortest_path(cm, self.graph.nodes[r1]['ring_atoms']['C1'], self.graph.nodes[r2]['ring_atoms']['C1'])
                    n = 1 ; term = False
                    while n < len(path):
                        at = path[-n]
                        #Check wheter path[n] is inside a cycle
                        c = 0 
                        for cycle in cycles_in_graph:
                            if at in cycle: 
                                if at in self.graph.nodes[r2]['ring_atoms'].values():
                                    linker_atoms.append(self.graph.nodes[r2]['ring_atoms']['O'])
                                    linker_atoms.append(at)
                                    n += 1 ; break 
                                elif at in self.graph.nodes[r1]['ring_atoms'].values():
                                    linker_atoms.append(at)
                                    linked = True ; term = True  
                                    linker_type = (list(self.graph.nodes[r1]['ring_atoms'].keys())[list(self.graph.nodes[r1]['ring_atoms'].values()).index(at)])[-1]
                                    C_psi = 'C'+str(int(linker_type)-1)
                                    linker_atoms.append(self.graph.nodes[r1]['ring_atoms'][C_psi])
                                    break
                                else:
                                    term = True ; break
                            else: c += 1 
                        if c == 3: 
                            linker_atoms.append(at) 
                            n += 1
                        if term == True: break
                    #print(linker_type)

                    if linked == True:
                        adj = adjacent_atoms(self.conn_mat, linker_atoms[1])
                        for at in adj:
                            if self.atoms[at] == 'H':
                                list_of_atoms = linker_atoms[:3] + [at]
                        idih = measure_dihedral( self, list_of_atoms )[0]
                        if linker_type == '5': linker_type = '6'
                        if self.atoms[linker_atoms[4]] == 'N':
                            linker_type += 'N' 
                        if idih < 0.0: linkage = 'b1'+linker_type
                        else: linkage = 'a1'+linker_type
                        self.graph.add_edge(r1, r2, linker_atoms = linker_atoms, linker_type = linkage ) 

        #determine anomaricity of the redicing end: 
        adj = adjacent_atoms(self.conn_mat, self.graph.nodes[0]['ring_atoms']['C1'])
        for at in adj:
            if self.atoms[at] == 'H': Ha = at
            if self.atoms[at] == 'O' and at not in self.graph.nodes[0]['ring_atoms'].values(): O = at
        list_of_atoms = [ self.graph.nodes[0]['ring_atoms']['O'], self.graph.nodes[0]['ring_atoms']['C1'], O, Ha] 
        idih = measure_dihedral( self, list_of_atoms )[0]
        if idih < 0.0: self.anomer = 'beta'
        else: self.anomer = 'alpha'

        #print (self.dih_atoms, self.dih, self.anomer)

    def measure_glycosidic(self):

        for e in self.graph.edges:
            atoms = self.graph.edges[e]['linker_atoms']
            phi, ax = measure_dihedral(self, atoms[:4])
            psi, ax = measure_dihedral(self, atoms[1:5])

            if len(atoms) == 6: #1-6 linkage
                omega, ax = measure_dihedral(self, atoms[2:6])
                self.graph.edges[e]['dihedral'] = [phi, psi, omega]

            elif len(atoms) == 7: # linkage at NAc
                omega, ax = measure_dihedral(self, atoms[2:6])
                gamma, ax = measure_dihedral(self, atoms[3:7])
                self.graph.edges[e]['dihedral'] = [phi, psi, omega, gamma]

            else: self.graph.edges[e]['dihedral'] = [phi, psi]

    def set_glycosidic(self, bond, phi, psi, omega=None, gamma=None):

        #atoms = sort_linkage_atoms(self.dih_atoms[bond])
        atoms = self.graph.edges[bond]['linker_atoms']
        set_dihedral(self, atoms[:4], phi)
        set_dihedral(self, atoms[1:5], psi)

        if omega != None: 
            set_dihedral(self, atoms[2:6], omega)
        if gamma != None: 
            set_dihedral(self, atoms[3:7], gamma)

        self.measure_glycosidic()

    def measure_ring(self):

        for n in self.graph.nodes:
            atoms = self.graph.nodes[n]['ring_atoms']
            phi, psi, R = calculate_ring(self.xyz, atoms)
            self.graph.nodes[n]['ring'] = R; self.graph.nodes[n]['CP'] = [phi, psi]

    def update_vector(self):

        self.ga_vector = []
        for e in self.graph.edges: self.ga_vector.append(self.graph.edges[e]['dihedral'])
        for n in self.graph.nodes: self.ga_vector.append(self.graph.nodes[n]['CP'])

    def show_xyz(self, width=400, height=400):

        import py3Dmol as p3D

        XYZ = "84\n{0:s}\n".format(self._id)
        for at, xyz in zip(self.atoms, self.xyz):
            XYZ += "{0:3s}{1:10.3f}{2:10.3f}{3:10.3f}\n".format(at, xyz[0], xyz[1], xyz[2] )
        xyzview = p3D.view(width=width,height=height)
        xyzview.addModel(XYZ,'xyz')
        xyzview.setStyle({'stick':{}})
        xyzview.zoomTo()
        xyzview.show()

    def plot_ir(self, xmin = 900, xmax = 1700, scaling_factor = 0.965,  plot_exp = False, exp_data = None):

        ''' Plots the IR spectrum in xmin -- xmax range,
        x-axis is multiplied by scaling factor, everything
        is normalized to 1. If exp_data is specified, 
        then the top panel is getting plotted too. 
        Need to add output directory. Default name is self._id'''

        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter

        fig, ax = plt.subplots(1, figsize=(8, 2))

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


    def plot_ir2(self,  xmin = 900, xmax = 1700, scaling_factor = 0.965,  plot_exp = False, exp_data = None, exp_int_split=False, normal_modes=False):

        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter

        fig, ax = plt.subplots(1, figsize=(8,2))

        #left, width = 0.02, 0.98 ; bottom, height = 0.15, 0.8
        #ax  = [left, bottom, width, height ]
        #ax  = plt.axes(ax)
        exten = 20

        ax.tick_params(axis='both', which='both', bottom=True, top=False, labelbottom=True, right=False, left=False, labelleft=False)
        ax.spines['top'].set_visible(False) ; ax.spines['right'].set_visible(False) ; ax.spines['left'].set_visible(False)
        ax.xaxis.set_tick_params(direction='out')
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.set_ylim(0,1.1)

        xticks = np.linspace(xmin,xmax,int((xmax-xmin+2*exten)/100)+1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([int(x) for x in xticks], fontsize=10)
        ax.set_xlim(xmin-exten, xmax+exten+10)


       #for t in xticks:
       #    ax.plot([t,t],[0,3], 'k--')

        shift = 0.05 ;         incr = (self.IR[-1,0] - self.IR[0,0])/(len(self.IR)-1)
        scale_t  =  1/np.amax(self.IR[int(xmin/incr):int(xmax/incr)+100,1])

        if plot_exp == True:
            if exp_int_split == False:  
                scale_exp=  1/np.amax(exp_data[:,1])
                ax.plot(exp_data[:,0], exp_data[:,1]*scale_exp+shift, color='r', alpha=0.5, linewidth=2)
                ax.fill_between(exp_data[:,0], exp_data[:,1]*scale_exp+shift, np.linspace(shift,shift, len(exp_data[:,1])), color='r', alpha=0.5)

            else:
                scale_expL=  1/np.amax(exp_data[:,1])
                scale_expH= scale_t * np.amax(self.IR[int(1200/incr):int(xmax/incr)+100,1]) /(np.amax(np.where(exp_data[:,0] > 1200, 0, exp_data[:,1])))
                split_wn = np.where(exp_data[:,0] == 1200) ; split_wn = split_wn[0][0]
                ax.plot(exp_data[:split_wn,0], exp_data[:split_wn,1]*scale_expL+shift, color='r', alpha=0.75, linewidth=2)
                ax.fill_between(exp_data[:split_wn,0], exp_data[:split_wn,1]*scale_expL+shift, np.linspace(shift,shift, len(exp_data[:split_wn,1])), color='r', alpha=0.5)

                ax.plot(exp_data[split_wn:,0], exp_data[split_wn:,1]*scale_expH+shift, color='r', alpha=0.75, linewidth=2)
                ax.fill_between(exp_data[split_wn:,0], exp_data[split_wn:,1]*scale_expH+shift, np.linspace(shift,shift, len(exp_data[split_wn:,1])), color='r', alpha=0.5)

        Xsc = self.IR[:,0]* scaling_factor ; IRsc = self.IR[:,1]*scale_t
        ir_theo = ax.plot(Xsc, IRsc+shift, color='0.25', linewidth=2)
        ax.fill_between(Xsc, np.linspace(shift, shift, len(IRsc)), IRsc+shift, color='0.5', alpha=0.5)


        if normal_modes == True:
            for l in range(len(self.Freq)):
                 ax.plot([scaling_factor*self.Freq[l], scaling_factor*self.Freq[l]], [shift, self.Ints[l]*scale_t+shift], linewidth=2, color='0.25')        

        fig.tight_layout()
        plt.savefig(self._id+'.png', dpi=200)
           
