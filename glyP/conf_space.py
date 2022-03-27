import os, sys
import numpy as np
from .conformer import *
from .utilities import *
import copy
import networkx as nx
from operator import itemgetter, attrgetter
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

class Space(list):

    """A conformational space consisting of all conformers found in specified directory.
    The directory tree should have a structure:
    'molecule'/*/*log
    if directory 'molecule' holds a directory 'experimental', an attibute self.expIR is 
    created using the data found there. 
    for different molecules, different lists can (meaning should!) be made.

    Generally constructing a conformer space is the first thing to do when using this package.
    A conformer space will generate a list of conformers. 
    """

    _temp = 298.15 #standard temperature Kelvin
    _kb=0.0019872036
    _kT=0.0019872036*_temp #boltzmann
    _Ha2kcal=627.5095  

    def __init__(self, path, load=True, software='g16'):

        self.path = path
        self.logfiles_dir = []
        try: os.makedirs(self.path)

        except: 
            if load == True: 
                print("{0:10s} directory already exists, load existing data".format(path))
                self.load_dir(path, topol=None, software=software)
            else: 
                pass

    def __str__(self):
         
        """Prints a nice table with coded molecular values
        """

        if hasattr(self[0], 'H'): 
            print ("%20s%20s%20s%20s" %('id', 'E [Ha]', 'H [Ha]', 'F [Ha]'))
            for conf in self:
                print ("%20s%20.8f%20.8f%20.8f" %(conf._id, conf.E, conf.H, conf.F))
        else:
            print ("%20s%20s" %('id', 'E [Ha]'))
            for conf in self:  print ("%20s%20.8f" %(conf._id, conf.E))


        return ''

    def __getitem__(self, select):
        """Finds a specific conformer in the space
        """
        if isinstance(select, int):
            rval = list.__getitem__(self, select)
            return rval

        elif isinstance(select, (list, tuple)):
            rval = [ list.__getitem__(self, i) for i in select ] 

        elif isinstance(select, slice):
            rval = super(Space, self).__getitem__(select)

        rret = Space(self.path, load = False)
        rret.__dict__ = self.__dict__.copy()
        for conf in rval: rret.append(conf)
        return rret

    def __add__(self, space):

        new_space = copy.deepcopy(self)
        for conf in space: 
            new_space.append(copy.deepcopy(conf))
        return new_space

    def __getslice__(self, i, j):
        """gets a set of conformers in the space
        """

        return self.__getitem(slice(i,j))

    def load_dir(self, path, topol=None, software='g16'):
        """Loads a directory with data files to be processed. The path is just the name of the directory, the function will handle the presence of multiple directories within it.

        :param path: (string) this is the path to the directory with all the conformer data. This directory should be filled with other directories with the intended name of the conformer and it's .xyz and input.log files

        """
        print("Loading {0:30s}".format(path))
        #check if this is wont lead to an error if the path doesnt exist 

        self.logfiles_dir.append(path)
        for (root, dirs, files) in os.walk('./'+path):
            for dirname in dirs:
                for ifiles in os.walk(path+'/'+dirname):
                    for filename in ifiles[2]:
                        if filename.endswith('.log'):
                            for line in open('/'.join([path, dirname, filename]), 'r').readlines()[-10:]:

                                if software == 'g16' and  re.search('Normal',  line):

                                    conf = Conformer(topol, '/'.join([path, dirname]))
                                    conf.load_log(software="g16")
                                    conf.connectivity_matrix(distXX=1.65, distXH=1.25)

                                    if conf.Nmols == 1:
                                        conf.assign_atoms() ; conf.measure_c6() ; conf.measure_glycosidic() ; conf.measure_ring()
                                        #print(conf)
                                        self.append(conf)
                                    else: 
                                        del conf

                                elif software == 'fhiaims' and re.search('Have a nice day.', line):

                                    conf = Conformer(topol, '/'.join([path, dirname]))
                                    conf.load_log(software="fhiaims")
                                    conf.connectivity_matrix(distXX=1.65, distXH=1.25)

                                    if conf.Nmols == 1:
                                        conf.assign_atoms() ; conf.measure_c6() ; conf.measure_glycosidic() ; conf.measure_ring()
                                        self.append(conf)
                                    else: 
                                        del conf


                        elif software == 'xyz' and filename.endswith('.xyz'): 

                            conf = Conformer(topol, '/'.join([path, dirname]))
                            conf.load_log(software="xyz")
                            conf.connectivity_matrix(distXX=1.65, distXH=1.25)

                            if conf.Nmols == 1:
                                conf.assign_atoms() ; conf.measure_c6() ; conf.measure_glycosidic() ; conf.measure_ring()
                                #print(conf)
                                self.append(conf)
                            else: 
                                del conf


    def load_exp(self, path, ir_resolution=1.0):
        """Selects the experimental conformer that other conformers will be compared to. Creates/updates the self.expIR member of this class

        :param path: (string) the path to a specific exp.dat file, the path should not just go to the general dir but also include the filename
        :param ir_resolution: (float) Resolution of the plot. The value is used to make a grid and the ir_resolution is the number of values skipped between each plotted point. Ex: assuming exp is an array size 10 exp = [1...10], and ir_resolution is 3 the values plotted will be [1,4,7]
        """
        self.ir_resolution = ir_resolution 
        expIR= np.genfromtxt(path)
        new_grid = np.arange(np.ceil(expIR[0,0]), np.floor(expIR[-1,0]), self.ir_resolution)
        self.expIR = np.vstack((new_grid, interpolate.griddata(expIR[:,0], expIR[:,1], new_grid, method='cubic'))).T #espec - experimental spectrum

    def dump_xyz(self):

        with open('/'.join([self.path, 'all_structures.xyz']),'w') as out: 

            for conf in self:
                out.write("{0:5d}\n".format(conf.NAtoms))
                out.write("{0:30s}\n".format(conf._id))
                for at in range(conf.NAtoms):
                    out.write("{0:3s}{1:12.3f}{2:12.3f}{3:12.3f}\n".format(conf.atoms[at], conf.xyz[at][0], conf.xyz[at][1], conf.xyz[at][2]))

    def load_models(self, path):
        """Loads a set of specific models used of analysis
        """
        self.models = []
        for (root, dirs, files) in os.walk('./'+path):
            for dirname in dirs:
                for ifiles in os.walk(path+'/'+dirname):
                    for filename in ifiles[2]:
                        if filename.endswith('.xyz'):
                            conf = Conformer(None, '/'.join([path, dirname]))
                            conf.load_log(software="xyz")
                            self.models.append(conf)

        self.Nmodels = len(self.models)

        print("Analyzing Models:", end="\n")

        for conf in self.models:

            print("{0:10s}:   ".format(conf._id), end='')
            conf.ring = [] ; conf.ring_angle = [] ; conf.dih_angle = []
            conf.connectivity_matrix(distXX=1.6, distXH=1.15)
            conf.assign_atoms() ; conf.measure_c6() ; conf.measure_ring() ; conf.measure_glycosidic()
            if hasattr(conf, 'graph'):
                for e in conf.graph.edges:
                    edge = conf.graph.edges[e]
                    print("{0:1d}->{1:1d}: {2:6s}".format(e[0], e[1], edge['linker_type']), end='')
            print('')

        if len(self) != 0:

            for conf in self: 
                conf.topol = 'unknown'
                #the iteration of the graphs edges can lead to bug because the order of the list is unknown. For the conformers to have the same shape the list must have the same content in the same order.
                conf_links =  [ conf.graph.edges[e]['linker_type'] for e in conf.graph.edges]
                for m in self.models:
                    m_links = [ m.graph.edges[e]['linker_type'] for e in m.graph.edges ]
                    mat = conf.conn_mat - m.conn_mat
                    if not np.any(mat) and conf_links == m_links and m.anomer == conf.anomer :
                        conf.topol = m.topol
                        break
                        #return 0
                    elif conf_links == m_links and m.anomer == conf.anomer: 
                        atc = 0 
                        acm = np.argwhere(np.abs(mat) == 1)
                        for at in acm:
                            if conf.atoms[at[0]] == 'H' or conf.atoms[at[1]] == 'H':
                                atc += 1 
                        if atc == len(acm):
                                conf.topol = m.topol+'_Hs'
                                break
                    
#if np.array_equal(conf.conn_mat, m.conn_mat) and conf_links == m_links : conf.topol = m.topol

    def load_Fmaps(self, path, Temp = None):

        """Loads a set of free energy maps for the glycosidic linkages located in path
           The directory name that contains eps file with the map should be the linkage.
        """

        def load_linkage(linkage_path, Temp):
           
            f = open(linkage_path, 'r') 
            pattern = None ; matrix_lett = []; matrix_dict = {}
            grid = None
            for line in f.readlines():

                if re.search(r'(\d\s\d\b)', line) and grid is None:
                    grid = int(line.split()[1])
                    pattern = grid * '[A-Z]'

                if pattern and re.search(pattern, line):
                    replL1 = line.replace('",','')
                    replL2 = line.replace('"','')
                    matrix_lett.append( '\n'.join(replL2.split()))

                if re.search('((".*")[0-9])', line): 
                    replL1 = line.replace('"', ' ') 
                    letters, energy = values = str(replL1.split()[0]) , float(replL1.split()[4])
                    matrix_dict.update(dict.fromkeys(letters,energy))

            matrix = np.zeros( [grid, grid ] )
            for i in range(grid):
                for j in range(grid):
                    matrix[i,j] = np.exp(-matrix_dict[matrix_lett[i][j]]/(self._kb*Temp))
            matrix = matrix/np.sum(matrix)
            #print(np.sum(matrix))
            return matrix

        if not Temp: Temp=self._temp

        self.linkages = {}

        for (root, dirs, files) in os.walk('./'+path):
            for dirname in dirs:
                for ifiles in os.walk(path+'/'+dirname):
                    for filename in ifiles[2]:
                        if filename.endswith('.xpm'):
                            self.linkages[dirname] = load_linkage('/'.join([path, dirname, filename]), Temp = Temp)


    def set_theory(self, software='g16', **kwargs):
        """Parameters for simulations
        """
        if software == 'g16':
            self.theory = { 'method': 'PBE1PBE', 
                        'basis_set':'6-31+G(d,p)', 
                        'jobtype':'opt freq', 
                        'disp': True, 
                        'other_options':'int=(grid=99590)', 
                        'charge':0, 
                        'multiplicity':1, 
                        'nprocs':24, 
                        'mem':'64GB',
                        'extra': None
                        }

        elif software == 'fhiaims':

            self.theory = { 
                        'exec': '/exports/apps/fhi-aims.210226/aims.210226.scalapack.mpi.x', 
                        'xc': 'pbe',
                        'basis_set':'light', 
                        'jobtype':"relax_geometry  trm 5E-3\n sc_accuracy_forces  5E-4\n output_level MD_light",
                        'disp': 'vdw_correction_hirshfeld', 
                        'convergence_options':
                                        "sc_accuracy_rho  1E-4\n sc_accuracy_eev  1E-3\n sc_accuracy_etot 1E-6\n sc_iter_limit    999", 
                        'charge': '0.0', 
                        'density_update_method': "orbital", 
                        'nprocs': 24, 
                        'check_cpu_consistency': ".false.",
                        'extra' : None
                        }

        for key in kwargs: 
            self.theory[key] = kwargs[key]



    def sort_energy(self, energy_function='E'):
        """Sorted the conformers according to selected energy_function
        """

        if energy_function == 'E':      self.sort(key = lambda x: x.E)
        elif energy_function == 'H':    self.sort(key = lambda x: x.H)
        elif energy_function == 'F':    self.sort(key = lambda x: x.F)

    def reference_to_zero(self, energy_function='E'):

        """Finds a conformer with the lowest specified energy function and 
        references remaining conformers to this.

        :param energy_function: (char) either E, F or H. The table will be sorted least to greatest in terms of the energy indicated by this parameter. 
        """

        Eref = 0.0 ; Fref = 0.0 ; Href = 0.0 

        if hasattr(self[0], 'F'):
            for conf in self:
                if energy_function == 'E' and  conf.E < Eref:
                    Eref = copy.copy(conf.E) ; Href = copy.copy(conf.H) ; Fref = copy.copy(conf.F)
                elif energy_function == 'H' and  conf.H < Href:
                    Eref = copy.copy(conf.E) ; Href = copy.copy(conf.H) ; Fref = copy.copy(conf.F)
                elif energy_function == 'F' and  conf.F < Fref:
                    Eref = copy.copy(conf.E) ; Href = copy.copy(conf.H) ; Fref = copy.copy(conf.F)
            for conf in self: 
                conf.Erel = conf.E -  Eref;  conf.Hrel = conf.H -  Href ;  conf.Frel = conf.F -  Fref
        else:
            for conf in self:
                if energy_function == 'E' and  conf.E < Eref:
                    Eref = copy.copy(conf.E)

            for conf in self: 
                conf.Erel = conf.E -  Eref

    def print_relative(self, alive=None, pucker=False, output = None):
        """Prints relative energies of each conformer, related to reference_to_zero
        """

        if output: 
            out = open('/'.join([self.path, output]),'w') 
        else:
            out = sys.stdout
 
        if len(self) != 0:
            try: hasattr(self[0], 'E') and not hasattr(self[0], 'Erel')
            except: 
                print("run reference_to_zero first")
                return None 
        else:
            return None 
        
        if   hasattr(self[0], 'Frel'):  out.write ("{0:>23s}{1:>16s}{2:>20s}{3:>8s}{4:>10s}{5:>10s}\n".format('id', 'topol',  'F-abs', 'E', 'H', 'F'))
        elif hasattr(self[0], 'Erel'):  out.write ("{0:>23s}{1:>16s}{2:>20s}{3:>8s}\n".format('id', 'topol',  'E-abs', 'E')) 
        else:                           out.write ("{0:>23s}{1:>16s}\n".format('id','topol'))
 
        for n, conf in enumerate(self): 
 
            if n == alive: out.write("---------------------------------")
 
            if hasattr(self[0], 'Frel'):
 
                try: hasattr(conf, 'Frel')
                except: continue
                out.writet("{0:3d}{1:>20s}{2:>16s}{3:20.8f}{4:8.2f}{5:8.2f}{6:8.2f}".format(n, conf._id, conf.topol, conf.F, conf.Erel*self._Ha2kcal, conf.Hrel*self._Ha2kcal, conf.Frel*self._Ha2kcal))
 
            elif hasattr(self[0], 'Erel'):
 
                try: hasattr(conf, 'Erel')
                except: continue

                out.write("{0:3d}{1:>20s}{2:>16s}{3:20.8f}{4:8.2f}".format(n, conf._id, conf.topol, conf.E, conf.Erel*self._Ha2kcal))
  
            else:
 
                out.write("{0:3d}{1:>20s}{2:>16s}".format(n, conf._id, conf.topol))
 
            if hasattr(conf, 'ccs'):
                out.write("{0:8.1f}".format(conf.ccs))
 
            if hasattr(conf, 'anomer'):
                out.write("{0:>5s} ".format(conf.anomer[0]))
 
            if hasattr(conf, 'graph'):
                for e in conf.graph.edges:
                    edge = conf.graph.edges[e]
                    out.write("{0:1d}->{1:1d}: {2:6s}".format(e[0], e[1], edge['linker_type']))
                if pucker == True:
                    for r in conf.graph.nodes:
                        ring = conf.graph.nodes[r]
                        out.write("{0:5s}{1:8.3f}{2:8.3f}{3:8.3f}".format(ring['ring'], *ring['pucker']))
            out.write('\n')



    def remove_duplicates(self, rmsd = 0.1):
        """Removes duplicate conformers from the space
        """
        to_be_removed = []
        for i, conf1  in enumerate(self):
            for j, conf2 in enumerate(self):
                if j <= i : continue
                if calculate_rmsd(conf1, conf2) < rmsd:
                    to_be_removed.append(j)

        to_be_removed.reverse() 
        for rem in to_be_removed:
            del self[rem]


    def rmsd_matrix(self, Hydrogen=True):


        atoms = set(self[0].atoms)
        if Hydrogen == False: 
            atoms.remove('H')
        print(atoms)
        self.rmsd_mat = np.zeros((len(self), len(self)))

        for i, conf1  in enumerate(self):
            for j, conf2 in enumerate(self):
                self.rmsd_mat[i, j]  = calculate_rmsd(conf1, conf2, atoms)

    def cluster(self, output="cluster", energy_function='E', E_cutoff=[100.00], Dist_cutoff=[1.0], plot=False):

        if len(E_cutoff) > len(Dist_cutoff):
            print("E_cutoff list longer than Dist_cutoff")
            return None

        if not hasattr(self, 'rmsd_mat'): self.rmsd_matrix(Hydrogen=False)

        Z = linkage(self.rmsd_mat, 'single', optimal_ordering=True)

        if plot == True: 
            fig = plt.figure(figsize=(25,10))
            dn = dendrogram (Z)
            fig.tight_layout()
            plt.savefig('/'.join([self.path, 'cluster.pdf']), dpi=300)

        unique_conf = Space(output, load=False)

        cluster_assign = []
        for d in Dist_cutoff: 
            cluster_assign.append(cut_tree(Z, height = d))

        for n, conf in enumerate(self):

            Erel = conf.Erel*self._Ha2kcal
            for ecut, dcut in zip(E_cutoff, cluster_assign):
                if (Erel < ecut) and ( dcut[n] not in dcut[:n] ):
                    unique_conf.append(copy.deepcopy(conf))
                    break 

        return unique_conf 

    def calculate_ccs(self, method = 'pa', accuracy = 1):
        """ Calculates the collision cross section for each conformer. The parameters passed should generally just remain as their defaults values

        :param methond: (string) pa or ehss, methods of calculation
        :param accuracy: dont change the default, return a value converged within 1%
        """
        for conf in self:  conf.calculate_ccs(method=method, accuracy=accuracy)

    def gaussian_broadening(self, broaden=3):
        """Performs gaussian broadening for the set
        """

        #checks if self.ir_resolution exists in the object, it would only exist if load_exp is called
        #works when no load_exp is called, need to test with load_exp
        if hasattr(self, 'self.ir_resolution'):
            for conf in self: conf.gaussian_broadening(broaden, resolution=self.ir_resolution)
        else:
            for conf in self: conf.gaussian_broadening(broaden, resolution=1)

    def create_connectivity_matrix(self, distXX=1.6, distXH=1.2): #1

        """Create a connectivity matrix as an attribute to the conf_space:
        distXX - cutoff distance between heavy atoms
        distXH - cutoff distance between heavy at - hydrogen
        """

        print('creating connectivity matrix')
        for conf in self: conf.connectivity_matrix(distXX, distXH)

    def assign_atoms(self): #2
        """Labels each atom for each conformer in the space
        """
        print('assigning atoms')
        for conf in self: conf.assign_atoms()

    def apply_rotation_matrix(self, index):
        """ Finds and applies the rotation matrix to each conformer being optimally rotated to match a conformer in the list of space with the index provided
        :param index: (int) the index in the list of conformers that each confromer will be rotated to match
        """
        #For each conformer store the rotation matrix and which conformer it is rotated to match
        conf_name = self[index]._id
        for conf in self:
            rotmat = rmsd.rmsd_qcp(self[0].xyz,conf.xyz,True)
            conf.rotation_operation(conf_name,index,rotmat)


    def assign_ring_puckers(self): #3
        """Identifies each ring of each conformer in the space
        """
        print('assigning rings')
        for conf in self: conf.measure_ring() 

    def assign_glycosidic_angles(self):
        """Calculates the dihedral angles for each conformer in the space
        """
        print('assigning dihs')
        for conf in self: conf.measure_glycosidic()


    def plot_ccs(self, output=None, energy_function='E', ccs_exp=141, xmin=130., xmax=150., ymin = -1., ymax=30., xlabel = 'CCS$^{PA}$ [$\AA{}^2$]', legend=True):

        """ Coformational free energy vs ccs, for every conformer
        """

        from matplotlib.ticker import NullFormatter, FormatStrFormatter

        #color = { 'LeA': '#a6cee3', 'LeX': '#1f78b4', 'BGH-1': '#b2df8a', 'BGH-2': '#33a02c', 'a16':'#fb9a99', 'b16':'#e31a1c', 'a14':'#fdbf6f' , 'b14': '#ff7f00', 'a13': '#cab2d6', 'b13': '#6a3d9a','a16n':'#ffff99','b16n':'#b15928', 'a12n': '#000000'}
        #marker ={ 'LeX': 'o',       'LeA': 'o'      , 'BGH-2': 'o'      , 'BGH-1':'o'      , 'a16': 'o',       'b16':'o'      , 'a14': 'o'      , 'b14': 'o'      , 'a13': 'o'      , 'b13': 'o'      ,'a16n':'o'      ,'b16n':'o'      , 'a12n': 'o'      }

        #color =  [ '#a6cee3', '#1f78b4', '#b2df8a','#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f' , '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99','#b15928']
        color = ['#1f78b4', '#e31a1c', '#33a02c']


        labels = [None] * len(self)
        for i, conf in enumerate(self):
            labels[i] = conf.topol
        labels = list(set(labels))
        labels.sort()

        Hs_label = []
        if 'unknown' in labels:  
            labels.remove('unknown') 
            Hs_label.append('unknown')

        for l in labels: 
            if l[-3:] == '_Hs' or l[-2:] == '_M': 
                labels.remove(l) 
                Hs_label.append(l)

        color = dict(zip(labels, color))
        for l in Hs_label: 
            color[l] = '#000000'

        print(color)

        #['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
        nullfmt = NullFormatter()         # no labels

        fig, ax = plt.subplots(1, figsize=(6,10))

        ax.plot([ccs_exp, ccs_exp], [ymin, ymax], 'k--')

        for conf in self:
            if energy_function == 'E':
                ax.scatter(conf.ccs, conf.Erel*self._Ha2kcal, s=20, color=color[conf.topol], marker='o', label=conf.topol)

            elif energy_function == 'F':
                ax.scatter(conf.ccs, conf.Frel*self._Ha2kcal, s=20, color=color[conf.topol], marker='o', label=conf.topol)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        if legend:  ax.legend(by_label.values(), by_label.keys(), fontsize=18)

        if   energy_function == 'E' : ylabel = '$\Delta$E PBE0+D3 [kcal/mol]'
        elif energy_function == 'F':  ylabel = '$\Delta$F PBE0+D3 [kcal/mol]'

        ax.set_ylabel(ylabel, fontsize=18) ; ax.set_xlabel(xlabel, fontsize=18)
        yaxis = np.linspace(ymin+1, ymax, 7)
        xaxis = np.linspace(xmin, xmax, 5)
        ax.set_xlim(xmin-2.5, xmax+2.5)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(xaxis); ax.set_yticks(yaxis)
        ax.set_xticklabels(xaxis, fontsize='16'); ax.set_yticklabels(yaxis, fontsize='16')
        ax.tick_params(axis='both', which='both', bottom=True, top=False, labelbottom=True, right=False, left=True, labelleft=True)
        for s in ['top', 'right', 'left', 'bottom']: ax.spines[s].set_visible(False)
        ax.xaxis.set_tick_params(direction='out')
        ax.yaxis.set_tick_params(direction='out')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.plot([xmin-2.5,xmax+2.5], [ymin+0.05, ymin+0.05], 'k', lw=1.5)
        ax.plot([xmin-2.5+0.1, xmin-2.5+0.1], [0,ymax], 'k', lw=1.5)
        fig.tight_layout()
        if output: 
            #fig.savefig('/'.join([output, 'ccs_plot.png']), dpi=200, transparent=True)
            fig.savefig('/'.join([output, 'ccs_plot.pdf']), dpi=200, transparent=True)
        else: 
            fig.savefig('/'.join([self.path, 'ccs_plot.pdf']), dpi=200, transparent=True)


