import os, sys
import numpy as np
from .conformer import *
from .utilities import *
from copy import copy as cp
import networkx as nx
from operator import itemgetter, attrgetter

class Space(list):

    '''A conformational space consisting of all conformers found in specified directory.
    The directory tree should have a structure:
    'molecule'/*/*log
    if directory 'molecule' holds a directory 'experimental', an attibute self.expIR is 
    created using the data found there. 
    for different molecules, different lists can (meaning should!) be made.'''

    _temp = 298.15 #standard temperature Kelvin
    _kT=0.0019872036*_temp #boltzmann
    _Ha2kcal=627.5095  

    def __init__(self, path, load=True):

        self.path = path
        try: os.makedirs(self.path)
        except: 
            if load == True: 
                print("{0:10s} directory already exists, load existing data".format(path))
                self.load_dir(path, None)

    def __str__(self):
         
        '''Prints a nice table with coded molecular values'''

        if hasattr(self[0], 'H'): 
            print ("%20s%20s%20s%20s" %('id', 'E [Ha]', 'H [Ha]', 'F [Ha]'))
            for conf in self:
                print ("%20s%20.8f%20.8f%20.8f" %(conf._id, conf.E, conf.H, conf.F))
        else:
            print ("%20s%20s" %('id', 'E [Ha]'))
            for conf in self:  print ("%20s%20.8f" %(conf._id, conf.E))


        return ''

    def __getitem__(self, select):

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

    def __getslice__(self, i, j):

        return self.__getitem(slice(i,j))

    def load_dir(self, path, topol=None):

        print("Loading {0:30s}".format(path))
        for (root, dirs, files) in os.walk('./'+path):
            #print (root, dirs, files)
            for dirname in dirs:
                #print(dirname)
                for ifiles in os.walk(path+'/'+dirname):
                    for filename in ifiles[2]:
                        if filename.endswith('.log'):
                            for line in open('/'.join([path, dirname, filename]), 'r').readlines()[-10:]:
                                if re.search('Normal',  line):
                                    conf = Conformer(topol)
                                    conf.load_log(path+'/'+dirname+'/'+filename)
                                    conf.connectivity_matrix(distXX=1.65, distXH=1.25)
                                    conf.assign_atoms() ; conf.measure_c6() ; conf.measure_glycosidic() ; conf.measure_ring()
                                    self.append(conf)
                                    

    def load_exp(self, path, ir_resolution=1.0):

        self.ir_resolution = ir_resolution 
        expIR= np.genfromtxt(path)
        new_grid = np.arange(np.ceil(expIR[0,0]), np.floor(expIR[-1,0]), self.ir_resolution)
        self.expIR = np.vstack((new_grid, interpolate.griddata(expIR[:,0], expIR[:,1], new_grid, method='cubic'))).T #espec - experimental spectrum

    def load_models(self, path):

        self.models = []
        for (root, dirs, files) in os.walk('./'+path):
            for dirname in dirs:
                for ifiles in os.walk(path+'/'+dirname):
                    for filename in ifiles[2]:
                        if filename.endswith('.xyz'):
                            conf = Conformer(None)
                            conf.load_model(path+'/'+dirname+'/'+filename)
                            self.models.append(conf)
        self.Nmodels = len(self.models)

        for conf in self.models:
            print("Analyze {0:10s}".format(conf._id))
            conf.ring = [] ; conf.ring_angle = [] ; conf.dih_angle = []
 
            conf.connectivity_matrix(distXX=1.65, distXH=1.25)
            conf.assign_atoms() ; conf.measure_c6() ; conf.measure_ring() ; conf.measure_glycosidic()

        if len(self) != 0: 
            for conf in self: 
                conf.topol = 'unknown'
                conf_links = [ conf.graph.edges[e]['linker_type'] for e in conf.graph.edges]
                for m in self.models:
                    m_links = [ m.graph.edges[e]['linker_type'] for e in m.graph.edges ]
                    mat = conf.conn_mat - m.conn_mat
                    if not np.any(mat) and conf_links == m_links :
                        conf.topol = m.topol
                        break
                        #return 0
                    elif conf_links == m_links: 
                        atc = 0 
                        acm = np.argwhere(np.abs(mat) == 1)
                        for at in acm:
                            if conf.atoms[at[0]] == 'H' or conf.atoms[at[1]] == 'H':
                                atc += 1 
                        if atc == len(acm):
                                conf.topol = m.topol+'_Hs'
                                break
                    
#                    if np.array_equal(conf.conn_mat, m.conn_mat) and conf_links == m_links : conf.topol = m.topol



    def set_theory(self, **kwargs):

        self.theory = { 'method': 'PBE1PBE', 
                        'basis_set':'6-31+G(d,p)', 
                        'jobtype':'opt freq', 
                        'disp': True, 
                        'other_options':'int=(grid=99590)', 
                        'charge':0, 
                        'multiplicity':1, 
                        'nprocs':24, 
                        'mem':'64GB'
                        }

        for key in kwargs: 
            self.theory[key] = kwargs[key]

    def sort_energy(self, energy_function='E'):

        '''Sorted the conformers according to selected energy_function'''

        if energy_function == 'E':      self.sort(key = lambda x: x.E)
        elif energy_function == 'H':    self.sort(key = lambda x: x.H)
        elif energy_function == 'F':    self.sort(key = lambda x: x.F)

    def reference_to_zero(self, energy_function='E'):

        '''Finds a conformer with the lowest specified energy function and 
        references remainins conformers to this.'''

        Eref = 0.0 ; Fref = 0.0 ; Href = 0.0 

        if hasattr(self[0], 'F'):
            for conf in self:
                if energy_function == 'E' and  conf.E < Eref:
                    Eref = cp(conf.E) ; Href = cp(conf.H) ; Fref = cp(conf.F)
                elif energy_function == 'H' and  conf.H < Href:
                    Eref = cp(conf.E) ; Href = cp(conf.H) ; Fref = cp(conf.F)
                elif energy_function == 'F' and  conf.F < Fref:
                    Eref = cp(conf.E) ; Href = cp(conf.H) ; Fref = cp(conf.F)
            for conf in self: 
                conf.Erel = conf.E -  Eref;  conf.Hrel = conf.H -  Href ;  conf.Frel = conf.F -  Fref
        else:
            for conf in self:
                if energy_function == 'E' and  conf.E < Eref:
                    Eref = cp(conf.E)

            for conf in self: 
                conf.Erel = conf.E -  Eref

    def print_relative(self, alive=None):

        if len(self) != 0:
            try: hasattr(self[0], 'Erel')
            except: 
                print("run reference_to_zero first")
                return None 
        else:
            return None 
        if hasattr(self[0], 'Frel'): print ("%20s%10s%20s%8s%8s%8s" %('id', 'topol',  'F-abs', 'E', 'H', 'F'))
        else:  print ("%20s%10s%20s%8s" %('id', 'topol',  'E-abs', 'E'))

        for n, conf in enumerate(self): 
            if n == alive: print("---------------------------------")
            if hasattr(self[0], 'Frel'):
               print("%20s%10s%20.8f%8.2f%8.2f%8.2f" %(conf._id, conf.topol, conf.F, conf.Erel*self._Ha2kcal, conf.Hrel*self._Ha2kcal, conf.Frel*self._Ha2kcal), end='')
            else: 
               print("%20s%10s%20.8f%8.2f" %(conf._id, conf.topol, conf.E, conf.Erel*self._Ha2kcal), end='')

            if hasattr(self[0], 'ccs'):
                print("{0:8.1f}".format(conf.ccs), end='')
            if hasattr(self[0], 'anomer'):
                print("{0:>10s} ".format(conf.anomer[0]), end='')

            if hasattr(self[0], 'graph'):
                for e in conf.graph.edges:
                    edge = conf.graph.edges[e]
                    print("{0:1d}->{1:1d}: {2:6s}".format(e[0], e[1], edge['linker_type']), end='')
            print(' ')

            #else: 
                #print ("%20s%20s" %('id', 'E [kcal/mol]'))
                #for conf in self: print("%20s%20.2f" %(conf._id, conf.Erel*self._Ha2kcal))

        #return ''

    def calculate_ccs(self, method = 'pa', accuracy = 1):

        for conf in self:  conf.calculate_ccs(self.path, method=method, accuracy=accuracy)

    def gaussian_broadening(self, broaden=3):

        ''' Performs gaussian broadening for the set'''

        #checks if self.ir_resolution exists in the object, it would only exist if load_exp is called
        #works when no load_exp is called, need to test with load_exp
        if hasattr(self, 'self.ir_resolution'):
            for conf in self: conf.gaussian_broadening(broaden, resolution=self.ir_resolution)
        else:
            for conf in self: conf.gaussian_broadening(broaden, resolution=1)

    def create_connectivity_matrix(self, distXX=1.6, distXH=1.2): #1

        '''Create a connectivity matrix as an attribute to the conf_space:
        distXX - cutoff distance between heavy atoms
        distXH - cutoff distance between heavy at - hydrogen '''

        print('creating connectivity matrix')
        for conf in self: conf.connectivity_matrix(distXX, distXH)

    def assign_atoms(self): #2

        print('assigning atoms')
        for conf in self: conf.assign_atoms()


    def assign_ring_puckers(self): #3

        print('assigning rings')
        for conf in self: conf.measure_ring() 

    def assign_glycosidic_angles(self):

        print('assigning dihs')
        for conf in self: conf.measure_glycosidic()


    def plot_ccs(self, energy_function='E', ccs_exp=141, xmin=130., xmax=150., ymin = -1., ymax=30., xlabel = 'CCS$^{PA}$ [$\AA{}^2$]'):

        from matplotlib.ticker import NullFormatter, FormatStrFormatter
        #color = { 'LeA': '#a6cee3', 'LeX': '#1f78b4', 'BGH-1': '#b2df8a', 'BGH-2': '#33a02c', 'a16':'#fb9a99', 'b16':'#e31a1c', 'a14':'#fdbf6f' , 'b14': '#ff7f00', 'a13': '#cab2d6', 'b13': '#6a3d9a','a16n':'#ffff99','b16n':'#b15928', 'a12n': '#000000'}
        #marker ={ 'LeX': 'o',       'LeA': 'o'      , 'BGH-2': 'o'      , 'BGH-1':'o'      , 'a16': 'o',       'b16':'o'      , 'a14': 'o'      , 'b14': 'o'      , 'a13': 'o'      , 'b13': 'o'      ,'a16n':'o'      ,'b16n':'o'      , 'a12n': 'o'      }

        color =  [ '#a6cee3', '#1f78b4', '#b2df8a','#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f' , '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99','#b15928']
        labels = [None] * len(self)
        for i,conf in enumerate(self):
            labels[i] = conf.topol
        labels = list(set(labels))
        Hs_label = []
        if 'unknown' in labels:  
            labels.remove('unknown') 
            Hs_label.append('unknown')
        for l in labels: 
            if l[-3:] == '_Hs': 
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
        ax.legend(by_label.values(), by_label.keys(), fontsize=18)

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
        fig.savefig('/'.join([self.path, 'ccs_plot.png']), dpi=200, transparent=True)









