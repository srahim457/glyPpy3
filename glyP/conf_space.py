import os, sys
import numpy as np
#from conformer import *
from .conformer import *
#from utilities import *
from .utilities import *
from copy import copy as cp

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

    def __init__(self):

        pass


    def __str__(self):
         
        '''Prints a nice table with coded molecular values'''

        print ("%20s%20s%20s%20s" %('id', 'E', 'H', 'F'))
        for conf in self: 
            print ("%20s%20.2f%20.2f%20.2f" %(conf._id, conf.E*self._Ha2kcal, conf.H*self._Ha2kcal, conf.F*self._Ha2kcal))
        return ''

    def load_dir(self, molecule):

        for (root, dirs, files) in os.walk('./'+molecule):
            for dirname in dirs:
                print (dirname)
                for ifiles in os.walk(molecule+'/'+dirname):
                    for filename in ifiles[2]:
                        if filename.endswith('.log'):
                            conf = Conformer('dummy')
                            conf.load_log(molecule+'/'+dirname+'/'+filename)
                            self.append(conf)

    def load_exp(self, path, ir_resolution=1.0):

        self.ir_resolution = ir_resolution 
        expIR= np.genfromtxt(path)
        new_grid = np.arange(np.ceil(expIR[0,0]), np.floor(expIR[-1,0]), self.ir_resolution)
        self.expIR = np.vstack((new_grid, interpolate.griddata(expIR[:,0], expIR[:,1], new_grid, method='cubic'))).T #espec - experimental spectrum

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

    def gaussian_broadening(self, broaden=1):

        ''' Performs gaussian broadening for the set''' 

        #for conf in self: conf.gaussian_broadening(broaden, resolution=self.ir_resolution)
        for conf in self: conf.gaussian_broadening(broaden, resolution=1)
                   
    def reference_to_zero(self, energy_function='E'):

        '''Finds a conformer with the lowest specified energy function and 
        references remainins conformers to this.'''

        Eref = 0.0 ; Fref = 0.0 ; Href = 0.0 
        for conf in self: 
              if energy_function == 'E' and  conf.E < Eref: 
                    Eref = cp(conf.E) ; Href = cp(conf.H) ; Fref = cp(conf.F)
              elif energy_function == 'H' and  conf.H < Href: 
                    Eref = cp(conf.E) ; Href = cp(conf.H) ; Fref = cp(conf.F)
              elif energy_function == 'F' and  conf.F < Fref: 
                    Eref = cp(conf.E) ; Href = cp(conf.H) ; Fref = cp(conf.F)
        for conf in self: 
              conf.E -= Eref;  conf.H -= Href ;  conf.F -= Fref

    def create_connectivity_matrix(self, distXX=1.6, distXH=1.2): #1

        '''Create a connectivity matrix as an attribute to the conf_space:
        distXX - cutoff distance between heavy atoms
        distXH - cutoff distance between heavy at - hydrogen '''

        print('creating connectivity matrix')
        for conf in self: conf.connectivity_matrix(distXX=1.6, distXH=1.2)

    def assign_pyranose_atoms(self): #2

        import networkx as nx
        print('assigning pyranose atoms')
        for conf in self: conf.assign_ring_atoms()

    def assign_ring_puckers(self): #3

        ''' assign rings to each conformer '''

        #try: self.ring (rings??)
        #except AttributeError: print "find ring_atoms first" ; sys.quit()

        print('assigning rings')

        for conf in self: 
            conf.ring = []
            conf.ring_angle = []
            for r in conf.ring_atoms:
                phi, psi, R = calculate_ring(conf.xyz, r)
                conf.ring.append(R) ; conf.ring_angle.append([phi, psi])

