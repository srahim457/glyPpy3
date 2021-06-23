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

    def __init__(self, path):

        self.path = path
        try: os.makedirs(self.path)
        except: 
            print("{0:10s} directory already exists".format(path))

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

    def load_dir(self, path):

        for (root, dirs, files) in os.walk('./'+path):
            for dirname in dirs:
                print (dirname)
                for ifiles in os.walk(path+'/'+dirname):
                    for filename in ifiles[2]:
                        if filename.endswith('.log'):
                            conf = Conformer('dummy')
                            conf.load_log(path+'/'+dirname+'/'+filename)
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
                            conf = Conformer('dummy')
                            conf.load_model(path+'/'+dirname+'/'+filename)
                            self.models.append(conf)
        self.Nmodels = len(self.models)

        for conf in self.models:
            print("Analyze {0:10s}".format(conf._id))
            conf.ring = [] ; conf.ring_angle = [] ; conf.dih_angle = []
 
            conf.connectivity_matrix(distXX=1.6, distXH=1.2)
            conf.assign_atoms()

            for r in conf.ring_atoms:                                               
                phi, psi, R = calculate_ring(conf.xyz, r)
                conf.ring.append(R) ; conf.ring_angle.append([phi, psi])    

            for d in conf.dih_atoms:

                atoms = sort_linkage_atoms(d)
                phi, ax = measure_dihedral(conf, atoms[:4])
                psi, ax = measure_dihedral(conf, atoms[1:5])
                conf.dih_angle.append([phi, psi])

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
        for conf in self: 
              if energy_function == 'E' and  conf.E < Eref: 
                    Eref = cp(conf.E) ; Href = cp(conf.H) ; Fref = cp(conf.F)
              elif energy_function == 'H' and  conf.H < Href: 
                    Eref = cp(conf.E) ; Href = cp(conf.H) ; Fref = cp(conf.F)
              elif energy_function == 'F' and  conf.F < Fref: 
                    Eref = cp(conf.E) ; Href = cp(conf.H) ; Fref = cp(conf.F)
        for conf in self: 
              conf.Erel = conf.E -  Eref;  conf.Hrel = conf.H -  Href ;  conf.Frel = conf.F -  Fref

    def print_relative(self):

        try: hasattr(self[0], 'Erel')
        except: 
            print("run reference_to_zero first")
            return None 

        if hasattr(self[0], 'Frel'):
            print ("%20s%20s%20s%20s" %('id', 'E [kcal/mol]', 'H [kcal/mol]', 'F [kac/mol]'))
            for conf in self:
                print ("%20s%20.2f%20.2f%20.2f" %(conf._id, conf.Erel*self._Ha2kcal, conf.Hrel*self._Ha2kcal, conf.Frel*self._Ha2kcal), end='')
                if hasattr(conf, 'dih'):  
                    for d in conf.dih:  print ("%20s" %(d), end='')
                    print(' ')
                else: print (' ')

        else: 
            print ("%20s%20s" %('id', 'E [kcal/mol]'))
            for conf in self: print("%20s%20.2f" %(conf._id, conf.Erel*self._Ha2kcal))

        return ''

    def gaussian_broadening(self, broaden=1):

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

