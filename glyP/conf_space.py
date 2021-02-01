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

    def __init__(self, molecule, ir_resolution=1.0):
        
        self.ir_resolution = ir_resolution 
        incr = self.ir_resolution

        for (root, dirs, files) in os.walk('./'+molecule):
            for dirname in dirs:
                print (dirname)
                #oldername = os.path.basename(dirpath)
                if dirname == 'experimental':
                    expIR= np.genfromtxt(molecule+'/'+dirname+'/exp.dat')
                    new_grid = np.arange(np.ceil(expIR[0,0]), np.floor(expIR[-1,0]), incr)
                    self.expIR = np.vstack((new_grid, interpolate.griddata(expIR[:,0], expIR[:,1], new_grid, method='cubic'))).T
                    #K = np.ceiling(expIR[:,0])
                    #I = expIR[:,1]
                    #expIR = np.column_stack((K,I))
                    #grid_old = np.arange(0,len(expIR))
                    #exp_incr = (expIR[-1,0] -  expIR[0,0])/len(expIR)
                    #grid_new = np.arange(grid_old[0],grid_old[-1]+incr/exp_incr,incr/exp_incr)
                    #spline_1D = interpolate.splrep(grid_old,expIR.T[1],k=3,s=0) 
                    #splrep finds spline of 1 d curve (x,y)--k repressents the recommended cubic spline, s represents the closeness vs smoothness tradeoff of k-- .T creates a transpose of the coordinates which you can then unpack and separate x and y
                    #spline_coef = interpolate.splev(grid_new,spline_1D,der=0) #--splev provides the knots and coefficients--der is the degree of the spline and must be less or equal to k
                    #self.expIR = np.vstack(( np.arange(expIR[0,0], expIR[0,0]+len(grid_new)*incr, incr), spline_coef)).T 
                for ifiles in os.walk(molecule+'/'+dirname):
                    for filename in ifiles[2]:
                        if filename.endswith('.log'):
                            self.append(Conformer(molecule+'/'+dirname+'/'+filename))

    def __str__(self):
         
        '''Prints a nice table with coded molecular values'''

        print ("%20s%20s%20s%20s\n" %('id', 'E', 'H', 'F'))
        #sorted the conformers stored in self by their energy
        self.sort(key = lambda x: x.E)
        for conf in self: 
            print ("%20s%20.2f%20.2f%20.2f\n" %(conf._id, conf.E*self._Ha2kcal, conf.H*self._Ha2kcal, conf.F*self._Ha2kcal))
        return ''

    def gaussian_broadening(self, broaden=1):

        ''' Performs gaussian broadening for the set''' 

        for conf in self: conf.gaussian_broadening(broaden, resolution=self.ir_resolution)
                   
    def assign_ring_puckers(self):

        ''' assign rings to each conformer '''

        #try: self.ring
        #except AttributeError: print "find ring_atoms first" ; sys.quit()

        for conf in self: 
            conf.ring = []
            conf.ring_angle = []
            for r in self.rings: 
                phi, psi, R = calculate_ring(conf.xyz, r)
                conf.ring.append(R) ; conf.ring_angle.append([phi, psi])

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

    def create_connectivity_matrix(self, index=0, distXX=1.6, distXH=1.2):

        '''Create a connectivity matrix as an attribute to the conf_space:
        index - index of conf used for the matrix 
        distXX - cutoff distance between heavy atoms
        distXH - cutoff distance between heavy at - hydrogen '''

        Nat = self[0].NAtoms
        conf = self[index]
        self.conn_mat = np.zeros((Nat, Nat)) 
        for at1 in range(Nat):
            for at2 in range(Nat):
                dist = get_distance(conf.xyz[at1], conf.xyz[at2]) 
                if at1 == at2: pass
                elif (conf.atoms[at1] == 'H' or conf.atoms[at2] == 'H') and dist < distXH: self.conn_mat[at1,at2] = 1; self.conn_mat[at2,at1] = 1
                elif dist < distXX: self.conn_mat[at1,at2] = 1; self.conn_mat[at2,at1] = 1

    def assign_pyranose_atoms(self):

        import networkx as nx
        
        cm = nx.graph.Graph(self.conn_mat)
        rings = nx.cycle_basis(cm)
        an = self[0].atoms
        self.rings = []
        n = 0
        for r in rings:
            self.rings.append({})
            rd = self.rings[n]
            for at in r:
                if an[at] == 'O': 
                    rd['O'] = at 
                else: 
                    for at2 in np.where(self.conn_mat[at] == 1)[0]: 
                        if an[at2] == 'C' and at2 not in r: 
                            rd['C4'] = at  
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
