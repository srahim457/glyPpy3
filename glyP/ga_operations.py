import math
from . import calc_cp
from . import rmsd, utilities
import numpy as np
import sys, copy
from scipy import interpolate
from scipy.linalg import expm
from optparse import OptionParser

def modify_glyc(conf, bond):

    edge = conf.graph.edges[bond]
    atoms = len(edge['linker_atoms']) ; angles = [] ; n=0

    while n < atoms-3: 
        angles.append((utilities.draw_random()*360)-180.0)
        n += 1 
    if atoms == 5:
        conf.set_glycosidic(bond, angles[0], angles[1])
    elif atoms== 6:
        conf.set_glycosidic(bond, angles[0], angles[1], angles[2])
    elif atoms == 7: #NAc linkage, set two bonds linear
        angles[1] = (utilities.draw_random_int(top=2)+0.005)*179.0
        angles[2] = (utilities.draw_random_int(top=2)+0.005)*179.0
        #print( angles )
        conf.set_glycosidic(bond, angles[0], angles[1], angles[2], angles[3])

def modify_ring(conf, ring, phi, psi):

    pass

def cross_over(conf1, conf2):

    #exchange glycosidic bond:
    bond = utilities.draw_random_int(len(conf1.dih))
    print("Modifying bond number {0:2d}".format(bond))
    phi1, psi1 = conf1.dih_angels[bond]
    phi2, psi2 = conf2.dih_angels[bond] 

    conf2.set_glycosidic(bond, phi1, psi1)
    conf1.set_glycosidic(bond, phi2, psi2)




