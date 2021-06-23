import math
from . import calc_cp
from . import rmsd
import numpy as np
import sys, copy
from scipy import interpolate
from scipy.linalg import expm
from optparse import OptionParser

def modify_glyc(conf, bond, phi, psi, omega=None):
 
    if omega == None: conf.set_glycosidic(bond, phi, psi)
    else: conf.set_glycosidic(bond, phi, psi, omega)

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




