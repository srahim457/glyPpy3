""" The script is based an extension of the cp.py script published in SI of:
Puckering Coordinates of Monocyclic Rings by Triangular Decomposition
Anthony D. Hill and Peter J. Reilly

Usage:
python cp.py xyz-file
or import as module:
from cp import cp_values
"""

#! /usr/bin/env python
import sys
import numpy
import math
from operator import itemgetter


## List of line numbers with the following atoms: O, C1, C2, C3, C4, C5


z = ['0E', '0,3B', '3E', '0H1', '3S1', '3H4', 'E1', 'B1,4', 'E4', '2H1', '5S1',
     '5H4', '2E', '2,5B', '5E', '2H3', '2S0', '5H0', 'E3', 'B0,3', 'E0', '4H3',
     '1S3', '1H0', '4E', '1,4B', '1E', '4H5', '1S5', '1H2', 'E5', 'B2,5', 'E2',
     '0H5', '0S2', '3H2', '1C4', '4C1']
list_of_data = [[0, 55], [0, 90], [0, 125], [30, 51], [30, 92], [30, 129],
                [60, 55], [60, 90], [60, 125], [90, 51], [90, 92], [90, 129],
                [120, 55], [120, 90], [120, 125], [150, 51], [150, 92],
                [150, 129], [180, 55], [180, 90], [180, 125], [210, 51],
                [210, 88], [210, 129], [240, 55], [240, 90], [240, 125],
                [270, 51], [270, 88], [270, 129], [300, 55], [300, 90],
                [300, 125], [330, 51], [330, 88], [330, 129], [180, 180],
                [180, 0]]

dict_canon = {}
for i in range(len(z)):
    dict_canon[str(z[i])] = list_of_data[i]


def haversine(lon1, lat1, lon2, lat2):
    """Calculates and returns a distance using the haversine formula

    :param lon1: (float) the longitude of position 1
    :param lat1: (float) the latitude of position 1
    :param lon2: (float) the longitude of position 2
    :param lat2: (float) the latitude of poition 2
    :return distance: (float) distance between two positions
    """
    lat1 = -lat1 + 90.0
    lat2 = -lat2 + 90.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    c = math.acos(math.sin(lat1)*math.sin(lat2)+math.cos(lat1)*math.cos(lat2)*math.cos(lon1-lon2))
    return c

def find_canon(n1, n2):
    """ Finds the canonical shape of a puckered ring using 2 angles from the paper by Anthony D. Hill and Peter J. Reilly mentioned in the description of this class

    :param n1: (float) phi angle
    :param n2: (float) theta angle
    :return: the canonical shape
    """

    new_dict = {}
    for elem in dict_canon:
        diff = haversine(n1, n2, dict_canon[elem][0], dict_canon[elem][1])
        new_dict[elem] = diff

    #return min(new_dict.iteritems(), key=itemgetter(1))[0]
    #Python2 dict.iteritems() became just dict.items() in Python3 because the old version of dict.items() was inefficient and nobody used it
    return min(new_dict.items(), key=itemgetter(1))[0]


def norm(a):
    """Norm of a vector, basically returns the non-negative value of a number

    :param a: (float) some value
    :return: (float) the normalized value
    """
    return math.sqrt(numpy.sum(a*a))


def cp_values(xyz, ring_atoms):
    """Calculate the puckering angles phi, theta and the canonical shape of the puckered ring

    :param xyz: the position of each atom
    :param ring_atoms: list of the atoms that are present in a ring 
    :return: phi, theta and canonical shape
    """

    atoms = numpy.zeros((6, 3), dtype='float64')
    for i  in range(xyz.shape[0]):
        if i in ring_atoms:
            atoms[ring_atoms.index(i)] = xyz[i,:]
    #print atoms

    center = numpy.add.reduce(atoms)/6.
    atoms = atoms - center
    r1a = numpy.zeros((3), dtype='float64')
    r2a = numpy.zeros((3), dtype='float64')
    for j, i in enumerate(atoms[0:6]):
        r1a += i * math.sin(2.*math.pi*j/6.)
        r2a += i * math.cos(2.*math.pi*j/6.)
    n = numpy.cross(r1a, r2a)
    n = n / norm(n)
    z = numpy.dot(atoms, n)
    q2cosphi = 0.
    q2sinphi = 0.
    q1cosphi = 0.
    q1sinphi = 0.
    q3 = 0.
    bigQ = 0.
    sqrt_2 = math.sqrt(2.)
    inv_sqrt_6 = math.sqrt(1./6.)
    for j, i in enumerate(z):
        q2cosphi += i*math.cos(2.*math.pi*2.*j/6.)
        q2sinphi -= i*math.sin(2.*math.pi*2.*j/6.)
        q1cosphi += i*math.cos(2.*math.pi*j/6.)
        q1sinphi -= i*math.sin(2.*math.pi*j/6.)
        q3 += i*math.cos(j*math.pi)
        bigQ += i*i
    q2cosphi = sqrt_2 * inv_sqrt_6 * q2cosphi
    q2sinphi = sqrt_2 * inv_sqrt_6 * q2sinphi
    q3 = inv_sqrt_6 * q3
    q2 = math.sqrt(q2cosphi*q2cosphi + q2sinphi*q2sinphi)
    q1 = math.sqrt(q1cosphi*q1cosphi + q1sinphi*q1sinphi)
    bigQ = math.sqrt(bigQ)
    if (q2cosphi > 0.):
        if (q2sinphi > 0.):
            phi = math.degrees(math.atan(q2sinphi/q2cosphi))
        else:
            phi = 360. - abs(math.degrees(math.atan(q2sinphi/q2cosphi)))
    else:
        if (q2sinphi > 0.):
            phi = 180. - abs(math.degrees(math.atan(q2sinphi/q2cosphi)))
        else:
            phi = 180. + abs(math.degrees(math.atan(q2sinphi/q2cosphi)))
    theta = math.degrees(math.atan(q2/q3))
    if (q3 > 0.):
        if (q2 > 0.):
            theta = math.degrees(math.atan(q2/q3))
        else:
            theta = 360. - abs(math.degrees(math.atan(q2/q3)))
    else:
        if (q2 > 0.):
            theta = 180. - abs(math.degrees(math.atan(q2/q3)))
        else:
            theta = 180. + abs(math.degrees(math.atan(q2/q3)))
    #bigQ2 = numpy.array([q1,q2,q3],dtype='float64')
    #bigQ2 = math.sqrt((bigQ2*bigQ2).sum())
    canon = find_canon(phi, theta)
    #return ' %7.3f %7.3f %s' % (phi, theta,canon)
    return phi, theta, canon



if __name__ == "__main__":

    f1 = sys.argv[1]
    cp_val = cp_values(f1)
    #print "%10.1f%10.1f%10s" %(cp_val[0], cp_val[1], cp_val[2])
