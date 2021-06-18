import math
from . import calc_cp
from . import rmsd
import numpy as np
import sys, copy
from scipy import interpolate
from scipy.linalg import expm
from optparse import OptionParser

def error(msg):
   """ write error message and quit
   """
   sys.stderr.write(msg + "\n")
   sys.exit(3)

def get_distance(at1, at2):

    return math.sqrt((at1[0]-at2[0])**2+(at1[1]-at2[1])**2+(at1[2]-at2[2])**2)

def clashcheck(conf, cutoff=1.0):

    Inv_cm = np.ones( (conf.NAtoms, conf.NAtoms) ) + 10*conf.conn_mat
    dist = np.zeros((conf.NAtoms, conf.NAtoms))

    for at1 in range(conf.NAtoms):
        for at2 in range(conf.NAtoms):
            if at1 != at2 : dist[at1,at2] = get_distance(conf.xyz[at1], conf.xyz[at2])
            else: dist[at1,at2] = 10.0

    Dist = dist*Inv_cm
    print( np.amin(Dist))
    if np.amin(Dist) > cutoff: 
        return False
    else: 
        return True

def adjacent_atoms(conn_mat, at):

    return np.nonzero(conn_mat[at,:])[0]

def draw_random():

    return np.random.random_sample()

def draw_random_int(top=1):

    return np.random.randint(top)

def element_symbol(A): 

    periodic_table = { '1' : 'H', '6' : 'C', '7' : 'N', '8' : 'O' , '9' : 'F', '14' : 'Si' }
    return periodic_table[A]

def calculate_ring(xyz, ring_atoms): 

    sorted_atoms = []
    for i in 'O', 'C1', 'C2', 'C3', 'C4', 'C5': sorted_atoms.append(ring_atoms[i])

    phi, psi, R = calc_cp.cp_values(xyz, sorted_atoms) 
    return phi, psi, R

def sort_linkage_atoms(dih_atoms):

    d = dih_atoms
    if 'C2l'   in dih_atoms.keys(): at1=d['O'] ; at2=d['C1l']; at3=d['Ol']; at4=d['C2l']; at5=d['C1']
    elif 'C3l' in dih_atoms.keys(): at1=d['O'] ; at2=d['C1l']; at3=d['Ol']; at4=d['C3l']; at5=d['C2']
    elif 'C4l' in dih_atoms.keys(): at1=d['O'] ; at2=d['C1l']; at3=d['Ol']; at4=d['C4l']; at5=d['C3']
    elif 'C5l' in dih_atoms.keys(): 
        at1=d['O'] ; at2=d['C1l']; at3=d['Ol']; at4=d['C6']; at5=d['C5l'] ; at6=d['C4']
        return [at1, at2, at3, at4, at5, at6] 

    return [ at1, at2, at3, at4, at5]

def determine_carried_atoms(at1, at2, conn_mat):

    """Find all atoms necessary to be carried over during rotation
    of an atom 2:

    Args:
        at1, at2: two atoms number
    """
    #   1. Zero the connections in connectivity matrix
    tmp_conn = np.copy(conn_mat)
    tmp_conn[at1, at2] = 0
    tmp_conn[at2, at1] = 0
    import networkx as nx
    cm = nx.graph.Graph(tmp_conn)
    if nx.is_connected(cm) == True:
        print("Matrix still connected")

    #   2. Determine the connected atoms:
    carried_atoms = [at2]
    start = True
    while start:
        start = False
        #   Always iterate over entire list because of branching
        for at in carried_atoms:
            #   List of indexes of connected atoms:
            conn_atoms = np.where(tmp_conn[at] != 0)[0]
            conn_atoms.tolist
            for x in conn_atoms:
                if x not in carried_atoms:
                    carried_atoms.append(x)
                    start = True
    return carried_atoms

def calculate_normal_vector(conf, list_of_atoms):
    """Calculate the normal vector of a plane by
    cross product of two vectors belonging to it.

    Args:
        list_of_atoms: list of 3 atoms
        xyz: numpy array with atoms xyz position
    """

    r0 = conf.xyz[list_of_atoms[1], :] - conf.xyz[list_of_atoms[0], :]
    r1 = conf.xyz[list_of_atoms[2], :] - conf.xyz[list_of_atoms[1], :]
    cross_product = np.cross(r1, r0)
    return cross_product

def measure_angle(conf, list_of_atoms):

    """Calculate an angle between three atoms:
    angle = acos(dot(X,Y)/(norm(X)*norm(Y)))

    Args:
        list_of_atoms: list of 3 atoms
        xyz: numpy array with atoms xyz positions
    """
    r0 = conf.xyz[list_of_atoms[0], :] - conf.xyz[list_of_atoms[1], :]
    r1 = conf.xyz[list_of_atoms[2], :] - conf.xyz[list_of_atoms[1], :]

    norm_r0 = np.sqrt(np.sum(r0**2))
    norm_r1 = np.sqrt(np.sum(r1**2))
    norm = norm_r0*norm_r1

    dot_product = np.dot(r0, r1)/norm
    angle = np.arccos(dot_product)

    #    Calculate the axis of rotation (axor):
    axor = np.cross(r0, r1)

    return angle*180.0/np.pi, axor

def measure_dihedral(conf, list_of_atoms):

    """Calculate a dihedral angle between two planes defined by
    a list of four atoms. It returns the angle and the rotation axis
    required to set a new dihedral.

    Args:
        list_of_atoms: list of 4 atoms
        xyz: numpy array with atom xyz positions
    """
    plane1 = calculate_normal_vector(conf, list_of_atoms[:3])
    plane2 = calculate_normal_vector(conf, list_of_atoms[1:])
    #   Calculate the axis of rotation (axor)
    axor = np.cross(plane1, plane2)

    #   Calculate a norm of normal vectors:
    norm_plane1 = np.sqrt(np.sum(plane1**2))
    norm_plane2 = np.sqrt(np.sum(plane2**2))
    norm = norm_plane1 * norm_plane2

    #   Measure the angle between two planes:
    dot_product = np.dot(plane1, plane2)/norm
    alpha = np.arccos(dot_product)

    #   The cosine function is symetric thus, to distinguish between
    #   negative and positive angles, one has to calculate if the fourth
    #   point is above or below the plane defined by first 3 points:

    ppoint = - np.dot(plane1, conf.xyz[list_of_atoms[0], :])
    dpoint = (np.dot(plane1, conf.xyz[list_of_atoms[3], :])+ppoint)/norm_plane1

    if dpoint >= 0:
        return -(alpha*180.0)/np.pi, axor
    else:
        return (alpha*180.0)/np.pi, axor


def set_angle(conf, list_of_atoms, new_ang):

    from scipy.linalg import expm

    at1 = list_of_atoms[0] 
    at2 = list_of_atoms[1] #midpoint 
    at3 = list_of_atoms[2]  
    #xyz = copy.copy(conf.xyz)
    xyz = conf.xyz

    if len(position) != 3:
        raise ValueError("The position needs to be defined by 4 integers")

    """Set a new angle between three atoms

    Args:
        list_of_atoms: list of three atoms
        new_ang: value of dihedral angle (in degrees) to be set
        atoms_ring: dictionary of atoms in the ring. It recognizes
                    if the last atom is 'C0O' (obsolete)
        xyz: numpy array with atoms xyz positions
        conn_mat: connectivity matrix
    Returns:
        xyz: modified numpy array with new atoms positions
    """
    #   Determine the axis of rotation:

    old_ang, axor = measure_angle(conf, [at1, at2, at3])
    norm_axor = np.sqrt(np.sum(axor**2))
    normalized_axor = axor/norm_axor

    #   Each carried_atom is rotated by euler-rodrigues formula:
    #   Also, I move the midpoint of the bond to the mid atom
    #   the rotation step and then move the atom back.

    rot_angle = np.pi*(new_ang - old_ang)/180.
    translation = xyz[at2, :]

    #apply rotations to at3. 
    rot = expm(np.cross(np.eye(3), normalized_axor*(rot_angle)))
    xyz[at3, :] = np.dot(rot, xyz[at3, :]-translation)
    xyz[at3, :] = xyz[at3, :]+translation

    #return xyz

def set_dihedral(conf, list_of_atoms, new_dih):

    """Set a new dihedral angle between two planes defined by
    atoms first and last three atoms of the supplied list.

    Args:
        list_of_atoms: list of four atoms
        new_dih: value of dihedral angle (in degrees) to be set
        xyz: numpy array with atoms xyz positions
        conn_mat: connectivity matrix
    Returns:
        xyz: modified numpy array with new atoms positions
    """
    at1 = list_of_atoms[0]
    at2 = list_of_atoms[1] #midpoint 
    at3 = list_of_atoms[2]
    at4 = list_of_atoms[3]
    #xyz = copy.copy(conf.xyz)
    xyz = conf.xyz

    #   Determine the axis of rotation:
    old_dih, axor = measure_dihedral(conf, [at1, at2, at3, at4])
    norm_axor = np.sqrt(np.sum(axor**2))
    normalized_axor = axor/norm_axor

    #   Determine which atoms should be dragged along with the bond:
    carried_atoms = determine_carried_atoms(at2,at3, conf.conn_mat)

    #   Each carried_atom is rotated by Euler-Rodrigues formula:
    #   Reverse if the angle is less than zero, so it rotates in
    #   right direction.
    #   Also, I move the midpoint of the bond to the center for
    #   the rotation step and then move the atom back.

    if old_dih >= 0.0:
        rot_angle = np.pi*(new_dih - old_dih)/180.
    else:
        rot_angle = -np.pi*(new_dih - old_dih)/180.

    rot = expm(np.cross(np.eye(3), normalized_axor*rot_angle))
    translation = (xyz[list_of_atoms[1], :]+xyz[list_of_atoms[2], :])/2

    for at in carried_atoms:
        xyz[at, :] = np.dot(rot, xyz[at, :]-translation)
        xyz[at, :] = xyz[at, :]+translation

    #return xyz

def calculate_rmsd(conf1, conf2, atoms=None): #pass 2 conformers instead of just the xyz list 

    xyz1 = []
    xyz2 = []
    #exclude the specified atom
    if atoms != None:
      for i in range(len(conf1.atoms)):
        if conf1.atoms[i] != atoms:
          xyz1.append(conf1.xyz[i])
      for i in range(len(conf2.atoms)):
        if conf2.atoms[i] != atoms:
          xyz2.append(conf2.xyz[i])
    else:
      xyz1 = conf1.xyz
      xyz2 = conf2.xyz

    return rmsd.rmsd_qcp(xyz1, xyz2)


def deriv(spec,h):
   """ calculate first derivative of function 'spec'
       using the central finite difference method up to 6th order,
       and for the first 3 and last 3 grid points the
       forward/backward finite difference method up to 2nd order.
       ...as used in f77-program and suggested by Zanazzi-Jona...
   """ 
   der_spec =[[i[0],0] for i in spec]

   length=len(spec)
   for i in range(3,length-3):
      der_spec[i][1]=(-1*spec[i-3][1]+9*spec[i-2][1]-45*spec[i-1][1]+45*spec[i+1][1]-9*spec[i+2][1]+1*spec[i+3][1])/(60*h)
   for i in range(0,3):
      der_spec[i][1]=(-11*spec[i][1]+18*spec[i+1][1]-9*spec[i+2][1]+2*spec[i+3][1])/(6*h)
   for i in range(length-3,length):
      der_spec[i][1]=(11*spec[i][1]-18*spec[i-1][1]+9*spec[i-2][1]-2*spec[i-3][1])/(6*h)

   return der_spec


def get_range(tspec,espec,w_incr,shift,start,stop):
   """ determine wavenumber range within the comparison between theoretical
       and experimental spectrum is performed (depends on the shift)
   """
   de1=start+shift-espec[0][0]
   if (de1 >= 0 ):
      de1=int((start+shift-espec[0][0])/w_incr+0.00001)
      enstart=de1
      tnstart=int((start-tspec[0][0])/w_incr+0.00001)
   else:
      de1=int((start+shift-espec[0][0])/w_incr-0.00001)
      enstart=0
      tnstart=int((start-tspec[0][0])/w_incr-de1+0.00001)
   de2=stop+shift-espec[-1][0]
   if (de2 <= 0 ):
      de2=int((stop+shift-espec[-1][0])/w_incr-0.00001)
      enstop=len(espec)+de2
      tnstop=len(tspec)+int((stop-tspec[-1][0])/w_incr-0.00001) 
   else:
      de2=int((stop+shift-espec[-1][0])/w_incr+0.00001)
      enstop=len(espec)
      tnstop=len(tspec)+int((stop-tspec[-1][0])/w_incr-de2-0.00001)
   return tnstart, tnstop, enstart, enstop
 

def integrate(integrand,delta):
   """ integrate using the trapezoid method as Zanazzi-Jona suggested and was used in the f77-program...
   """
   integral = 0.5*(integrand[0][1]+integrand[-1][1])   
   for i in range(1,len(integrand)-1):
      integral += integrand[i][1]
   return integral*delta

def ypendry(spec,d1_spec,VI):
   """ calculate the Pendry Y-function: y=l^-1/(l^-2+VI^2) with l=I'/I (logarithmic derivative),
       J.B. Pendry, J. Phys. C: Solid St. Phys. 13 (1980) 937-44
   """
   y=[[i[0],0] for i in spec]

   for i in range(len(spec)):
      if (abs(spec[i][1]) <= 1.E-7):
         if (abs(d1_spec[i][1]) <= 1.E-7):
            y[i][1] = 0 
         else:
            y[i][1] = (spec[i][1]/d1_spec[i][1])/((spec[i][1]/d1_spec[i][1])**2+VI**2)
      else:
         y[i][1] = (d1_spec[i][1]/spec[i][1])/(1+(d1_spec[i][1]/spec[i][1])**2*(VI**2))
   return y


def rfac(espec, tspec, start=1000, stop=1800, w_incr=1.0, shift_min=-10, shift_max=+10, shift_incr=1, r="pendry", VI=10):

   """ %prog [options] r-fac.in
        Reads two spectra and calculates various R-factors -- FS 2011
        Attention: both spectra have to be given on the same, equidistant grid!
        NOTE: in the f77-program R1 is scaled by 0.75 and R2 is scaled by 0.5; this is not done here
        Please provide a file r-fac.in with the following specifications (without the comment lines!!!) 
        (the numbers are just examples, choose them according to your particular case)
        start=1000       # where to start the comparison
        stop=1800        # where to stop the comparison
        w_incr=0.5       # grid interval of the spectra -- should be 1 or smaller! (otherwise integrations/derivatives are not accurate)
        shift_min=-10    # minimal shift of the theoretical spectrum 
        shift_max=+10    # maximal shift of the experimental spectrum
        shift_incr=1     # shift interval
        r=pendry         # which r-factor should be calculated? options: pendry, ZJ, R1, R2 (give a list of the requested r-factors separated by comma)
        VI=10            # approximate half-width of the peaks (needed for pendry r-factor)
        """
   #for shift in numpy.arange(shift_min,shift_max+shift_incr,shift_incr):# get the interval within the two spectra are compared
                #tnstart,tnstop,enstart,enstop = get_range(tspec,espec,w_incr,shift,start,stop) 
 
   
# perform some checks of the input data...
   if (int(shift_incr/w_incr+0.00001) == 0):
      error("Error: shift_incr cannot be smaller than w_incr!")
   if (start-espec[0][0] < 0) or (espec[-1][0]-stop < 0):
      error("check experimental spectrum!!")
   if (start-tspec[0][0] < 0) or (tspec[-1][0]-stop < 0):
      error("check theoretical spectrum!!")
   if (int((espec[-1][0]-espec[0][0])/w_incr+0.0001) != len(espec)-1 ) or (int((tspec[-1][0]-tspec[0][0])/w_incr+0.0001) != len(tspec)-1 ):
      error("check w_incr!!")

 
# cut out data points that are not needed in order to save time...
   if (espec[0][0]-(start+shift_min-w_incr*25) < 0):
         espec=espec[-1*int((espec[0][0]-(start+shift_min-w_incr*25))/w_incr-0.00001):][:]
   if (espec[-1][0]-(stop+shift_max+w_incr*25) > 0):
         espec=espec[:-1*(int((espec[-1][0]-(stop+shift_max+w_incr*25))/w_incr+0.00001)+1)][:] 
   if (tspec[0][0]-(start-w_incr*25) < 0):
         tspec=tspec[-1*int((tspec[0][0]-(start-w_incr*25))/w_incr-0.00001):][:]
   if (tspec[-1][0]-(stop+w_incr*25) > 0):
         tspec=tspec[:-1*(int((tspec[-1][0]-(stop+w_incr*25))/w_incr+0.00001)+1)][:]

   
# set negative intensity values to zero
   for i in range(0,len(espec)):
      if (espec[i][1]<0):
         espec[i][1]=0
   for i in range(0,len(tspec)):
      if (tspec[i][1]<0):
         tspec[i][1]=0
   
# start calculating derivatives...
   d1_espec = deriv(espec,w_incr)   
   d1_tspec = deriv(tspec,w_incr)
# calculate the second derivatives if the Zanazzi-Jona R-factor is requested   
   if "ZJ" in r:
      d2_tspec = deriv(d1_tspec,w_incr)
      d2_espec = deriv(d1_espec,w_incr)
# calculate Pendry Y-function if Pendry R-factor is requested      
   if "pendry" in r:
      ye = ypendry(espec,d1_espec,VI)
      yt = ypendry(tspec,d1_tspec,VI)
   


   min_pendry = [1.E100,0]
   min_r1     = [1.E100,0]
   min_r2     = [1.E100,0]
   min_zj     = [1.E100,0]
# start with loop over x-axis shifts
   for shift in np.arange(shift_min,shift_max+shift_incr,shift_incr):
      # get the interval within the two spectra are compared
      tnstart,tnstop,enstart,enstop = get_range(tspec,espec,w_incr,shift,start,stop) 
      #sys.stdout.write("\nshift: %9.3f, theory-start: %5d, theory-end: %5d, exp-start: %5d, exp-end: %5d\n" % (shift,tspec[tnstart][0],tspec[tnstop-1][0],espec[enstart][0],espec[enstop-1][0]))
      s_espec = np.array(espec[enstart:enstop]) # cut out the interval within which the comparison takes place
      s_tspec = np.array(tspec[tnstart:tnstop])
      s_d1_espec = np.array(d1_espec[enstart:enstop])
      s_d1_tspec = np.array(d1_tspec[tnstart:tnstop])
      c_scale=integrate(s_espec,w_incr)/integrate(s_tspec,w_incr)
      if "pendry" in r:
         # see J.B. Pendry, J. Phys. C: Solid St. Phys. 13 (1980) 937-44
         s_yt = np.array(yt[tnstart:tnstop]) # cut out the interval within which the comparison takes place
         s_ye = np.array(ye[enstart:enstop])
         te2 = integrate((s_yt-s_ye)**2,w_incr) # integrate (yt-ye)^2
         t2e2 = integrate(s_yt**2+s_ye**2,w_incr) # integrate yt^2+ye^2
         r_pend = te2/t2e2
         #sys.stdout.write("Pendry R-factor : %f, shift: %f\n" % (r_pend,shift))
         if (r_pend < min_pendry[0] ):
            min_pendry=[r_pend,shift]
      if "R1" in r:
         # see  M.A. van Hove, S.Y. Tong, and M.H. Elconin, Surfac Science 64 (1977) 85-95
         r1 = integrate(abs(s_espec-c_scale*s_tspec),w_incr)/integrate(abs(s_espec),w_incr)
         sys.stdout.write("R1 R-factor     : %f, shift: %f\n" % (r1,shift))
         if (r1 < min_r1[0]):
            min_r1=[r1,shift]
      if "R2" in r:
         # see  M.A. van Hove, S.Y. Tong, and M.H. Elconin, Surfac Science 64 (1977) 85-95
         r2 = integrate((s_espec-c_scale*s_tspec)**2,w_incr)/integrate(s_espec**2,w_incr)
         sys.stdout.write("R2 R-factor     : %f, shift: %f\n" % (r2,shift))
         if (r2 < min_r2[0]):
            min_r2=[r2,shift]
      if "ZJ" in r:      
         # E. Zanazzi, F. Jona, Surface Science 62 (1977), 61-88
         s_d2_tspec = np.array(d2_tspec[tnstart:tnstop])
         s_d2_espec = np.array(d2_espec[enstart:enstop])

         epsilon = 0
         for i in s_d1_espec:
            if abs(i[1]) > epsilon:
               epsilon = abs(i[1])
         
         integrand = abs(c_scale*s_d2_tspec-s_d2_espec)*abs(c_scale*s_d1_tspec-s_d1_espec)/(abs(s_d1_espec)+epsilon)
         # interpolate integrand onto 10 times denser grid, see publication by Zanazzi & Jona
         incr = 0.1*w_incr
         grid_old = np.arange(0,len(integrand))*w_incr
         grid_new = np.arange(grid_old[0],grid_old[-1]+incr,incr)
         spl = interpolate.splrep(grid_old,integrand.T[1],k=3,s=0)
         integrand_dense = interpolate.splev(grid_new,spl,der=0)
         integrand_dense = np.vstack((grid_new,integrand_dense)).T
         # calculate reduced Zanazzi-Jona R-factor r=r/0.027
         r_zj = integrate(integrand_dense,incr)/(0.027*integrate(abs(s_espec),w_incr))
         sys.stdout.write("red. ZJ R-factor: %f, shift %f\n" % (r_zj,shift))
         if (r_zj < min_zj[0]):
            min_zj=[r_zj,shift]


# find minimal r-factor and write it out
   #sys.stdout.write("\nMinimal r-factors:\n")
   if "pendry" in r:
      #sys.stdout.write("minimal r-factor: Delta = %8.5f, Pendry R-factor = %7.5f \n" % ( min_pendry[1], min_pendry[0]))
        #print  (min_pendry[1], min_pendry[0])
        #I'm adding a return statement
        return min_pendry[0]
   if "R1" in r:
      sys.stdout.write("minimal r-factor: Delta = %8.5f, R1 R-factor = %7.5f \n" % ( min_r1[1], min_r1[0]))
   if "R2" in r:
      sys.stdout.write("minimal r-factor: Delta = %8.5f, R2 R-factor = %7.5f \n" % ( min_r2[1], min_r2[0]))
   if "ZJ" in r:
      sys.stdout.write("minimal r-factor: Delta = %8.5f, ZJ R-factor = %7.5f \n" % ( min_zj[1], min_zj[0]))















