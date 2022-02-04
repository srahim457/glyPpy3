#Scanning 3D surface of the ring puckering angles

import glyP
import copy, sys, argparse

def puck_scan(in_dir,ring,detail,out_dir): #take a command line argument for the ring number
	#creates a space
	space = glyP.Space(out_dir)
	space.load_models(in_dir)
	#working with glucose so this makes a copy of the single model created; subject to change
	conf0 = space.models[0]
	pucker_conformations = glyP.utilities.pucker_scan(detail)

	#the theory is set here, however it can also be made a command line argument
    #space.set_theory(nprocs=12, mem='16GB', charge=0, basis_set='STO-3G', jobtype='opt=loose freq')
    #space.set_theory(nprocs=24, mem='64GB', charge=1, method='PBE1PBE' , basis_set='6-31G(d)', jobtype='opt=loose', disp=True)
	space.set_theory(software='g16', nprocs=24)

	#creates copies of the glucose model and alters the ring pucker, saves an input.com file and the geom.xyz file
	for count, degrees in enumerate(pucker_conformations):
		space.append(copy.deepcopy(conf0))
		new_conf=space[-1]
		new_conf._id=str(count)
		glyP.utilities.set_ring_pucker(new_conf,ring,degrees)
		new_conf.create_input(space.theory, space.path)
		new_conf.save_xyz()

def main():
	#parses the command line arguments, every argument is required
	#python3 puck_scan.py --in_dir glucose -r 0 -d low --out_dir g
	#for more specifics: python3 puck_scan.py -h 
	parser = argparse.ArgumentParser(description='Selects the ring number, level of detail of the scan and the input/output directories')
	parser.add_argument('--in_dir', required=True, help='the name of the input directory')
	parser.add_argument('--ring', '-r', type=int, required=True, help='the index of which ring to select on a molecule')
	parser.add_argument('--detail', '-d', choices=['low','medium','high'], required=True, help='the level of detail of the pucker scan, more detail means more conformations')
	parser.add_argument('--out_dir', required=True, help='the name of the output directory')
	args = parser.parse_args()

	puck_scan(args.in_dir,args.ring,args.detail,args.out_dir)

if __name__ == '__main__':
	main()



