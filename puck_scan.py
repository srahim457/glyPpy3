#Scanning 3D surface of the ring puckering angles

import glyP
import copy, sys, argparse

def puck_scan(in_dir, out_dir, ring, detail): #take a command line argument for the ring number

	#creates a space
    ring_scan = glyP.Space(out_dir)
    ring_scan.load_models(in_dir)
	#working with glucose so this makes a copy of the single model created; subject to change
    reference_conf = ring_scan.models[0]
    #3 sets of the 4 atoms that make dihedral, followed by F\n
    # '1 2 3 4 F\n' + '2 3 4 5 F\n' + '3 4 5 1 F\n'
    ra = reference_conf.graph.nodes[ring]['ring_atoms']
    
    freeze = 'atoms'
    
    if freeze == 'dih':
        freeze_dih=glyP.utilities.gaussian_string_parameter(freeze,ra)
        print(freeze_dih)
        ring_scan.set_theory(software='g16', nprocs=24, jobtype='opt=modredundant', extra=freeze_dih )
    elif freeze == 'atoms':
        freeze_atoms=glyP.utilities.gaussian_string_parameter(freeze,ra)
        print(freeze_atoms)
        #ring_scan.set_theory(software='g16', nprocs=24, jobtype='opt=readopt', extra='notatoms=1,3,4,8,12,16' )
        ring_scan.set_theory(software='g16', nprocs=24, jobtype='opt=modredundant', extra=freeze_dih )

    pucker_conformations = glyP.utilities.pucker_scan(detail)
    for count, degrees in enumerate(pucker_conformations):
        ring_scan.append(copy.deepcopy(reference_conf))
        new_conf=ring_scan[-1]

        new_conf._id = '-'.join([new_conf._id,'{0:03d}'.format(count)])
        new_conf.path= '/'.join([ring_scan.path, new_conf._id])

        glyP.utilities.set_ring_pucker(new_conf, ring, degrees)

        new_conf.create_input(ring_scan.theory, ring_scan.path)
        
        succ_job = new_conf.run_qm(ring_scan.theory, software='g16')

        if succ_job:
            new_conf.load_log(software='g16')
            new_conf.measure_ring()
            print(new_conf)
        else:
            print(new_conf._id + " failed.")
		
    print(ring_scan)
    ring_scan.reference_to_zero()
    ring_scan.print_relative()

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

    with open(args.in_dir+"scan.log",'w') as f:
        puck_scan(args.in_dir, args.out_dir, args.ring, args.detail)
        


if __name__ == '__main__':
    main()



