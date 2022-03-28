#Scanning 3D surface of the ring puckering angles

import glyP
import copy, sys, os, shutil, argparse


def puck_scan(**kwargs): #take a command line argument for the ring number

    scan_settings = {
        "in_dir"   : "models",
        "out_dir"  : "output-atoms",
        "output"   : "scan-atoms.log",
        "detail"  : "low",
        "ring"     : 0,
        "sample_c6": True,
        "freeze"   : "atoms",
        "software" : "g16"
        }

    for key in kwargs:
        if kwargs[key] != None: scan_settings[key] = kwargs[key]

    output = scan_settings["output"]
    ring = scan_settings["ring"]

    with open(output, 'w') as out: 

        sys.stdout = out
        #sys.stderr = out
        print("Scan settings:\n", scan_settings)

        #creates a space
        dih_scan = glyP.Space(scan_settings["out_dir"], software=scan_settings["software"])

        #loads a model from in_dir that will be used to generate puckers
        dih_scan.load_dir(scan_settings["in_dir"])
        reference_conf = dih_scan[0]

        #atoms in the edge: 
        linker_atoms = reference_conf.graph.edges[(ring, ring+1)]['linker_atoms']
        phi_atoms = linker_atoms[:3]
        psi_atoms = linker_atoms[1:]
        freeze = glyP.utilities.select_freeze_linker(scan_settings["freeze"],linker_atoms)
        jobtype = {"dih" : "opt=modredundant", "atoms": "opt=readopt"}

        ring_scan.set_theory(software='g16', method='PM3', basis_set= '', disp=False,  nprocs=12, mem='24GB', jobtype=jobtype[scan_settings["freeze"]], extra=freeze)
        #ring_scan.set_theory(software='fhiaims', nprocs=12, charge=0.0, extra=freeze)

        out.flush()

        detail = scan_settings["detail"]
        #determines the degrees that will be used
        if detail == 'low':
            points = np.linspace(-180,180,num=37)
        elif detail == 'med':
            points = np.linspace(-180,180,num=73)
        elif detail == 'high':
            points = np.linspace(-180,180,num=361)

        workable_dihedrals=[]
        for phi in points:
            for psi in points:
                conf.set_glycosidic((ring,ring+1),phi,psi)
                if glyP.utilities.clashcheck(reference_conf) == False:
                    workable_dihedrals.append([phi,psi])

        for n, angle in enumerate(workable_dihedrals):

            _id = '-'.join([reference_conf._id,'{0:03d}'.format(n)])

            ring_scan.append(copy.deepcopy(reference_conf))
            new_conf=ring_scan[-1]
            new_conf._id = '-'.join([new_conf._id,'{0:03d}'.format(n)])
            new_conf.path= '/'.join([ring_scan.path, new_conf._id])

            if os.path.exists(new_conf.path): 
                print("delete", new_conf._id)
                shutil.rmtree(new_conf.path)

            #set phi
            glyP.utilities.set_dihedral(new_conf,phi_atoms,angle[0])
            #set psi
            glyP.utilities.set_dihedral(new_conf,psi_atoms,angle[1])            
            new_conf.create_input(ring_scan.theory, ring_scan.path, software=scan_settings["software"])
            succ_job = new_conf.run_qm(ring_scan.theory, software=scan_settings["software"])

            if not succ_job:
                new_conf.load_log(software=scan_settings["software"])
                new_conf.measure_ring() ; new_conf.measure_c6() 
                print(new_conf)
            else:
                print(new_conf._id + " failed. Moving to the next pucker.")
                del ring_scan[-1]
            out.flush()

        ring_scan.reference_to_zero()
        ring_scan.sort_energy()
        ring_scan.print_relative(pucker=True)


def main():

	#parses the command line arguments, every argument is required
	#python3 puck_scan.py --in_dir glucose -r 0 -d low --out_dir g
	#for more specifics: python3 puck_scan.py -h 

    parser = argparse.ArgumentParser(description='Selects the ring number, level of detail of the scan and the input/output directories')

    parser.add_argument('--in_dir', required=False, help='the name of the input directory')
    parser.add_argument('--out_dir', required=False, help='the name of the output directory')
    parser.add_argument('--output',  required=False, help='output log-file')
    parser.add_argument('--ring', '-r', type=int, required=False, help='the index of which ring to select on a molecule')
    parser.add_argument('--detail', '-d', choices=['low','medium','high'], required=False, help='the level of detail of the pucker scan, more detail means more conformations')
    args = parser.parse_args()

    puck_scan(**vars(args))

if __name__ == '__main__':
    main()



