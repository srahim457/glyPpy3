import glyP
import copy, sys, os, shutil, argparse
import numpy as np

def scan(**kwargs): #take a command line argument for the ring number

    scan_settings = {
        "scan_type": "",
        "in_dir"   : "models",
        "out_dir"  : "output-atoms",
        "output"   : "scan-atoms.log",
        "details"  : "low",
        "ring"     : 0,
        "sample_c6": True,
        "freeze"   : "atoms",
        "software" : "g16"
        }

    for key in kwargs:
        if kwargs[key] != None: scan_settings[key] = kwargs[key]

    output = scan_settings["output"]
    scan_type = scan_settings["scan_type"]

    with open(output, 'w') as out: 

        sys.stdout = out
        #sys.stderr = out
        print("Scan settings:\n", scan_settings)

        #creates a space
        scan = glyP.Space(scan_settings["out_dir"], software=scan_settings["software"])
        #checks the existing files in the dir
        finished_conformers = [conf._id for conf in scan]
        print("Complete puckers:\n", finished_conformers)


        #loads a model from in_dir that will be used to generate puckers
        scan.load_dir(scan_settings["in_dir"], software='xyz')
        reference_conf = scan[0]

        if scan_type == 'pucker':
            ra = reference_conf.graph.nodes[scan_settings["ring"]]['ring_atoms']
            freeze = glyP.utilities.select_freeze(scan_settings["freeze"],ra)

            conformations = glyP.utilities.pucker_scan(scan_settings["details"])

        elif scan_type == 'torsion':
            ring = int(scan_settings["ring"])
            linker_atoms = reference_conf.graph.edges[(ring, ring+1)]['linker_atoms']
            phi_atoms = linker_atoms[:3]
            psi_atoms = linker_atoms[1:]
            freeze = glyP.utilities.select_freeze(scan_settings["freeze"],linker_atoms)

            detail = scan_settings["detail"]
            #determines the degrees that will be used
            if detail == 'low':
                points = np.linspace(-180,180,num=37)
            elif detail == 'med':
                points = np.linspace(-180,180,num=73)
            elif detail == 'high':
                points = np.linspace(-180,180,num=361)

            conformations=[]
            for phi in points:
                if phi == 180:
                    phi = 179.99
                if phi == -180:
                    phi = -179.99
                for psi in points:
                    if psi == 180:
                        psi = 179.99
                    if psi == -180:
                        psi = -179.99
                    reference_conf.set_glycosidic((ring,ring+1),phi,psi)
                    if glyP.utilities.clashcheck(reference_conf) == False:
                        conformations.append([phi,psi])

        
        jobtype = {"dih" : "opt=modredundant", "atoms": "opt=readopt"}
        scan.set_theory(software='g16', method='PM3', basis_set= '', disp=False,  nprocs=12, mem='24GB', jobtype=jobtype[scan_settings["freeze"]], extra=freeze)

        out.flush()

        for n, thetas in enumerate(conformations):

            _id = '-'.join([reference_conf._id,'{0:03d}'.format(n)])
            if _id in finished_conformers: continue

            scan.append(copy.deepcopy(reference_conf))
            new_conf=scan[-1]
            new_conf._id = '-'.join([new_conf._id,'{0:03d}'.format(n)])
            new_conf.path= '/'.join([scan.path, new_conf._id])

            if os.path.exists(new_conf.path): 
                print("delete", new_conf._id)
                shutil.rmtree(new_conf.path)

            if scan_type == 'pucker':
                glyP.utilities.set_ring_pucker(new_conf, scan_settings["ring"], thetas)
            elif scan_type == 'torsion':
                new_conf.set_glycosidic((ring,ring+1),thetas[0],thetas[1])

            new_conf.create_input(scan.theory, scan.path, software=scan_settings["software"])
            succ_job = new_conf.run_qm(scan.theory, software=scan_settings["software"])

            if not succ_job:
                new_conf.load_log(software=scan_settings["software"])
                new_conf.measure_ring() ; new_conf.measure_c6() 
                print(new_conf)
            else:
                print(new_conf._id + " failed. Moving to the next pucker.")
                del scan[-1]
            out.flush()

        scan.reference_to_zero()
        scan.sort_energy()
        scan.print_relative(pucker=True)


def main():

	#parses the command line arguments, every argument is required
	#python3 puck_scan.py --in_dir glucose -r 0 -d low --out_dir g
	#for more specifics: python3 puck_scan.py -h 

    parser = argparse.ArgumentParser(description='Selects the ring number, level of detail of the scan and the input/output directories')

    parser.add_argument('--scan_type', required=True, help='the type of scan, either torsion or pucker', choices=['pucker', 'torsion'])   
    parser.add_argument('--in_dir', required=False, help='the name of the input directory')
    parser.add_argument('--out_dir', required=False, help='the name of the output directory')
    parser.add_argument('--output',  required=False, help='output log-file')
    parser.add_argument('--ring', '-r', type=int, required=False, help='the index of which ring to select on a molecule')
    parser.add_argument('--detail', '-d', choices=['low','medium','high'], required=False, help='the level of detail of the scan, more detail means more conformations')
    args = parser.parse_args()

    scan(**vars(args))

if __name__ == '__main__':
    main()


