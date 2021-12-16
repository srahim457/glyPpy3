import glyP
import copy, sys


GAsettings = {
            "initial_pool"  : 10,
            "alive_pool"    : 10,
            "generations"   : 60,
            "prob_dih_mut"  : 0.65,
            "prob_c6_mut"   : 0.65,
            "prob_ring_mut" : 0.33,
            "pucker_P_model": [0.5, 0.20, 0.20, 0.05, 0.05],
            "rmsd_cutoff"   : 0.1,
            "output"        : "GAout.log",
            "software"      : "fhiaims"
            }

dtime = glyP.utilities.dtime

def  spawn_initial(GArun, n):

    m = glyP.utilities.draw_random_int(len(GArun.models))
    GArun.append(copy.deepcopy(GArun.models[m]))
    GArun[-1]._id = "initial-{0:02d}".format(n)
    GArun[-1].path= '/'.join([GArun.path, GArun[-1]._id])
    print("Generate initial-{0:02d} form {1:20s}  Date: {2:30s}".format(n, GArun.models[m]._id, dtime()))

def spawn_offspring(GArun, n, IP=GAsettings['initial_pool']):

    m = glyP.utilities.draw_random_int(GAsettings['alive_pool'])
    GArun.append(copy.deepcopy(GArun[m]))
    GArun[-1]._id = "offspring-{0:02d}".format(n-IP)
    GArun[-1].path= '/'.join([GArun.path, GArun[-1]._id])
    print("Generate offspring-{0:02d} from {1:20s} Date: {2:30s}".format(n-IP, GArun[m]._id, dtime()))

def remove_duplicates(GArun):

    for i in range(len(GArun)-1):
        rmsd = glyP.utilities.calculate_rmsd(GArun[-1], GArun[i])
        if rmsd < GAsettings['rmsd_cutoff']:
            print("RMSD: {0:6.3f} already exist as {1:20s}".format(rmsd, GArun[i]._id))
            return True 
    return False

def run_ga():
    """A genetic algorithm script that implements functions of the glyP package.
    """
    output = GAsettings["output"]
    with open(output, 'w') as out: 

        #sys.stdout = out 
        #sys.stderr = out
        print("Initialize:", dtime())

        #Generation
        #Creates a conformer space and fills it with models of conformers. Models are theoretically generated conformers saved as files in a different directory
        GArun = glyP.Space('GA-test', software=GAsettings["software"])
        GArun.load_models('models_tri')
        GArun.load_Fmaps('Fmaps')

        GArun.set_theory(nprocs=12, mem='16GB', charge=1, method='PM3', basis_set='', jobtype='opt=loose', disp=False)
        #GArun.set_theory(nprocs=24, mem='64GB', charge=1, method='PBE1PBE' , basis_set='6-31G(d)', jobtype='opt=loose', disp=True)
        #GArun.set_theory(software='fhiaims', charge='1.0', basis_set='light', nprocs=24) 

        n = len(GArun) #the number of models in the directory, the following checks if the directory is empty
        if n != 0: #if the models exits in folder it will print out the energy of everything
            GArun.sort_energy(energy_function='E')
            GArun.reference_to_zero(energy_function='E')
            GArun.print_relative(alive=GAsettings['alive_pool'])
        out.flush()

        populate = True ; evolve = False ; IP = GAsettings['initial_pool']

        while True:

            if n >= IP: populate = False ; evolve = True
            if n >= (GAsettings['generations'] + IP) : break

            succ_job = 1
            while succ_job: 

                if    populate == True: spawn_initial(GArun, n)
                elif  evolve   == True: spawn_offspring(GArun,n)

                offspring = GArun[-1]

                clash = True ; new_parent = False ; attempt = 0 
                xyz_backup = copy.copy(offspring.xyz)

                while clash:

                    #print("Attempt {0:3d}".format(attempt), end='\n')
                    if attempt > 100: 
                         print("More attempts than threshold, trying another parent.")
                         del offspring ; del GArun[-1] ; new_parent = True ; break

                    if attempt > 0:
                        offspring.xyz = copy.copy(xyz_backup)

                    if populate == True:

                        for e in offspring.graph.edges:
                            glyP.ga_operations.modify_glyc(offspring, e)

                        for r in offspring.graph.nodes:
                            glyP.ga_operations.modify_c6(offspring, r)
                            #glyP.ga_operations.modify_ring(offspring, r)

                    elif evolve == True: 

                        for e in offspring.graph.edges: 
 
                            if glyP.utilities.draw_random() < GAsettings['prob_dih_mut']:
                                glyP.ga_operations.modify_glyc(offspring, e, model = "Fmaps", Fmap = GArun.linkages)

                        for r in offspring.graph.nodes:

                            if glyP.utilities.draw_random() < GAsettings['prob_c6_mut']:
                                glyP.ga_operations.modify_c6(offspring, r)

                            if glyP.utilities.draw_random() < GAsettings['prob_ring_mut']:
                                glyP.ga_operations.modify_ring(offspring, r, GAsettings['pucker_P_model'])

                    attempt += 1 #records the number of attempts before a modification without clashes occurs
                    clash = glyP.utilities.clashcheck(offspring)

                if new_parent == True: continue 

                #print("Attempt {0:3d} successful".format(attempt), end='\n')
                offspring.measure_c6(); offspring.measure_glycosidic() ; offspring.measure_ring()

                offspring.create_input(GArun.theory, GArun.path, software=GAsettings["software"])
                succ_job = offspring.run_qm(GArun.theory, software=GAsettings["software"]) #with a proper execution of gaussian

            offspring.load_log(software=GAsettings["software"])
            offspring.measure_c6() ; offspring.measure_glycosidic() ; offspring.measure_ring()
            offspring.update_topol(GArun.models)

            #sometimes in calculation a mol will break into 2 molecules (ex torsion), this checks if the current molecule disociates into 2 fragments. If so it is removed.
            if offspring.Nmols > 1: 
                print("{0:3d} molecules present, remove".format(offspring.Nmols))
                del offspring ; del GArun[-1] ; continue 

            #checks for copies
            if n > 0: 
                if remove_duplicates(GArun): 
                    del GArun[-1] ; continue 

            if populate == True: print("Finished initial-{0:02d}{1:27s} Date: {2:30s}".format(n, '', dtime()))
            else:                print("Finished offspring-{0:02d}{1:27s} Date: {2:30s}".format(n-IP, '', dtime()))
            print(offspring)

            GArun.sort_energy(energy_function='E')
            GArun.reference_to_zero(energy_function='E')
            GArun.print_relative(alive=GAsettings['alive_pool'])
            out.flush()
            n += 1

if __name__ == '__main__':

    run_ga()
