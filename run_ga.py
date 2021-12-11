import glyP
import copy, sys

def run_ga():
    """A genetic algorithm script that implements functions of the glyP package.
    """

    #parameters for the algorithm
    GAsettings = {
            "initial_pool"  : 10,
            "alive_pool"    : 10,
            "generations"   : 60,
            "prob_dih_mut"  : 0.65,
            "prob_c6_mut"   : 0.65,
            "prob_ring_mut" : 0.33,
            "rmsd_cutoff"   : 0.1,
            }


    output = "GAout.log"
    with open(output, 'w') as out: 

        sys.stdout = out 
        sys.stderr = out
        dtime = glyP.utilities.dtime
        print("Initialize:", dtime())

        #Generation
        #Creates a conformer space and fills it with models of conformers. Models are theoretically generated conformers saved as files in a different directory
        GArun = glyP.Space('GA-test', software='fhiaims')
        GArun.load_models('models_tri')

        #GArun.set_theory(nprocs=12, mem='16GB', charge=0, basis_set='STO-3G', jobtype='opt=loose freq')
        #GArun.set_theory(nprocs=24, mem='64GB', charge=1, method='PBE1PBE' , basis_set='6-31G(d)', jobtype='opt=loose', disp=True)
        GArun.set_theory(software='fhiaims', nprocs=24) 

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

                if populate == True:

                    #generate new conformers with existing topol, in order to have num of conformers == size of initial pool 

                    print("Generate initial-{0:02d} Date: {1:30s}".format(n, dtime()))
                    m = glyP.utilities.draw_random_int(len(GArun.models))
                    GArun.append(copy.deepcopy(GArun.models[m]))
                    offspring = GArun[-1]
                    offspring._id = "initial-{0:02d}".format(n)
                    #deepcopy of random conformer in the list

                elif evolve == True:

                     print("Generate offspring-{0:02d} Date: {1:30s}".format(n-IP, dtime()))
                     m = glyP.utilities.draw_random_int(GAsettings['alive_pool'])
                     GArun.append(copy.deepcopy(GArun[m]))
                     offspring = GArun[-1]
                     offspring._id = "offspring-{0:02d}".format(n-IP)

                clash = True ; new_parent = False ; attempt = 0 

                #Attempt to modify the glycosidic bond while avoiding any clashes

                clash = True ; attempt = 0
                while clash: 

                    if attempt > 100: 
                         print("More attempts than threshold, try another parent")
                         del offspring ; del GArun[-1] ; new_parent = True ; break

                    if populate == True:

                        for e in offspring.graph.edges:
                            glyP.ga_operations.modify_glyc(offspring, e)

                        for r in offspring.graph.nodes:
                            glyP.ga_operations.modify_c6(offspring, r)
                            #glyP.ga_operations.modify_ring(GArun[n], r)

                    elif evolve == True: 

                        for e in offspring.graph.edges: 
 
                            if glyP.utilities.draw_random() < GAsettings['prob_dih_mut']:
                                glyP.ga_operations.modify_glyc(offspring, e)

                        for r in offspring.graph.nodes:

                            if glyP.utilities.draw_random() < GAsettings['prob_c6_mut']:
                                glyP.ga_operations.modify_c6(offspring, r)

                            if glyP.utilities.draw_random() < GAsettings['prob_ring_mut']:
                                glyP.ga_operations.modify_ring(offspring, r)

                    attempt += 1 #records the number of attempts before a modification without clashes occurs
                    clash = glyP.utilities.clashcheck(offspring)

                if new_parent == True: continue 

                print("Attempt {0:3d}".format(attempt), end='\n')
                offspring.measure_c6(); offspring.measure_glycosidic() ; offspring.measure_ring()
                offspring.update_vector()

                offspring.create_input(GArun.theory, GArun.path, software='fhiaims')
                succ_job = offspring.run_qm(GArun.theory, software='fhiaims') #with a proper execution of gaussian


            offspring.load_aims('/'.join([GArun.path, offspring._id, 'aims.log']))
            offspring.measure_c6() ; offspring.measure_glycosidic() ; offspring.measure_ring()
            offspring.update_topol(GArun.models)
            offspring.update_vector()
            if populate == True: print("Finished initial-{0:02d} Date: {1:30s}".format(n, dtime()))
            else:                print("Finished offspring-{0:02d} Date: {1:30s}".format(n-IP, dtime()))

            print(offspring)

            #sometimes in calculation a mol will break into 2 molecules (ex torsion), this checks if the current molecule disociates into 2 fragments. If so it is removed.
            if offspring.Nmols > 1: 
                print("{0:3d} molecules present, remove".format(offspring.Nmols))
                del offspring ; del GArun[-1] ; continue 

            #checks for copies
            if n > 0: 
                duplicate = False
                for i in range(n):
                    rmsd = glyP.utilities.calculate_rmsd(offspring, GArun[i])
                    if rmsd < GAsettings['rmsd_cutoff']:
                        print("RMSD: {0:6.3f} already exist as {1:20s}".format(rmsd, GArun[i]._id))
                        duplicate = True ; break
                if duplicate == True: 
                     del offspring ; del GArun[-1] ; continue 

            GArun.sort_energy(energy_function='E')
            GArun.reference_to_zero(energy_function='E')
            GArun.print_relative(alive=GAsettings['alive_pool'])
            out.flush()
            n += 1

if __name__ == '__main__':

    run_ga()
