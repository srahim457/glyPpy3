import glyP
import copy, sys

def run_ga():
    """A genetic algorithm script that implements functions of the glyP package.
    """

    #parameters for the algorithm
    GAsettings = {
            "initial_pool"  : 10,
            "alive_pool"    : 6,
            "generations"   : 10,
            "prob_dih_mut"  : 0.65,
            "rmsd_cutoff"   : 0.1,
            "prob_rattle_C6": 0.65
            }


    #nnodes = 2
    output = "GAout.log"
    with open(output, 'w') as out: 

        #sys.stdout = out 
        #sys.stderr = out
        #Initialization:
        dtime = glyP.utilities.dtime
        print("Initialize:", dtime())
        #Generation
        #Creates a conformer space and fills it with models of conformers. Models are theoretically generated conformers saved as files in a different directory
        GArun = glyP.Space('GA-test')
        GArun.load_models('models_tri')
        #GArun.set_theory(nprocs=12, mem='16GB', charge=0, basis_set='STO-3G', jobtype='opt=loose freq')
        GArun.set_theory(nprocs=12, mem='16GB', charge=1, method='pm6' , basis_set=' ', jobtype='opt=loose', disp=False)

        n = len(GArun) #the number of models in the directory, the following checks if the directory is empty
        if n != 0: #if the models exits in folder it will print out the energy of everything
            GArun.sort_energy(energy_function='E')
            GArun.reference_to_zero(energy_function='E')
            GArun.print_relative(alive=10)
        else:
            glyP.utilities.error("The directory of models is empty")

        out.flush()
        while n < GAsettings['initial_pool']:

            #generate new conformers with existing topol, in order to have num of conformers == size of initial pool 
            print("Generate initial-{0:02d} Date: {1:30s}".format(n, dtime()))
            m = glyP.utilities.draw_random_int(len(GArun.models))
            GArun.append(copy.deepcopy(GArun.models[m]))
            GArun[n]._id = "initial-{0:02d}".format(n)
            #deepcopy of random conformer in the list

            succ_job = 1
            while succ_job: 

                #Attempt to modify the glycosidic bond while avoiding any clashes
                #GArun[n] is used because that should be the most recently generated conformer appended to the list GArun
                clash = True ; attempt = 0
                while clash: 

                    for e in GArun[n].graph.edges:
                        glyP.ga_operations.modify_glyc(GArun[n], e)

                    for r in GArun[n].graph.nodes:
                        glyP.ga_operations.modify_c6(GArun[n], r)

                    attempt += 1 #records the number of attempts before a modification without clashes occurs
                    clash = glyP.utilities.clashcheck(GArun[n])

                print("Attempt {0:3d}".format(attempt), end='\n')
                GArun[n].measure_c6(); GArun[n].measure_glycosidic() ; GArun[n].measure_ring()
                GArun[n].update_vector()
                GArun[n].create_input(GArun.theory, GArun.path)
                succ_job = GArun[n].run_gaussian() #with a proper execution of gaussian

            GArun[n].load_log('/'.join([GArun.path, GArun[n]._id, 'input.log']))
            GArun[n].measure_c6() ; GArun[n].measure_glycosidic() ; GArun[n].measure_ring()
            GArun[n].update_topol(GArun.models)
            GArun[n].update_vector()
            print("Finished initial-{0:02d} Date: {1:30s}".format(n, dtime()))
            print(GArun[n])

            #sometimes in calculation a mol will break into 2 molecules (ex torsion), this checks if the current molecule disociates into 2 fragments. If so it is removed.
            if GArun[n].Nmols > 1: 
                print("{0:3d} molecules present, remove".format(GArun[n].Nmols)
                del GArun[n] ; continue 

            #checks for copies
            if n > 0: 
                duplicate = False
                for i in range(n):
                    rmsd = glyP.utilities.calculate_rmsd(GArun[n], GArun[i])
                    if rmsd < GAsettings['rmsd_cutoff']:
                        print("RMSD: {0:6.3f} already exist as {1:20s}".format(rmsd, GArun[i]._id))
                        duplicate = True ; break
                if duplicate == True: 
                     del GArun[n] ; continue 

            GArun.sort_energy(energy_function='E')
            GArun.reference_to_zero(energy_function='E')
            GArun.print_relative()
            out.flush()
            n += 1

        n -= GAsettings['initial_pool'] #this should set n back to 0

        #This limits the number of generations the algorithm will run for
        #The actual genetic algorithm starts here
        #the implementation below follows that of the set up above where the initial size of the pool is populated. 
        #The only difference is that only 1 conformer is mutated per generation and the random selection is narrowed a few of the most optimal conformers
        while n < GAsettings['generations']:

            print("Generate offspring-{0:02d} Date: {1:30s}".format(n, dtime()))
            total_population = n + GAsettings['initial_pool'] #the number of all conformers generated
        
            #draw random structure from the alive pool
            m = glyP.utilities.draw_random_int(GAsettings['alive_pool'])
            GArun.append(copy.deepcopy(GArun[m]))
            offspring = GArun[-1]
            offspring._id = "offspring-{0:02d}".format(n)     
        
            succ_job = 1
            #implement a condition attempt mutating the bond 50 times and stop it if it doesnt work
            while succ_job:
                #Random mutation at each glyc bond
                clash = True ; attempt = 0 
                while clash:
                    #check for clashing after the mutation
                    for e in offspring.graph.edges: 
                        if glyP.utilities.draw_random() < GAsettings['prob_dih_mut']:
                            glyP.ga_operations.modify_glyc(offspring, e)

                    for r in GArun[n].graph.nodes:
                        if glyP.utilities.draw_random() < GAsettings['prob_rattle_C6']:
                            glyP.ga_operations.modify_c6(offspring, r)

                    attempt += 1
                    clash = glyP.utilities.clashcheck(offspring)
        
                print("Attempt {0:3d}".format(attempt), end='\n')
                offspring.measure_c6() ; offspring.measure_glycosidic() ; offspring.measure_ring()
                offspring.update_vector()
                offspring.create_input(GArun.theory, GArun.path)
                succ_job = offspring.run_gaussian()
        
            offspring.load_log('/'.join([GArun.path, offspring._id, 'input.log']))
            offspring.measure_c6() ; offspring.measure_glycosidic() ; offspring.measure_ring()
            offspring.update_topol(GArun.models)
            offspring.update_vector()
            print("Finished offspring-{0:02d} Date: {1:30s}".format(n, dtime()))
            print(offspring)

            if offspring.Nmols > 1: 
                print("{0:3d} molecules present, remove".format(offspring.Nmols)
                del offspring ; continue 


            duplicate = False
            for i in range(total_population):
                rmsd = glyP.utilities.calculate_rmsd(offspring, GArun[i])
                if rmsd < GAsettings['rmsd_cutoff']:
                    print("RMSD: {0:6.3f} already exist as {1:20s}".format(rmsd, GArun[i]._id))
                    duplicate = True ; break
            if duplicate == True:
                 del GArun[-1] ; continue

            #The conformer mutated to make a new conformer in the pool is selected randomly of the indices [0-alive_pool] so for example this would be the first 6 conformers on the list if alive_pool=6
            #after the new mutated conformer is added to the end of the list and the list is sorted only the best conformers will occupy the top of the list. 
            GArun.sort_energy(energy_function='E')
            GArun.reference_to_zero(energy_function='E')
            GArun.print_relative(alive=10)
            out.flush()

            n += 1 

if __name__ == '__main__':

    run_ga()
