import glyP
import copy, sys


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

     #Generation
    GArun = glyP.Space('GA-test')
    GArun.load_models('models_tri')
    #GArun.set_theory(nprocs=12, mem='16GB', charge=0, basis_set='STO-3G', jobtype='opt=loose freq')
    GArun.set_theory(nprocs=12, mem='16GB', charge=1, method='pm6' , basis_set=' ', jobtype='opt=loose', disp=False)

    n = len(GArun)
    if n != 0: 
        GArun.sort_energy(energy_function='E')
        GArun.reference_to_zero(energy_function='E')
        GArun.print_relative(alive=10)

    out.flush()
    while n < GAsettings['initial_pool']:

  
        print("generate initial {0:5d}".format(n))
        m = glyP.utilities.draw_random_int(len(GArun.models))
        GArun.append(copy.deepcopy(GArun.models[m]))
        GArun[n]._id = "initial-{0:02d}".format(n)

        succ_job = 1
        while succ_job: 

            clash = True ; attempt = 0
            while clash: 

                for e in GArun[n].graph.edges:
                    glyP.ga_operations.modify_glyc(GArun[n], e)

                for r in GArun[n].graph.nodes:
                    glyP.ga_operations.modify_c6(GArun[n], r)

                attempt += 1
                clash = glyP.utilities.clashcheck(GArun[n])

            print("Attempt {0:3d}".format(attempt), end='\n')
            GArun[n].measure_c6(); GArun[n].measure_glycosidic() ; GArun[n].measure_ring()
            GArun[n].update_vector()
            GArun[n].create_input(GArun.theory, GArun.path)
            succ_job = GArun[n].run_gaussian()

        GArun[n].load_log('/'.join([GArun.path, GArun[n]._id, 'input.log']))
        GArun[n].measure_c6() ; GArun[n].measure_glycosidic() ; GArun[n].measure_ring()
        GArun[n].update_topol(GArun.models)
        GArun[n].update_vector()
        print(GArun[n])

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

    n -= GAsettings['initial_pool']
    while n < GAsettings['generations']:

        print("generate offspring {0:5d}".format(n))
        N = n + GAsettings['initial_pool']
    
        #draw random structure from the alive pool
        m = glyP.utilities.draw_random_int(GAsettings['alive_pool'])
        GArun.append(copy.deepcopy(GArun[m]))
        offspring = GArun[-1]
        offspring._id = "offspring-{0:02d}".format(n)     
    
        succ_job = 1
        while succ_job:
            #1. Random mutation at each glyc bond
            clash = True ; attempt = 0 
            while clash: 
                for e in offspring.graph.edges: 
                    if glyP.utilities.draw_random() < GAsettings['prob_dih_mut']:
                        glyP.ga_operations.modify_glyc(offspring, e)

                for r in GArun[n].graph.nodes:
                    if glyP.utilities.draw_random() < GAsettings['prob_rattle_C6']:
                        glyP.ga_operations.modify_c6(GArun[n], r)

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
        print(offspring)

        duplicate = False
        for i in range(N):
            rmsd = glyP.utilities.calculate_rmsd(offspring, GArun[i])
            if rmsd < GAsettings['rmsd_cutoff']:
                print("RMSD: {0:6.3f} already exist as {1:20s}".format(rmsd, GArun[i]._id))
                duplicate = True ; break
        if duplicate == True:
             del GArun[-1] ; continue

        GArun.sort_energy(energy_function='E')
        GArun.reference_to_zero(energy_function='E')
        GArun.print_relative(alive=10)
        out.flush()

        n += 1 

