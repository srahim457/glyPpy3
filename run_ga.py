import glyP
import copy


GAsettings = {
        "initial_pool"  : 20,
        "alive_pool"    : 10,
        "generations"   : 50,
        "prob_dih_mut"  : 0.65,
        }
        


n = 0 #Generation

GArun = glyP.Space('GA-test')
GArun.load_models('models_tri')
#GArun.set_theory(nprocs=12, mem='16GB', charge=0, basis_set='STO-3G', jobtype='opt=loose freq')
GArun.set_theory(nprocs=12, mem='16GB', charge=1, method='pm6' , basis_set=' ', jobtype='opt=loose freq', disp=False)

while n < GAsettings['initial_pool']:
    print("generate structure {0:5d}".format(n))
    m = glyP.utilities.draw_random_int(len(GArun.models))
    GArun.append(copy.deepcopy(GArun.models[m]))
    GArun[n]._id = "initial-{0:02d}".format(n)

    succ_job = 1 
    while succ_job: 
        clash = True ; attempt = 0
        while clash: 
            for e in GArun[n].graph.edges:
                glyP.ga_operations.modify_glyc(GArun[n], e)
            attempt += 1
            clash = glyP.utilities.clashcheck(GArun[n])
        print("Attempt {0:3d}".format(attempt), end='\n')
        GArun[n].measure_glycosidic() ; GArun[n].measure_ring()
        GArun[n].update_vector()
        #print("Initial vector:", GArun[n].ga_vector)
        #print(GArun[n])
        GArun[n].create_input(GArun.theory, GArun.path)
        succ_job = GArun[n].run_gaussian()

    GArun[n].load_log('/'.join([GArun.path, GArun[n]._id, 'input.log']))
    GArun[n].measure_glycosidic() ; GArun[n].measure_ring()
    GArun[n].update_vector()
    #print("Final vector", GArun[n].ga_vector)
    print(GArun[n])

    GArun.sort_energy(energy_function='F')
    GArun.reference_to_zero(energy_function='F')
    GArun.print_relative()
    n += 1

n = 0

GArun.print_relative()
#for conf in GArun: print(conf)

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
        clash = True
        for e in offspring.graph.edges: 
            edge = offspring.graph.edges[e]
            link_lenght = len(edge['linker_atoms'])
            if glyP.utilities.draw_random() < GAsettings['prob_dih_mut']:
                while clash:
                    glyP.ga_operations.modify_glyc(GArun[n], e)
                    clash = glyP.utilities.clashcheck(offspring)

        offspring.measure_glycosidic() ; offspring.measure_ring()
        offspring.update_vector()
        #print(iniitial vector:", offspring.ga_vector)   
        offspring.create_input(GArun.theory, GArun.path)
        succ_job = offspring.run_gaussian()

    offspring.load_log('/'.join([GArun.path, offspring._id, 'input.log']))
    offspring.measure_glycosidic() ; offspring.measure_ring()
    offspring.update_vector()
    print(offspring)
    #print("Final vector", offspring.ga_vector)


    GArun.sort_energy(energy_function='F')
    GArun.reference_to_zero(energy_function='F')
    GArun.print_relative()


    n += 1 

