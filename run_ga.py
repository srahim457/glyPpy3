import glyP
import copy


GAsettings = {
        "initial_pool"  : 10,
        "alive_pool"    : 6,
        "generations"   : 30,
        "prob_dih_mut"  : 0.65,
        }
        


n = 0 #Generation

GArun = glyP.Space('GA-test')
GArun.load_models('models')
GArun.set_theory(nprocs=12, mem='16GB', charge=0, basis_set='STO-3G', jobtype='opt=loose freq')

while n < GAsettings['initial_pool']:
    print("generate structure {0:5d}".format(n))

    m = glyP.utilities.draw_random_int(len(GArun.models))
    GArun.append(copy.deepcopy(GArun.models[m]))
    GArun[n]._id = "initial-{0:02d}".format(n)
    clash = True
    while clash: 
        for d in range(len(GArun[n].dih_atoms)):
            phi, psi = ((glyP.utilities.draw_random()*360)-180.0, (glyP.utilities.draw_random()*360)-180.0)
            glyP.ga_operations.modify_glyc(GArun[n], d, phi, psi)
        clash = glyP.utilities.clashcheck(GArun[n])

    GArun[n].measure_glycosidic()
    GArun[n].update_vector()

    GArun[n].create_input(GArun.theory, GArun.path)
    GArun[n].run_gaussian()
    GArun[n].load_log('/'.join([GArun.path, GArun[n]._id, 'input.log']))
    GArun[n].udate_vector()

    GArun.sort_energy(energy_function='E')
    print(GArun)
    n += 1

n = 0

while n < GAsetting['generations']:
    print("generate offspring {0:5d}".format(n))

    #draw random structure:

    m = glyP.utilities.draw_random_int(GAsettings['alive_pool'])
    GArun.append(copy.deepcopy(GArun.models[m]))
    N = n + GAsettings['initial_pool']
    GArun[N]._id = "offspring-{0:02d}".format(n)     

    #1. Random mutation at each glyc bond
    clash = True
    for d in range(len(GArun[N].dih_atoms)):
        if utilities.draw.random() < GAsetting['prob_dih_mut']:
            while clash: 
                phi, psi = ((glyP.utilities.draw_random()*360)-180.0, (glyP.utilities.draw_random()*360)-180.0)
                glyP.ga_operations.modify_glyc(GArun[N], d, phi, psi)
            clash = glyP.utilities.clashcheck(GArun[N])

    GArun[N].measure_glycosidic()
    GArun[N].update_vector()
    
    GArun[N].create_input(GArun.theory, GArun.path)
    GArun[N].run_gaussian()
    GArun[N].load_log('/'.join([GArun.path, GArun[N]._id, 'input.log']))
    GArun[N].udate_vector()



    GArun.sort_energy(energy_function='E')
    print(GArun)
    n += 1 

