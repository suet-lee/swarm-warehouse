import os

from simulator.wh_proc import *
from simulator.wh_sim import *
from simulator.lib import Config, SaveSample
from simulator import CFG_FILES

###### Hardcoded parameters ######

iterations = 20
it_offset = 0
verbose = False    
fault_range = range(1,11) # inject 0-10 faults
batch_id = 'metrics_dist'
data_dir = "data"

###### Config class ######

default_cfg_file = CFG_FILES['default']
cfg_file = CFG_FILES['ex_1']


###### Run baseline experiment ######

random_seed = 0
faults = 0
data = []
print("-- running %s"%"baseline")
cfg_obj = Config(cfg_file, default_cfg_file, ex_id='e_1')
for it in range(iterations):
    dm = DataModel(store_internal=True, compute_roc=True)
    # Create simulator object
    sim = Simulator(cfg_obj, 
        verbose=verbose,
        data_model=dm,
        random_seed=random_seed,
        fault_count=[0])
    sim.run()
    data.append(sim.warehouse.delivered)
    data.append(sim.delivered_in)
    random_seed += 1

    data_dir = "data/"+batch_id+"/baseline"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    print("saving...")
    filepath = os.path.join(data_dir,"data_%d.txt"%it)
    dm.get_dataframe().to_csv(filepath)
    print("-- current random seed : %d"%random_seed)

###### Run script ######

ex_ids = ['e_1','e_2','e_3','e_4','e_5','e_6']

for ex_id in ex_ids:
    print("-- running %s"%ex_id)
    cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id)
    for f_number in fault_range:
        print("---- fault no. %d"%f_number)
        for it in range(iterations):
            dm = DataModel(store_internal=True, compute_roc=True)
            # Create simulator object
            sim = Simulator(cfg_obj, 
                verbose=verbose,
                data_model=dm,
                random_seed=random_seed,
                fault_count=[f_number])
            sim.run()
            random_seed += 1

            data_dir = "data/"+batch_id+"_2/%s/%d"%(ex_id,f_number)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            print("saving...")
            filepath = os.path.join(data_dir,"data_%d.txt"%it)
            dm.get_dataframe().to_csv(filepath)
            print("-- current random seed : %d"%random_seed)