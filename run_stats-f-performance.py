from simulator.wh_sim import *
from simulator.lib import Config
from simulator import CFG_FILES
import os

###### Hardcoded parameters ######

iterations = 100
it_offset = 0
verbose = False    
fault_range = range(1,11) # inject 0-10 faults
batch_id = 'faults_perf'

###### Config class ######

default_cfg_file = CFG_FILES['default']
cfg_file = CFG_FILES['ex_1']

    
###### Create data directory ######

data_dir = "data/"+batch_id
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

###### Run baseline experiment ######

random_seed = 0
faults = 0
data = []
print("-- running %s"%"baseline")
cfg_file = CFG_FILES['ex_1']
cfg_obj = Config(cfg_file, default_cfg_file, ex_id='e_1')
for it in range(iterations):
    # Create simulator object
    sim = Simulator(cfg_obj, 
        verbose=verbose,
        data_model=None,
        random_seed=random_seed,
        fault_count=[0])
    sim.run()
    data.append(sim.warehouse.delivery_log)
    random_seed += 1

print("saving...")
filepath = os.path.join(data_dir,"data.txt")
with open(filepath, 'a') as f:
    f.write(str(data))
    f.write("\n")
print("-- current random seed : %d"%random_seed)

###### Run fault experiments ######

ex_ids = ['e_1','e_2','e_3','e_4','e_5','e_6']

for ex_id in ex_ids:
    print("-- running %s"%ex_id)
    cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id)
    data = []
    for f_number in fault_range:
        print("---- fault no. %d"%f_number)
        for it in range(iterations):
            # Create simulator object
            sim = Simulator(cfg_obj, 
                verbose=verbose,
                data_model=None,
                random_seed=random_seed,
                fault_count=[f_number])
            sim.run()
            data.append(sim.warehouse.delivery_log)
            random_seed += 1

    print("saving...")
    filepath = os.path.join(data_dir,"data.txt")
    with open(filepath, 'a') as f:
        f.write(str(data))
        f.write("\n")
    print("-- current random seed : %d"%random_seed)
