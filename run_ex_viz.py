from simulator.wh_sim import *
from simulator.lib import Config, SaveSample
from simulator import CFG_FILES, MODEL_ROOT, STATS_ROOT

###### Experiment parameters ######

ex_id = 'e_1' # Set the experiment ID
verbose = False  # Set verbosity
faults = [0] # Set the number of faults

###### Config class ######

default_cfg_file = CFG_FILES['default'] # Set default config file for experiment
cfg_file = CFG_FILES['ex_1'] # Set experiment config file
cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id)

###### Functions ######

# data_model = None # 
thresh_file = None#os.path.join(MODEL_ROOT, "%s_%s.txt"%(ex_id, "threshold_file"))
stats_file = None#os.path.join(STATS_ROOT, "%s_%s.txt"%(ex_id, "stats_file"))

# Set the fault prediction model
# Dummy model: outputs null predictions
# Simple threshold model: reads threshold from thresh_file, reads saliency statistics from stats_file
ad_model = DummyModel(cfg_obj.warehouse.get('number_of_agents')) 
# ad_model = ExportThresholdModel(10, thresh_file, stats_file, 3, 0.15, 2) 

# Create simulator object
sim = VizSim(cfg_obj,
    # data_model=data_model,
    fault_count=faults,
    ad_model=ad_model,
    random_seed=66764970)

sim.run()
