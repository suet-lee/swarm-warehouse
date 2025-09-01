### Overview

The code in this repository simulates an intralogistics task for a robot swarm. The robots operate in a 500 by 500 centimeter bounded arena, referred to as the warehouse: the task is to retrieve and deliver them to the drop-off zone (a 25cm-wide vertical strip extending along the length of the right-hand wall). The simulator provides tools for data generation for fault detection.

This simulator supports the work published at: 10.1109/LRA.2022.3189789 

### Requirements

Python modules:
- numpy
- pandas
- scipy
- seaborn
- yaml

### Scripts

Simulation run scripts:

| Name | Description |
| ----------- | ----------- |
| run_ex.py | Run the experiment with given configuration, no visualization | 
| run_ex_multic.py | Run the experiment with given configuration, no visualization, multicore processing option available |
| run_ex_viz.py | Run the experiment with Matplotlib visualization | 
| run_thresh.py | Run experiment with fault detection (threshold model) | 


Data processing scripts:

| Name | Description |
| ----------- | ----------- |
| run_sampler.py | Processes standard data samples (generated from DataModel class) | 
| run_min_sampler.py | Processes minimal data samples (generated from MinimalDataModel and ExtremeMinDataModel classes) | 
| run_stats.py | Computes effect size and thresholds for metrics for given experiment | 

### Data generation and analysis pipeline

Standard pipeline:  

1. Generate data by running a simulation script for experimental configuration e.g. N iterations, faults_number = {0,1,...,n}.  
2. Process generated data with the appropriate data processing script (dependent on the DataModel class used).  
3. Compute statistics: effect size and thresholds.
4. Run threshold model for fault prediction.

