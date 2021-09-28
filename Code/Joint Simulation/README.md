# Transmission and Distribution (T+D) Joint Simulation
Here we provide the source code and simulation case files for PSS/E and OpenDSS used to perfrom joint simulation

## Prerequisite
- A Windows environment is required since both simulators do not support Linux or MacOS
- Install PSS/E and the Python modules for Python 3.4 (psspy34): https://new.siemens.com/global/en/products/energy/energy-automation-and-smart-grid/pss-software/pss-e.html
- Install OpenDSS: https://www.epri.com/pages/sa/opendss
- - Install required Python dependencies
```angular2html
pip install -r requirements.txt
```

## Usage
The core co-simulation engine is programmed in code script `cosim.py` There are 3 ways to run the code for different purposes:
- Test, develop and debug: `main.py` gives a simple example to initialize the co-simulation engine and run a steady-state or dynamic simulation. In the header the user will need to set the path to the 1) `case_T`: PSS/E transmission system case file (.sav / .raw); 2) `case_D`: OpenDSS distribution system case file (.dss); 'data_path': 3) Input profile data from the PSML dataset. The data handle `data` and co-simulation handle `env` is then created using the input paths. Then the user can use one of the two functions to create and inspect simulated data: 1) `env.solve_ss(hour_num)` which attempts to solve the T+D steady-state power flow for the hour in input profile data specified in the argument. 2) `env.dynsim(hour_num, step_num)` performs a dynamic simulation of a random disturbance event for the hour `hour_num`. The simulation is continued for `step_num` timesteps.
- Create consecutive steady-state output data: In `ss_data.py` set the four path variables in the header: `case_T`, `case_D`, `data_path` are the same as described above. `out_path` is the csv file where simulation output is to be stored. Then, in `run_sspf.bat` set the amount of rows in input profile data to be simulated. Last, execute `run_sspf.bat` and wait for result. Note that if the simulation is interrupted the code is able to find the breakpoint and continue from the row where the last simulation ended, so the user can simply double-click the batch again to continue simulation. 
- Create consecutive transient output data: In `one.py` set the four path variables in the header as described above. Then, set the index of the starting and ending rows in the input profile data in the loop in `run_scenarios.bat` and execute the batch file. The result folders will automatically be created in the specified path.

## References

- User manual for OpenDSS and its Python interface: https://sourceforge.net/p/electricdss/code/HEAD/tree/trunk/Distrib/Doc/OpenDSSManual.pdf
- Tutorials for the Python interface of PSS/E (psspy): http://www.whit.com.au/blog/
- A review paper about T+D joint simulation: https://www.mdpi.com/1996-1073/14/1/12


