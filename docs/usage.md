# Running Scenarios

## Learning Policies
- Before running any executable you should source the proper paths: `source ./scripts/paths.sh` **(the script assumes that you are in the root of the repo)**
- All the executables including your own new scenarios (assuming the compilation produced no errors) should be located in the `deps/limbo/build/exp/blackdrops/src/` folder
- For example if we want to run the peg insertion scenario without any visualization, we should use: `./deps/limbo/build/exp/blackdrops/src/dart/iiwa_skills_peg_simu [args]` (you can get help on what arguments to use, with `/path/to/binary --help`)
- For your convenience there is a script that sets some reasonable default options in `./scripts/experiments/run_iiwa.sh {peg, obstacle, combined}`

### Learning Output
Each learning process will create a folder in `/tmp/blackdrops/{experiment}/{date}{time}`. It will contain these folders and files:

* `optimizations`: Rewards for all as well as the best optimization iterations
* `parameters`: Parameter configurations from before (initial) and after (best) the optimization. Eigen vectors in binary format.
* `trajectories`:
  * `traj_opt_final_0.dat`: The states and actions of the episode with parameters after the learning
  * `traj_opt_final_ee_0.dat`: End effector trajectory of that episode
* `DONE`: File indicating that the learning has finished
* `ERROR_{date}`: Error log file
* `estimates.dat`: Rewards of the post-learning episode for every time step
* `expected.dat`: Total reward of that episode
* `INFO_{data}`: Info log message
* `times.dat`: Learning time in seconds
* `WARNING_{date}`: Warning log file

There are some additional files that are currently not used.

**Warning:** The results are saved in `/tmp` and nowhere else unless you copy them. This means that on most systems they are lost when rebooting the machine.

## Running a Specific Parameter Set
After learning or for debugging and design reasons, it can be desirable to run some specific parameter sets. Therefore, each experiment has a matching `play_param_config_{experiment}` executable.

- Before running any executable you should source the proper paths: `source ./scripts/paths.sh` **(the script assumes that you are in the root of the repo)**
- For example the executable for the peg insertion scenario is located in: `./deps/limbo/build/exp/blackdrops/src/dart/play_params_skills_peg_graphic [args]`
- For your convenience there is a script in `./scripts/experiments/play_param_config.sh {peg, obstacle, combined}`
- The executable will look for the learning output folders mentioned above in `/tmp/blackdrops/`. You'd be interactively asked which learning process folder to use and which parameter set to run.

You will see the simulation window pop up and the whole episode will be visualized. By default, domain randomization is turned on, so every run will be slightly different.

### Quick Start with Example Parameters

This repository includes some example parameter sets that can be copied in the appropriate place (instead of learning them) with this script:
```
./scripts/experiments/install_example_parameters.sh
```
Subsequently they can be played as usual:
```
./scripts/experiments/play_param_config.sh peg
./scripts/experiments/play_param_config.sh obstacle
```

## Run the Execution on a real System
Every robot setup is different and since our torso arrangement is unique, we decided to not include launch files for the robot setup. However, we provide the full implementation of the controller and the tooling used to run the experiments, so you could re-run everything on your setup.

- Before running any executable you should source the proper paths: `source ./scripts/paths.sh`
- The executable is located in: `./deps/limbo/build/exp/blackdrops/src/dart/iiwa_skills_execution_simu [args]`
- Please note that it is only built if ROS is detected when executing the `configure.sh` script.
- For your convenience there is a script `./scripts/experiments/run_execution.sh {peg, obstacle, combined} {parameters}.csv`.
- Additionally, there is a script `./scripts/experiments/run_execution_all.sh` that cycles through all the start positions of an experiment for evaluation purposes.