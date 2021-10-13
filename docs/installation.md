## Installation of Behavior Tree Policy Learning Code

Since this is a `limbo` experiment (check the [docs](http://www.resibots.eu/limbo/index.html) of limbo for details), there needs to be no installation. Nevertheless, the dependencies must be installed. We provide scripts for easy installation of the dependencies, configuration and compilation of the source code on Ubuntu-based systems. If you need a more controlled installation of the dependencies and/or configuration/compilation, please check the [advanced installation tutorial](advanced_installation.md) (this is recommended for users experienced with building systems and command line usage).

### How to properly clone this repo

- Clone the repo *recursively*:
  - `git clone --recursive https://github.com/matthias-mayr/behavior-tree-policy-learning.git`
  - or `git clone --recursive git@github.com:matthias-mayr/behavior-tree-policy-learning.git`)

### Dependencies

#### Required
- Ubuntu (it works on version 18.04, possibly newer ones as well)
- limbo, https://github.com/resibots/limbo (for high-performing Gaussian process regression)
- libcmaes, https://github.com/beniz/libcmaes (for high-quality implementations of CMA-ES variants) --- recommended to use with TBB
- Eigen3 (needed by limbo and libcmaes)
- Boost (needed by limbo)
- NLOpt, http://ab-initio.mit.edu/wiki/index.php/NLopt (needed by limbo)
- DART, http://dartsim.github.io/ (for scenarios based on DART)
- robot\_dart, https://github.com/resibots/robot_dart (for the DART integration)
- SDL2 (for visualization)
- BehaviorTree.CPP, https://github.com/BehaviorTree/BehaviorTree.CPP/ (BT implementation)

#### Optional
- TBB, https://www.threadingbuildingblocks.org/ (for parallelization) --- highly recommended
- ROS, https://ros.org - ROS for execution or learning on a real system

### Installation of the dependencies

Some of the dependencies (libcmaes, DART, NLOpt, robot\_dart) require specific installation steps (e.g., compilation from sources). As such, we provide some scripts (under the `scripts` folder) for automatic installation of the dependencies:

#### Install the recommended dependencies

- `cd /path/to/repo/root` **(this is very important as the script assumes that you are in the root of the repo)**
- `./scripts/installation/install_deps.sh`

Using the scripts, all of the custom dependencies (limbo, libcmaes, DART, NLOpt, robot\_dart) will be installed in `/path/to/repo/root/install` in order not to pollute your linux distribution. As such, you should update your `LD_LIBRARY_PATH` (or you can source the proper script --- see below). Consequently no `sudo` is required for these dependencies; nevertheless, `sudo` is still required for installing standard packages (like boost-dev packages, libeigen3-dev, etc).

#### Install ROS

If you want to build the executable used for the execution - learning or final policy - on a real system, ROS is required. Other than that, ROS is an optional dependency.

- `cd /path/to/repo/root` **(this is very important as the script assumes that you are in the root of the repo)**
- `./scripts/installation/install_ros.sh`

### Compilation

As this code is a `limbo` experiment and can sometimes be a bit tricky to compile, we provide the `configure.sh` and `compile.sh` scripts. The former needs to be ran once. The latter will compile all the behavior tree policy learning code. Even your own new scenarios can be compiled with this script (if the files are in the added to the `wscript` file in `src/dart`. In short you should do the following:

- `cd /path/to/repo/root` **(this is very important as the scripts assume that you are in the root of the repo)**
- `./scripts/configure.sh`
- `./scripts/compile.sh`

And then every time you make a change to a source file (*\*.hpp or \*.cpp*), you should re-run the compilation script. If you want to know in more detail how to compile limbo experiments (i.e, not with the scripts), please check the quite extensive [documentation](http://www.resibots.eu/limbo/index.html) of limbo. In addition, if you want more fine-tuned compilation of your own scenarios, please check the [advanced installation tutorial](advanced_installation.md).

