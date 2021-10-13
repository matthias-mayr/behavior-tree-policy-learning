## Implemented Scenarios

In this page, we will briefly present all the already implemented scenarios and how to run/use them properly. Note that if you have installed the code following the advanced installation procedure, you should replace all occurrences of `./deps/limbo/` with `./path/to/limbo/`.

This figure shows the combined task with obstacle avoidance and peg insertion:
<center>
<img src="../res/fig/experiment.svg.png" width="500">
</center>

For the individual tasks, the unneeded objects are removed.

All tasks share the following properties:
- Robot is controlled in joint velocity (servo) mode in simulation
- The RL step rate and therefore also the control rate is 50Hz
- Action space consists of the 7 commanded joint velocities
- State space are the 7 joint positions + 7 joint velocities
- There are several possible start configurations of an episode 

### Obstacle Avoidance Task

This task is designed to that both parameters in a movement skill as well as conditions in a behavior tree can be learned. Furthermore, assuming that the red object is a fragile object in the workspace, it should be avoided to learn potentially dangerous policies in reality.

- Duration of 16 seconds for each episode
- 1 evaluation per parameter configuration

#### How to run it

The recommended parameters to use are the following:

- 5000 iterations
- 5 restart for CMA-ES
- Enable elitism for CMA-ES
- [-2,2] the boundaries for the parameters of the policy
- Enable CMA-ES stochasticity

In short, you should run: `./deps/limbo/build/exp/blackdrops/src/dart/iiwa_skills_obstacle_simu -m 5000 -r 5 -e 1 -b 2 -d -1 -s`


### Peg Insertion Task

This task shows that this algorithm and the policy formulation can perform when doing contact-rich tasks. We learn this task in simulation. However, to demonstrate that this policy can be learned directly on a real system if an accurate simulation is not accessible, we also learned it with the real system. Note that the parameter space in this task does not allow to produce dangerous policies.

- Duration of 15 seconds for each episode
- 1 evaluation per parameter configuration

#### How to run it on the real System

The recommended parameters to use are the following:

- 200 iterations
- 5 restart for CMA-ES
- Disable elitism for CMA-ES
- Population size (lambda) of 10
- [-2,2] the boundaries for the parameters of the policy. These will be internally scaled to reasonable values for this task.
- Enable CMA-ES stochasticity
- Enable learning on the real system

In short, you should run: `./deps/limbo/build/exp/blackdrops/src/dart/iiwa_skills_peg_simu -m 200 -l 10 -r 5 -e 0 -b 2 -d 1 -s -v -y`

#### How to run it in Simulation

The recommended parameters to use are the following:

- 5000 iterations
- 5 restart for CMA-ES
- Enable elitism for CMA-ES
- [-2,2] the boundaries for the parameters of the policy. These will be internally scaled to reasonable values for this task.
- Enable CMA-ES stochasticity

In short, you should run: `./deps/limbo/build/exp/blackdrops/src/dart/iiwa_skills_peg_simu -m 5000 -r 5 -e 1 -b 2 -d -1 -s`

### Combined Task

This task combines the policies of the two aforementioned tasks. It was used the execution of separately learned policies in simulation and on the real system only. The setup correspons to the figure shown above.