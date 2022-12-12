## EC545: Cyber-Physical Systems
#### Final Project: ROSMASTER x3 Tag

| Branch Name | Languages | Aim | Assumptions |
| -- | -- | -- | -- |
| 1. soft-test | Python | testing algorithms | complete info: postition, borders, no transfer lag |
| 2. hard-test | Python, C++, ROS | other programs are 'safe' | algorithms that are not being tested will be set to 'safe' outputs |
| 3. master | Python, C++, ROS | final testing | no assumed information |

#### Directory: Simulation
The simulation of the tag-planner is straight forward to run. The *CVXOPT* Python library must be installed. The simulation can then be run from the *root.py* executable. When run the user will have two options for the output, and three for the type of environment being implemented.

1. Threshold Plots:
    - options: polygon-world (p), sphere-world (s) or all-world (a)
2. Animation:
    - options: polygon-world (p), sphere-world (s) or all-world (a)


#### Directory: MATLAB
The MATLAB directory contains the Simulink and init.m files which launches the Manager FSM. To launch...

1. Launch MATLAB and Simulink.
2. Open the FSM diagram titled *game.slx*.
3. Before running the FSM, initialize the game variables by running the *init.m* script.
4. The robots must have position data posted to the *amcl_pose* ROSTOPIC posted before the FSM can be run correctly.
5. The FSM can now run.

The terminal commands that list the outputs of the FSM are:

1. *rostopic echo bernard_pose*
2. *rostopic echo scrappy_pose*
3. *rostopic echo oswaldo_pose*

#### Directory: project_ws
This folder communicates directly to the *roscore* during game run-time. Assuming the *roscore* and *rospkg* is initialized properly, and the robots are localizing to the same */map* topic, the tag planners can be launched using the following command.

*rosrun tag planner.py RobotName __name:=RobotName*

Where *RobotName* is the unique identifier for the robot being launched. Currently, the robots are strictly named **bernard**, **scrappy** and **oswaldo**. All robots that will be playing tag should be launched individually.
