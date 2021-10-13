#!/usr/bin/env python
# | Copyright Matthias Mayr October 2021
# |
# | Code repository: https://github.com/matthias-mayr/behavior-tree-policy-learning
# | Preprint: https://arxiv.org/abs/2109.13050
# |
# | This software is governed by the CeCILL-C license under French law and
# | abiding by the rules of distribution of free software.  You can  use,
# | modify and/ or redistribute the software under the terms of the CeCILL-C
# | license as circulated by CEA, CNRS and INRIA at the following URL
# | "http://www.cecill.info".
# |
# | As a counterpart to the access to the source code and  rights to copy,
# | modify and redistribute granted by the license, users are provided only
# | with a limited warranty  and the software's author,  the holder of the
# | economic rights,  and the successive licensors  have only  limited
# | liability.
# |
# | In this respect, the user's attention is drawn to the risks associated
# | with loading,  using,  modifying and/or developing or reproducing the
# | software by the user in light of its specific status of free software,
# | that may mean  that it is complicated to manipulate,  and  that  also
# | therefore means  that it is reserved for developers  and  experienced
# | professionals having in-depth computer knowledge. Users are therefore
# | encouraged to load and test the software's suitability as regards their
# | requirements in conditions enabling the security of their systems and/or
# | data to be ensured and,  more generally, to use and operate it in the
# | same conditions as regards security.
# |
# | The fact that you are presently reading this means that you have had
# | knowledge of the CeCILL-C license and that you accept its terms.
# |
import rospy
import time
import numpy as np
import sys

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

configs = dict()
configs["obstacle1_1"] = [-0.03, 0.64, -0.12, -1.27, 0.96, 1.80, -1.08]
configs["obstacle2_1"] = [-0.03, 0.64, -0.12, -1.27, 0.96, 1.80, -1.08]
configs["obstacle3_1"] = [-0.03, 0.64, -0.12, -1.27, 0.96, 1.80, -1.08]
configs["obstacle4_1"] = [-0.03, 0.64, -0.12, -1.27, 0.96, 1.80, -1.08]
configs["obstacle5_1"] = [-0.03, 0.64, -0.12, -1.27, 0.96, 1.80, -1.08]

configs["obstacle1_2"] = [0.08, 1.11, -0.12, -0.78, 0.97, 1.45, -1.08]
configs["obstacle2_2"] = [0.08, 1.11, -0.12, -0.78, 0.97, 1.45, -1.08]
configs["obstacle3_2"] = [0.08, 1.11, -0.12, -0.78, 0.97, 1.45, -1.08]
configs["obstacle4_2"] = [0.08, 1.11, -0.12, -0.78, 0.97, 1.45, -1.08]
configs["obstacle5_2"] = [0.08, 1.11, -0.12, -0.78, 0.97, 1.45, -1.08]

configs["obstacle1_3"] = [0.47, 1.54, -0.30, -0.84, 1.48, 1.34, -1.28]
configs["obstacle2_2"] = [0.43, 1.65, -0.32, -0.56, 1.46, 1.37, -1.11]
configs["obstacle3_2"] = [0.44, 1.58, -0.30, -0.90, 1.49, 1.34, -1.38]
configs["obstacle4_2"] = [0.49, 1.45, -0.28, -1.10, 1.50, 1.32, -1.45]
configs["obstacle5_3"] = [0.48, 1.52, -0.30, -0.74, 1.46, 1.35, -1.15]


configs["peg1_1"] = [1.10, 0.41, -0.42, -1.52, 1.30, 1.14, -0.68]
configs["peg2_1"] = [1.10, 0.41, -0.42, -1.52, 1.30, 1.14, -0.68]
configs["peg3_1"] = [1.10, 0.41, -0.42, -1.52, 1.30, 1.14, -0.68]
configs["peg4_1"] = [1.10, 0.41, -0.42, -1.52, 1.30, 1.14, -0.68]
configs["peg5_1"] = [1.10, 0.41, -0.42, -1.52, 1.30, 1.14, -0.68]
configs["peg6_1"] = [1.10, 0.41, -0.42, -1.52, 1.30, 1.14, -0.68]
configs["peg7_1"] = [1.10, 0.41, -0.42, -1.52, 1.30, 1.14, -0.68]
configs["peg8_1"] = [1.10, 0.41, -0.42, -1.52, 1.30, 1.14, -0.68]
configs["peg9_1"] = [1.10, 0.41, -0.42, -1.52, 1.30, 1.14, -0.68] 
configs["peg10_1"] = [1.10, 0.41, -0.42, -1.52, 1.30, 1.14, -0.68]
configs["peg11_1"] = [1.10, 0.41, -0.42, -1.52, 1.30, 1.14, -0.68]
configs["peg12_1"] = [1.10, 0.41, -0.42, -1.52, 1.30, 1.14, -0.68]
configs["peg13_1"] = [1.10, 0.41, -0.42, -1.52, 1.30, 1.14, -0.68]
configs["peg14_1"] = [1.10, 0.41, -0.42, -1.52, 1.30, 1.14, -0.68]
configs["peg15_1"] = [1.10, 0.41, -0.42, -1.52, 1.30, 1.14, -0.68]

configs["peg1_2"] = [0.67, 0.27, -0.27, -1.40, 1.11, 1.49, -0.58]
configs["peg2_2"] = [0.70, 0.21, -0.27, -1.46, 1.09, 1.47, -0.56]
configs["peg3_2"] = [0.69, 0.25, -0.27, -1.39, 1.10, 1.49, -0.54]
configs["peg4_2"] = [0.52, 0.30, -0.15, -1.48, 1.10, 1.46, -0.69]
configs["peg5_2"] = [0.52, 0.26, -0.15, -1.55, 1.09, 1.45, -0.72]
configs["peg6_2"] = [0.65, 0.22, -0.27, -1.54, 1.01, 1.47, -0.67]
configs["peg7_2"] = [0.65, 0.30, -0.27, -1.40, 1.11, 1.50, -0.61]
configs["peg8_2"] = [0.70, 0.165, -0.27, -1.53, 1.08, 1.45, -0.59]
configs["peg9_2"] = [0.57, 0.09, -0.15, -1.72, 1.07, 1.41, -0.69]
configs["peg10_2"] = [0.50, 0.17, -0.15, -1.72, 1.09, 1.42, -0.80]
configs["peg11_2"] = [0.51, 0.21, -0.15, -1.65, 1.09, 1.43, -0.77]
configs["peg12_2"] = [0.57, 0.13, -0.15, -1.65, 1.07, 1.42, -0.66]
configs["peg13_2"] = [0.67, 0.23, -0.27, -1.47, 1.10, 1.48, -0.60]
configs["peg14_2"] = [0.67, 0.19, -0.27, -1.54, 1.09, 1.46, -0.63]
configs["peg15_2"] = [0.65, 0.26, -0.27, -1.47, 1.11, 1.48, -0.64]

positions = None
if len(sys.argv) is not 2:
    print "Experiment 'peg' or 'obstacle' needs to be specified."
    exit()
experiment = sys.argv[1]
if experiment == "peg":
    print "Peg experiment - default configuration"
    experiment = "peg1"
elif experiment == "obstacle":
    print "Obstacle experiment - default configuration"
    experiment = "obstacle1"

print "Experiment: ", experiment
first_key = experiment + "_1"
if first_key in configs:
    desired_pos = configs[first_key]
    print "Moving to", first_key, ":", desired_pos
else:
    print "No configuration found for key", first_key
    print "Did you mispell the experiment? Exiting."
    exit()
accepted_diff = 0.001
step = 0.01


def callback(data):
    global positions
    positions = data.position[0:7]


rospy.init_node('send_joint_pos')
r = rospy.Rate(200)
pub = rospy.Publisher('/iiwa/PositionController/command', Float64MultiArray, queue_size=10)
sub = rospy.Subscriber("/bh/joint_states", JointState, callback)
while positions is None:
    time.sleep(0.01)
print "Moving to configuration: ", desired_pos

msg = Float64MultiArray()
msg.data = positions
pub.publish(msg)

target_it = 1
while not rospy.is_shutdown():
    cur_pos = np.array(positions)
    diff = np.array(desired_pos) - cur_pos
    reached = True
    for i in range(7):
        if diff[i] > step:
            diff[i] = step
        elif diff[i] < -step:
            diff[i] = -step
        if np.abs(diff[i]) < accepted_diff:
            diff[i] = 0
        if np.abs(diff[i]) > 0.8 * step:
            reached = False
    joint_target = cur_pos + np.array(diff)
    msg.data = [joint_target[0], joint_target[1], joint_target[2], joint_target[3],
                joint_target[4], joint_target[5], joint_target[6]]
    pub.publish(msg)

    # Switch way points. Some experiment need intermediate ones.
    if reached:
        target_it += 1
        next_key = experiment + "_" + str(target_it)
        if next_key in configs:
            desired_pos = configs[next_key]
            print "Moving to", next_key, ":", desired_pos
        else:
            print "Reached position. Leaving."
            exit()
    r.sleep()
