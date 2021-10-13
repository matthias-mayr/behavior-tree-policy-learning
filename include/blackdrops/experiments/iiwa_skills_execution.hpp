
//| Copyright Matthias Mayr October 2021
//|
//| Code repository: https://github.com/matthias-mayr/behavior-tree-policy-learning
//| Preprint: https://arxiv.org/abs/2109.13050
//|
//| This software is governed by the CeCILL-C license under French law and
//| abiding by the rules of distribution of free software.  You can  use,
//| modify and/ or redistribute the software under the terms of the CeCILL-C
//| license as circulated by CEA, CNRS and INRIA at the following URL
//| "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and  rights to copy,
//| modify and redistribute granted by the license, users are provided only
//| with a limited warranty  and the software's author,  the holder of the
//| economic rights,  and the successive licensors  have only  limited
//| liability.
//|
//| In this respect, the user's attention is drawn to the risks associated
//| with loading,  using,  modifying and/or developing or reproducing the
//| software by the user in light of its specific status of free software,
//| that may mean  that it is complicated to manipulate,  and  that  also
//| therefore means  that it is reserved for developers  and  experienced
//| professionals having in-depth computer knowledge. Users are therefore
//| encouraged to load and test the software's suitability as regards their
//| requirements in conditions enabling the security of their systems and/or
//| data to be ensured and,  more generally, to use and operate it in the
//| same conditions as regards security.
//|
//| The fact that you are presently reading this means that you have had
//| knowledge of the CeCILL-C license and that you accept its terms.
//|
#ifndef BLACKDROPS_IIWA_SKILLS_EXECUTION_HPP
#define BLACKDROPS_IIWA_SKILLS_EXECUTION_HPP

#include <chrono>
#include <exception>

#include <robot_dart/robot.hpp>

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Bool.h>

#include <blackdrops/utils/utils.hpp>

#include <stdlib.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace std::chrono;

#define degreesToRadians(angleDegrees) ((angleDegrees) * M_PI / 180.0)
#define radiansToDegrees(angleRadians) ((angleRadians) * 180.0 / M_PI)

namespace blackdrops {
    namespace execution {
        bool reset_to_start_pos() {
            // Reset to start position
            int positions = 5;
            Eigen::VectorXd r_nr_v = blackdrops::utils::uniform_rand(1);
            double r_nr = r_nr_v[0] * positions;
            r_nr += 0.5;
            int index = static_cast<int>(std::round(r_nr));
            // Cover edge cases:
            if (index < 1) {
                index = 1;
            } else if (index > positions) {
                index = static_cast<int>(positions);
            }
            // Python file takes the start position as an argument
            std::string cmd("python ");
            cmd += "./scripts/experiments/move_to_experiment_start.py";
            cmd += " peg" + std::to_string(index);

            LOG(INFO) << "Executing command: " << cmd << std::endl;
            // Call python script and wait for it to finish
            struct blackdrops::utils::popen2 start_script;
            blackdrops::utils::popen2(cmd.c_str(), &start_script);
            waitpid(start_script.child_pid, NULL, 0 );
            return true;
        }
    }
}

template <class PolicyController, typename Params>
struct iiwa_skills_execution {

    // Helper function to get value in a MultiArray
    double get_multi_array(const std_msgs::Float64MultiArray& array, size_t i, size_t j)
    {
        assert(array.layout.dim.size() == 2);
        size_t offset = array.layout.data_offset;

        return array.data[offset + i * array.layout.dim[0].stride + j];
    }

    // Helper function to set value in a MultiArray
    void set_multi_array(std_msgs::Float64MultiArray& array, size_t i, size_t j, double val)
    {
        assert(array.layout.dim.size() == 2);
        size_t offset = array.layout.data_offset;

        array.data[offset + i * array.layout.dim[0].stride + j] = val;
    }

    // Translates a JointState message to our state descriptions
    void states_callback(const sensor_msgs::JointStatePtr& states) {
        _real_states[0] = states->position[0];
        _real_states[1] = states->position[1];
        _real_states[2] = states->position[2];
        _real_states[3] = states->position[3];
        _real_states[4] = states->position[4];
        _real_states[5] = states->position[5];
        _real_states[6] = states->position[6];
        _real_states[7] = states->velocity[0];
        _real_states[8] = states->velocity[1];
        _real_states[9] = states->velocity[2];
        _real_states[10] = states->velocity[3];
        _real_states[11] = states->velocity[4];
        _real_states[12] = states->velocity[5];
        _real_states[13] = states->velocity[6];
    }

    // Saves if the robot can be commanded
    void status_callback(const std_msgs::BoolConstPtr& status) {
        _commanding_status = status->data;
    }

    template <typename T>
    std::string vector_to_string(const std::vector<T>& vector) {
        std::stringstream param_str;
        for (const auto& el : vector) {
            param_str << el <<  ", ";
        }
        return param_str.str();
    }

    iiwa_skills_execution() {
        _pub_commands = _node_handle.advertise<std_msgs::Float64MultiArray>("/iiwa/PositionController/command", 2);
        _sub_states = _node_handle.subscribe("/bh/joint_states", 1, &iiwa_skills_execution::states_callback, this);
        _sub_status = _node_handle.subscribe("/iiwa/commanding_status", 1, &iiwa_skills_execution::status_callback, this);
    }

    void print_config() {
        LOG(INFO) << "Parameters: " << vector_to_string(_params);
        LOG(INFO) << "Joint limits in deg: " << vector_to_string(_joint_limits);
        LOG(INFO) << "Maximum joint velocity in rad/s: " << _max_joint_vel_value;
        LOG(INFO) << "FRI control mode: " << _fri_control_mode;
        LOG(INFO) << "FRI joint stiffness: " << _fri_joint_stiffness;
        LOG(INFO) << "Control rate in Hz: " << _rate;
        LOG(INFO) << "Gains for joints: " << _gains.transpose();
    }


    void set_params(std::vector<double> params) {
        _params = params;
    }

    void set_robot_state(const std::shared_ptr<robot_dart::Robot>& robot, const Eigen::VectorXd& state) const
    {
        robot->skeleton()->setPositions(state.head(7));
        robot->skeleton()->setVelocities(state.tail(7));
    }

    void update_commands(std_msgs::Float64MultiArray& commands, const Eigen::VectorXd& diff) {
        for (size_t joint = 0; joint < 7; ++joint) {
            double value = get_multi_array(commands, 0, joint);
            value += diff[joint];
            set_multi_array(commands, 0, joint, value);
        }
    }

    void overwrite_commands_with_pos(std_msgs::Float64MultiArray& commands) {
        for (size_t joint = 0; joint < 7; ++joint) {
                set_multi_array(commands, 0, joint, _real_states[joint]);
        }
    }

    void clamp_joint_positions(std_msgs::Float64MultiArray& commands) {
        for (size_t joint = 0; joint < 7; ++joint) {
            double value = get_multi_array(commands, 0, joint);
            if (value > degreesToRadians(_joint_limits[joint])) {
                value = degreesToRadians(_joint_limits[joint]);
                LOG(WARNING) << "Upper clamp from " << get_multi_array(commands, 0, joint) << " to " << value;
            } else if (value < -degreesToRadians(_joint_limits[joint])) {
                value = -degreesToRadians(_joint_limits[joint]);
                LOG(WARNING) << "Lower clamp from " << get_multi_array(commands, 0, joint) << " to " << value;
            }
            set_multi_array(commands, 0, joint, value);
        }
    }

    void save_data(std::shared_ptr<PolicyController> policy_control, const std::string& traj_sparse_filename, const std::string& traj_dense_filename, const std::string& ee_dense_filename) {
        // Saving the data
        blackdrops::utils::save_traj_to_file(traj_sparse_filename, policy_control->get_states(), policy_control->get_commands());
        std::ofstream outfile;
        outfile.open(traj_sparse_filename, std::ios_base::app); // append instead of overwrite
        outfile << "bt_finish_time:" << policy_control->bt_finish_time;
        LOG(INFO) << "bt_finish_time:" << policy_control->bt_finish_time;
        blackdrops::utils::save_traj_to_file(traj_dense_filename, policy_control->get_dense_states(), policy_control->get_dense_commands());
        blackdrops::utils::save_traj_to_file(ee_dense_filename, policy_control->get_dense_ee_pos(), policy_control->get_dense_ee_rot());
    }

    std::shared_ptr<PolicyController> configure_controller(std::shared_ptr<robot_dart::Robot> robot, std::vector<double> params) {
        std::shared_ptr<PolicyController> policy_control = std::make_shared<PolicyController>(params);
        policy_control->set_robot_in_policy(robot);
        policy_control->set_controller_real_params();
        robot->add_controller(policy_control);
        return policy_control;
    }

    bool run(std::shared_ptr<PolicyController> policy_control, std::shared_ptr<robot_dart::Robot> robot, bool exit_on_success = false) {
        double t{0};
        Eigen::VectorXd vel_com;
        std_msgs::Float64MultiArray commands;
        commands.layout.dim.resize(2);
        commands.layout.data_offset = 0;
        commands.layout.dim[0].size = 1;
        commands.layout.dim[0].stride = 7;
        commands.layout.dim[1].size = 7;
        commands.layout.dim[1].stride = 0;
        commands.data.resize(1 * 7);

        // Check if we can connect to the robot with timeout
        auto start = ros::Time::now();
        while (_real_states[0] == 0.0) {
            ros::spinOnce();
            if ((ros::Time::now()-start).toSec() > 5.) {
                LOG(ERROR) << "Could not connect to the robot. Quitting.";
                return false;
            }
        }
        overwrite_commands_with_pos(commands);
        ros::Rate r(_rate);
        set_robot_state(robot, _real_states);
        vel_com = policy_control->calculate(t);
        start = ros::Time::now();

        // Execute the policy on the real system until time T
        while (t < Params::blackdrops::T()) {
            r.sleep();
            if (r.cycleTime() > r.expectedCycleTime()) {
                LOG(WARNING) << "Loop took longer than cycle time. It took: " << r.cycleTime() << "ms";
            }
            t = (ros::Time::now()-start).toSec();
            ros::spinOnce();
            if (!ros::ok() || !_commanding_status) {
                LOG(ERROR) << "Error when executing policy. Exiting.";
                return false;
            }
            if (exit_on_success && policy_control->bt_finish_time > 0) {
                LOG(INFO) << "Successfully finished task. Exiting.";
                return true;
            }
            if (Params::meta_conf::verbose()) {
                LOG_EVERY_N(INFO, _rate) << "t: " << t << "\n";
            }

            set_robot_state(robot, _real_states);
            
            // Get velocity commands
            vel_com = policy_control->calculate(t);
            // Apply additional gain
            vel_com = vel_com.cwiseProduct(_gains);
            // Clamp them for safety reasons
            vel_com = vel_com.cwiseMin(_max_joint_vel).cwiseMax(-_max_joint_vel);
            // Calculate diff in joint positions
            vel_com *= r.expectedCycleTime().toSec();
            
            overwrite_commands_with_pos(commands);
            update_commands(commands, vel_com);
            clamp_joint_positions(commands);
            _pub_commands.publish(commands);
        }
        return true;
    }

    ros::NodeHandle _node_handle{ros::NodeHandle("~")};
    ros::Publisher _pub_commands;
    ros::Subscriber _sub_states;
    ros::Subscriber _sub_status;
    std::vector<double> _params;
    Eigen::VectorXd _real_states{Eigen::VectorXd::Zero(14)};
    bool _commanding_status{false};
    const std::string _experiment;
    const double _rate{200};
    // Safety limits in the URDF
    const std::vector<double> _joint_limits{161.5, 114, 161.5, 114, 161.5, 114, 166.25};
    const double _max_joint_vel_value{3.4};
    const std::string _fri_control_mode{"Joint Impedance"};
    const double _fri_joint_stiffness{400}; // Joint stiffness value the gains are chosen for
    const Eigen::VectorXd _gains = (Eigen::VectorXd(7) <<
    8.5, 8.0, 6.5, 6.5, 4.5, 4.0, 4.0).finished(); // Additional gain to make real world executation closer to the simulated.
    const Eigen::VectorXd _max_joint_vel{Eigen::VectorXd::Constant(7, _max_joint_vel_value)};
};

#endif