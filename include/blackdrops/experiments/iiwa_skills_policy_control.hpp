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
#ifndef BLACKDROPS_IIWA_SKILLS_POLICY_CONTROL_HPP
#define BLACKDROPS_IIWA_SKILLS_POLICY_CONTROL_HPP

#include <blackdrops/system/dart_system.hpp>
#include <blackdrops/controller/iiwa_skills_velocity_controller.hpp>


// Controls the robot in DART and handles the queries to the policy.
template <typename Params, typename Policy>
struct PolicyControl : public blackdrops::system::BaseDARTPolicyControl<Params, Policy> {
    using base_t = blackdrops::system::BaseDARTPolicyControl<Params, Policy>;

    PolicyControl() : base_t() {
    }
    PolicyControl(const std::vector<double>& ctrl) : base_t(ctrl) {
    }

    void configure() override
    {
        this->_prev_time = 0.0;
        this->_t = 0.0;
        this->_first = true;

        this->_policy.set_params(Eigen::VectorXd::Map(this->_ctrl.data(), this->_ctrl.size()));

        this->_states.clear();
        this->_noiseless_states.clear();
        this->_coms.clear();
        this->_dense_states.clear();
        this->_dense_coms.clear();
        this->_applied_residual_mean.clear();
        this->_applied_residual_variance.clear();
        cart_controller.set_max_joint_vel(Params::blackdrops::max_joint_vel());

        if (this->_control_dof == 7)
            this->_active = true;
    }

    Eigen::VectorXd get_state(const std::shared_ptr<robot_dart::Robot>& robot) const
    {
        Eigen::VectorXd state(Params::blackdrops::model_pred_dim());
        state << robot->skeleton()->getPositions(), robot->skeleton()->getVelocities();
        return state;
    }

    std::vector<Eigen::VectorXd> get_dense_states() const
    {
        return _dense_states;
    }

    std::vector<Eigen::VectorXd> get_dense_commands() const
    {
        return _dense_coms;
    }

    std::vector<Eigen::VectorXd> get_dense_ee_pos() const
    {
        return _dense_ee_pos;
    }

    std::vector<Eigen::VectorXd> get_dense_ee_rot() const
    {
        return _dense_ee_rot;
    }

    std::vector<Eigen::VectorXd> get_applied_residual_mean() const
    {
        return _applied_residual_mean;
    }

    std::vector<Eigen::VectorXd> get_applied_residual_variance() const
    {
        return _applied_residual_variance;
    }

    std::shared_ptr<robot_dart::control::RobotControl> clone() const override
    {
        return std::make_shared<PolicyControl>(*this);
    }

    void set_skip_next() {
        skip_next = true;
    }

    void set_skip_calculate() {
        skip_calculate = true;
    }

    void set_fixed_commands(const Eigen::VectorXd& coms) {
        _fixed_commands = coms;
    }

    void append_applied_residual(const Eigen::VectorXd& mean, const Eigen::VectorXd& variance) {
        _applied_residual_mean.push_back(mean);
        _applied_residual_variance.push_back(variance);
    }

    void process_command(const Eigen::VectorXd& command, bool hard_set_desired = false) {
        cart_controller.process_command(command, hard_set_desired);
    }

    void set_controller_step(const double step) {
        cart_controller.set_step(step);
    }

    void set_controller_real_params() {
        cart_controller.complianceParamUpdate(false);
    }
    
    void logging(const Eigen::VectorXd& position, const Eigen::Quaterniond& orientation, const Eigen::VectorXd& q_problem) {
        _dense_states.push_back(q_problem);
        _dense_coms.push_back(this->_prev_commands);
        _dense_ee_pos.push_back(position);
        _dense_ee_rot.push_back(orientation.toRotationMatrix().eulerAngles(0, 1, 2));
    }

    Eigen::VectorXd calculate(double t)
    {
        this->_t = t;
        this->_update_func(t);

        double dt = Params::blackdrops::dt();

        if (skip_calculate) {
            return _fixed_commands;
        }

        std::shared_ptr<robot_dart::Robot> robot = this->_robot.lock();
        Eigen::Matrix<double, 7, 1> q = robot->skeleton()->getPositions();
        Eigen::VectorXd dq = robot->skeleton()->getVelocities();

        /* Get full Jacobian of our end-effector */
        Eigen::MatrixXd J = robot->skeleton()->getBodyNode("bh_link_ee")->getWorldJacobian();

        /* Get current _state of the end-effector */
        Eigen::MatrixXd currentWorldTransformation = robot->skeleton()->getBodyNode("bh_link_ee")->getWorldTransform().matrix();
        Eigen::VectorXd currentWorldPosition = currentWorldTransformation.block(0, 3, 3, 1);
        Eigen::Quaterniond currentWorldOrientationQ (Eigen::Matrix3d(currentWorldTransformation.block(0, 0, 3, 3)));
        Eigen::VectorXd q_problem = get_state(this->_robot.lock());

        // When using NN policy, this checks wheter the goal is reached
        if (Params::blackdrops::nn_policy()) {
            Eigen::VectorXd v = _goal;
            v[2] += 0.1;
            double dee = (currentWorldPosition.head(3) - v).squaredNorm();
            if (dee < 0.05) {   // If distance < 0.05m, we reached the goal
                this->_policy.bt_response = _bt_success;
            }
        }

        // Queries the policy every dt for a new command
        if ((this->_first || (this->_t - this->_prev_time - dt) > -Params::dart_system::sim_step() / 2.0) && !skip_next) {
            this->_noiseless_states.push_back(q_problem);
            q_problem = this->_add_noise(q_problem);
            Eigen::VectorXd commands = this->_policy.next(this->_policy_state(this->_tranform_state(q_problem)), t);
            if (this->_policy.bt_response == _bt_success && bt_finish_time < 0) {
                bt_finish_time = this->_t;
            }
            process_command(commands, this->_first);
            this->_prev_commands = cart_controller.update(q, dq, J, currentWorldPosition, currentWorldOrientationQ);
            this->_states.push_back(q_problem);
            this->_coms.push_back(this->_prev_commands);

            this->_prev_time = this->_t;
            this->_first = false;
        }
        logging(currentWorldPosition, currentWorldOrientationQ, q_problem);
        return this->_prev_commands;
    }

    void set_robot_in_policy(std::shared_ptr<robot_dart::Robot> simulated_robot) {
        this->_policy.set_robot(simulated_robot);
    }

    void set_goal(const Eigen::Vector3d goal) {
        _goal = goal;
    }
        double bt_finish_time{-1};
    private:
        std::vector<Eigen::VectorXd> _dense_coms;
        std::vector<Eigen::VectorXd> _dense_states;
        std::vector<Eigen::VectorXd> _dense_ee_pos;
        std::vector<Eigen::VectorXd> _dense_ee_rot;
        std::vector<Eigen::VectorXd> _applied_residual_mean;
        std::vector<Eigen::VectorXd> _applied_residual_variance;
        Eigen::Vector3d _goal = Eigen::Vector3d::Zero();
        Eigen::VectorXd _fixed_commands;
        bool skip_next = false;
        bool skip_calculate = false;
        cartesian_controller<Params> cart_controller;
        // BehaviorTrees-Cpp defines success as 2
        const int _bt_success {2};
};

#endif