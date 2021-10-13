//| Copyright Inria July 2017
//|
//| Contributor(s):
//|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Rituraj Kaushik (rituraj.kaushik@inria.fr)
//|   - Roberto Rama (bertoski@gmail.com)
//|
//| Base repository: http://github.com/resibots/blackdrops
//| Preprint: https://arxiv.org/abs/1703.07261
//|
//|
//| Copyright Matthias Mayr October 2021
//|
//| Code repository: https://github.com/matthias-mayr/behavior-tree-policy-learning
//| Preprint: https://arxiv.org/abs/2109.13050
//|
//| Adaption of Black-DROPS to learning behavior trees and contact-rich tasks.
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
#ifndef BLACKDROPS_SYSTEM_DART_SYSTEM_HPP
#define BLACKDROPS_SYSTEM_DART_SYSTEM_HPP

#include <functional>

#include <robot_dart/control/robot_control.hpp>
#include <robot_dart/robot.hpp>
#include <robot_dart/robot_dart_simu.hpp>
#include <robot_dart/utils.hpp>

#include <dart/dynamics/BodyNode.hpp>
#include <dart/collision/CollisionResult.hpp>

#ifdef GRAPHIC
#include <robot_dart/graphics/graphics.hpp>
#endif

#include <blackdrops/system/system.hpp>
#include <blackdrops/utils/utils.hpp>
#ifdef USE_ROS
#include <blackdrops/experiments/iiwa_skills_execution.hpp>
#endif

#include <glog/logging.h>

namespace blackdrops {
    namespace system {
        template <typename Params, typename PolicyController, typename RolloutInfo>
        struct DARTSystem : public System<Params, DARTSystem<Params, PolicyController, RolloutInfo>, RolloutInfo> {
            using robot_simu_t = robot_dart::RobotDARTSimu;
            
            DARTSystem() {
#ifdef USE_ROS
                if (Params::blackdrops::learn_real_system()) {
                    _execution = std::make_shared<iiwa_skills_execution<PolicyController, Params>>();
                }
#endif
            }

#ifdef USE_ROS
            template <typename Policy, typename Reward>
            std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute_with_real_system(const Policy& policy, Reward& world, double T, std::vector<double>& R) {
                    // Consists of: init_transformed_state, u, diff_state
                    std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> res;

                    R = std::vector<double>();
                    // Get the information of the rollout
                    RolloutInfo rollout_info = this->get_rollout_info();
                    std::shared_ptr<robot_dart::Robot> robot_model = this->get_robot();

                    // setup robot
                    Eigen::VectorXd pp = policy.params();
                    std::vector<double> params(pp.size());
                    Eigen::VectorXd::Map(params.data(), pp.size()) = pp;

                    // setup the controller
                    auto controller = std::make_shared<PolicyController>(params);
                    controller->set_robot_in_policy(robot_model);
                    controller->set_controller_real_params();
                    controller->set_goal(_goal);
                    robot_model->add_controller(controller);
                    bool not_done{true};
                    
                    while(not_done) {
                        controller->configure();
                        blackdrops::execution::reset_to_start_pos();
                        not_done = !_execution->run(controller, robot_model);
                    }

                    std::vector<Eigen::VectorXd> states = controller->get_states();
                    std::vector<Eigen::VectorXd> noiseless_states = controller->get_noiseless_states();
                    std::vector<Eigen::VectorXd> commands = controller->get_commands();

                    rollout_info.bt_finish_time = controller->bt_finish_time;
                    // Order results and calculate reward
                    for (size_t j = 0; j < states.size() - 1; j++) {
                        Eigen::VectorXd init = states[j];
                        Eigen::VectorXd init_full = this->transform_state(init);

                        Eigen::VectorXd u = commands[j];
                        Eigen::VectorXd final = states[j + 1];

                        double r = world.observe(rollout_info, noiseless_states[j], u, noiseless_states[j + 1], false);
                        R.push_back(r);
                        res.push_back(std::make_tuple(init_full, u, final - init));
                    }
                    if (!policy.random() && false) {
                        rollout_info.print_rewards();
                    }
                    return res;
            }
#endif

            template <typename Policy, typename Reward>
            std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute(const Policy& policy, Reward& world, double T, std::vector<double>& R, bool display = true)
            {
                // Make sure that the simulation step is smaller than the sampling/control rate
                assert(Params::dart_system::sim_step() < Params::blackdrops::dt());

                // Consists of: init_transformed_state, u, diff_state
                std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> res;

                R = std::vector<double>();

                robot_simu_t simu;
#ifdef GRAPHIC
                simu.set_graphics(std::make_shared<robot_dart::graphics::Graphics>(simu.world()));
                simu.graphics()->set_enable(display);
#endif
                // simulation step different from sampling rate -- we need a stable simulation
                simu.set_step(Params::dart_system::sim_step());

                // Get the information of the rollout
                RolloutInfo rollout_info = this->get_rollout_info();

                // setup robot
                Eigen::VectorXd pp = policy.params();
                std::vector<double> params(pp.size());
                Eigen::VectorXd::Map(params.data(), pp.size()) = pp;

                std::shared_ptr<robot_dart::Robot> simulated_robot = this->get_robot();
                simulated_robot->set_actuator_types(Params::dart_policy_control::joint_type());

                // setup the controller
                auto controller = std::make_shared<PolicyController>(params);
                controller->set_robot_in_policy(this->get_robot());
                controller->set_transform_state(std::bind(&DARTSystem::transform_state, this, std::placeholders::_1));
                controller->set_noise_function(std::bind(&DARTSystem::add_noise, this, std::placeholders::_1));
                controller->set_update_function(std::bind([&](double t) { rollout_info.t = t; }, std::placeholders::_1));
                controller->set_policy_function(std::bind(&DARTSystem::policy_transform, this, std::placeholders::_1, &rollout_info));
                controller->set_goal(_goal);

                // Add extra to simu object
                this->add_extra_to_simu(simu, rollout_info);

                // add the controller to the robot
                simulated_robot->add_controller(controller);
                // add the robot to the simulation
                simu.add_robot(simulated_robot);

                // Get initial state from info and add noise
                Eigen::VectorXd init_diff = rollout_info.init_state;
                this->set_robot_state(simulated_robot, init_diff);
                init_diff = this->add_noise(init_diff);

                // Runs the simulation for a maximum of T + sim_step time
                simu.run(T + Params::dart_system::sim_step());

                std::vector<Eigen::VectorXd> states = controller->get_states();
                std::vector<Eigen::VectorXd> noiseless_states = controller->get_noiseless_states();
                std::vector<Eigen::VectorXd> commands = controller->get_commands();
                if (display)  {
                    this->_last_states = states;
                    this->_last_dense_states = controller->get_dense_states();
                    this->_last_commands = commands;
                    this->_last_dense_commands = controller->get_dense_commands();
                    this->_last_dense_ee_pos = controller->get_dense_ee_pos();
                    this->_last_dense_ee_rot = controller->get_dense_ee_rot();
                }
                rollout_info.bt_finish_time = controller->bt_finish_time;
                // Order results and calculate reward
                for (size_t j = 0; j < states.size() - 1; j++) {
                    Eigen::VectorXd init = states[j];

                    Eigen::VectorXd init_full = this->transform_state(init);

                    Eigen::VectorXd u = commands[j];
                    Eigen::VectorXd final = states[j + 1];

                    double r = world.observe(rollout_info, noiseless_states[j], u, noiseless_states[j + 1], display);
                    R.push_back(r);
                    res.push_back(std::make_tuple(init_full, u, final - init));
                }
                if (!policy.random() && display) {
                    rollout_info.print_rewards();
                }

                return res;
            }

            template <typename Model>
            std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute_with_sequence(const std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>>& observation, const Eigen::VectorXd& params, bool display, const Model& model, const std::string& residual_file = "", bool with_variance = false) {
                if (Params::meta_conf::verbose()) {
                    LOG(INFO) << "Rollout of an observed episode in simulation";
                }
                // Make sure that the simulation step is smaller than the sampling/control rate
                assert(Params::dart_system::sim_step() < Params::blackdrops::dt());

                robot_simu_t simu;
#ifdef GRAPHIC
                simu.set_graphics(std::make_shared<robot_dart::graphics::Graphics>(simu.world()));
                simu.graphics()->set_enable(display);
#endif
                // simulation step different from sampling rate -- we need a stable simulation
                simu.set_step(Params::dart_system::sim_step());

                // Get the information of the rollout
                RolloutInfo rollout_info = this->get_rollout_info();

                std::vector<double> params_vec(params.size());
                Eigen::VectorXd::Map(params_vec.data(), params.size()) = params;

                std::shared_ptr<robot_dart::Robot> simulated_robot = this->get_robot();
                simulated_robot->set_actuator_types(Params::dart_policy_control::joint_type());

                // setup the controller
                auto controller = std::make_shared<PolicyController>(params_vec);
                controller->set_robot_in_policy(this->get_robot());
                controller->set_skip_next();
                controller->set_transform_state(std::bind(&DARTSystem::transform_state, this, std::placeholders::_1));
                controller->set_noise_function(std::bind(&DARTSystem::add_noise, this, std::placeholders::_1));
                controller->set_update_function(std::bind([&](double t) { rollout_info.t = t; }, std::placeholders::_1));
                controller->set_policy_function(std::bind(&DARTSystem::policy_transform, this, std::placeholders::_1, &rollout_info));
                controller->set_goal(_goal);

                // Add extra to simu object
                this->add_extra_to_simu(simu, rollout_info);

                // add the controller to the robot
                simulated_robot->add_controller(controller);
                // add the robot to the simulation
                simu.add_robot(simulated_robot);

                std::vector<Eigen::VectorXd> states;
                double dt = Params::blackdrops::dt();
                double T_passed{0};
                for (size_t i = 0; i < observation.size(); i++, T_passed+= dt) {
                    // Setting State and action
                    this->set_robot_state(simulated_robot, std::get<0>(observation[i]));
                    controller->set_skip_calculate();
                    controller->set_fixed_commands(std::get<1>(observation[i]));
                    if (!residual_file.empty()) {
                        Eigen::VectorXd query_vec(Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim());
                        query_vec.head(Params::blackdrops::model_input_dim()) = std::get<0>(observation[i]);
                        query_vec.tail(Params::blackdrops::action_dim()) = std::get<1>(observation[i]);
                        Eigen::VectorXd mu;
                        Eigen::VectorXd sigma;
                        std::tie(mu, sigma) = model.predict(query_vec, with_variance);
                        Eigen::VectorXd res_mean{mu};
                        if (with_variance) {
                            sigma = sigma.array().sqrt();
                            for (int i = 0; i < mu.size(); i++) {
                                double s = utils::gaussian_rand(mu(i), sigma(i));
                                mu(i) = std::max(mu(i) - sigma(i),
                                    std::min(s, mu(i) + sigma(i)));
                            }
                        }
                        controller->append_applied_residual(res_mean, res_mean - mu);
                    }

                    simu.run(dt);
                    states.push_back(controller->get_state(simulated_robot));
                    if (Params::meta_conf::verbose()) {
                        LOG(INFO) << "State before: " << std::get<0>(observation[i]).format(OneLine);
                        LOG(INFO) << "Action : " << std::get<1>(observation[i]).format(OneLine);
                        LOG(INFO) << "State after: " << controller->get_state(simulated_robot).format(OneLine);
                    }
                }

                // Consists of: init_state_t, u_t, simulation_state_t+1
                std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> res;
                for (size_t j = 0; j < observation.size(); j++) {
                    res.push_back(std::make_tuple(std::get<0>(observation[j]), std::get<1>(observation[j]), states[j]));
                }

                if (!residual_file.empty()) {
                    utils::save_traj_to_file(residual_file, controller->get_applied_residual_mean(), controller->get_applied_residual_variance());
                }

                return res;
            }

            template <typename Policy, typename Model, typename Reward>
            std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute_with_model(const Policy& policy, const Model& model, Reward& world, double T, std::vector<double>& R, bool display = true, bool with_variance = true) {
                // Make sure that the simulation step is smaller than the sampling/control rate
                assert(Params::dart_system::sim_step() < Params::blackdrops::dt());

                // Consists of: init_transformed_state, u, diff_state
                std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> res;

                R = std::vector<double>();

                robot_simu_t simu;
#ifdef GRAPHIC
                simu.set_graphics(std::make_shared<robot_dart::graphics::Graphics>(simu.world()));
                simu.graphics()->set_enable(display);
#endif
                // simulation step different from sampling rate -- we need a stable simulation
                simu.set_step(Params::dart_system::sim_step());

                // Get the information of the rollout
                RolloutInfo rollout_info = this->get_rollout_info();

                // setup robot
                Eigen::VectorXd pp = policy.params();
                std::vector<double> params(pp.size());
                Eigen::VectorXd::Map(params.data(), pp.size()) = pp;

                std::shared_ptr<robot_dart::Robot> simulated_robot = this->get_robot();
                simulated_robot->set_actuator_types(Params::dart_policy_control::joint_type());

                // setup the controller
                auto controller = std::make_shared<PolicyController>(params);
                controller->set_robot_in_policy(this->get_robot());
                controller->set_transform_state(std::bind(&DARTSystem::transform_state, this, std::placeholders::_1));
                controller->set_noise_function(std::bind(&DARTSystem::add_noise, this, std::placeholders::_1));
                controller->set_update_function(std::bind([&](double t) { rollout_info.t = t; }, std::placeholders::_1));
                controller->set_policy_function(std::bind(&DARTSystem::policy_transform, this, std::placeholders::_1, &rollout_info));
                controller->set_goal(_goal);

                // Add extra to simu object
                this->add_extra_to_simu(simu, rollout_info);

                // add the controller to the robot
                simulated_robot->add_controller(controller);
                // add the robot to the simulation
                simu.add_robot(simulated_robot);

                // Get initial state from info and add noise
                Eigen::VectorXd init_diff = rollout_info.init_state;
                this->set_robot_state(simulated_robot, init_diff);
                init_diff = this->add_noise(init_diff);

                if (Params::meta_conf::verbose()) {
                    LOG(INFO) << "Finished setup of the simulation. Running it now." << std::flush;
                }
                double dt = Params::blackdrops::dt();
                bool deep_collision{false};
                for (double T_passed = 0; T_passed < T; T_passed+= dt) {
                    simu.run(dt);
                    Eigen::VectorXd state = controller->get_state(simulated_robot);
                    Eigen::VectorXd action = controller->get_last_command();
                    if (action.size() < static_cast<int>(Params::blackdrops::action_dim())) {
                        LOG(WARNING) << "Controller returned an action of size " << action.size() << " which is smaller than the action dim.: " << Params::blackdrops::action_dim();
                        action = Eigen::VectorXd(Params::blackdrops::action_dim());
                    }
                    Eigen::VectorXd query_vec(Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim());

                    query_vec.head(Params::blackdrops::model_input_dim()) = state;
                    query_vec.tail(Params::blackdrops::action_dim()) = action;

                    Eigen::VectorXd mu;
                    Eigen::VectorXd sigma;
                    std::tie(mu, sigma) = model.predict(query_vec, with_variance);
                    Eigen::VectorXd res_mean{mu};
                    if (with_variance) {
                        sigma = sigma.array().sqrt();
                        for (int i = 0; i < mu.size(); i++) {
                            double s = utils::gaussian_rand(mu(i), sigma(i));
                            mu(i) = std::max(mu(i) - sigma(i),
                                std::min(s, mu(i) + sigma(i)));
                        }
                    }
                    controller->append_applied_residual(res_mean, res_mean - mu);
                    Eigen::VectorXd final = state + mu;
                    this->set_robot_state(simulated_robot, final);
                    if (check_for_deep_collisions(simu.world()->getLastCollisionResult(), Params::blackdrops::max_collision_depth())) {
                        deep_collision = true;
                        break;
                    }
                }
                if (Params::meta_conf::verbose()) {
                    LOG(INFO) << "Simulation done. Postprocessing." << std::flush;
                }
                if (deep_collision) {
                    R.push_back(-20000);
                    LOG(WARNING) << "Detected deep collision and aborted execution. Parameters are: " << pp.array().format(OneLine);
                }

                std::vector<Eigen::VectorXd> states = controller->get_states();
                std::vector<Eigen::VectorXd> noiseless_states = controller->get_noiseless_states();
                std::vector<Eigen::VectorXd> commands = controller->get_commands();
                if (display) {
                    this->_last_states = states;
                    this->_last_dense_states = controller->get_dense_states();
                    this->_last_commands = commands;
                    this->_last_dense_commands = controller->get_dense_commands();
                    this->_last_dense_ee_pos = controller->get_dense_ee_pos();
                    this->_last_dense_ee_rot = controller->get_dense_ee_rot();
                    this->_last_applied_residual_mean = controller->get_applied_residual_mean();
                    this->_last_applied_residual_variance = controller->get_applied_residual_variance();
                }
                rollout_info.bt_finish_time = controller->bt_finish_time;
                // Order results and calculate reward
                for (size_t j = 0; j < states.size() - 1; j++) {
                    Eigen::VectorXd init = states[j];

                    Eigen::VectorXd init_full = this->transform_state(init);

                    Eigen::VectorXd u = commands[j];
                    Eigen::VectorXd final = states[j + 1];

                    double r = world.observe(rollout_info, noiseless_states[j], u, noiseless_states[j + 1], display);
                    R.push_back(r);
                    res.push_back(std::make_tuple(init_full, u, final - init));
                }

                if (!policy.random() && display) {
                    double rr = std::accumulate(R.begin(), R.end(), 0.0);
                    LOG(INFO) << "Total reward combined:\t\t" << rr;
                    rollout_info.print_rewards();
                }
                return res;
            }

            bool check_for_deep_collisions(const dart::collision::CollisionResult& col_res, double depth) {
                const auto contacts = col_res.getContacts();
                for (const auto& contact : contacts) {
                    if (contact.penetrationDepth > depth) {
                        return true;
                    }
                }
                return false;
            }

            // override this to add extra stuff to the robot_dart simulator
            virtual void add_extra_to_simu(robot_simu_t& simu, const RolloutInfo& rollout_info) const {}

            // you should override this, to define how your simulated robot_dart::Robot will be constructed
            virtual std::shared_ptr<robot_dart::Robot> get_robot() const = 0;

            // override this if you want to set in a specific way the initial state of your robot
            virtual void set_robot_state(const std::shared_ptr<robot_dart::Robot>& robot, const Eigen::VectorXd& state) const {}

            // get states from last execution
            std::vector<Eigen::VectorXd> get_last_dense_states() const
            {
                return _last_dense_states;
            }

            // get commands from last execution
            std::vector<Eigen::VectorXd> get_last_dense_commands() const
            {
                return _last_dense_commands;
            }

            // get ee pos from last execution
            std::vector<Eigen::VectorXd> get_last_dense_ee_pos() const
            {
                return _last_dense_ee_pos;
            }

            // get ee rot from last execution
            std::vector<Eigen::VectorXd> get_last_dense_ee_rot() const
            {
                return _last_dense_ee_rot;
            }

            // get applied residual mean
            std::vector<Eigen::VectorXd> get_last_applied_residual_mean() const
            {
                return _last_applied_residual_mean;
            }

            // get applied residual variance
            std::vector<Eigen::VectorXd> get_last_applied_residual_variance() const
            {
                return _last_applied_residual_variance;
            }

            std::vector<Eigen::VectorXd> _last_dense_states, _last_dense_commands, _last_dense_ee_pos, _last_dense_ee_rot, _last_applied_residual_mean, _last_applied_residual_variance;
#ifdef USE_ROS
            std::shared_ptr<iiwa_skills_execution<PolicyController, Params>> _execution;
#endif
            Eigen::Vector3d _goal = Eigen::Vector3d::Zero();
        };

        template <typename Params, typename Policy>
        class BaseDARTPolicyControl : public robot_dart::control::RobotControl {
        public:
            using robot_t = std::shared_ptr<robot_dart::Robot>;

            BaseDARTPolicyControl() {}
            BaseDARTPolicyControl(const std::vector<double>& ctrl, bool full_control = false)
                : robot_dart::control::RobotControl(ctrl, full_control)
            {
                // set some default functions in case the user does not define them
                set_transform_state(std::bind(&BaseDARTPolicyControl::transform_state, this, std::placeholders::_1));
                set_noise_function(std::bind(&BaseDARTPolicyControl::transform_state, this, std::placeholders::_1));
                set_policy_function(std::bind(&BaseDARTPolicyControl::transform_state, this, std::placeholders::_1));
                set_update_function(std::bind(&BaseDARTPolicyControl::dummy, this, std::placeholders::_1));
            }

            void configure() override
            {
                _prev_time = 0.0;
                _t = 0.0;
                _first = true;

                _policy.set_params(Eigen::VectorXd::Map(_ctrl.data(), _ctrl.size()));

                _states.clear();
                _noiseless_states.clear();
                _coms.clear();

                if (Params::blackdrops::action_dim() == _control_dof)
                    _active = true;
            }

            Eigen::VectorXd calculate(double t) override
            {
                _t = t;
                _update_func(t);

                double dt = Params::blackdrops::dt();

                if (_first || (_t - _prev_time - dt) > -Params::dart_system::sim_step() / 2.0) {
                    Eigen::VectorXd q = this->get_state(_robot.lock());
                    _noiseless_states.push_back(q);
                    q = _add_noise(q);
                    Eigen::VectorXd commands = _policy.next(_policy_state(_tranform_state(q)), t);
                    _states.push_back(q);
                    _coms.push_back(commands);

                    ROBOT_DART_ASSERT(_control_dof == static_cast<size_t>(commands.size()), "BaseDARTPolicyControl: Policy output size is not the same as the control DOFs of the robot", Eigen::VectorXd::Zero(_control_dof));
                    _prev_commands = commands;
                    _prev_time = _t;
                    _first = false;
                }

                return _prev_commands;
            }

            std::vector<Eigen::VectorXd> get_states() const
            {
                return _states;
            }

            std::vector<Eigen::VectorXd> get_noiseless_states() const
            {
                return _noiseless_states;
            }

            std::vector<Eigen::VectorXd> get_commands() const
            {
                return _coms;
            }

            Eigen::VectorXd get_last_command() const
            {
                if (!_coms.empty()) {
                    return _coms.back();
                } else {
                    return Eigen::VectorXd();
                }
            }

            void set_transform_state(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> func)
            {
                _tranform_state = func;
            }

            void set_noise_function(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> func)
            {
                _add_noise = func;
            }

            void set_policy_function(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> func)
            {
                _policy_state = func;
            }

            void set_update_function(std::function<void(double)> func)
            {
                _update_func = func;
            }

            virtual Eigen::VectorXd get_state(const robot_t& robot) const = 0;

        protected:
            double _prev_time;
            double _t;
            bool _first;
            Eigen::VectorXd _prev_commands;
            Policy _policy;
            std::vector<Eigen::VectorXd> _coms;
            std::vector<Eigen::VectorXd> _states, _noiseless_states;
            std::function<Eigen::VectorXd(const Eigen::VectorXd&)> _tranform_state;
            std::function<Eigen::VectorXd(const Eigen::VectorXd&)> _add_noise;
            std::function<Eigen::VectorXd(const Eigen::VectorXd&)> _policy_state;
            std::function<void(double)> _update_func;

            Eigen::VectorXd transform_state(const Eigen::VectorXd& original_state) const
            {
                return original_state;
            }

            void dummy(double) const {}
        };
    } // namespace system
} // namespace blackdrops

#endif