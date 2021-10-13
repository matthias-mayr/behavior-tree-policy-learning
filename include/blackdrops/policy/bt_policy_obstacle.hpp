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
#ifndef BLACKDROPS_POLICY_BT_POLICY_OBSTACLE_HPP
#define BLACKDROPS_POLICY_BT_POLICY_OBSTACLE_HPP

#include <Eigen/Core>
#include <limbo/tools/random_generator.hpp>
#include <glog/logging.h>
#include <robot_dart/robot.hpp>
#include <behaviortree_cpp_v3/bt_factory.h>
#include <blackdrops/policy/bt_nodes.hpp>
#include <behaviortree_cpp_v3/loggers/bt_cout_logger.h>

namespace blackdrops {
    namespace policy {

        template <typename Params>
        struct BTPolicyObstacle {
        public:
            BTPolicyObstacle(bool unneeded = false)
            {
                // Amount of parameters we use.
                _params = Eigen::VectorXd::Zero(Params::bt_policy::num_params());

                BT::BehaviorTreeFactory factory;
                using namespace bt_learning_nodes;
                factory.registerNodeType<SetMGGoal>("SetMGGoal");
                factory.registerNodeType<EEYThreshold>("EEYThreshold");
                factory.registerNodeType<EEZThreshold>("EEZThreshold");
                factory.registerNodeType<EEPositionDistance>("EEPositionDistance");

                _tree = std::make_shared<BT::Tree>(factory.createTreeFromFile("./res/BTs/bt_obstacle.xml"));
            }

            Eigen::VectorXd next(const Eigen::VectorXd& state, const double time = 0.0)
            {
                _simulated_robot->skeleton()->setPositions(state.head(7));
                Eigen::Vector3d ee_pos = _simulated_robot->skeleton()->getBodyNode("peg")->getTransform().translation();

                _tree->rootBlackboard()->set("ee_x", ee_pos[0]);
                _tree->rootBlackboard()->set("ee_y", ee_pos[1]);
                _tree->rootBlackboard()->set("ee_z", ee_pos[2]);

                BT::NodeStatus status = _tree->tickRoot();
                bt_response = static_cast<int>(status);
                // IDLE = 0,
                // RUNNING,
                // SUCCESS,
                // FAILURE

                Eigen::VectorXd action{Eigen::VectorXd::Zero(9)};
                action[0] = ee_pos[0];
                action[1] = ee_pos[1];
                action[2] = ee_pos[2];
                // Quaternion that points down.
                action[3] = 1.0;
                try {
                    _tree->rootBlackboard()->get("mg_trans_x", action[0]);
                    _tree->rootBlackboard()->get("mg_trans_y", action[1]);
                    _tree->rootBlackboard()->get("mg_trans_z", action[2]);
                } catch (const std::runtime_error& e) {
                    LOG(ERROR) << "Could not retrieve MG config from blackboard.";
                }
                return action;
            }

            void set_random_policy()
            {
                LOG(ERROR) << "BTPolicyObstacle does not support random.";
            }

            bool random() const
            {
                return false;
            }

            void set_params(const Eigen::VectorXd& params)
            {
                _params = params;
                if (_params.size() == 6) {
                    const double x_offset = -0.5;
                    _tree->rootBlackboard()->set("z_threshold", _params[0]);
                    _tree->rootBlackboard()->set("y_threshold", _params[1]);

                    _tree->rootBlackboard()->set("mp_1_y", _params[2]);
                    _tree->rootBlackboard()->set("mp_1_z", _params[3]);

                    _tree->rootBlackboard()->set("mp_2_y", _params[4]);
                    _tree->rootBlackboard()->set("mp_2_z", _params[5]);

                    _tree->rootBlackboard()->set("mp_3_x", x_offset);
                    _tree->rootBlackboard()->set("mp_3_y", 0.0);
                    _tree->rootBlackboard()->set("mp_3_z", 0.8);

                    _tree->rootBlackboard()->set("mp_1_x", x_offset);
                    _tree->rootBlackboard()->set("mp_2_x", x_offset);
                } else {
                    LOG(FATAL) << "Parameter vector does not have a size of 6 or 8. Can't set blackboard.";
                }
            }

            Eigen::VectorXd params() const
            {
                return _params;
            }

            void set_robot(std::shared_ptr<robot_dart::Robot> simulated_robot) {
                _simulated_robot = simulated_robot->clone();
            }
            int bt_response{0};

        protected:
            std::shared_ptr<BT::Tree> _tree;
            Eigen::VectorXd _params;
            std::shared_ptr<robot_dart::Robot> _simulated_robot;
        };
    } // namespace policy
} // namespace blackdrops
#endif
