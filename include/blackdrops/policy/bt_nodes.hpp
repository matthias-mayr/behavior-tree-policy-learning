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
#ifndef BLACKDROPS_POLICY_BT_NODES_HPP
#define BLACKDROPS_POLICY_BT_NODES_HPP


#include <behaviortree_cpp_v3/action_node.h>
#include <behaviortree_cpp_v3/tree_node.h>

namespace blackdrops {
    namespace policy {
        namespace bt_learning_nodes {

            class EEPositionDistance : public BT::ConditionNode {
            public:
                EEPositionDistance(const std::string& name, const BT::NodeConfiguration& config) :
                    BT::ConditionNode(name, config) {}

                static BT::PortsList providedPorts() { 
                    return {
                        BT::InputPort<double>("threshold"),
                        BT::InputPort<double>("target_x"),
                        BT::InputPort<double>("target_y"),
                        BT::InputPort<double>("target_z"),
                        BT::InputPort<double>("ee_x"),
                        BT::InputPort<double>("ee_y"),
                        BT::InputPort<double>("ee_z")};
                }

                BT::NodeStatus tick() override {
                    BT::Optional<double> threshold = getInput<double>("threshold");
                    BT::Optional<double> ee_x = getInput<double>("ee_x");
                    BT::Optional<double> ee_y = getInput<double>("ee_y");
                    BT::Optional<double> ee_z = getInput<double>("ee_z");
                    BT::Optional<double> target_x = getInput<double>("target_x");
                    BT::Optional<double> target_y = getInput<double>("target_y");
                    BT::Optional<double> target_z = getInput<double>("target_z");
                    if (!threshold || !ee_x || !ee_y || !ee_z || !target_x || !target_y || !target_z)
                    {
                        throw BT::RuntimeError("Missing required input.");
                    }
                    Eigen::Vector3d target, ee_pos;
                    target << target_x.value(), target_y.value(), target_z.value();
                    ee_pos << ee_x.value(), ee_y.value(), ee_z.value();
                    double diff = (target-ee_pos).norm();
                    if (diff < threshold.value()) {
                        return BT::NodeStatus::SUCCESS;
                    } else {
                        return BT::NodeStatus::FAILURE;
                    }
                }
            };

            class EEOutsideCube : public BT::ConditionNode {
            public:
                EEOutsideCube(const std::string& name, const BT::NodeConfiguration& config) :
                    BT::ConditionNode(name, config) {}

                static BT::PortsList providedPorts() { 
                    return {
                        BT::InputPort<double>("length"),
                        BT::InputPort<double>("cube_x"),
                        BT::InputPort<double>("cube_y"),
                        BT::InputPort<double>("cube_z"),
                        BT::InputPort<double>("ee_x"),
                        BT::InputPort<double>("ee_y"),
                        BT::InputPort<double>("ee_z")};
                }

                BT::NodeStatus tick() override {
                    BT::Optional<double> length = getInput<double>("length");
                    BT::Optional<double> ee_x = getInput<double>("ee_x");
                    BT::Optional<double> ee_y = getInput<double>("ee_y");
                    BT::Optional<double> ee_z = getInput<double>("ee_z");
                    BT::Optional<double> cube_x = getInput<double>("cube_x");
                    BT::Optional<double> cube_y = getInput<double>("cube_y");
                    BT::Optional<double> cube_z = getInput<double>("cube_z");
                    if (!length || !ee_x || !ee_y || !ee_z || !cube_x || !cube_y || !cube_z)
                    {
                        throw BT::RuntimeError("Missing required input.");
                    }
                    if (ee_x.value() > cube_x.value() + length.value() || ee_x.value() < cube_x.value() - length.value()) {
                        return BT::NodeStatus::SUCCESS;
                    } else if (ee_y.value() > cube_y.value() + length.value() || ee_y.value() < cube_y.value() - length.value()) {
                        return BT::NodeStatus::SUCCESS;
                    } else if (ee_z.value() > cube_z.value() + length.value() || ee_z.value() < cube_z.value() - length.value()) {
                        return BT::NodeStatus::SUCCESS;
                    } else {
                        return BT::NodeStatus::FAILURE;
                    }
                }
            };

            class EEYThreshold : public BT::ConditionNode {
            public:
                EEYThreshold(const std::string& name, const BT::NodeConfiguration& config) :
                    BT::ConditionNode(name, config) {}

                static BT::PortsList providedPorts() { 
                    return {
                        BT::InputPort<double>("threshold"),
                        BT::InputPort<double>("ee_y"),
                        BT::InputPort<bool>("above")};
                }

                BT::NodeStatus tick() override {
                    BT::Optional<double> threshold = getInput<double>("threshold");
                    BT::Optional<double> ee_y = getInput<double>("ee_y");
                    BT::Optional<bool> above = getInput<bool>("above");
                    if (!threshold || !ee_y || !above)
                    {
                        throw BT::RuntimeError("Missing required input [threshold], [ee_y] or [above].");
                    }
                    if (above.value() && ee_y.value() > threshold.value()) {
                        return BT::NodeStatus::SUCCESS;
                    } else if (!above.value() && ee_y.value() < threshold.value()) {
                        return BT::NodeStatus::SUCCESS;
                    } else {
                        return BT::NodeStatus::FAILURE;
                    }
                }
            };

            class EEZThreshold : public BT::ConditionNode {
            public:
                EEZThreshold(const std::string& name, const BT::NodeConfiguration& config) :
                    BT::ConditionNode(name, config) {}

                static BT::PortsList providedPorts() { 
                    return {
                        BT::InputPort<double>("threshold"),
                        BT::InputPort<double>("ee_z"),
                        BT::InputPort<bool>("above")};
                }

                BT::NodeStatus tick() override {
                    BT::Optional<double> threshold = getInput<double>("threshold");
                    BT::Optional<double> ee_z = getInput<double>("ee_z");
                    BT::Optional<bool> above = getInput<bool>("above");
                    if (!threshold || !ee_z || !above)
                    {
                        throw BT::RuntimeError("Missing required input [threshold], [ee_z] or [above].");
                    }
                    if (above.value() && ee_z.value() > threshold.value()) {
                        return BT::NodeStatus::SUCCESS;
                    } else if (!above.value() && ee_z.value() < threshold.value()) {
                        return BT::NodeStatus::SUCCESS;
                    } else {
                        return BT::NodeStatus::FAILURE;
                    }
                }
            };

            class SetMGGoal : public BT::ActionNodeBase
            {
            public:
                SetMGGoal(const std::string& name, const BT::NodeConfiguration& config) :
                    BT::ActionNodeBase(name, config)
                {
                }

                static BT::PortsList providedPorts() { return {
                    BT::InputPort<double>("mp_param_x"),
                    BT::InputPort<double>("mp_param_y"),
                    BT::InputPort<double>("mp_param_z"),
                    BT::OutputPort<double>("mg_trans_x"),
                    BT::OutputPort<double>("mg_trans_y"),
                    BT::OutputPort<double>("mg_trans_z"),
                }; }

                void halt()
                {
                    setStatus(BT::NodeStatus::IDLE);
                }

                BT::NodeStatus tick() override
                {
                    BT::Optional<double> param_x = getInput<double>("mp_param_x");
                    BT::Optional<double> param_y = getInput<double>("mp_param_y");
                    BT::Optional<double> param_z = getInput<double>("mp_param_z");
                    // Check if optional is valid. If not, throw its error
                    if (!param_x || !param_y || !param_z)
                    {
                        throw BT::RuntimeError("missing required input [mg_param_x] or [mg_param_y] or [mg_param_z]: ");
                    }
                    setOutput("mg_trans_x", param_x.value());
                    setOutput("mg_trans_y", param_y.value());
                    setOutput("mg_trans_z", param_z.value());
                    return BT::NodeStatus::RUNNING;
                }
            };

            class SetMGSin : public BT::ActionNodeBase
            {
            public:
                SetMGSin(const std::string& name, const BT::NodeConfiguration& config) :
                    BT::ActionNodeBase(name, config)
                {
                }

                static BT::PortsList providedPorts() { return {
                    BT::InputPort<double>("mp_ampl"),
                    BT::InputPort<double>("mp_freq"),
                    BT::OutputPort<double>("mg_ampl"),
                    BT::OutputPort<double>("mg_freq"),
                }; }

                void halt()
                {
                    setStatus(BT::NodeStatus::IDLE);
                }

                BT::NodeStatus tick() override
                {
                    BT::Optional<double> ampl = getInput<double>("mp_ampl");
                    BT::Optional<double> freq = getInput<double>("mp_freq");
                    // Check if optional is valid. If not, throw its error
                    if (!ampl || !freq)
                    {
                        throw BT::RuntimeError("Missing required input.");
                    }
                    setOutput("mg_ampl", ampl.value());
                    setOutput("mg_freq", freq.value());
                    return BT::NodeStatus::RUNNING;
                }
            };
        } // namespace bt_learning_nodes
    } // namespace policy
} // namespace blackdrops
#endif
