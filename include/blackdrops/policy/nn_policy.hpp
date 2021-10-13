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
#ifndef BLACKDROPS_POLICY_NN_POLICY_HPP
#define BLACKDROPS_POLICY_NN_POLICY_HPP

#include <Eigen/Core>

#include <limbo/tools/random_generator.hpp>

#include <simple_nn/neural_net.hpp>

namespace blackdrops {
    namespace policy {

        template <typename Params>
        struct Tanh {
            static Eigen::MatrixXd f(const Eigen::MatrixXd& input)
            {
                return (Params::nn_policy::af() * input.array()).tanh();
            }

            static Eigen::MatrixXd df(const Eigen::MatrixXd& input)
            {
                Eigen::MatrixXd value = f(input);
                return 1. - value.array().square();
            }
        };

        template <typename Params>
        struct NNPolicy {
        public:
            using nn_t = simple_nn::NeuralNet;

            NNPolicy(bool unneeded = false)
            {
                _boundary = Params::blackdrops::boundary();
                _random = false;

                _nn.add_layer<simple_nn::FullyConnectedLayer<Tanh<Params>>>(Params::nn_policy::state_dim(), Params::nn_policy::hidden_neurons());
                _nn.add_layer<simple_nn::FullyConnectedLayer<Tanh<Params>>>(Params::nn_policy::hidden_neurons(), Params::nn_policy::action_dim());

                _params = Eigen::VectorXd::Zero(_nn.num_weights());
                _limits = Eigen::VectorXd::Constant(Params::nn_policy::state_dim(), 1.0);

                // Get the limits
                for (int i = 0; i < _limits.size(); i++) {
                    _limits(i) = Params::nn_policy::limits(i);
                }
            }

            Eigen::VectorXd next(const Eigen::VectorXd& state, double t = 0) const
            {
                if (_random || _params.size() == 0) {
                    Eigen::VectorXd act = (limbo::tools::random_vector(Params::nn_policy::action_dim()).array() * 2 - 1.0);
                    for (int i = 0; i < act.size(); i++) {
                        act(i) = act(i) * Params::nn_policy::max_u(i);
                    }
                    return act;
                }

                Eigen::VectorXd nstate = state.array().head(7) / _limits.array();
                Eigen::VectorXd act = _nn.forward(nstate);

                for (int i = 0; i < act.size(); i++) {
                    act(i) = act(i) * Params::nn_policy::max_u(i);
                }
                Eigen::VectorXd controller_act = Eigen::VectorXd::Zero(9);
                controller_act[3] = 1.0;
                controller_act.head(3) = act;
                return controller_act;
            }

            void set_random_policy()
            {
                _random = true;
            }

            bool random() const
            {
                return _random;
            }

            void set_robot(std::shared_ptr<robot_dart::Robot> simulated_robot) {
            }

            void set_params(const Eigen::VectorXd& params)
            {
                _params = params;
                _random = false;
                _nn.set_weights(params);
            }

            Eigen::VectorXd params() const
            {
                if (_random || _params.size() == 0)
                    return limbo::tools::random_vector(_nn.num_weights()).array() * 2.0 * _boundary - _boundary;
                return _params;
            }

            int bt_response{0};

        protected:
            nn_t _nn;
            Eigen::VectorXd _params;
            bool _random;

            Eigen::VectorXd _means;
            Eigen::MatrixXd _sigmas;
            Eigen::VectorXd _limits;

            double _boundary;
        };
    } // namespace policy
} // namespace blackdrops
#endif
