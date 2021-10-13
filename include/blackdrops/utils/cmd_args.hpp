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
#ifndef BLACKDROPS_UTILS_CMD_ARGS_HPP
#define BLACKDROPS_UTILS_CMD_ARGS_HPP

#include <boost/program_options.hpp>
#include <string>

namespace po = boost::program_options;

namespace blackdrops {
    namespace utils {
        class CmdArgs {
        public:
            CmdArgs() : _verbose(false), _stochastic(false), _uncertainty(false), _real_rollout(false), _threads(tbb::task_scheduler_init::automatic), _desc("Command line arguments") { _set_defaults(); }

            int parse(int argc, char** argv)
            {
                try {
                    po::variables_map vm;
                    po::store(po::parse_command_line(argc, argv, _desc), vm);
                    if (vm.count("help")) {
                        std::cout << _desc << std::endl;
                        return 0;
                    }

                    po::notify(vm);

                    if (vm.count("threads")) {
                        _threads = vm["threads"].as<int>();
                    }
                    if (vm.count("data")) {
                        _data = vm["data"].as<std::string>();
                    }
                    if (vm.count("pseudo_samples")) {
                        int c = vm["pseudo_samples"].as<int>();
                        if (c < 1)
                            c = 1;
                        _pseudo_samples = c;
                    }
                    else {
                        _pseudo_samples = 10;
                    }
                    if (vm.count("boundary")) {
                        double c = vm["boundary"].as<double>();
                        if (c < 0)
                            c = 0;
                        _boundary = c;
                    }
                    else {
                        _boundary = 0.;
                    }

                    // Cmaes parameters
                    if (vm.count("max_evals")) {
                        int c = vm["max_evals"].as<int>();
                        _max_fun_evals = c;
                    }
                    else {
                        _max_fun_evals = -1;
                    }
                    if (vm.count("tolerance")) {
                        double c = vm["tolerance"].as<double>();
                        if (c < 0.)
                            c = 0.;
                        _fun_tolerance = c;
                    }
                    else {
                        _fun_tolerance = 1.;
                    }
                    if (vm.count("restarts")) {
                        int c = vm["restarts"].as<int>();
                        if (c < 1)
                            c = 1;
                        _restarts = c;
                    }
                    else {
                        _restarts = 1;
                    }
                    if (vm.count("elitism")) {
                        int c = vm["elitism"].as<int>();
                        if (c < 0 || c > 3)
                            c = 0;
                        _elitism = c;
                    }
                    else {
                        _elitism = 0;
                    }
                    if (vm.count("lambda")) {
                        int l = vm["lambda"].as<int>();
                        if (l < 0)
                            l = -1;
                        _lambda = l;
                    }
                    else {
                        _lambda = -1;
                    }
                    if (vm.count("iterations")) {
                        int i = vm["iterations"].as<int>();
                        if (i < 0)
                            i = 0;
                        _iterations = i;
                    }
                    else {
                        _iterations = 0;
                    }
                }
                catch (po::error& e) {
                    std::cerr << "[Exception caught while parsing command line arguments]: " << e.what() << std::endl;
                    return 1;
                }

                return -1;
            }

            bool verbose() const { return _verbose; }
            bool stochastic() const { return _stochastic; }
            bool real_rollout() const { return _real_rollout; }
            bool learn_real() const {return _learn_real; }
            bool uncertainty() const { return _uncertainty; }

            int threads() const { return _threads; }
            bool neural_net() const { return _nn; }
            int pseudo_samples() const { return _pseudo_samples; }
            int max_fun_evals() const { return _max_fun_evals; }
            int restarts() const { return _restarts; }
            int elitism() const { return _elitism; }
            int lambda() const { return _lambda; }
            int iterations() const {return _iterations; }

            double boundary() const { return _boundary; }
            double fun_tolerance() const { return _fun_tolerance; }

            std::string data() const { return _data; }

        protected:
            bool _verbose, _stochastic, _uncertainty, _real_rollout, _cma_es, _learn_real, _nn;
            int _threads, _pseudo_samples, _max_fun_evals, _restarts, _elitism, _lambda, _iterations;
            double _boundary, _fun_tolerance;
            std::string _data;

            po::options_description _desc;

            void _set_defaults()
            {
                // clang-format off
                _desc.add_options()("help,h", "Prints this help message")
                                ("neural_net,n", po::bool_switch(&_nn)->default_value(false), "Use NN policy.")
                                ("pseudo_samples,p", po::value<int>(), "Number of pseudo samples in GP policy.")
                                ("boundary,b", po::value<double>(), "Boundary of the values during the optimization.")
                                ("max_evals,m", po::value<int>(), "Max function evaluations to optimize the policy.")
                                ("tolerance,t", po::value<double>(), "Maximum tolerance to continue optimizing the function.")
                                ("restarts,r", po::value<int>(), "Max number of restarts to use during optimization.")
                                ("elitism,e", po::value<int>(), "Elitism mode to use [0 to 3].")
                                ("lambda,l", po::value<int>(), "Initial population of CMA-ES. Defaults to -1 (i.e., automatically determined).")
                                ("uncertainty,u", po::bool_switch(&_uncertainty)->default_value(false), "Enable uncertainty handling in CMA-ES.")
                                ("stochastic,s", po::bool_switch(&_stochastic)->default_value(false), "Enable stochastic rollouts (i.e., not use the mean model).")
                                ("iterations,i", po::value<int>(), "The number of learning & executions iterations to run.")
                                ("real_rollout,z", po::bool_switch(&_real_rollout)->default_value(false), "Enables rollout on the real robot instead of simulation.")
                                ("learn_real,y", po::bool_switch(&_learn_real)->default_value(false), "Learn with the real robot instead of simulation.")
                                ("threads,d", po::value<int>(), "Max number of threads used by TBB")
                                ("data", po::value<std::string>(), "An experiment folder to continue from.")
                                ("verbose,v", po::bool_switch(&_verbose)->default_value(false), "Enable verbose mode.");
                // clang-format on
            }
        };
    } // namespace utils
} // namespace blackdrops

#endif