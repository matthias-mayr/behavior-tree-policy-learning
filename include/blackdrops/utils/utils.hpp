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
#ifndef UTILS_UTILS_HPP
#define UTILS_UTILS_HPP

#include <string>
#include <sys/stat.h>
#include <utility>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <limbo/tools/random_generator.hpp>

namespace blackdrops {
    namespace rng {
        static thread_local limbo::tools::rgen_gauss_t gauss_rng(0., 1.);
        static thread_local limbo::tools::rgen_double_t uniform_rng(0., 1.);
    } // namespace rng

    namespace utils {
        inline Eigen::VectorXd uniform_rand(int size, limbo::tools::rgen_double_t& rgen = rng::uniform_rng)
        {
            return limbo::tools::random_vec(size, rgen);
        }

        inline Eigen::VectorXd gaussian_rand(const Eigen::VectorXd& mean, limbo::tools::rgen_gauss_t& rgen = rng::gauss_rng)
        {
            return limbo::tools::random_vec(mean.size(), rgen) + mean;
        }

        inline Eigen::VectorXd gaussian_rand(const Eigen::VectorXd& mean, const Eigen::MatrixXd& covar, limbo::tools::rgen_gauss_t& rgen = rng::gauss_rng)
        {
            assert(mean.size() == covar.rows() && covar.rows() == covar.cols());

            Eigen::LLT<Eigen::MatrixXd> cholSolver(covar);
            Eigen::MatrixXd transform = cholSolver.matrixL();

            return transform * gaussian_rand(Eigen::VectorXd::Zero(mean.size()), rgen) + mean;
        }

        inline Eigen::VectorXd gaussian_rand(const Eigen::VectorXd& mean, const Eigen::VectorXd& sigma, limbo::tools::rgen_gauss_t& rgen = rng::gauss_rng)
        {
            assert(mean.size() == sigma.size());

            Eigen::MatrixXd covar = Eigen::MatrixXd::Zero(mean.size(), mean.size());
            covar.diagonal() = sigma.array().square();

            return gaussian_rand(mean, covar, rgen);
        }

        inline Eigen::VectorXd gaussian_rand(const Eigen::VectorXd& mean, double sigma, limbo::tools::rgen_gauss_t& rgen = rng::gauss_rng)
        {
            Eigen::VectorXd sig = Eigen::VectorXd::Constant(mean.size(), sigma);

            return gaussian_rand(mean, sig, rgen);
        }

        inline double gaussian_rand(double mean, double sigma, limbo::tools::rgen_gauss_t& rgen = rng::gauss_rng)
        {
            Eigen::VectorXd m(1);
            m << mean;

            return gaussian_rand(m, sigma, rgen)[0];
        }

        inline double angle_dist(double a, double b)
        {
            double theta = b - a;
            while (theta < -M_PI)
                theta += 2 * M_PI;
            while (theta > M_PI)
                theta -= 2 * M_PI;
            return theta;
        }

        // Sample mean and covariance
        inline std::pair<Eigen::VectorXd, Eigen::MatrixXd> sample_statistics(const std::vector<Eigen::VectorXd>& points)
        {
            assert(points.size());

            // Get the sample mean
            Eigen::VectorXd mean = Eigen::VectorXd::Zero(points[0].size());

            for (size_t i = 0; i < points.size(); i++) {
                mean.array() += points[i].array();
            }

            mean = mean.array() / double(points.size());

            // Calculate the sample covariance matrix
            Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(points[0].size(), points[0].size());
            for (size_t i = 0; i < points.size(); i++) {
                cov = cov + points[i] * points[i].transpose();
            }

            cov = (cov.array() - (double(points.size()) * mean * mean.transpose()).array()) / (double(points.size()) - 1.0);

            return {mean, cov};
        }

        inline bool file_exists(const std::string& name)
        {
            struct stat buffer;
            return (stat(name.c_str(), &buffer) == 0);
        }

        bool replace_string(std::string& str, const std::string& from, const std::string& to)
        {
            size_t start_pos = str.find(from);
            if (start_pos == std::string::npos)
                return false;
            str.replace(start_pos, from.length(), to);
            return true;
        }

        void save_traj_to_file(const std::string& filename, const std::vector<Eigen::VectorXd>& states, const std::vector<Eigen::VectorXd>& commands) {
            std::ofstream file;
            file.open(filename);
            for (size_t i = 0; i < states.size()-1; i++) {
                Eigen::VectorXd state = states[i];
                Eigen::VectorXd command = commands[i];

                for (int j = 0; j < state.size(); j++)
                    file << state(j) << " ";
                for (int j = 0; j < command.size(); j++)
                    file << command(j) << " ";
                file << std::endl;
            }
            // In the last timestep we do not do any action
            Eigen::VectorXd state = states.back();
            for (int j = 0; j < state.size(); j++)
                file << state(j) << " ";
            for (int j = 0; j < commands.back().size(); j++)
                file << "0.0 ";
            file << std::endl;
            file.close();
        }

        template<typename T>
        void save_vec_to_file(const std::string& filename, const T& data) {
            std::ofstream file;
            file.open(filename);
            for (const auto& e : data) {
                file << e.c_sep() << std::endl;
            }
            file.close();
        }

        // Infrastructure to call python files
        struct popen2 {
            pid_t child_pid;
            int from_child, to_child;
        };

        int popen2(const char *cmdline, struct popen2 *childinfo) {
            pid_t p;
            int pipe_stdin[2], pipe_stdout[2];

            if (pipe(pipe_stdin))
                return -1;
            if (pipe(pipe_stdout))
                return -1;

            // printf("pipe_stdin[0] = %d, pipe_stdin[1] = %d\n", pipe_stdin[0],
            //         pipe_stdin[1]);
            // printf("pipe_stdout[0] = %d, pipe_stdout[1] = %d\n", pipe_stdout[0],
            //         pipe_stdout[1]);

            p = fork();
            if (p < 0)
                return p;   /* Fork failed */
            if (p == 0) { /* child */
                close(pipe_stdin[1]);
                dup2(pipe_stdin[0], 0);
                close(pipe_stdout[0]);
                dup2(pipe_stdout[1], 1);
                execl("/bin/sh", "sh", "-c", cmdline, 0);
                perror("execl");
                exit(99);
            }
            childinfo->child_pid = p;
            childinfo->to_child = pipe_stdin[1];
            childinfo->from_child = pipe_stdout[0];
            return 0;
        }

    } // namespace utils
} // namespace blackdrops

#endif