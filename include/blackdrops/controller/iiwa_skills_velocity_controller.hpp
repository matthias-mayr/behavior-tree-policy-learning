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
#ifndef BLACKDROPS_IIWA_SKILLS_CONTROLLER_HPP
#define BLACKDROPS_IIWA_SKILLS_CONTROLLER_HPP

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <cmath>

#include <glog/logging.h>

template <typename Params>
struct cartesian_controller {
    #define degreesToRadians(angleDegrees) ((angleDegrees) * M_PI / 180.0)
    #define radiansToDegrees(angleRadians) ((angleRadians) * 180.0 / M_PI)

    cartesian_controller() {
        complianceParamUpdate();
        // Initializes a permutation matrix needed for the Jacobian.
        // DART returns the linear part in the last 3 lines, but we expect it in the first.
        Eigen::VectorXi perm_indices = Eigen::VectorXi(6);
        perm_indices << 3,4,5,0,1,2;
        jacobian_perm_ = Eigen::PermutationMatrix<Eigen::Dynamic,6>(perm_indices);
    }

    void complianceParamUpdate(bool sim_params = true) {
        simulation_ = sim_params;
        double rotational_stiffness, damping, translational_stiffness_xy, translational_stiffness_z;
        const double scaling = 4.;
        translational_stiffness_xy = 6.;
        translational_stiffness_z = 5.;
        if (sim_params) {
            rotational_stiffness = 3.5;
            damping = 0.05;
        } else {
            rotational_stiffness = 4.5;
            damping = 0.005;
        }
        cartesian_stiffness_.setIdentity();
        cartesian_stiffness_.topLeftCorner(2, 2)
            << translational_stiffness_xy * Eigen::Matrix2d::Identity();
        cartesian_stiffness_(2, 2)= translational_stiffness_z;
        cartesian_stiffness_.bottomRightCorner(3, 3)
            << rotational_stiffness * Eigen::Matrix3d::Identity();
        cartesian_stiffness_ *= scaling;
        cartesian_damping_.setIdentity();
        cartesian_damping_.topLeftCorner(3, 3)
            << damping * Eigen::Matrix3d::Identity();
        cartesian_damping_.bottomRightCorner(3, 3)
            << damping * Eigen::Matrix3d::Identity();
    }

    Eigen::Matrix<double, 7, 1> saturateCommandRate(const Eigen::Matrix<double, 7, 1>& tau_d_calculated, const Eigen::Matrix<double, 7, 1>& tau_J_d){  // NOLINT (readability-identifier-naming)
        Eigen::Matrix<double, 7, 1> tau_d_saturated{};
        for (size_t i = 0; i < 7; i++) {
            double difference = tau_d_calculated[i] - tau_J_d[i];
            tau_d_saturated[i] =
                tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
        }
        return tau_d_saturated;
    }

    inline void pseudoInverse(const Eigen::MatrixXd& M_, Eigen::MatrixXd& M_pinv_, bool damped = true) {
        double lambda_ = damped ? 0.2 : 0.0;

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(M_, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType sing_vals_ = svd.singularValues();
        Eigen::MatrixXd S_ = M_;  // copying the dimensions of M_, its content is not needed.
        S_.setZero();

        for (int i = 0; i < sing_vals_.size(); i++)
            S_(i, i) = (sing_vals_(i)) / (sing_vals_(i) * sing_vals_(i) + lambda_ * lambda_);

        M_pinv_ = Eigen::MatrixXd(svd.matrixV() * S_.transpose() * svd.matrixU().transpose());
    }

    Eigen::Vector3d sinus_overlay() {
        Eigen::Vector3d add{Eigen::Vector3d::Zero()};
        double sin_amplitude = 0.8*sin_amplitude_;
        double t_spiral = 4.6;
        if (iteration_*step_ < t_spiral/2.0) {
            sin_amplitude *= iteration_*step_/ (t_spiral/2.0);
        } else {
            sin_amplitude *= (2-(iteration_*step_/ (t_spiral/2.0)));
        }
        sin_amplitude += 0.2*sin_amplitude_;
        add[0] = sin_amplitude*std::cos(sin_frequency_*iteration_*step_);
        add[1] = sin_amplitude*std::sin(sin_frequency_*iteration_*step_);
        if (iteration_*step_ >= t_spiral) {
            iteration_ = 0;
        }
        return add;
    }

    Eigen::Vector3d spiral_overlay() {
        Eigen::Vector3d add{Eigen::Vector3d::Zero()};
        double v{sin_frequency_};
        double theta_d = v * step_ / (last_r_+ 0.001);
        double r_d = theta_d * 0.003 / (2 * M_PI);
        theta_ += theta_d;
        spiral_increase_ ? last_r_ += r_d : last_r_ -= r_d;
        add[0] = last_r_*std::cos(theta_);
        add[1] = last_r_*std::sin(theta_);
        if (last_r_ > sin_amplitude_) {
            spiral_increase_ = false;
        }
        if (last_r_ < 0) {
            spiral_increase_ = true;
            last_r_ = 0.0;
        }
        if (!simulation_) {
            add[2] = 0.6 * last_r_ * std::cos(theta_ + M_PI/2.2);
        }
        if (simulation_ && add[1] > 0.) {
            add[1] *= 1.3;
        }
        if (simulation_ && add[0] < 0.) {
            add[0] *= 2.0;
        }
        return add;
    }

    void set_goal_direction() {
        travelled_dist_ = 0.0;
        position_d_dir_ = position_d_target_ - pos_;
        position_d_dist_ = position_d_dir_.norm();
        position_d_dir_.normalize();
    }

    void update_goal_and_parameters() {
        travelled_dist_ += step_ * cart_ref_vel_;
        if (travelled_dist_ > position_d_dist_) {
            position_d_ = position_d_target_;
        } else {
            position_d_ = position_d_target_ - (position_d_dist_ - travelled_dist_) * position_d_dir_;
        }
        position_d_ += spiral_overlay();
    }

    // Implements a Cartesian admittance controller inspired by Cartesian impedance
    Eigen::VectorXd update(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::MatrixXd& jacobian, const Eigen::VectorXd& pos, Eigen::Quaterniond rot) {
        pos_ = pos;
        if (first_) {
            tau_J_d_ = dq;
            set_goal_direction();
        }
        update_goal_and_parameters();
        // We are expecting a Jacobian with the linear part in the first 3 lines
        Eigen::MatrixXd J = jacobian_perm_ * jacobian;

        // compute error to desired pose
        // position error
        Eigen::Matrix<double, 6, 1> error;
        error.head(3) << pos - position_d_;

        // orientation error
        if (orientation_d_.coeffs().dot(rot.coeffs()) < 0.0) {
            rot.coeffs() << -rot.coeffs();
        }
        // "difference" quaternion
        Eigen::Quaterniond error_quaternion(rot * orientation_d_.inverse());

        // convert to axis angle
        Eigen::AngleAxisd error_quaternion_angle_axis(error_quaternion);
        // compute "orientation error"
        error.tail(3) << error_quaternion_angle_axis.axis() * error_quaternion_angle_axis.angle();

        // compute control
        // allocate variables
        Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7);

        // Cartesian PD control with damping ratio = 1
        tau_task << J.transpose() *
                        (-cartesian_stiffness_ * error - cartesian_damping_ * (J * dq));

        // Desired torque.
        tau_d << tau_task;

        double tau_max = tau_d.cwiseAbs().maxCoeff();
        double scale{1.};
        if (tau_max > max_joint_vel_value) {
            scale = max_joint_vel_value/tau_max;
        }
        tau_d = tau_d*scale;

        // reduce velocities next to joint limits
        for (size_t i{0}; i < 7; i++) {
            if (q[i] > degreesToRadians(_joint_limits[i]) && tau_d[i] > 0.) {
                tau_d[i] = 0.0;
            } else if (q[i] < -degreesToRadians(_joint_limits[i]) && tau_d[i] < 0.) {
                tau_d[i] = 0.0;
            }
        }
        first_ = false;
        iteration_++;
        tau_J_d_ = tau_d;
        return tau_d;
    }

    void process_command(const Eigen::VectorXd& command, bool hard_set_desired = true) {
        if (command != prev_command_) {
            if (prev_command_.head(3) != command.head(3)) {
                position_d_target_ = command.head(3);
                first_ = true;
            }
            orientation_d_ = Eigen::Quaternion<double>(command.segment(3,4).data());
            sin_amplitude_ = command[7];
            sin_frequency_ = command[8];
            prev_command_ = command;
        }
        if(hard_set_desired) {
            first_ = true;
        }
    }

    void set_step(const double step) {
        step_ = step;
    }

    void set_max_joint_vel(const double vel) {
        max_joint_vel_value = vel;
    }

    private:
        const double pi_{3.14159265358979323846};
        double step_{Params::blackdrops::dt()};
        double sin_amplitude_{0.0};
        double sin_frequency_{0.0};
        double last_r_{0};
        double theta_{0};
        bool simulation_{true};
        bool spiral_increase_{true};
        unsigned int iteration_{0};
        bool first_{true};
        // double filter_params_{0.001};
        const double delta_tau_max_{0.01};  // Maximum difference per timestep
        double max_joint_vel_value{0.4};
        double cart_ref_vel_{0.15};
        const Eigen::VectorXd max_joint_vel{Eigen::VectorXd::Constant(7, max_joint_vel_value)};
        // Safety configuration in the URDF minus 1 deg
        const std::vector<double> _joint_limits{160.5, 113, 160.5, 113, 160.5, 113, 165.25};
        Eigen::VectorXd position_d_{Eigen::VectorXd::Zero(3)};
        double position_d_dist_{0.0};
        double travelled_dist_{0.0};
        Eigen::Vector3d position_d_dir_{Eigen::Vector3d::Zero()};
        Eigen::VectorXd pos_{Eigen::VectorXd::Zero(3)};
        Eigen::VectorXd prev_command_{Eigen::VectorXd::Zero(9)};
        Eigen::VectorXd position_d_target_{Eigen::VectorXd::Zero(3)};
        Eigen::Quaterniond orientation_d_{Eigen::Quaterniond::Identity()};
        Eigen::Matrix<double, 6, 6> cartesian_stiffness_{Eigen::Matrix<double, 6,6>::Zero()};
        Eigen::Matrix<double, 6, 6> cartesian_damping_{Eigen::Matrix<double, 6,6>::Zero()};
        Eigen::Matrix<double, 7, 1> tau_J_d_{Eigen::Matrix<double, 7,1>::Zero()};   // Last command. Used for saturation
        Eigen::PermutationMatrix<Eigen::Dynamic,6> jacobian_perm_;
};

#endif