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
#ifndef BLACKDROPS_IIWA_SKILLS_OBSTACLE_HPP
#define BLACKDROPS_IIWA_SKILLS_OBSTACLE_HPP

#include <sstream>
#include <cmath>
#include <assert.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/constant.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/model/multi_gp.hpp>
#include <limbo/model/multi_gp/parallel_lf_opt.hpp>
#include <limbo/opt/cmaes.hpp>

#include <blackdrops/blackdrops_for_bt.hpp>
#include <blackdrops/model/gp/kernel_lf_opt.hpp>
#include <blackdrops/model/gp_model.hpp>
#include <blackdrops/system/dart_system.hpp>
#include <blackdrops/policy/bt_policy_obstacle.hpp>
#include <blackdrops/policy/nn_policy.hpp>
#include <blackdrops/reward/gp_reward.hpp>

#include <blackdrops/utils/cmd_args.hpp>
#include <blackdrops/utils/utils.hpp>
#include <blackdrops/experiments/iiwa_skills_policy_control.hpp>

#include <dart/constraint/ConstraintSolver.hpp>
#include <dart/collision/fcl/FCLCollisionDetector.hpp>

#include <glog/logging.h>

#include <boost/filesystem.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>


namespace iiwa_skills_obstacle{

#define degreesToRadians(angleDegrees) ((angleDegrees) * M_PI / 180.0)
#define radiansToDegrees(angleRadians) ((angleRadians) * 180.0 / M_PI)

struct Params {
    struct meta_conf {
        // Folders to save results of an experiment
        BO_DYN_PARAM(std::string, folder);
        BO_DYN_PARAM(bool, verbose);
        BO_DYN_PARAM(std::string, model_folder);
        BO_DYN_PARAM(std::string, params_folder);
        BO_DYN_PARAM(std::string, traj_folder);
        BO_DYN_PARAM(std::string, opt_folder);
        BO_DYN_PARAM(int, threads);
    };

    struct blackdrops : public ::blackdrops::defaults::blackdrops {
        BO_PARAM(size_t, num_params, 6);
        BO_PARAM(size_t, action_dim, 7);
        BO_PARAM(size_t, model_input_dim, 14);
        BO_PARAM(size_t, model_pred_dim, 14);
        // single time step for querying the action from the policy
        BO_PARAM(double, dt, 0.02);
        // episode length in seconds
        BO_PARAM(double, T, 16.0);
        BO_DYN_PARAM(double, boundary);
        BO_DYN_PARAM(bool, learn_real_system);
        // assume stochastic or deterministig episodes
        // if stochastic: there's a number of rollouts parameters
        BO_DYN_PARAM(bool, stochastic);
        BO_DYN_PARAM(bool, real_rollout);
        BO_PARAM(bool, stochastic_evaluation, true);// Enables the two behaviors below.
        BO_PARAM(int, num_evals, 10);                // How often will a faked real execution or the final evaluation be executed?
        BO_PARAM(int, opt_evals, 1);                // How often will optimization evaluation be executed?
        BO_DYN_PARAM(bool, domain_randomization);   // Enable domain randomization
        BO_PARAM(double, max_collision_depth, 0.04);
        BO_PARAM(double, max_joint_vel, 0.4);
        BO_DYN_PARAM(bool, nn_policy);
    };

    struct dart_system {
        BO_PARAM(double, sim_step, 0.001);
    };

    struct dart_policy_control {
        BO_PARAM(dart::dynamics::Joint::ActuatorType, joint_type, dart::dynamics::Joint::SERVO);
    };

    struct gp_model {
        BO_PARAM(double, noise, 0.0000001);
    };

    struct mean_constant {
        BO_PARAM(double, constant, 0.0);
    };

    struct kernel : public limbo::defaults::kernel {
        BO_PARAM(double, noise, gp_model::noise());
        BO_PARAM(bool, optimize_noise, true);
    };

    struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
    };

    struct opt_cmaes : public limbo::defaults::opt_cmaes {
        BO_DYN_PARAM(int, max_fun_evals);
        BO_DYN_PARAM(double, fun_tolerance);
        BO_DYN_PARAM(int, restarts);
        BO_DYN_PARAM(int, elitism);
        BO_DYN_PARAM(bool, handle_uncertainty);

        BO_DYN_PARAM(int, lambda);

        BO_PARAM(int, variant, aIPOP_CMAES);
        BO_PARAM(bool, verbose, false);
        BO_PARAM(bool, fun_compute_initial, true);
        BO_DYN_PARAM(double, ubound);
        BO_DYN_PARAM(double, lbound);
        BO_DYN_PARAM(bool, stochastic);
    };

    struct opt_rprop : public limbo::defaults::opt_rprop {
        BO_PARAM(int, iterations, 300);
        BO_PARAM(double, eps_stop, 1e-4);
    };
};

struct PolicyParams {
    struct blackdrops : public Params::blackdrops {
    };

    struct bt_policy {
        BO_DYN_PARAM(bool, verbose);
        BO_PARAM(size_t, num_params, Params::blackdrops::num_params());
        BO_PARAM(size_t, state_dim, Params::blackdrops::model_input_dim());
        BO_PARAM(size_t, action_dim, Params::blackdrops::action_dim());
    };

    struct nn_policy {
        BO_PARAM(size_t, state_dim, 7);
        BO_PARAM(size_t, hidden_neurons, 10);
        BO_PARAM(size_t, action_dim, 3);
        BO_PARAM_ARRAY(double, max_u, 2.0, 2.0, 2.0);
        BO_PARAM_ARRAY(double, limits, 3.0, 2.1, 3.0, 2.1, 3.0, 2.1, 3.0);
        BO_PARAM(double, af, 1.0);
    };
};

namespace global {
    std::shared_ptr<robot_dart::Robot> global_robot;

    using policy_t = blackdrops::policy::BTPolicyObstacle<PolicyParams>;
    //using policy_t = blackdrops::policy::NNPolicy<PolicyParams>;

    Eigen::VectorXd goal(3);
} // namespace global

Eigen::VectorXd get_random_vector(size_t dim, Eigen::VectorXd bounds)
{
    Eigen::VectorXd rv = (limbo::tools::random_vector(dim).array() * 2 - 1);
    // rv(0) *= 3; rv(1) *= 5; rv(2) *= 6; rv(3) *= M_PI; rv(4) *= 10;
    return rv.cwiseProduct(bounds);
}

std::vector<Eigen::VectorXd> random_vectors(size_t dim, size_t q, Eigen::VectorXd bounds)
{
    std::vector<Eigen::VectorXd> result(q);
    for (size_t i = 0; i < q; i++) {
        result[i] = get_random_vector(dim, bounds);
    }
    return result;
}

// The experiment definition
struct SimpleArm : public blackdrops::system::DARTSystem<Params, PolicyControl<Params, global::policy_t>, blackdrops::RolloutInfo> {
    using base_t = blackdrops::system::DARTSystem<Params, PolicyControl<Params, global::policy_t>, blackdrops::RolloutInfo>;

    SimpleArm() {
        _goal = global::goal;
    }

    // Initialize the state, e.g. joints. Additionally there's a simulation setup function.
    Eigen::VectorXd init_state() const
    {
        // This corresponds to the "set_robot_state" function below
        // The last 7 entries will be 0 for joint velocities
        Eigen::VectorXd init_state(14);
        if (Params::blackdrops::domain_randomization()) {
            std::vector<Eigen::VectorXd> start_pos;
            Eigen::VectorXd q(7);
            q << 0.47, 1.54, -0.30, -0.84, 1.48, 1.34, -1.28;
            start_pos.push_back(q);
            q << 0.43, 1.65, -0.32, -0.56, 1.46, 1.37, -1.11;
            start_pos.push_back(q);
            q << 0.44, 1.58, -0.30, -0.90, 1.49, 1.34, -1.38;
            start_pos.push_back(q);
            q << 0.49, 1.45, -0.28, -1.10, 1.50, 1.32, -1.45;
            start_pos.push_back(q);
            q << 0.48, 1.52, -0.30, -0.74, 1.46, 1.35, -1.15;
            start_pos.push_back(q);
            // Get random number
            Eigen::VectorXd r_nr_v = blackdrops::utils::uniform_rand(1);
            double r_nr = r_nr_v[0] * start_pos.size();
            // The range for index 0 shall be from -0.5 to 0.5.
            r_nr -= 0.5;
            int index = static_cast<int>(std::round(r_nr));
            // Cover edge cases:
            if (index < 0) {
                index = 0;
            } else if (index >= (int)start_pos.size()) {
                index = static_cast<int>(start_pos.size()) - 1;
            }
            if (Params::meta_conf::verbose()) {
                LOG(INFO) << "Start position number [1-?]: " << index + 1;
            }
            init_state << start_pos[index], Eigen::VectorXd::Zero(7);
        } else {
            init_state << 0.47, 1.54, -0.30, -0.84, 1.48, 1.34, -1.28, Eigen::VectorXd::Zero(7);
        }
        return init_state;
    }

    // Returns a fixed robot
    std::shared_ptr<robot_dart::Robot> get_robot() const
    {
        std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();
        simulated_robot->fix_to_world();
        simulated_robot->set_position_enforced(true);
        simulated_robot->set_actuator_types(Params::dart_policy_control::joint_type());
        return simulated_robot;
    }

    // If you want, you can add some extra to your simulator object (this is called once before its episode on a newly-created simulator object)
    void add_extra_to_simu(base_t::robot_simu_t& simu, const blackdrops::RolloutInfo& info) const
    {
        // Add collision detector that works with meshes
        simu.world()->getConstraintSolver()->setCollisionDetector( dart::collision::FCLCollisionDetector::create());
        // Disable gravity, because we stronly limit the joint torques in the URDF
        simu.world()->setGravity(Eigen::Vector3d::Zero());

        // Add goal marker
        Eigen::Vector6d goal_pose = Eigen::Vector6d::Zero();
        goal_pose.tail(3) = global::goal;
        // dims, pose, type, mass, color, name
        auto ellipsoid = robot_dart::Robot::create_ellipsoid({0.1, 0.1, 0.1}, goal_pose, "fixed", 1., dart::Color::Green(1.0), "goal_marker");
        // remove collisions from goal marker
        ellipsoid->skeleton()->getRootBodyNode()->setCollidable(false);
        // add ellipsoid to simu
        simu.add_robot(ellipsoid);

        // Insert table
        Eigen::Vector6d table_pose;
        table_pose << 0., 0., 0., 0.8, 0.4, 0.415;
        auto table = robot_dart::Robot::create_box({table_pose[3]*2, table_pose[4]*2, table_pose[5]*2}, table_pose, "fixed", 100., dart::Color::White(1.0), "table");
        table->skeleton()->getRootBodyNode()->setCollidable(true);
        simu.add_robot(table);

        // Insert workbox
        Eigen::Vector6d workbox_pose;
        workbox_pose << 0., 0., 0., -(0.942/2.0), 0.48, 0.3;
        auto workbox = robot_dart::Robot::create_box({workbox_pose[3]*2, 2.0, workbox_pose[5]*2}, workbox_pose, "fixed", 100., dart::Color::Gray(1.0), "workbox");
        workbox->skeleton()->getRootBodyNode()->setCollidable(false);
        simu.add_robot(workbox);

        // Insert obstacle
        Eigen::Vector6d engine_pose;
        engine_pose << 0., 0., 0., -0.52, 0.3, workbox_pose[5]*2+0.151;
        if (Params::blackdrops::domain_randomization()) {
            // Randomly add offset in x and y direction
            double sigma{0.04};
            for (size_t i = 3; i <= 4; i++) {
                double offset = blackdrops::utils::gaussian_rand(0, sigma);
                offset = std::max(-sigma, std::min(offset, sigma));
                engine_pose[i] = engine_pose[i] + offset;
            }
        }
        auto engine = robot_dart::Robot::create_box({0.38, 0.3, 0.3}, engine_pose, "fixed", 100., dart::Color::Red(1.0), "engine");
        engine->skeleton()->getRootBodyNode()->setCollidable(true);
        simu.add_robot(engine);

#ifdef GRAPHIC
        Eigen::Vector3d camera_pos = Eigen::Vector3d(-4, 0.4, 1.7);
        Eigen::Vector3d look_at = Eigen::Vector3d(0., 0.4, 1.);
        Eigen::Vector3d up = Eigen::Vector3d(0., 0., 1.);
        std::static_pointer_cast<robot_dart::graphics::Graphics>(simu.graphics())->look_at(camera_pos, look_at, up);
        // slow down visualization. Smaller values make it faster. 0.015 is the upper limit.
        simu.graphics()->set_render_period(0.001);
#endif
    }

    void set_robot_state(const std::shared_ptr<robot_dart::Robot>& robot, const Eigen::VectorXd& state) const
    {
        robot->skeleton()->setPositions(state.head(7));
        robot->skeleton()->setVelocities(state.tail(7));
    }
};


struct RewardFunction : public blackdrops::reward::Reward<RewardFunction> {
    typedef boost::geometry::model::point<double, 3, boost::geometry::cs::cartesian> point_t;
    typedef boost::geometry::model::box<point_t> box_t;

    RewardFunction(){
        // WARNING: boost:geometry is picky about min, max. Sort your values.
        // Defines the box we want to avoid.
        point_t min_corner_engine {-0.71,0.15,0.6};
        point_t max_corner_engine {-0.33,0.45,0.9};
        _box_engine = box_t(min_corner_engine, max_corner_engine);

        point_t min_corner_workbox {-0.93, -0.7, 0.0};
        point_t max_corner_workbox {-0.13, 0.5, 0.6};
        _box_workbox = box_t(min_corner_workbox, max_corner_workbox);
        Eigen::VectorXd v = global::goal;
        point_t min_corner_goal {v[0] - 0.02, v[1] - 0.02, v[2] -0.02};
        point_t max_corner_goal {v[0] + 0.02, v[1] + 0.02, v[2] +0.02};
        _box_goal = box_t(min_corner_goal, max_corner_goal);
    }

    double goal_distance_reward(const Eigen::VectorXd& ee_pos) const {
        double s_c_sq = 0.4 * 0.4;
        double dee = (ee_pos - global::goal).squaredNorm();
        // When changing this, remember to update the print function.
        double reward = std::exp(-0.5 / s_c_sq * (dee+0.25));
        //LOG(INFO) << "Distance to goal: " << dee << " Reward: " << _mult_goal_distance*reward;
        return reward;
    }

    double avoidance_reward_box(const Eigen::VectorXd& ee_position, const box_t& box) const {
        point_t ee_pos{ee_position[0], ee_position[1], ee_position[2]};
        double distance = boost::geometry::distance(ee_pos, box);
        if (distance < 0) {
            distance = 0.0;
        }
        // When changing this, remember to update the print function.
        return -1/(7*std::pow(distance + 0.03, 2.0));
    }

    template <typename RolloutInfo>
    double operator()(RolloutInfo& info, const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state) const
    {
        // Information allows more sophisticated calculation of rewards
        std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();
        simulated_robot->fix_to_world();
        simulated_robot->set_position_enforced(true);
        const Eigen::VectorXd positions{to_state.head(7)};
        simulated_robot->skeleton()->setPositions(positions);
        auto bd = simulated_robot->skeleton()->getBodyNode("peg");
        Eigen::VectorXd eef = bd->getTransform().translation();

        double r_goal_distance = goal_distance_reward(eef);
        double r_special = -avoidance_reward_box(eef, _box_goal);
        double r_avoidance = avoidance_reward_box(eef, _box_engine) + _mult_workbox_avoidance*avoidance_reward_box(eef, _box_workbox);
        double t_goal_distance = _mult_goal_distance * r_goal_distance;
        double t_avoidance = _mult_avoidance * r_avoidance;
        double t_special = _mult_close_goal * r_special;
        double t_finish {0};
        if (info.bt_finish_time > 0) {
            t_finish = _finished_reward;
        }
        info.r_goal += t_goal_distance;
        info.r_avoidance += t_avoidance;
        info.r_special += t_special;
        info.r_finish += t_finish;

        return t_goal_distance + t_avoidance + t_finish;
    }

    void print() const{
        LOG(INFO) << "Reward parameters:";
        LOG(INFO) << "  BT finish reward per time step: " << _finished_reward;
        LOG(INFO) << "  Multiplication goal distance: " << _mult_goal_distance;
        LOG(INFO) << "  Multiplication avoidance: " << _mult_avoidance;
        LOG(INFO) << "  Multiplied close goal: " << _mult_close_goal;
        LOG(INFO) << "  Workbox avoidance reward factor: " << _mult_workbox_avoidance;
        LOG(INFO) << "  Formula goal distance: std::exp(-0.5 / s_c_sq * (dee+0.25))";
        LOG(INFO) << "  Formula avoidance: -1/(7*std::pow(distance + 0.03, 2.0)) with distance >= 0";
    }

    private:
        const double _finished_reward{8}; // per time step
        box_t _box_engine;     // The obstacle we want to avoid
        box_t _box_workbox;
        box_t _box_goal;
        // Factors
        const double _mult_goal_distance {50};
        const double _mult_avoidance {1.5};
        const double _mult_workbox_avoidance{0.1};
        const double _mult_close_goal{0.05};
};

// Called even before the experiment is set up.
void init_simu(const std::string& robot_file)
{
    // We need packages to desribe where the stl files are located.
    // They are copied into blackdrops to avoid dependencies.
    std::vector<std::pair<std::string, std::string>> packages;
    packages.push_back(std::make_pair(std::string("bh_description"), std::string(RESPATH)));
    packages.push_back(std::make_pair(std::string("iiwa_description"), std::string(RESPATH)));

    global::global_robot = std::make_shared<robot_dart::Robot>(robot_file, packages, "arm");

    Eigen::VectorXd goal_pos (3);
    goal_pos << -0.5, 0.0, 0.7;
    global::goal = goal_pos;

    LOG(INFO) << "Goal is at: " << global::goal.transpose();
}

BO_DECLARE_DYN_PARAM(bool, Params::meta_conf, verbose);
BO_DECLARE_DYN_PARAM(std::string, Params::meta_conf, folder);
BO_DECLARE_DYN_PARAM(std::string, Params::meta_conf, model_folder);
BO_DECLARE_DYN_PARAM(std::string, Params::meta_conf, params_folder);
BO_DECLARE_DYN_PARAM(std::string, Params::meta_conf, traj_folder);
BO_DECLARE_DYN_PARAM(std::string, Params::meta_conf, opt_folder);
BO_DECLARE_DYN_PARAM(int, Params::meta_conf, threads);

BO_DECLARE_DYN_PARAM(bool, PolicyParams::bt_policy, verbose);
BO_DECLARE_DYN_PARAM(double, Params::blackdrops, boundary);
BO_DECLARE_DYN_PARAM(bool, Params::blackdrops, learn_real_system);
BO_DECLARE_DYN_PARAM(bool, Params::blackdrops, stochastic);
BO_DECLARE_DYN_PARAM(bool, Params::blackdrops, real_rollout);
BO_DECLARE_DYN_PARAM(bool, Params::blackdrops, domain_randomization);
BO_DECLARE_DYN_PARAM(bool, Params::blackdrops, nn_policy);

BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, max_fun_evals);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, fun_tolerance);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, lbound);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, ubound);
BO_DECLARE_DYN_PARAM(bool, Params::opt_cmaes, stochastic);
BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, restarts);
BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, elitism);
BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, lambda);
BO_DECLARE_DYN_PARAM(bool, Params::opt_cmaes, handle_uncertainty);

} // namespace iiwa_skills_obstacle

#endif
