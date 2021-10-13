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
#include <blackdrops/experiments/iiwa_skills_obstacle_and_peg.hpp>
#ifdef USE_ROS
#include <ros/ros.h>
#endif

using namespace iiwa_skills_obstacle_and_peg;

template <typename Params>
int get_iteration() {
    int iteration{0};
    // Get parameter files
    std::vector<std::string> param_files;
    boost::filesystem::recursive_directory_iterator param_it(Params::meta_conf::params_folder());
    boost::filesystem::recursive_directory_iterator endit;
    while (param_it != endit) {
        if(boost::filesystem::is_regular_file(*param_it) && (param_it->path().string().find("_opt_final.bin") != std::string::npos)) {
            iteration++;
        }
        ++param_it;
    }
    return iteration;
}

template <typename Params, typename T>
int read_existing_data(T& system) {
    LOG(INFO) << "Continuing from existing experiment folder.";
    boost::filesystem::recursive_directory_iterator endit;
    // Get trajectory files
    std::vector<std::string> traj_files;
    boost::filesystem::recursive_directory_iterator traj_it(Params::meta_conf::traj_folder());
    while (traj_it != endit) {
        if(boost::filesystem::is_regular_file(*traj_it) && (traj_it->path().string().find("traj_real_") != std::string::npos)) {
            traj_files.push_back(traj_it->path().string());
        }
        ++traj_it;
    }
    // Get parameter files
    std::vector<std::string> param_files;
    boost::filesystem::recursive_directory_iterator param_it(Params::meta_conf::params_folder());
    while (param_it != endit) {
        if(boost::filesystem::is_regular_file(*param_it) && (param_it->path().string().find("_opt_final.bin") != std::string::npos)) {
            param_files.push_back(param_it->path().string());
        }
        ++param_it;
    }
    std::sort(traj_files.begin(), traj_files.end());
    std::sort(param_files.begin(), param_files.end());

    Eigen::VectorXd params;
    for (size_t i = 0; i < param_files.size()-1; i++) {
        LOG(INFO) << "Loading trajectory from: " << traj_files[i];
        LOG(INFO) << "Loading parameters from: " << param_files[i];
        Eigen::read_binary(param_files[i].c_str(), params);
        LOG(INFO) << "Parameter: " << params.format(OneLine);
        system.populate_observations(traj_files[i], params);
    }
    // Set policy parameters
    Eigen::read_binary(param_files.back().c_str(), params);
    system.set_policy_params(params);

    LOG(INFO) << "Restarting learning from iteration " << param_files.size();
    return param_files.size();
}

int main(int argc, char** argv)
{
    blackdrops::utils::CmdArgs cmd_arguments;
    int ret = cmd_arguments.parse(argc, argv);
    if (ret >= 0)
        return ret;

    std::string folder("/tmp/blackdrops/iiwa_skills_combined/");
    if (boost::filesystem::create_directories(folder)) {
        boost::filesystem::permissions(folder, boost::filesystem::add_perms|boost::filesystem::all_all);
    }
    std::string exp_name;
    if (cmd_arguments.neural_net()) {
        exp_name = "neural_net";
        Params::blackdrops::set_nn_policy(true);
    } else {
        exp_name = std::to_string(Params::blackdrops::num_params());
        exp_name += "_params";
        Params::blackdrops::set_nn_policy(false);
    }
    if (cmd_arguments.data().empty()) {
        std::time_t t
        = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::stringstream timestr;
        timestr << std::put_time( std::localtime( &t ), "%Y%m%d-%H%M%S");
        boost::filesystem::create_directories(folder);    
        boost::filesystem::permissions(folder, boost::filesystem::add_perms|boost::filesystem::all_all);
        folder += timestr.str()+"_"+exp_name;
    } else {
        if (!boost::filesystem::is_directory(cmd_arguments.data())) {
            std::cerr << "Fatal: Specified data location " << cmd_arguments.data() << " is not a directory." << std::endl;
            return -1;
        }
        folder = cmd_arguments.data();
    }
    if (folder.back() != std::string("/").back()) {
        folder += "/";
    }
    Params::meta_conf::set_folder(folder);
    Params::meta_conf::set_model_folder(folder+"models/");
    Params::meta_conf::set_params_folder(folder+"parameters/");
    Params::meta_conf::set_traj_folder(folder+"trajectories/");
    Params::meta_conf::set_opt_folder(folder+"optimizations/");
    boost::filesystem::create_directories(Params::meta_conf::model_folder());
    boost::filesystem::create_directories(Params::meta_conf::params_folder());
    boost::filesystem::create_directories(Params::meta_conf::traj_folder());
    boost::filesystem::create_directories(Params::meta_conf::opt_folder());
    google::SetLogDestination(google::GLOG_INFO, std::string(folder+"INFO_").c_str());
    google::SetLogDestination(google::GLOG_WARNING, std::string(folder+"WARNING_").c_str());
    google::SetLogDestination(google::GLOG_ERROR, std::string(folder+"ERROR_").c_str());
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = 1;

    Params::meta_conf::set_threads(cmd_arguments.threads());

    PolicyParams::bt_policy::set_verbose(cmd_arguments.verbose());

    Params::blackdrops::set_boundary(cmd_arguments.boundary());
    Params::opt_cmaes::set_lbound(-cmd_arguments.boundary());
    Params::opt_cmaes::set_ubound(cmd_arguments.boundary());
    Params::opt_cmaes::set_stochastic(cmd_arguments.stochastic());
    Params::opt_cmaes::set_max_fun_evals(cmd_arguments.max_fun_evals());
    Params::opt_cmaes::set_fun_tolerance(cmd_arguments.fun_tolerance());
    Params::opt_cmaes::set_restarts(cmd_arguments.restarts());
    Params::opt_cmaes::set_elitism(cmd_arguments.elitism());
    Params::opt_cmaes::set_lambda(cmd_arguments.lambda());
    Params::opt_cmaes::set_handle_uncertainty(cmd_arguments.uncertainty());

#ifdef USE_TBB
    static tbb::task_scheduler_init init(Params::meta_conf::threads());
#endif

    Params::meta_conf::set_verbose(cmd_arguments.verbose());
    Params::blackdrops::set_stochastic(cmd_arguments.stochastic());
    Params::blackdrops::set_real_rollout(cmd_arguments.real_rollout());
    Params::blackdrops::set_domain_randomization(false);
    Params::blackdrops::set_learn_real_system(true);

    LOG(INFO) << "CMA-ES parameters:";
    LOG(INFO) << "  max_fun_evals = " << Params::opt_cmaes::max_fun_evals();
    LOG(INFO) << "  fun_tolerance = " << Params::opt_cmaes::fun_tolerance();
    LOG(INFO) << "  restarts = " << Params::opt_cmaes::restarts();
    LOG(INFO) << "  elitism = " << Params::opt_cmaes::elitism();
    LOG(INFO) << "  lambda = " << Params::opt_cmaes::lambda();
    LOG(INFO) << "  handle_uncertainty = " << Params::opt_cmaes::handle_uncertainty();

    LOG(INFO) << "Blackdrops parameters:";
    LOG(INFO) << "  Stochastic rollouts = " << Params::blackdrops::stochastic();
    LOG(INFO) << "  Boundary = " << Params::blackdrops::boundary();
    LOG(INFO) << "  TBB threads = " << cmd_arguments.threads();
    LOG(INFO) << "  Iterations = " << cmd_arguments.iterations();
    LOG(INFO) << "  Do a real rollout = " << Params::blackdrops::real_rollout();
    LOG(INFO) << "  Domain Randomization = " << Params::blackdrops::domain_randomization();
    LOG(INFO) << "  Maximum collision depth = " << Params::blackdrops::max_collision_depth();

    int N = (Params::blackdrops::stochastic_evaluation()) ? Params::blackdrops::opt_evals() : 1;
    LOG(INFO) << "  Rolling out policy " << N << " times.";
    
#ifdef USE_ROS
    ros::init(argc, argv, "blackdrops", ros::init_options::AnonymousName);
#endif

    //
    // Set up simulation and other components
    //
    LOG(INFO) << "Setting up simulation and other components.";
    std::string robot_urdf{"/URDF/iiwa_with_peg/iiwa_left_cad_5mm.urdf"};
    LOG(INFO) << "Robot URDF: " << std::string(RESPATH) + robot_urdf;
    init_simu(std::string(RESPATH) + robot_urdf);

    using policy_t = blackdrops::policy::BTPolicyObstacleAndPeg<PolicyParams>;

    using kernel_t = limbo::kernel::SquaredExpARD<Params>;
    using mean_t = limbo::mean::Constant<Params>;

    using GP_t = limbo::model::MultiGP<Params, limbo::model::GP, kernel_t, mean_t, limbo::model::multi_gp::ParallelLFOpt<Params, blackdrops::model::gp::KernelLFOpt<Params>>>;

    using MGP_t = blackdrops::model::GPModel<Params, GP_t>;
    
    //
    // Start learning
    //
    // Arg 1: Iteration to start from
    // Arg 2: Learning iterations
    // Arg 3: Random policies
    int start_it = 0;
    int iterations = cmd_arguments.iterations();
    LOG(INFO) << "Leaving 'main', entering 'learn' function.";
    using policy_opt_t = limbo::opt::Cmaes<Params>;
    blackdrops::BlackDROPS<Params, MGP_t, SimpleArm, policy_t, policy_opt_t, RewardFunction> arm_system;
    if (!cmd_arguments.data().empty()) {
        start_it = read_existing_data<Params,blackdrops::BlackDROPS<Params, MGP_t, SimpleArm, policy_t, policy_opt_t, RewardFunction>> (arm_system);
    }
    arm_system.learn(start_it, iterations, true);

    // Place DONE file as an easy indicator for a finished experiment
    boost::filesystem::ofstream(Params::meta_conf::folder() + "DONE");
    return 0;
}
