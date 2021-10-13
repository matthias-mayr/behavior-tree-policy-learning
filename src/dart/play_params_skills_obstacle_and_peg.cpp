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

namespace fs = boost::filesystem;

void copyDirectoryRecursively(const fs::path& sourceDir, const fs::path& destinationDir)
{
    if (!fs::exists(sourceDir) || !fs::is_directory(sourceDir))
    {
        throw std::runtime_error("Source directory " + sourceDir.string() + " does not exist or is not a directory");
    }
    if (fs::exists(destinationDir))
    {
        throw std::runtime_error("Destination directory " + destinationDir.string() + " already exists");
    }
    if (!fs::create_directory(destinationDir))
    {
        throw std::runtime_error("Cannot create destination directory " + destinationDir.string());
    }

    for (const auto& dirEnt : fs::recursive_directory_iterator{sourceDir})
    {
        const auto& path = dirEnt.path();
        auto relativePathStr = path.string();
        boost::replace_first(relativePathStr, sourceDir.string(), "");
        fs::copy(path, destinationDir / relativePathStr);
    }
}

int main(int argc, char** argv)
{
    //
    // Parse things and set parameters
    //
    blackdrops::utils::CmdArgs cmd_arguments;
    int ret = cmd_arguments.parse(argc, argv);
    if (ret >= 0)
        return ret;

    Params::blackdrops::set_boundary(cmd_arguments.boundary());
    Params::opt_cmaes::set_lbound(-cmd_arguments.boundary());
    Params::opt_cmaes::set_ubound(cmd_arguments.boundary());

    Params::opt_cmaes::set_max_fun_evals(cmd_arguments.max_fun_evals());
    Params::opt_cmaes::set_fun_tolerance(cmd_arguments.fun_tolerance());
    Params::opt_cmaes::set_restarts(cmd_arguments.restarts());
    Params::opt_cmaes::set_elitism(cmd_arguments.elitism());
    Params::opt_cmaes::set_lambda(cmd_arguments.lambda());
    Params::opt_cmaes::set_handle_uncertainty(cmd_arguments.uncertainty());

#ifdef USE_TBB
    static tbb::task_scheduler_init init(cmd_arguments.threads());
#endif

    Params::meta_conf::set_verbose(cmd_arguments.verbose());
    Params::blackdrops::set_stochastic(cmd_arguments.stochastic());
    Params::blackdrops::set_real_rollout(cmd_arguments.real_rollout());
    Params::blackdrops::set_domain_randomization(true);

#ifdef USE_ROS
    ros::init(argc, argv, "blackdrops", ros::init_options::AnonymousName);
#endif

    std::cout << std::endl;
    std::cout << "Cmaes parameters:" << std::endl;
    std::cout << "  max_fun_evals = " << Params::opt_cmaes::max_fun_evals() << std::endl;
    std::cout << "  fun_tolerance = " << Params::opt_cmaes::fun_tolerance() << std::endl;
    std::cout << "  restarts = " << Params::opt_cmaes::restarts() << std::endl;
    std::cout << "  elitism = " << Params::opt_cmaes::elitism() << std::endl;
    std::cout << "  handle_uncertainty = " << Params::opt_cmaes::handle_uncertainty() << std::endl;

    std::cout << "  Domain randomization = " << Params::blackdrops::domain_randomization() << std::endl;
    std::cout << "  Stochastic rollouts = " << Params::blackdrops::stochastic() << std::endl;
    std::cout << "  Do a real rollout = " << Params::blackdrops::real_rollout() << std::endl;
    std::cout << "  Boundary = " << Params::blackdrops::boundary() << std::endl;
    std::cout << "  TBB threads = " << cmd_arguments.threads() << std::endl;
    std::cout << std::endl;

    int N = (Params::blackdrops::stochastic_evaluation()) ? Params::blackdrops::opt_evals() : 1;
    if (Params::meta_conf::verbose())
        std::cout << "Rolling out policy " << N << " times." << std::endl;
    
    //
    // Set up simulation and other components
    //
    std::time_t t
    = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream timestr;
    timestr << std::put_time( std::localtime( &t ), "%Y%m%d-%H%M%S");
    std::string folder;
    std::string exp_name;
    if (cmd_arguments.data().empty()) {
        folder = "/tmp/blackdrops_replay/iiwa_skills_obstacle_and_peg/";
        if (boost::filesystem::create_directories(folder)) {
            boost::filesystem::permissions(folder, boost::filesystem::add_perms|boost::filesystem::all_all);
        }
    } else {
        folder = cmd_arguments.data() + "/replay/";
    }
    if (cmd_arguments.real_rollout()) {
        exp_name = "_trajectory/";
    } else {
        exp_name = "_params/";
    }
    folder += timestr.str()+exp_name;
    Params::meta_conf::set_folder(folder);
    Params::meta_conf::set_traj_folder(folder+"/trajectories/");
    Params::meta_conf::set_model_folder(folder+"/models/");
    boost::filesystem::create_directories(folder);
    boost::filesystem::create_directories(Params::meta_conf::traj_folder());
    google::SetLogDestination(google::GLOG_INFO, std::string(folder+"INFO_").c_str());
    google::SetLogDestination(google::GLOG_WARNING, std::string(folder+"WARNING_").c_str());
    google::SetLogDestination(google::GLOG_ERROR, std::string(folder+"ERROR_").c_str());
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = 1;
    LOG(INFO) << "Setting up simulation and other components.";
    std::string robot_urdf{"/URDF/iiwa_with_peg/iiwa_left_cad_5mm.urdf"};
    LOG(INFO) << "Robot URDF: " << std::string(RESPATH) + robot_urdf;
    init_simu(std::string(RESPATH) + robot_urdf);

    using policy_opt_t = limbo::opt::Cmaes<Params>;

    using kernel_t = limbo::kernel::SquaredExpARD<Params>;
    using mean_t = limbo::mean::Constant<Params>;

    using GP_t = limbo::model::MultiGP<Params, limbo::model::GP, kernel_t, mean_t, limbo::model::multi_gp::ParallelLFOpt<Params, blackdrops::model::gp::KernelLFOpt<Params>>>;

    using MGP_t = blackdrops::model::GPModel<Params, GP_t>;
    blackdrops::BlackDROPS<Params, MGP_t, SimpleArm, blackdrops::policy::BTPolicyObstacleAndPeg<PolicyParams>, policy_opt_t, RewardFunction> arm_system;
    // Check for available folders
    std::string exp_folder;
    if (cmd_arguments.data().empty()) {
        exp_folder = "/tmp/blackdrops/iiwa_skills_obstacle_and_peg/";
        std::cout << "\nChecking for log files in: " << exp_folder << std::endl;
        std::vector<boost::filesystem::directory_entry> experiments;
        boost::filesystem::directory_iterator exp_dir_it{exp_folder};
        int ignored{0};
        int folder_per_line{3};
        for (int i = 0; exp_dir_it != boost::filesystem::directory_iterator(); exp_dir_it++) {
            std::string dir_name {exp_dir_it->path().filename().c_str()};
            if (dir_name.find("_replay") == std::string::npos) {
                std::cout << i << ") " << dir_name;
                if ((i+1) % folder_per_line) {
                    std::cout << "\t";
                } else {
                    std::cout << std::endl;
                }
                experiments.push_back(*exp_dir_it);
                i++;
            } else {
                ignored++;
            }
        }
        std::cout << "\nIgnored " << ignored << " folders." << std::endl;
        std::cout << "Choose an experiment to load by number. Press [enter] to load the latest." << std::endl;
        std::string exp_choice;
        std::getline( std::cin, exp_choice);
        if (exp_choice.empty()) {
            exp_folder = experiments.back().path().c_str();
        } else {
            exp_folder = experiments[std::stoi(exp_choice)].path().c_str();
        }
        std::cout << "Experiment folder: " << exp_folder << std::endl;
    } else {
        exp_folder = cmd_arguments.data();
    }

    boost::filesystem::directory_iterator model_dir_it{exp_folder + "/models"};
    std::vector<boost::filesystem::directory_entry> model_dirs;
    std::cout << "Choose a GP model to load:" << std::endl;
    std::cout << "0) No model"  << std::endl;
    for (int i = 1; model_dir_it != boost::filesystem::directory_iterator(); model_dir_it++) {
        std::string dir_name {model_dir_it->path().filename().c_str()};
        if (dir_name.find("model_learn") != std::string::npos) {
            std::cout << i << ") " << dir_name << std::endl;
            model_dirs.push_back(*model_dir_it);
            i++;
        }
    }
    std::string model_folder;
    if (model_dirs.empty()) {
        std::cout << "\nNo model found in this directory. Continue with parameters.\n" << std::endl;
    } else {
        std::cout << "Choose a model iteration to load by number. Press [enter] to load the latest." << std::endl;
        std::string model_choice;
        std::getline( std::cin, model_choice);
        if (model_choice.empty()) {
            model_folder = model_dirs.back().path().c_str();
            std::cout << "Model folder: " << model_folder << std::endl;
        } else if (std::stoi(model_choice) > 0) {
            model_folder = model_dirs[std::stoi(model_choice)-1].path().c_str();
            std::cout << "Model folder: " << model_folder << std::endl;
        } else {
            std::cout << "Not loading any model." << std::endl;
        }
    }
    // Copy model folder
    if(!model_folder.empty()) {
        copyDirectoryRecursively(boost::filesystem::path(model_folder), boost::filesystem::path(Params::meta_conf::model_folder()));
    }
    std::string traj_sparse_filename = std::string(Params::meta_conf::traj_folder() + "traj_sparse.dat");
    std::string traj_dense_filename = std::string(Params::meta_conf::traj_folder() + "traj_dense.dat");
    std::string traj_dense_ee_filename = std::string(Params::meta_conf::traj_folder() + "traj_dense_ee.dat");
    std::string residual_filename = std::string(Params::meta_conf::traj_folder() + "residual.dat");
    std::string abort;
    if (!cmd_arguments.real_rollout()) {
        boost::filesystem::directory_iterator param_dir_it{exp_folder+"/parameters"};
        std::vector<boost::filesystem::directory_entry> param_files;
        for (int i = 0; param_dir_it != boost::filesystem::directory_iterator(); param_dir_it++) {
            std::string param_name{param_dir_it->path().filename().c_str()};
            if (param_name.find(".bin") != std::string::npos) {
                std::cout << i << ") " << param_name << std::endl;
                param_files.push_back(*param_dir_it);
                i++;
            }
        }
        if (param_files.empty()) {
            std::cout << "No parameter files found. Exiting.";
            return -1;
        }
        std::cout << "Choose a parameter file to load by number. Press [enter] to load the latest." << std::endl;
        std::string param_filename;
        std::string param_choice;
        std::getline( std::cin, param_choice);
        if (param_choice.empty()) {
            param_filename = param_files.back().path().c_str();
        } else {
            param_filename = param_files[std::stoi(param_choice)].path().c_str();
        }
        std::cout << "Parameter file: " << param_filename << std::endl;

        while(abort.empty()) {
            arm_system.play_iteration(model_folder, param_filename, traj_sparse_filename, traj_dense_filename, traj_dense_ee_filename, residual_filename);
            std::cout << "\nPress [enter] to run again. Any other response exits." << std::endl;
            std::getline( std::cin, abort);
        }
    } else {
        boost::filesystem::directory_iterator traj_dir_it{exp_folder+"/trajectories"};
        std::vector<boost::filesystem::directory_entry> traj_files;
        for (int i = 0; traj_dir_it != boost::filesystem::directory_iterator(); traj_dir_it++) {
            std::string traj_name{traj_dir_it->path().filename().c_str()};
            if (traj_name.find("_real") != std::string::npos) {
                std::cout << i << ") " << traj_name << std::endl;
                traj_files.push_back(*traj_dir_it);
                i++;
            }
        }
        std::cout << "Choose a trajectory file to load by number. Press [enter] to load the latest." << std::endl;
        std::string traj_filename;
        std::string param_choice;
        std::getline( std::cin, param_choice);
        if (param_choice.empty()) {
            traj_filename = traj_files.back().path().c_str();
        } else {
            traj_filename = traj_files[std::stoi(param_choice)].path().c_str();
        }
        std::cout << "Trajectory file: " << traj_filename << std::endl;
        // Copy trajectory file
        boost::filesystem::copy_file(boost::filesystem::path(traj_filename), boost::filesystem::path(Params::meta_conf::traj_folder() + "traj_real.dat"));

        while(abort.empty()) {
            arm_system.replay_trajectory_with_model(model_folder, traj_filename, traj_sparse_filename, traj_dense_filename, traj_dense_ee_filename, residual_filename);
            std::cout << "\nPress [enter] to run again. Any other response exits." << std::endl;
            std::getline( std::cin, abort);
        }
    }
    return 0;
}