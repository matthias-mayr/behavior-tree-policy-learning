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
#include <chrono>
#include <exception>

#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>

#include <blackdrops/utils/utils.hpp>
#include <blackdrops/experiments/iiwa_skills_peg.hpp>
#include <blackdrops/experiments/iiwa_skills_obstacle.hpp>
#include <blackdrops/experiments/iiwa_skills_obstacle_and_peg.hpp>
#include <blackdrops/experiments/iiwa_skills_execution.hpp>


template <typename Params>
std::vector<double> get_parameters(std::string experiment, const std::string parameter_file) {
        std::vector<double> params = std::vector<double>(Params::blackdrops::num_params());
        if (experiment == "peg") {
            params[0] = -0.6;
            params[1] = 0.0;
        } else if (experiment == "obstacle") {
            params[0] = 1.1;
            params[1] = 0.1;
            params[2] = 0.6;
            params[3] = 1.5;
            params[4] = -0.1;
            params[5] = 1.3;
        } else if (experiment == "obstacle_and_peg") {
            params[0] = 1.0;
            params[1] = 0.1;
            params[2] = 0.6;
            params[3] = 1.3;
            params[4] = -0.1;
            params[5] = 1.1;
            params[6] = -0.6;
        } else {
            LOG(FATAL) << "No valid experiment specified. Aborting.";
        }
        if (!parameter_file.empty()) {
            std::ifstream param_file(parameter_file);
            if(!param_file.fail()) {
                std::string line;
                std::getline(param_file, line);
                std::stringstream ss;
                ss.str(line);
                for (size_t i = 0; i < Params::blackdrops::num_params(); i++) {
                    double c_s;
                    char comma;
                    ss >> c_s;
                    ss >> comma;
                    params[i] = c_s;
                }
                bool failed = ss.fail();
                std::string extra;
                ss >> extra;
                if (failed || extra.size() > 0) {
                    LOG(ERROR) << "Current line: " << line;
                    if (extra.size() > 0) {
                        LOG(ERROR) << "There were extra characters in the parameter file: " << extra;
                    }
                    if (line[line.size()-1] != ',') {
                        LOG(ERROR) << "The last character needs to be a comma.";
                    }
                    LOG(FATAL) << "Failure when reading from file: " << parameter_file;
                }
            } else {
                throw std::runtime_error("Could not read from specified parameter file.");
            }
        }
        return params; 
}

int main(int argc, char** argv)
{
	if (argc <= 1)
	{
        std::cout << "Usage: <program name> <experiment> [parameter csv file]" << '\n';
		return -1;
	}
    std::string experiment;
    std::string parameter_file;
    try {
        std::stringstream convert{argv[1]};
        if(convert.good()){
            experiment = convert.str();
        } else {
            throw std::runtime_error("Stringstream was not good.");
        }
        if(argc > 2) {
            std::stringstream convert{argv[2]};
            if(convert.good()){
                parameter_file = convert.str();
            } else {
                throw std::runtime_error("Stringstream was not good.");
            }
        }
    } catch (const std::exception& e) {
        std::cout << "Exception when parsing arguments: " << e.what();
        return -2;
    }
    // ROS
    ros::init(argc, argv, "blackdrops_execution", ros::init_options::AnonymousName);

    //
    // Set up experiment
    //
    std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream timestr;
    timestr << std::put_time( std::localtime( &t ), "%Y%m%d-%H%M%S");
    std::string folder("/tmp/blackdrops_execution/");
    if (boost::filesystem::create_directories(folder)) {
        boost::filesystem::permissions(folder, boost::filesystem::add_perms|boost::filesystem::all_all);
    }
    folder += timestr.str() + "_" + experiment + "/";
    boost::filesystem::create_directories(folder);
    google::SetLogDestination(google::GLOG_INFO, std::string(folder+"INFO_").c_str());
    google::SetLogDestination(google::GLOG_WARNING, std::string(folder+"WARNING_").c_str());
    google::SetLogDestination(google::GLOG_ERROR, std::string(folder+"ERROR_").c_str());
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = 1;

    if (experiment == "peg") {
        using namespace iiwa_skills_peg;
        Params::meta_conf::set_verbose(true);
        Params::meta_conf::set_folder(folder);
        Params::meta_conf::set_traj_folder(folder+"/trajectories/");
        boost::filesystem::create_directories(Params::meta_conf::traj_folder());
        iiwa_skills_execution<PolicyControl<Params, global::policy_t>, Params> execution;
        iiwa_skills_peg::init_simu(std::string(RESPATH) + "/URDF/iiwa_left_cad.urdf");
        auto robot = iiwa_skills_peg::global::global_robot->clone();
        std::shared_ptr<PolicyControl<Params, global::policy_t>> controller = execution.configure_controller(robot, get_parameters<Params>(experiment, parameter_file)); 
        execution.set_params(get_parameters<Params>(experiment, parameter_file));
        execution.print_config();

        std::string traj_sparse_filename(Params::meta_conf::traj_folder() + "traj_sparse.dat");
        std::string traj_dense_filename(Params::meta_conf::traj_folder() + "traj_dense.dat");
        std::string traj_dense_ee_filename(Params::meta_conf::traj_folder() + "traj_dense_ee.dat");
        execution.run(controller, robot, true);
        execution.save_data(controller, traj_sparse_filename, traj_dense_filename, traj_dense_ee_filename);
    } else if (experiment == "obstacle") {
        using namespace iiwa_skills_obstacle;
        Params::meta_conf::set_verbose(true);
        Params::meta_conf::set_folder(folder);
        Params::meta_conf::set_traj_folder(folder+"/trajectories/");
        boost::filesystem::create_directories(Params::meta_conf::traj_folder());
        iiwa_skills_execution<PolicyControl<Params, global::policy_t>, Params> execution;
        iiwa_skills_obstacle::init_simu(std::string(RESPATH) + "/URDF/iiwa_left_cad.urdf");
        auto robot = iiwa_skills_obstacle::global::global_robot->clone();
        std::shared_ptr<PolicyControl<Params, global::policy_t>> controller = execution.configure_controller(robot, get_parameters<Params>(experiment, parameter_file));
        execution.set_params(get_parameters<Params>(experiment, parameter_file));
        execution.print_config();

        std::string traj_sparse_filename(Params::meta_conf::traj_folder() + "traj_sparse.dat");
        std::string traj_dense_filename(Params::meta_conf::traj_folder() + "traj_dense.dat");
        std::string traj_dense_ee_filename(Params::meta_conf::traj_folder() + "traj_dense_ee.dat");
        execution.run(controller, robot, true);
        execution.save_data(controller, traj_sparse_filename, traj_dense_filename, traj_dense_ee_filename);
    } else if (experiment == "obstacle_and_peg") {
        using namespace iiwa_skills_obstacle_and_peg;
        Params::meta_conf::set_verbose(true);
        Params::meta_conf::set_folder(folder);
        Params::meta_conf::set_traj_folder(folder+"/trajectories/");
        boost::filesystem::create_directories(Params::meta_conf::traj_folder());
        iiwa_skills_execution<PolicyControl<Params, global::policy_t>, Params> execution;
        iiwa_skills_obstacle_and_peg::init_simu(std::string(RESPATH) + "/URDF/iiwa_left_cad.urdf");
        auto robot = iiwa_skills_obstacle_and_peg::global::global_robot->clone();
        std::shared_ptr<PolicyControl<Params, global::policy_t>> controller = execution.configure_controller(robot, get_parameters<Params>(experiment, parameter_file));
        execution.set_params(get_parameters<Params>(experiment, parameter_file));
        execution.print_config();

        std::string traj_sparse_filename(Params::meta_conf::traj_folder() + "traj_sparse.dat");
        std::string traj_dense_filename(Params::meta_conf::traj_folder() + "traj_dense.dat");
        std::string traj_dense_ee_filename(Params::meta_conf::traj_folder() + "traj_dense_ee.dat");
        execution.run(controller, robot, true);
        execution.save_data(controller, traj_sparse_filename, traj_dense_filename, traj_dense_ee_filename);
    } else {
        LOG(WARNING) << "No known experiment was specified with the first command line argument.";
    }
    return 0;
}
