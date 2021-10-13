#!/usr/bin/env python
# encoding: utf-8
# | Copyright Matthias Mayr October 2021
# |
# | Code repository: https://github.com/matthias-mayr/behavior-tree-policy-learning
# | Preprint: https://arxiv.org/abs/2109.13050
# |
# | This software is governed by the CeCILL-C license under French law and
# | abiding by the rules of distribution of free software.  You can  use,
# | modify and/ or redistribute the software under the terms of the CeCILL-C
# | license as circulated by CEA, CNRS and INRIA at the following URL
# | "http://www.cecill.info".
# |
# | As a counterpart to the access to the source code and  rights to copy,
# | modify and redistribute granted by the license, users are provided only
# | with a limited warranty  and the software's author,  the holder of the
# | economic rights,  and the successive licensors  have only  limited
# | liability.
# |
# | In this respect, the user's attention is drawn to the risks associated
# | with loading,  using,  modifying and/or developing or reproducing the
# | software by the user in light of its specific status of free software,
# | that may mean  that it is complicated to manipulate,  and  that  also
# | therefore means  that it is reserved for developers  and  experienced
# | professionals having in-depth computer knowledge. Users are therefore
# | encouraged to load and test the software's suitability as regards their
# | requirements in conditions enabling the security of their systems and/or
# | data to be ensured and,  more generally, to use and operate it in the
# | same conditions as regards security.
# |
# | The fact that you are presently reading this means that you have had
# | knowledge of the CeCILL-C license and that you accept its terms.
# |
import os
from waflib.Configure import conf


def options(opt):
  opt.add_option('--ros_path', type='string', help='Path to ROS', dest='ros_path')
  opt.add_option('--add_packages', type='string', help='Additional packages. Comma separated.', dest='add_packages')

@conf
def check_ros(conf):
  # Get locations where to search for ROS's header and lib files
  if conf.options.ros_path:
    includes_check = [conf.options.ros_path + '/include']
    libs_check = [conf.options.ros_path + '/lib']
  else:
    if 'CMAKE_PREFIX_PATH' not in os.environ:
      conf.start_msg('Checking for ROS')
      conf.end_msg('CMAKE_PREFIX_PATH not in environment variables. Was ROS sourced?', 'RED')
      return
    includes_check = list()
    libs_check = list()
    paths = os.environ['CMAKE_PREFIX_PATH'].split(":")
    for path in paths:
      includes_check.append(path+'/include')
      libs_check.append(path+'/lib/')

  try:
    # Find the header for ROS
    conf.start_msg('Checking for ROS base includes')
    conf.find_file('ros/ros.h', includes_check)
    conf.end_msg('ok')

    # Find the lib files
    base_libs = ['roscpp','rosconsole','roscpp_serialization','rostime','xmlrpcpp',
            'rosconsole_log4cxx', 'rosconsole_backend_interface']
    conf.start_msg('Checking for ROS base libs')
    for lib in base_libs:
      conf.find_file('lib'+lib+'.so', libs_check)
    conf.end_msg('ok')

  except:
    conf.end_msg('ROS base not found', 'RED')
    return

  req_libs = list()
  if conf.options.add_packages:
    req_libs = conf.options.add_packages.split(",")
    # remove spaces for better usability
    req_libs = [x.strip(' ') for x in req_libs]

  try:
    # Find the lib files
    conf.start_msg('Checking for ROS additional libs')
    for lib in req_libs:
      conf.find_file('lib'+lib+'.so', libs_check)
    conf.end_msg('ok')

  except:
    conf.end_msg('ROS additional libs not found', 'RED')
    return

  conf.env.INCLUDES_ROS = includes_check
  conf.env.LIBPATH_ROS = libs_check
  libs = base_libs + req_libs
  conf.env.LIB_ROS = libs
  conf.env.DEFINES_ROS = ['USE_ROS']