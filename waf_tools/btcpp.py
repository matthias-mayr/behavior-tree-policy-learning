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
    opt.add_option('--btcpp', type='string', help='path to BehaviorTree.CPP', dest='btcpp')


@conf
def check_btcpp(conf):
    if conf.options.btcpp:
        includes_check = [conf.options.btcpp + '/include']
        libs_check = [conf.options.btcpp + '/lib']
    else:
        includes_check = ['/usr/include/glog']
        libs_check = ['/usr/lib/x86_64-linux-gnu']

    try:
        # Find the header for ROS
        conf.start_msg('Checking for BehaviorTree.CPP includes')
        conf.find_file('behaviortree_cpp_v3/behavior_tree.h', includes_check)
        conf.end_msg('ok')

        # Find the lib files
        lib = 'behaviortree_cpp_v3'
        conf.start_msg('Checking for BehaviorTree.CPP lib')
        conf.find_file('lib' + lib + '.so', libs_check)
        conf.end_msg('ok')

        conf.env.INCLUDES_BTCPP = includes_check
        conf.env.LIBPATH_BTCPP = libs_check
        conf.env.LIB_BTCPP = lib
        conf.env.DEFINES_BTCPP = ['USE_BTCPP']
    except:
        conf.end_msg('Not found', 'RED')
        return
