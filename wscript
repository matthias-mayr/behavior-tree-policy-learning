#!/usr/bin/env python
# encoding: utf-8
# | Copyright Inria July 2017
# | This project has received funding from the European Research Council (ERC) under
# | the European Union's Horizon 2020 research and innovation programme (grant
# | agreement No 637972) - see http://www.resibots.eu
# |
# | Contributor(s):
# |   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
# |   - Rituraj Kaushik (rituraj.kaushik@inria.fr)
# |   - Roberto Rama (bertoski@gmail.com)
# |
# | This software is the implementation of the Black-DROPS algorithm, which is
# | a model-based policy search algorithm with the following main properties:
# |   - uses Gaussian processes (GPs) to model the dynamics of the robot/system
# |   - takes into account the uncertainty of the dynamical model when
# |                                                      searching for a policy
# |   - is data-efficient or sample-efficient; i.e., it requires very small
# |     interaction time with the system to find a working policy (e.g.,
# |     around 16-20 seconds to learn a policy for the cart-pole swing up task)
# |   - when several cores are available, it can be faster than analytical
# |                                                    approaches (e.g., PILCO)
# |   - it imposes no constraints on the type of the reward function (it can
# |                                                  also be learned from data)
# |   - it imposes no constraints on the type of the policy representation
# |     (any parameterized policy can be used --- e.g., dynamic movement
# |                                              primitives or neural networks)
# |
# | Main repository: http://github.com/resibots/blackdrops
# | Preprint: https://arxiv.org/abs/1703.07261
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
#! /usr/bin/env python
from waflib import Logs
import limbo
import os
import sys
sys.path.insert(0, sys.path[0] + '/waf_tools')


def options(opt):
    opt.load('sdl')
    opt.load('dart')
    opt.load('robot_dart')
    opt.load('simple_nn')
    opt.load('ros')
    opt.load('glog')
    opt.load('btcpp')


def configure(conf):
    conf.load('sdl')
    conf.load('dart')
    conf.load('robot_dart')
    conf.load('simple_nn')
    conf.load('ros')
    conf.load('glog')
    conf.load('btcpp')

    conf.check_sdl()
    conf.check_dart()
    conf.check_robot_dart()
    conf.check_simple_nn()
    conf.check_ros()
    conf.check_glog()
    conf.check_btcpp()

    conf.env.LIB_THREADS = ['pthread']

    if conf.env.CXX_NAME in ["clang"]:
        conf.env['CXXFLAGS'] = conf.env['CXXFLAGS'] + ['-faligned-new']
    elif not (conf.env.CXX_NAME in ["icc", "icpc"]):
        gcc_version = int(conf.env['CC_VERSION'][0] + conf.env['CC_VERSION'][1])
        if gcc_version >= 71 and "-march=native" in conf.env['CXXFLAGS']:
            conf.env['CXXFLAGS'] = conf.env['CXXFLAGS'] + ['-faligned-new']
    conf.env['CXXFLAGS'] = conf.env['CXXFLAGS'] + ['-Wno-noexcept-type']
    Logs.pprint('NORMAL', 'CXXFLAGS (Blackdrops): %s' % conf.env['CXXFLAGS'])


def build(bld):
    bld.recurse('src/')
