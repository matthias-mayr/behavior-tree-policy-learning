#!/usr/bin/env python
# encoding: utf-8

import os
from waflib.Configure import conf

@conf
def check_glog(conf):
  includes_check = ['/usr/include/glog']
  libs_check = ['/usr/lib/x86_64-linux-gnu']

  try:
    # Find the header for ROS
    conf.start_msg('Checking for Glog includes')
    conf.find_file('logging.h', includes_check)
    conf.end_msg('ok')

    # Find the lib files
    lib = 'glog'
    conf.start_msg('Checking for Glog lib')
    conf.find_file('lib' + lib + '.so', libs_check)
    conf.end_msg('ok')

    conf.env.INCLUDES_GLOG = includes_check
    conf.env.LIBPATH_GLOG = libs_check
    conf.env.LIB_GLOG = lib
    conf.env.DEFINES_GLOG = ['USE_GLOG']
  except:
    conf.end_msg('Not found', 'RED')
    return