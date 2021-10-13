#!/usr/bin/env python
# encoding: utf-8
#| Copyright Inria July 2017
#| This project has received funding from the European Research Council (ERC) under
#| the European Union's Horizon 2020 research and innovation programme (grant
#| agreement No 637972) - see http://www.resibots.eu
#|
#| Contributor(s):
#|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
#|   - Rituraj Kaushik (rituraj.kaushik@inria.fr)
#|   - Roberto Rama (bertoski@gmail.com)
#|
#| This software is the implementation of the Black-DROPS algorithm, which is
#| a model-based policy search algorithm with the following main properties:
#|   - uses Gaussian processes (GPs) to model the dynamics of the robot/system
#|   - takes into account the uncertainty of the dynamical model when
#|                                                      searching for a policy
#|   - is data-efficient or sample-efficient; i.e., it requires very small
#|     interaction time with the system to find a working policy (e.g.,
#|     around 16-20 seconds to learn a policy for the cart-pole swing up task)
#|   - when several cores are available, it can be faster than analytical
#|                                                    approaches (e.g., PILCO)
#|   - it imposes no constraints on the type of the reward function (it can
#|                                                  also be learned from data)
#|   - it imposes no constraints on the type of the policy representation
#|     (any parameterized policy can be used --- e.g., dynamic movement
#|                                              primitives or neural networks)
#|
#| Main repository: http://github.com/resibots/blackdrops
#| Preprint: https://arxiv.org/abs/1703.07261
#|
#| This software is governed by the CeCILL-C license under French law and
#| abiding by the rules of distribution of free software.  You can  use,
#| modify and/ or redistribute the software under the terms of the CeCILL-C
#| license as circulated by CEA, CNRS and INRIA at the following URL
#| "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and  rights to copy,
#| modify and redistribute granted by the license, users are provided only
#| with a limited warranty  and the software's author,  the holder of the
#| economic rights,  and the successive licensors  have only  limited
#| liability.
#|
#| In this respect, the user's attention is drawn to the risks associated
#| with loading,  using,  modifying and/or developing or reproducing the
#| software by the user in light of its specific status of free software,
#| that may mean  that it is complicated to manipulate,  and  that  also
#| therefore means  that it is reserved for developers  and  experienced
#| professionals having in-depth computer knowledge. Users are therefore
#| encouraged to load and test the software's suitability as regards their
#| requirements in conditions enabling the security of their systems and/or
#| data to be ensured and,  more generally, to use and operate it in the
#| same conditions as regards security.
#|
#| The fact that you are presently reading this means that you have had
#| knowledge of the CeCILL-C license and that you accept its terms.
#|
#
# partially based on boost.py written by Gernot Vormayr
# written by Ruediger Sonderfeld <ruediger@c-plusplus.de>, 2008
# modified by Bjoern Michaelsen, 2008
# modified by Luca Fossati, 2008
# rewritten for waf 1.5.1, Thomas Nagy, 2008
# rewritten for waf 1.6.2, Sylvain Rouquette, 2011

'''
To add the boost tool to the waf file:
$ ./waf-light --tools=compat15,boost
	or, if you have waf >= 1.6.2
$ ./waf update --files=boost

The wscript will look like:

def options(opt):
	opt.load('compiler_cxx boost')

def configure(conf):
	conf.load('compiler_cxx boost')
	conf.check_boost(lib='system filesystem', mt=True, static=True)

def build(bld):
	bld(source='main.cpp', target='app', use='BOOST')
'''

import sys
import re
from waflib import Utils, Logs
from waflib.Configure import conf

BOOST_LIBS = ['/usr/lib', '/usr/local/lib', '/opt/local/lib', '/sw/lib', '/lib', '/usr/lib/x86_64-linux-gnu/']
BOOST_INCLUDES = ['/usr/include', '/usr/local/include', '/opt/local/include', '/sw/include']
BOOST_VERSION_FILE = 'boost/version.hpp'
BOOST_VERSION_CODE = '''
#include <iostream>
#include <boost/version.hpp>
int main() { std::cout << BOOST_LIB_VERSION << std::endl; }
'''

# toolsets from {boost_dir}/tools/build/v2/tools/common.jam
PLATFORM = Utils.unversioned_sys_platform()
detect_intel = lambda env: (PLATFORM == 'win32') and 'iw' or 'il'
detect_clang = lambda env: (PLATFORM == 'darwin') and 'clang-darwin' or 'clang'
detect_mingw = lambda env: (re.search('MinGW', env.CXX[0])) and 'mgw' or 'gcc'
BOOST_TOOLSETS = {
	'borland':  'bcb',
	'clang':	detect_clang,
	'como':	 'como',
	'cw':	   'cw',
	'darwin':   'xgcc',
	'edg':	  'edg',
	'g++':	  detect_mingw,
	'gcc':	  detect_mingw,
	'icc': detect_intel,
	'icpc':	 detect_intel,
	'intel':	detect_intel,
	'kcc':	  'kcc',
	'kylix':	'bck',
	'mipspro':  'mp',
	'mingw':	'mgw',
	'msvc':	 'vc',
	'qcc':	  'qcc',
	'sun':	  'sw',
	'sunc++':   'sw',
	'tru64cxx': 'tru',
	'vacpp':	'xlc'
}


def options(opt):
	opt.add_option('--boost-includes', type='string',
				   default='', dest='boost_includes',
				   help='''path to the boost directory where the includes are
				   e.g. /boost_1_45_0/include''')
	opt.add_option('--boost-libs', type='string',
				   default='', dest='boost_libs',
				   help='''path to the directory where the boost libs are
				   e.g. /boost_1_45_0/stage/lib''')
	opt.add_option('--boost-static', action='store_true',
				   default=False, dest='boost_static',
				   help='link static libraries')
	opt.add_option('--boost-mt', action='store_true',
				   default=False, dest='boost_mt',
				   help='select multi-threaded libraries')
	opt.add_option('--boost-abi', type='string', default='', dest='boost_abi',
				   help='''select libraries with tags (dgsyp, d for debug),
				   see doc Boost, Getting Started, chapter 6.1''')
	opt.add_option('--boost-toolset', type='string',
				   default='', dest='boost_toolset',
				   help='force a toolset e.g. msvc, vc90, \
						gcc, mingw, mgw45 (default: auto)')
	py_version = '%d%d' % (sys.version_info[0], sys.version_info[1])
	opt.add_option('--boost-python', type='string',
				   default=py_version, dest='boost_python',
				   help='select the lib python with this version \
						(default: %s)' % py_version)


@conf
def __boost_get_version_file(self, dir):
	try:
		return self.root.find_dir(dir).find_node(BOOST_VERSION_FILE)
	except:
		return None


@conf
def boost_get_version(self, dir):
	"""silently retrieve the boost version number"""
	re_but = re.compile('^#define\\s+BOOST_LIB_VERSION\\s+"(.*)"$', re.M)
	try:
		val = re_but.search(self.__boost_get_version_file(dir).read()).group(1)
	except:
		val = self.check_cxx(fragment=BOOST_VERSION_CODE, includes=[dir],
							 execute=True, define_ret=True)
	return val


@conf
def boost_get_includes(self, *k, **kw):
	includes = k and k[0] or kw.get('includes', None)
	if includes and self.__boost_get_version_file(includes):
		return includes
	for dir in BOOST_INCLUDES:
		if self.__boost_get_version_file(dir):
			return dir
	if includes:
		self.fatal('headers not found in %s' % includes)
	else:
		self.fatal('headers not found, use --boost-includes=/path/to/boost')


@conf
def boost_get_toolset(self, cc):
	toolset = cc
	if not cc:
		build_platform = Utils.unversioned_sys_platform()
		if build_platform in BOOST_TOOLSETS:
			cc = build_platform
		else:
			cc = self.env.CXX_NAME
	if cc in BOOST_TOOLSETS:
		toolset = BOOST_TOOLSETS[cc]
	return isinstance(toolset, str) and toolset or toolset(self.env)


@conf
def __boost_get_libs_path(self, *k, **kw):
	''' return the lib path and all the files in it '''
	if 'files' in kw:
		return self.root.find_dir('.'), Utils.to_list(kw['files'])
	libs = k and k[0] or kw.get('libs', None)
	if libs:
		path = self.root.find_dir(libs)
		files = path.ant_glob('*boost_*')
	if not libs or not files:
		for dir in BOOST_LIBS:
			try:
				path = self.root.find_dir(dir)
				files = path.ant_glob('*boost_*')
				if files:
					break
				path = self.root.find_dir(dir + '64')
				files = path.ant_glob('*boost_*')
				if files:
					break
			except:
				path = None
	if not path:
		if libs:
			self.fatal('libs not found in %s' % libs)
		else:
			self.fatal('libs not found, use --boost-includes=/path/to/boost/lib')

	self.to_log('Found the boost path in %r with the libraries:' % path)
	for x in files:
		self.to_log('    %r' % x)
	return path, files

@conf
def boost_get_libs(self, *k, **kw):
	'''
	return the lib path and the required libs
	according to the parameters
	'''
	path, files = self.__boost_get_libs_path(**kw)
	t = []
	if kw.get('mt', False):
		t.append('mt')
	if kw.get('abi', None):
		t.append(kw['abi'])
	tags = t and '(-%s)+' % '-'.join(t) or ''
	toolset = '(-%s[0-9]{0,3})+' % self.boost_get_toolset(kw.get('toolset', ''))
	version = '(-%s)+' % self.env.BOOST_VERSION

	def find_lib(re_lib, files):
		for file in files:
			if re_lib.search(file.name):
				self.to_log('Found boost lib %s' % file)
				return file
		return None

	def format_lib_name(name):
		if name.startswith('lib'):
			name = name[3:]
		return name.split('.')[0]

	libs = []
	for lib in Utils.to_list(k and k[0] or kw.get('lib', None)):
		py = (lib == 'python') and '(-py%s)+' % kw['python'] or ''
		# Trying libraries, from most strict match to least one
		for pattern in ['boost_%s%s%s%s%s' % (lib, toolset, tags, py, version),
						'boost_%s%s%s%s' % (lib, tags, py, version),
						'boost_%s%s%s' % (lib, tags, version),
						# Give up trying to find the right version
						'boost_%s%s%s%s' % (lib, toolset, tags, py),
						'boost_%s%s%s' % (lib, tags, py),
						'boost_%s%s' % (lib, tags)]:
			self.to_log('Trying pattern %s' % pattern)
			file = find_lib(re.compile(pattern), files)
			if file:
				libs.append(format_lib_name(file.name))
				break
		else:
			self.fatal('lib %s not found in %s' % (lib, path))

	return path.abspath(), libs


@conf
def check_boost(self, *k, **kw):
	"""
	initialize boost

	You can pass the same parameters as the command line (without "--boost-"),
	but the command line has the priority.
	"""
	if not self.env['CXX']:
		self.fatal('load a c++ compiler first, conf.load("compiler_cxx")')

	params = {'lib': k and k[0] or kw.get('lib', None)}
	for key, value in self.options.__dict__.items():
		if not key.startswith('boost_'):
			continue
		key = key[len('boost_'):]
		params[key] = value and value or kw.get(key, '')

	var = kw.get('uselib_store', 'BOOST')

	self.start_msg('Checking boost includes')
	self.env['INCLUDES_%s' % var] = self.boost_get_includes(**params)
	self.env.BOOST_VERSION = self.boost_get_version(self.env['INCLUDES_%s' % var])
	self.end_msg(self.env.BOOST_VERSION)
	if Logs.verbose:
		Logs.pprint('CYAN', '	path : %s' % self.env['INCLUDES_%s' % var])

	if not params['lib']:
		return
	self.start_msg('Checking boost libs')
	suffix = params.get('static', 'ST') or ''
	path, libs = self.boost_get_libs(**params)
	self.env['%sLIBPATH_%s' % (suffix, var)] = [path]
	self.env['%sLIB_%s' % (suffix, var)] = libs
	self.end_msg('ok')
	if Logs.verbose:
		Logs.pprint('CYAN', '	path : %s' % path)
		Logs.pprint('CYAN', '	libs : %s' % libs)