#!/bin/bash

OS=$(uname)
echo "Detected OS: $OS"

if [ $OS = "Darwin" ]; then
    echo "ERROR: OSX is not supported"
    exit 1
fi

# check if we have Ubuntu or not
distro_str="$(cat /etc/*-release | grep -s DISTRIB_ID)"
distro=$(echo $distro_str | cut -f2 -d'=')

if [ "$distro" != "Ubuntu" ]; then
    echo "ERROR: We need an Ubuntu system to use this script"
    exit 1
fi

sudo apt-get -qq update
# install Eigen 3, Boost and TBB
sudo apt-get --yes --force-yes install cmake libeigen3-dev libtbb-dev libboost-serialization-dev libboost-filesystem-dev libboost-test-dev libboost-program-options-dev libboost-thread-dev libboost-regex-dev libsdl2-dev
# install google tests for libcmaes
sudo apt-get --yes --force-yes install libgtest-dev autoconf automake libtool libgoogle-glog-dev libgflags-dev

# save current directory
cwd=$(pwd)
# create install dir
mkdir -p install

# do libgtest fix for libcmaes
cd /usr/src/gtest
sudo mkdir -p build && cd build
sudo cmake ..
sudo make
sudo cp *.a /usr/lib
# install libcmaes
cd ${cwd}/deps/libcmaes
mkdir -p build && cd build
cmake -DUSE_TBB=ON -DUSE_OPENMP=OFF -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX=${cwd}/install ..
make -j8
make install
# go back to original directory
cd ../../..

# configure paths
source ./scripts/paths.sh

# installing NLOpt
cd deps
wget http://members.loria.fr/JBMouret/mirrors/nlopt-2.4.2.tar.gz
tar -zxvf nlopt-2.4.2.tar.gz && cd nlopt-2.4.2
./configure -with-cxx --enable-shared --without-python --without-matlab --without-octave --prefix=${cwd}/install
make install
# go back to original directory
cd ../..

# get ubuntu version
version_str="$(cat /etc/*-release | grep -s DISTRIB_RELEASE)"
version=$(echo $version_str | cut -f2 -d'=')
major_version=$(echo $version | cut -f1 -d'.')
minor_version=$(echo $version | cut -f2 -d'.')

# if less than 14.04, exit
if [ "$(($major_version))" -lt "14" ]; then
    echo "ERROR: We need Ubuntu >= 14.04 for this script to work"
    exit 1
fi

# install DART dependencies
# if we have less than 16.04, we need some extra stuff
if [ "$(($major_version))" -lt "16" ]; then
    sudo apt-add-repository ppa:libccd-debs/ppa -y
    sudo apt-add-repository ppa:fcl-debs/ppa -y
fi
sudo apt-add-repository ppa:dartsim/ppa -y
sudo apt-get -qq update
sudo apt-get --yes --force-yes install build-essential pkg-config libassimp-dev libccd-dev libfcl-dev
sudo apt-get --yes --force-yes install libnlopt-dev libbullet-dev libtinyxml-dev libtinyxml2-dev liburdfdom-dev liburdfdom-headers-dev libxi-dev libxmu-dev freeglut3-dev libopenscenegraph-dev
# install DART
cd deps/dart
mkdir -p build && cd build
cmake -DDART_ENABLE_SIMD=ON -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX=${cwd}/install ..
make -j8
make install
# go back to original directory
cd ../../..

# just as fail-safe
sudo ldconfig

# configure paths to find DART related libraries properly
source ./scripts/paths.sh

# install robot_dart
cd deps/robot_dart
./waf configure --dart=${cwd}/install --prefix=${cwd}/install
./waf
./waf install
# go back to original directory
cd ../..

# install simple_nn
cd deps/simple_nn
./waf configure --prefix=${cwd}/install
./waf
./waf install
# go back to original directory
cd ../..

# install BehaviorTree.CPP
sudo apt-get install -y libzmq3-dev libboost-dev
cd deps/BehaviorTree.CPP
mkdir build; cd build
cmake -DCMAKE_INSTALL_PREFIX=${cwd}/install ..
make
make install