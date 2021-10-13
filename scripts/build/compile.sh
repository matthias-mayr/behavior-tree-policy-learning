#!/bin/bash
cwd=$(pwd)

# configure paths
source ./scripts/paths.sh

# go to limbo directory
cd deps/limbo

# Make the number of jobs an optional argument
if [ $# -eq 0 ]; then
    JOBS=6
else
    JOBS=$1
fi

# compile
./waf --libcmaes=${cwd}/install --nlopt=${cwd}/install --dart=${cwd}/install --robot_dart=${cwd}/install --exp blackdrops -j$JOBS

# go back to original directory
cd ../..
echo "Do not forget to source scripts/paths.sh to run an experiment"
