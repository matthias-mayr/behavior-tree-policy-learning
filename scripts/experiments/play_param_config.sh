#!/bin/bash
cwd=$(pwd)

if [ $# -eq 0 ]; then
    echo "Please specficy an experiment. 'obstacle', 'peg' or 'combined'".
    exit -1
fi

if [ $1 == "obstacle" ]; then
    ./deps/limbo/build/exp/blackdrops/src/dart/play_params_skills_obstacle_graphic $2 $3 $4 $5 -v
elif [ $1 == "peg" ]; then
    ./deps/limbo/build/exp/blackdrops/src/dart/play_params_skills_peg_graphic $2 $3 $4 $5 -v
elif [ $1 == "combined" ]; then
    ./deps/limbo/build/exp/blackdrops/src/dart/play_params_skills_combined_graphic $2 $3 $4 $5 -v
else
    echo "Please specficy an experiment. 'obstacle' or 'peg'".
fi
