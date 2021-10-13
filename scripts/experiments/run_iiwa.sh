#!/bin/bash
cwd=$(pwd)

#PREFIX="valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-leak.txt"

if [ $# -eq 0 ]; then
    echo "Please specficy an experiment. 'obstacle', 'peg' or 'combined'".
    exit -1
fi

if [ $# -eq 2 ]; then
    RUNS=$2
else
    RUNS=1
fi

if [ $1 == "obstacle" ]; then
	for i in $(eval echo "{1..$RUNS}")
	do
	    echo "Experiment $i:"
        $PREFIX ./deps/limbo/build/exp/blackdrops/src/dart/iiwa_skills_obstacle_simu -m 5000 -r 5 -e 1 -b 2 -d -1 -s
	done
elif [ $1 == "peg" ]; then
	for i in $(eval echo "{1..$RUNS}")
	do
	    echo "Experiment $i:"
        $PREFIX ./deps/limbo/build/exp/blackdrops/src/dart/iiwa_skills_peg_simu -m 200 -l 10 -r 5 -e 0 -b 2 -d -1 -s -v -y
	done
elif [ $1 == "combined" ]; then
	for i in $(eval echo "{1..$RUNS}")
	do
	    echo "Experiment $i:"
        $PREFIX ./deps/limbo/build/exp/blackdrops/src/dart/iiwa_skills_combined_simu -m 8000 -r 5 -e 0 -b 2 -d -1 -s
	done
else
    echo "Please specficy an experiment. 'obstacle', 'peg' or 'combined'".
fi
