#!/bin/bash

FOLDER=/tmp/blackdrops/iiwa_skills_peg/example_config
mkdir -p $FOLDER/parameters
mkdir -p $FOLDER/models
cp res/examples/peg_params.bin /tmp/blackdrops/iiwa_skills_peg/example_config/parameters

FOLDER=/tmp/blackdrops/iiwa_skills_obstacle/example_config
mkdir -p $FOLDER/parameters
mkdir -p $FOLDER/models
cp res/examples/obstacle_params.bin /tmp/blackdrops/iiwa_skills_obstacle/example_config/parameters