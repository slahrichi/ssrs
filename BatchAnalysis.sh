#!/bin/bash

TASK=building
OUTPUT='building_all_results.csv'

python3 BatchAnalysis.py --task $TASK --output $OUTPUT
