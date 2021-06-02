#!/bin/bash

. start_env.sh

for f in scripts/plot_*.py; do python $f; done
