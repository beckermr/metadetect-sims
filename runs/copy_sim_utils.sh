#!/bin/bash

cp ../sim_utils/* example/.

for dr in `ls -1d run_*`; do
    echo $dr
    cp ../sim_utils/* $dr/.
done
