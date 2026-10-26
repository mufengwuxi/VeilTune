#!/bin/bash
cd /workspace/veiltune
for t in sst2 rte mrpc yelp agnews; do zo8/run_zo.sh $t 1e-4 3e-4 1e-3; done
touch zo8/done_all
