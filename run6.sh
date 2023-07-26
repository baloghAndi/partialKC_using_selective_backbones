#!/bin/bash

seed=1234
for d in "./Planning/pddlXpSym/flip/" "./Planning/pddlXpSym/ring/"  ; do
    for alg_type in "random_ratio_selection"  ; do
      for i in $d*.cnf; do
            timeout 3600 python3 ./greedy_selective_backbone.py $d $i $alg_type
     done
  done
done
