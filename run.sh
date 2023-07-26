#!/bin/bash

seed=1234
  for alg_type in "random_ratio_selection" ; do
     for d in "./DatasetA/"   "./DatasetB/" "./iscas/iscas89/" ; do
      	  for i in $d*.cnf; do
        echo $i
        timeout 3600 python3 ./greedy_selective_backbone.py $d $i $alg_type
     done
  done
done
