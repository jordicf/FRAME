#!/bin/bash

python=/c/Users/Lenovo/AppData/Local/Programs/Python/Python311/python.exe

$python -m pip install ../..

num_iter=60
radii=(0.01 0.02 0.05 0.1 0.2 0.5 1.0)
num_tests=5
netlist=./examples/example2_netlist.yaml
die=./examples/example2_die.yaml

for radius in ${radii[@]}
do
	for (( test=1; test<=$num_tests; test++ ))
	do
		echo $radius, $test
		$python legalfloor.py $netlist $die --num_iter $num_iter --radius $radius --small_steps > ./outputs/test.$radius.$test.txt
	done
done

