PYTHON=/c/Users/Lenovo/AppData/Local/Programs/Python/Python311/python.exe


for N in 10 12 14 16 18 20 22 24 26 28 30
do
	for (( c=1; c<=5; c++ ))
	do
		declare -i A=$N/$c
		echo $N $c $A
		$PYTHON ./examplegenerator.py --width $N --height $N --maxw $A --outfile ./benchmarks/test/test$N.$c.yaml
	done
done