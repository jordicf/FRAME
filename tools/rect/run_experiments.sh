echo "Generating data..."

mkdir ./data/
echo "file cubic slicing complete" > ./data/times.csv

for I in 10 20 30 40 50 60 70 80 90 100
do
	for J in 0 1 2 3 4 5 6 7 8 9
	do
		echo "random$I.M$J.txt"
		./cpp_bin/experiment.exe "./benchmarks/greedy_inputs/random$I.M$J.txt" "random$I.M$J.txt" >> ./data/times.csv
	done
done
