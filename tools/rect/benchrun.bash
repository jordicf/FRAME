PYTHON=/c/Users/Lenovo/AppData/Local/Programs/Python/Python311/python.exe

for N in 10 12 14 16 18 20 22 24 26 28 30
do
	for (( c=1; c<=5; c++ ))
	do
		echo $N $c
		start_minarea=`date +%s%N`
		$PYTHON ./rect.py ./benchmarks/test/test$N.$c.yaml --module M0 --minarea > ./benchmarks/test_out/test_minarea$N.$c.yaml
		end_minarea=`date +%s%N`
		start_maxdif=`date +%s%N`
		$PYTHON ./rect.py ./benchmarks/test/test$N.$c.yaml --module M0 --maxdiff > ./benchmarks/test_out/test_maxdiff$N.$c.yaml
		end_maxdif=`date +%s%N`
		start_minerr=`date +%s%N`
		$PYTHON ./rect.py ./benchmarks/test/test$N.$c.yaml --module M0 --minerr > ./benchmarks/test_out/test_minerr$N.$c.yaml
		end_minerr=`date +%s%N`
		echo $N,$c,`expr $end_minarea - $start_minarea`,`expr $end_maxdif - $start_maxdif`,`expr $end_minerr - $start_minerr` >> ./results.css
	done
done