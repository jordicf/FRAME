PYTHON=/c/Users/Lenovo/AppData/Local/Programs/Python/Python311/python.exe

for N in 10 12 14 16 18 20 22 24 26 28 30
do
	for (( c=1; c<=5; c++ ))
	do
		echo $N $c
		minarea=$($PYTHON ./rect.py ./benchmarks/test/test$N.$c.yaml --module M0 --minarea | grep " area:    ")
		minarea_sa=$(echo "$minarea" | grep "Selected area:" | tail -1 | awk '{print $3}')
		minarea_ra=$(echo "$minarea" | grep "Real area:" | tail -1 | awk '{print $3}')
		
		maxdiff=$($PYTHON ./rect.py ./benchmarks/test/test$N.$c.yaml --module M0 --maxdiff | grep " area:    ")
		maxdiff_sa=$(echo "$maxdiff" | grep "Selected area:" | tail -1 | awk '{print $3}')
		maxdiff_ra=$(echo "$maxdiff" | grep "Real area:" | tail -1 | awk '{print $3}')

		minerr=$($PYTHON ./rect.py ./benchmarks/test/test$N.$c.yaml --module M0 --minerr | grep " area:    ")
		minerr_sa=$(echo "$minerr" | grep "Selected area:" | tail -1 | awk '{print $3}')
		minerr_ra=$(echo "$minerr" | grep "Real area:" | tail -1 | awk '{print $3}')
		
		echo $N,minarea,$minarea_sa,$minarea_ra >> ./results2.css
		echo $N,maxdiff,$maxdiff_sa,$maxdiff_ra >> ./results2.css
		echo $N,minerr,$minerr_sa,$minerr_ra >> ./results2.css
	done
done