echo "N,class,selected area,real area" > results2.css

for (( c=1; c<=5; c++ ))
do
	./benchgen.bash
	./benchrun2.bash
done