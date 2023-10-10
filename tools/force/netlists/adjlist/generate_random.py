import random
import sys
import os
from tools.force.netlists.adjlist.adjlist_translate import translate

def generate(n: int, e: int, seed: int, filename: str | None = None):
	"""
	Generates a random graph with n vertices and 
	an expected number of edges e, using a random
	seed, and writing the output in a file.
	"""

	random.seed(seed)
	assert 1 <= n

	p = e / (n*(n-1)/2)
	assert 0 <= p <= 1

	if filename is None:
		filename = f"random/r_{n}_{e}_{seed}"

	with open(filename+".list", 'w') as f:
		for i in range(n):
			for j in range(i+1, n):
				if random.random() < p:
					f.write(f"{i} {j}\n")

	translate(filename+".list", filename+".yaml", seed=random.randint(1, 100000000000))


def main():
	assert len(sys.argv) == 4
	n = int(sys.argv[1])
	e = int(sys.argv[2])
	s = int(sys.argv[3])
	generate(n, e, s)

if __name__ == '__main__':
	main()
