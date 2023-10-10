import tools.force.netlists.adjlist.generate_random as generate_random
import os
import sys
import random

def generate(seed: str, directory: str = "") -> list[str]:
    """
    Generates a set of random graphs using a seed 
    and storing them at a certain directory.
    """

    random.seed(seed)

    iterations = 2
    sizes = [50, 100, 150, 200]
    edge_ratios = [1.5, 2, 3, 5]

    if len(directory) == 0:
        directory = f"benchmark_{seed}"

    os.system(f"mkdir {directory}")

    files = []

    for _ in range(iterations):
        for n in sizes:
            for r in edge_ratios:
                e = int(n*r)
                s = random.randint(1, 1000)
                filename = f"{directory}/g-{n}_{e}_{s}"
                files.append(filename)
                generate_random.generate(n, e, s, filename)

    os.system(f"rm {directory}/g-*.list")

    return files


def main():
    assert len(sys.argv) == 2
    seed = sys.argv[1]
    generate(seed)

if __name__ == "__main__":
    main()