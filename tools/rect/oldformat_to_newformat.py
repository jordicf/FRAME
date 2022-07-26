import sys
from frame.utils.utils import read_yaml, write_yaml

file = sys.argv[1]
    
ifile = read_yaml(file)

width = ifile['Width']
height = ifile['Height']

blocks = []

it = 0
for b in ifile['Rectangles']:
    for bname in b:
        [x1, y1, x2, y2, l] = b[bname]
        item1 = [float(x1 + x2) / 2.0, float(y1 + y2) / 2.0, float(x2 - x1), float(y2 - y1)]
        item2 = {}
        for i in range(0, len(l)):
            if l[i] > 0:
                item2['M' + str(i)] = l[i]
        blocks.append([item1, item2])
        it = it + 1

print(write_yaml(blocks))
