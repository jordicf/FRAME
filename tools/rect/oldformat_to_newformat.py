
import sys
import yaml

file = sys.argv[1]
    
ifile = None
with open(file, 'r') as file:
    ifile = yaml.load(file, Loader=yaml.FullLoader)

width = ifile['Width']
height = ifile['Height']

print('Width:', width)
print('Height:', height)
print('Rectangles:')

blocks = [ ( [0,0,0,0], {} ) ] * len(ifile['Rectangles'])

it = 0
for b in ifile['Rectangles']:
    for bname in b:
        [x1, y1, x2, y2, l] = b[bname]
        item1 = [ float(x1 + x2) / 2.0, float(y1 + y2) / 2.0, x2 - x1, y2 - y1 ]
        print('  - ', bname, ':', sep="")
        print('    - dim:', item1)
        print('    - mod:')
        for i in range(0, len(l)):
            if l[i] > 0:
                print('      - M' + str(i) + ":", l[i])
        it = it + 1
        


