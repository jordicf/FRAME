# (c) Víctor Franco Sanchez 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).


import random
from argparse import ArgumentParser
from frame.utils.utils import write_yaml
import typing


def transpose(matrix: list[list[typing.Any]]) -> list[list[typing.Any]]:
    res: list[list[typing.Any]] = []
    for i in range(0, len(matrix[0])):
        res.append([])
        for j in range(0, len(matrix)):
            res[i].append(matrix[j][i])
    return res


def tabulate(matrix: list[list[typing.Any]]) -> None:
    mlen = 0
    for vec in matrix:
        for elem in vec:
            if len(str(elem)) > mlen:
                mlen = len(str(elem))

    for vec in matrix:
        line, word = "", ""
        for elem in vec:
            tmp = str(elem)
            while len(tmp) < mlen:
                tmp = " " + tmp
            word += tmp
            line += word
            word = " "
        print(line)


def isvaliddirection(layout: list[list[int]], block: int, position: tuple[int, int]) -> bool:
    (i, j) = position
    if i < 0 or i >= len(layout) or j < 0 or j >= len(layout[0]):
        return False
    if layout[i][j] != -1:
        return False
    if i > 0 and layout[i-1][j] == block:
        return True
    if j > 0 and layout[i][j-1] == block:
        return True
    if i < len(layout) - 1 and layout[i+1][j] == block:
        return True
    if j < len(layout[i]) - 1 and layout[i][j+1] == block:
        return True
    return False


def randompos(layout: list[list[int]]) -> tuple[int, int]:
    allpos = []
    for i in range(0, len(layout)):
        for j in range(0, len(layout[i])):
            if layout[i][j] == -1:
                allpos.append((i, j))
    return random.choice(allpos)


def randomadjacentpos(layout: list[list[int]], box: int) -> tuple[int, int] | None:
    allpos = []
    for i in range(0, len(layout)):
        for j in range(0, len(layout[i])):
            if isvaliddirection(layout, box, (i, j)):
                allpos.append((i, j))
    if len(allpos) == 0:
        return None
    return random.choice(allpos)


def fuzzify(options: dict[str, typing.Any], layout: list[list[int]], nblocks: int) -> list[list[list[float]]]:
    """
    Receives a deterministic layout and makes it fuzzy by
    adding noise.
    """
    noise: float = options['noise']

    # Type 0 noise is no noise. The higher this number is,
    # the more similar the resulting layout will be from the
    # unaltered one
    type0: float = (1 - noise) * (1 - noise)
    
    # Type 1 noise is proximity noise. The higher this number is,
    # the more the modules will mix up with nearby modules
    type1: float = 2 * noise * (1 - noise)

    # Type 2 noise is white-ish noise. The higher this number is,
    # the more the resulting layout will just look like a
    # jarbled mess
    type2: float = noise * noise

    # Step one, construct the mixed layout
    flayout: list[list[list[float]]] = []
    resout: list[list[list[float]]] = []
    for i in range(0, options['height']):
        flayout.append([])
        resout.append([])
        for j in range(0, options['width']):
            flayout[i].append([0.0] * nblocks)
            flayout[i][j][layout[i][j]] = 1.0
            resout[i].append([0.0] * nblocks)

    # Step two, start mixing the different noises:
    for i in range(0, options['height']):
        for j in range(0, options['width']):
            # Type 2 noise
            rnoise = [0.0] * nblocks 
            rnsum = 0.0
            for k in range(0, nblocks):
                rnoise[k] = random.random() ** 20
                rnsum += rnoise[k]
            for k in range(0, nblocks):
                rnoise[k] = type2 * rnoise[k] / rnsum

            # Type 1 noise
            pnoise = [0.0] * nblocks
            pnsum = 0.0
            for i2 in range(i - 1, i + 2):
                if i2 < 0 or i2 >= len(flayout):
                    continue
                for j2 in range(j - 1, j + 2):
                    if j2 < 0 or j2 >= len(flayout[i2]):
                        continue
                    pnsum += 1.0
                    for k in range(0, nblocks):
                        pnoise[k] += flayout[i2][j2][k]
            for k in range(0, nblocks):
                pnoise[k] = type1 * pnoise[k] / pnsum

            # Type 0 noise
            nnoise = flayout[i][j].copy()
            for k in range(0, nblocks):
                nnoise[k] *= type0

            resum = 0.0
            for k in range(0, nblocks):
                resout[i][j][k] = round((rnoise[k] + pnoise[k] + nnoise[k]) * 100.0) / 100.0
                resum += resout[i][j][k]
            if resum > 1:
                for k in range(0, nblocks):
                    resout[i][j][k] = float(int(resout[i][j][k] / resum * 100.0)) / 100.0
    return resout


def addblock(layout: list[list[int]], block: int, bounds: list[int]) -> int:
    [minx, miny, maxx, maxy] = bounds
    maxx = random.randrange(minx+1, maxx+2)
    maxy = random.randrange(miny+1, maxy+2)
    for x in range(minx, maxx):
        for y in range(miny, maxy):
            layout[x][y] = block
    return (maxx - minx) * (maxy - miny)


def generatedie(options: dict[str, typing.Any]) -> list[list[list[float]]]:
    height: int = options['height']
    width: int = options['width']
    maxw: int = options['maxw']
    maxn: int = options['maxn']
    random.seed(options['seed'])
    layout = []
    for i in range(0, height):
        layout.append([-1] * width)
    counter = height * width
    block = 0
    while counter > 0:
        (minx, miny) = randompos(layout)
        maxx, maxy = minx, miny
        for i in range(minx, minx + maxw):
            if i >= width:
                break
            if layout[i][miny] == -1:
                maxx = i
            else:
                break
        for j in range(miny, miny + maxw):
            if j >= height:
                break
            ok = True
            for i in range(minx, maxx+1):
                if layout[i][j] != -1:
                    ok = False
                    break
            if not ok:
                break
            maxy = j
        counter -= addblock(layout, block, [minx, miny, maxx, maxy])
        for m in range(1, maxn):
            f = maxn + 1 - m
            p = float(f - 1) / float(f)
            if random.random() > p:
                break
            rap = randomadjacentpos(layout, block)
            if rap is None:
                break
            (i, j) = rap
            if isvaliddirection(layout, block, (i + 1, j)):
                # We expand to the right
                minx, miny = i, j
                maxx, maxy = minx, miny
                for i2 in range(minx, minx + maxw):
                    if not isvaliddirection(layout, block, (i2, miny)):
                        break
                    if layout[i2][miny] == -1:
                        maxx = i2
                    else:
                        break
                for j2 in range(miny, miny + maxw):
                    if j2 >= height:
                        break
                    ok = True
                    for i2 in range(minx, maxx+1):
                        if layout[i2][j2] != -1:
                            ok = False
                            break
                    if not ok:
                        break
                    maxy = j2
                counter -= addblock(layout, block, [minx, miny, maxx, maxy])
            elif isvaliddirection(layout, block, (i, j + 1)):
                # We expand to the bottom
                minx, miny = i, j
                maxx, maxy = minx, miny
                for j2 in range(miny, miny + maxw):
                    if not isvaliddirection(layout, block, (minx, j2)):
                        break
                    if layout[minx][j2] == -1:
                        maxy = j2
                    else:
                        break
                for i2 in range(minx, minx + maxw):
                    if i2 >= width:
                        break
                    ok = True
                    for j2 in range(miny, maxy+1):
                        if layout[i2][j2] != -1:
                            ok = False
                            break
                    if not ok:
                        break
                    maxx = i2
                counter -= addblock(layout, block, [minx, miny, maxx, maxy])
            else:
                # We don't expand
                counter -= addblock(layout, block, [i, j, i, j])
        block += 1
    tabulate(transpose(layout))
    return fuzzify(options, layout, block)


def check_integrity(options: dict[str, typing.Any]) -> None:
    if options['width'] < 1:
        raise Exception("Width option must be minimum 1")
    if options['height'] < 1:
        raise Exception("Height option must be minimum 1")
    if options['maxw'] < 1:
        raise Exception("Maxw option must be minimum 1")
    if options['maxn'] < 1:
        raise Exception("Maxn option must be minimum 1")
    if options['noise'] < 0 or options['noise'] > 1:
        raise Exception("Noise option must be between 0 and 1")


def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, typing.Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = ArgumentParser(prog=prog, description="An example generation tool", usage='%(prog)s [options]')
    parser.add_argument("--width",  type=int,   default=10,   help="width of the die                    (minimum 1)")
    parser.add_argument("--height", type=int,   default=10,   help="height of the die                   (minimum 1)")
    parser.add_argument("--maxw",   type=int,   default=7,    help="maximal width of a block            (minimum 1)")
    parser.add_argument("--maxn",   type=int,   default=3,    help="maximal number of blocks per module (minimum 1)")
    parser.add_argument("--noise",  type=float, default=0.3,  help="intensity of the noise              "
                                                                   "(between 0 and 1, > 0.3 not recommended)")
    parser.add_argument("--seed",   type=int,   default=None, help="seed for the random number generator")
    parser.add_argument("--outfile",            default=None, help="output file")
    return vars(parser.parse_args(args))


def initout(options: dict[str, typing.Any]):
    if options['outfile'] is None:
        return print, lambda: print()
    else:
        f = open(options['outfile'], 'w')

        def swrite(*args, sep: str = ' ', end: str = '\n'):
            if len(args) == 0:
                return f.write(end)
            strr = str(args[0])
            for i in range(1, len(args)):
                strr += sep + str(args[i])
            return f.write(strr + end)
        return swrite, f.close


def main(prog: str | None = None, args: list[str] | None = None) -> int:
    """Main function."""
    options = parse_options(prog, args)
    check_integrity(options)
    die = generatedie(options)
    print()
    pout, pend = initout(options)
    rlist = []
    for i in range(0, options['height']):
        for j in range(0, options['width']):
            item1 = [i + 0.5, j + 0.5, 1.0, 1.0]
            item2 = {}
            for k in range(0, len(die[i][j])):
                if die[i][j][k] > 0:
                    item2["M" + str(k)] = die[i][j][k]
            rlist.append([item1, item2])
    pout(write_yaml(rlist))
    pend()
    return 1


if __name__ == "__main__":
    main()
