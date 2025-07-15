# (c) Jordi Cortadella 2022
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import unittest

from frame.netlist.netlist import Netlist
from frame.geometry.geometry import Rectangle


def getkey(i):
    return str(i)


def yaml_diff(yamla, yamlb, path='', key=getkey):
    if isinstance(yamla, list) and isinstance(yamlb, list):
        for d in listdiff(yamla, yamlb, path=path, key=key):
            yield d
    elif isinstance(yamla, dict) and isinstance(yamlb, dict):
        for d in dictdiff(yamla, yamlb, path=path, key=key):
            yield d
    else:
        if yamla != yamlb:
            yield path, 'value_difference', yamla, '-->', yamlb


def listdiff(yamla, yamlb, path, key):
    """
    Compute the differences between two lists, as a generator
    :param list yamla: list of python objects
    :param list yamlb: list of python objects
    :param str path: path of current node (same idea as XPath)
    :param callable key: key lambda used for sorting
    """

    yamla = {key(v, i): v for i, v in enumerate(yamla)}
    yamlb = {key(v, i): v for i, v in enumerate(yamlb)}
    keysa = set(yamla.keys())
    keysb = set(yamlb.keys())

    for a in [k for k in keysb.difference(keysa)]:
        yield path + '/' + str(a), 'added'
    for m in [k for k in keysa.difference(keysb)]:
        yield path + '/' + str(m), 'missing'

    common = [k for k in keysa.intersection(keysb) if k]
    for k in common:
        if (isinstance(yamla[k], list) and isinstance(yamlb, list)) \
                or (isinstance(yamla[k], dict) and isinstance(yamlb, dict)):
            for d in yaml_diff(yamla[k], yamlb[k], path + '/' + k):
                yield d
        else:
            for d in yaml_diff(yamla[k], yamlb[k], path + '/' + k, key):
                yield d


def dictdiff(yamla, yamlb, path, key):
    """
    Compute the difference between two dictionaries, as a generator
    :param dict yamla: first dictionary
    :param dict yamlb: second dictionary to compare with first one
    :param str path: path of current node (same idea as XPath)
    :param callable key: key lambda used for sorting
    """
    keysa = set(yamla.keys())
    keysb = set(yamlb.keys())

    for a in [k for k in keysb.difference(keysa)]:
        yield path + '/' + a, 'added'
    for m in [k for k in keysa.difference(keysb)]:
        yield path + '/' + m, 'missing'

    common = [k for k in keysa.intersection(keysb) if k]
    for k in common:
        if (isinstance(yamla[k], list) and isinstance(yamlb[k], list)) \
                or (isinstance(yamla[k], dict) and isinstance(yamlb[k], dict)):
            for d in yaml_diff(yamla[k], yamlb[k], path + '/' + k, key):
                yield d
        else:
            for d in yaml_diff(yamla[k], yamlb[k], path + '/' + k, key):
                yield d


class TestYaml(unittest.TestCase):
    def read_write_netlist(self, yaml_netlist: str) -> None:
        n1 = Netlist(yaml_netlist)
        n1_yaml = n1.write_yaml()
        assert n1_yaml is not None  # Assertion to suppress Mypy error
        n2 = Netlist(n1_yaml)
        n2_yaml = n2.write_yaml()
        diffs = [x for x in yaml_diff(n1_yaml, n2_yaml)]
        if len(diffs) > 0:
            print(diffs)
        self.assertEqual(len(diffs), 0)

    def test_read_write(self):
        self.read_write_netlist(netlist1)
        self.read_write_netlist(netlist2)

    def test_epsilon(self):
        Rectangle.undefine_epsilon()
        Netlist(netlist3)

    def test_terminal(self):
        n = Netlist(netlist4)
        m = n.get_module("T2")
        self.assertTrue(m.is_terminal and m.is_hard and not m.is_fixed and m.area() == 0)


if __name__ == '__main__':
    unittest.main()

netlist1 = """
Modules: {
  B1: {
    area: 18,
    rectangles: [[3,3,6,3]],
    fixed: false
  },
  B2: {
    rectangles: [[4,2.5,4,5]],
    fixed: true
  },
  B3: {
    area: {"DSP": 20, "BRAM": 50, "LUT": 40}
  }
}

Nets: [
  [B1, B2, 5]
]
"""

netlist2 = """
Modules: {
  B1: {
    area: 6,
    center: [2,3]
  },
  B2: {
    area: 3,
    center: [3,3],
    fixed: false
  },
  B3: {
    area: 5,
    center: [1,1]
  },
  B4: {
    rectangles: [[3,0.5,2,1]],
    fixed: True
  }

}

Nets: [
  [B1, B2, 5],
  [B2, B3, B4, 10],
  [B4, B1, B2]
]
"""

netlist3 = """
Modules:
  M0:
    rectangles: [
      [1, 1, 1, 1],
      [1, 2, 2, 1]
    ]
    hard: true
Nets: []
"""

netlist4 = """
Modules: {
  M0: {
    area: 10,
    aspect_ratio: [0.5, 2]
  },
  M1: {
    area: 24,
    aspect_ratio: 2
  },
  T2: {
    terminal: true
  }
}   

Nets: [
  [M0, M1, 16],
  [M0, T2, M1, 8]
]
"""