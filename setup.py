"""
Setup script.

It sets up the FRAME package and its subpackages, and specifies metadata and third party dependencies.
"""

from distutils.core import setup

setup(name='frame',
      version='0.1',
      description='Floorplanning with RectilineAr ModulEs',
      url='https://github.com/jordicf/FRAME',
      packages=[
        'frame',
        'frame.die',
        'frame.geometry',
        'frame.netlist',
        'frame.geometry',
        'frame.netlist',
        'frame.utils'],
      install_requires=[
          "networkx",
          "ruamel.yaml",
          "numpy",
          "gekko",
          "matplotlib",
          "seaborn"]
      )
