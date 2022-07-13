"""
Setup script.

It sets up both the FRAME Python package (and subpackages) and the FRAME command-line utility,
and specifies metadata and third party dependencies.
"""

from setuptools import setup

NAME = 'frame'
DESCRIPTION = 'Floorplanning with RectilineAr ModulEs'
URL = 'https://github.com/jordicf/FRAME'
VERSION = '0.1'

PACKAGES = [
    'frame',
    'frame.die',
    'frame.geometry',
    'frame.netlist',
    'frame.utils',
    'tools.hello'
]

INSTALL_REQUIRES = [
    "networkx",
    "ruamel.yaml",
    "numpy",
    "gekko",
    "matplotlib",
    "seaborn",
    "distinctipy"
]

ENTRY_POINTS = {
    "console_scripts": ["frame = tools.frame:main"]
}

setup(name=NAME,
      description=DESCRIPTION,
      url=URL,
      version=VERSION,
      packages=PACKAGES,
      install_requires=INSTALL_REQUIRES,
      entry_points=ENTRY_POINTS
      )
