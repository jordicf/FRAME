"""
Setup script.

It sets up both the FRAME Python package (and subpackages) and the FRAME command-line utility,
and specifies metadata and third party dependencies.
"""

from setuptools import setup, Extension
from distutils.command.build_ext import build_ext as build_ext_orig


class BuildExt(build_ext_orig):
    def build_extension(self, ext):
        self._ctypes = isinstance(ext, Extension)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + '.pyd'
        return super().get_ext_filename(ext_name)


NAME = 'frame'
DESCRIPTION = 'Floorplanning with RectilineAr ModulEs'
URL = 'https://github.com/jordicf/FRAME'
VERSION = '0.1'

PACKAGES = [
    'frame',
    'frame.allocation',
    'frame.die',
    'frame.geometry',
    'frame.netlist',
    'frame.utils',
    'tools.draw',
    'tools.hello',
    'tools.netgen',
    'tools.rect',
    'tools.spectral'
]

INSTALL_REQUIRES = [
    "ruamel.yaml",
    "gekko",
    "matplotlib",
    "distinctipy",
    "Pillow",
    "python-sat"
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
      entry_points=ENTRY_POINTS,
      ext_modules=[
          Extension(
              "rect_greedy",
              ["tools/rect/cpp_src/greedy_lib.cpp"]
          )
      ],
      cmdclass={'build_ext': BuildExt}
      )
