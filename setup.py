# (c) Marçal Comajoan Cara 2022
# For the FRAME Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

"""This file is only used to set up C++ extension modules.
The rest of the setup is done through the pyproject.toml file."""

import sysconfig

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class NoSuffixBuilder(build_ext):
    def get_ext_filename(self, ext_name):
        # Remove the platform dependent suffix from the extension filename
        return super().get_ext_filename(ext_name).replace(sysconfig.get_config_var("EXT_SUFFIX"), "") + ".pyd"


setup(ext_modules=[Extension("tools.rect.rect_greedy", ["tools/rect/cpp_src/greedy_lib.cpp"])],
      cmdclass={"build_ext": NoSuffixBuilder})
