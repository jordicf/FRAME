import sysconfig

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class NoSuffixBuilder(build_ext):
    def get_ext_filename(self, ext_name):
        # Remove the platform dependent suffix from the extension filename
        return super().get_ext_filename(ext_name).replace(sysconfig.get_config_var("EXT_SUFFIX"), "") + ".pyd"


setup(ext_modules=[Extension("tools.rect.rect_greedy", ["tools/rect/cpp_src/greedy_lib.cpp"])],
      cmdclass={"build_ext": NoSuffixBuilder})
