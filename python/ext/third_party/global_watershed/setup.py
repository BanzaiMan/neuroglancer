# Cython compile instructions

from distutils.core import setup
from Cython.Build import cythonize

# Use python setup.py build --inplace
# to compile

setup(
  ext_modules = cythonize('*.pyx', include_path=["/usr/people/it2/code/neuroglancer/python/ext/third_party/zi_lib","/usr/people/it2/code/neuroglancer/python/ext/third_party/watershed/"]),
)