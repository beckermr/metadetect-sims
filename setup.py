from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "mdetsims._lanczos",
        sources=["mdetsims/_lanczos.pyx"],
        libraries=["m"]  # Unix-like specific
    )
]

setup(
    name="mdetsims",
    version="0.1",
    packages=find_packages(),
    ext_modules=cythonize(ext_modules)
)
