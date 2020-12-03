from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

extensions = [
    Extension(
        "eski.base", ["eski/base.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c++",
        include_dirs=[np.get_include()]
        )
]

compiler_directives = {
    "language_level": 3,
    "embedsignature": True,
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "nonecheck": False
    }

extensions = cythonize(extensions, compiler_directives=compiler_directives)


setup(
    name='eski',
    version="0.0.1",
    author="Jan-Oliver Joswig",
    author_email="jan.joswig@fu-berlin.de",
    packages=find_packages(),
    ext_modules=extensions,
    cmdclass=dict(build_ext=build_ext)
)