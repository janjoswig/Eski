import sysconfig
from setuptools import setup, find_packages, Extension
from typing import List, Optional, Tuple

from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np

Options.fast_fail = False

cython_macros: List[Tuple[str, Optional[str]]] = [
    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
]

extra_compile_args = set(sysconfig.get_config_var('CFLAGS').split())
# extra_compile_args.add("-Wfatal-errors")

extensions = [
    Extension(
        "*", ["eski/*.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=list(extra_compile_args),
        )
]

compiler_directives = {
    "language_level": 3,
    "embedsignature": True,
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "nonecheck": False,
    "linetrace": True
    }

extensions = cythonize(extensions, compiler_directives=compiler_directives)


setup(
    name='eski',
    version="0.0.1",
    author="Jan-Oliver Joswig",
    author_email="jan.joswig@fu-berlin.de",
    packages=find_packages(),
    package_data={'eski': ['*.pxd']},
    ext_modules=extensions,
)
