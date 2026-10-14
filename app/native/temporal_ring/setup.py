from pathlib import Path

from Cython.Build import cythonize
import numpy
from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("temporal_ring.pyx")

setup(
    name="megaglitch-native-temporal",
    version="0.1.0",
    ext_modules=cythonize(
        [
            Extension(
                "megaglitch._native_temporal",
                [str(SOURCE)],
                include_dirs=[numpy.get_include()],
                language="c++",
                extra_compile_args=["/O2", "/std:c++17"] if __import__("os").name == "nt" else ["-O3", "-std=c++17"],
            )
        ],
        compiler_directives={"language_level": "3"},
    ),
)
