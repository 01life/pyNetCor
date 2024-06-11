import os
import pathlib
import platform
import re
import sys
import subprocess
from distutils.version import LooseVersion
from distutils.dir_util import copy_tree
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"cmake version (\d+\.\d+)", out.decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                self.build_extension(ext)

    def build_extension(self, ext):
        cwd = pathlib.Path().absolute()

        build_temp = pathlib.Path(self.build_temp).joinpath(ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        sourcedir = pathlib.Path(ext.sourcedir).absolute()
        python_bindings_dir = sourcedir.joinpath("python_bindings")
        copy_tree(python_bindings_dir, str(extdir.parent))

        cfg = "Debug" if self.debug else "Release"
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir.parent.absolute()}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(
                    cfg.upper(), extdir.parent
                )
            ]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]

        env = os.environ.copy()
        env["CXXFLAGS"] = "{} -DVERSION_INFO={}".format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )

        self.spawn(["cmake", "-B", build_temp] + cmake_args)
        self.spawn(["cmake", "--build", build_temp] + build_args)
        os.chdir(str(cwd))


setup(
    name="pyNetCor",
    version="0.0.1",
    author="Long",
    description="This is a python package for correlation analysis",
    ext_modules=[CMakeExtension("pyNetCor")],
    cmdclass={"build_ext": CMakeBuild},
)
