import os
import sys
import platform
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


def get_version():
    if os.getenv("RELEASE_VERSION"):
        version = os.environ["RELEASE_VERSION"]
    else:
        with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as f:
            version = f.read().strip()
    return version.lstrip("v")


def read_readme():
    """Read the contents of the README.md file"""
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()


def clean_cmake_cache(build_temp):
    """Clean up CMakeCache.txt and related CMakeFiles directory."""
    cache_file = os.path.join(build_temp, "CMakeCache.txt")
    if os.path.exists(cache_file):
        os.remove(cache_file)
        cmake_files_dir = os.path.join(build_temp, "CMakeFiles")
        if os.path.exists(cmake_files_dir):
            import shutil

            shutil.rmtree(cmake_files_dir)


# Parse environment variables
CMAKE_ENV_VARS = [
    "CMAKE_TOOLCHAIN_FILE",
    "PLATFORM",
    "ARCHS",
    "DEPLOYMENT_TARGET",
    "OpenMP_C_FLAGS",
    "OpenMP_CXX_FLAGS",
    "OpenMP_C_LIB_NAMES",
    "OpenMP_CXX_LIB_NAMES",
    "OpenMP_libomp_LIBRARY",
]
cmake_env = {var: os.environ.get(var, "") for var in CMAKE_ENV_VARS}

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        print("Building extension '{}'".format(ext.sourcedir))
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Ensure the extension is built directly in the package directory
        extdir = os.path.join(extdir, "pynetcor")
        print("Extension installation directory: {}", extdir)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
        ]

        # Add version info
        version = get_version()
        version_parts = version.split(".")
        cmake_args.extend(
            [
                f"-DPNC_VERSION_MAJOR={version_parts[0]}",
                f"-DPNC_VERSION_MINOR={version_parts[1]}",
                f"-DPNC_VERSION_PATCH={version_parts[2]}",
            ]
        )

        # Add environment-specific CMake arguments
        cmake_args.extend(
            f"-D{var}={value}" for var, value in cmake_env.items() if value
        )

        build_args = []

        if platform.system() == "Windows":
            # Use the default Visual Studio generator and set the architecture
            # generator = "Visual Studio 16 2019"
            arch = PLAT_TO_CMAKE.get(self.plat_name, "Win32")
            # cmake_args += ["-G", generator, "-A", arch]
            cmake_args += ["-A", arch]
            build_args += ["--config", "Debug" if self.debug else "Release"]
        elif platform.system() == "Linux":
            # Use Unix Makefiles generator for Linux
            cmake_args += ["-G", "Unix Makefiles"]
        elif platform.system() == "Darwin":
            # Use Xcode generator for macOS and set the architecture
            cmake_args += ["-G", "Xcode"]
            cmake_args += ["-DCMAKE_OSX_ARCHITECTURES=x86_64"]

        build_args += ["-j", str(os.cpu_count() or 2)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Clean existing CMakeCache.txt and CMakeFiles directory
        clean_cmake_cache(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


# Ensure the bdist_wheel command is available
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False

except ImportError:
    bdist_wheel = None


setup(
    name="pynetcor",
    version=get_version(),
    author="longshibin",
    author_email="longshibin@01lifetech.com",
    description="PyNetCor is a fast Python C++ extension for correlation and network analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/01life/pyNetCor",
    packages=["pynetcor"],
    package_data={
        "pynetcor": ["*.so", "*.pyd", "*.dylib"],
    },
    install_requires=["numpy"],
    ext_modules=[CMakeExtension("pynetcor")],
    cmdclass={"build_ext": CMakeBuild, "bdist_wheel": bdist_wheel},
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    license="MIT",
)
