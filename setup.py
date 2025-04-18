# NARX/setup.py

import setuptools
import os
import re

def get_version(package_path):
    """Reads the __version__ variable from the package's __init__.py."""
    version_file = os.path.join(package_path, '__init__.py')
    if not os.path.isfile(version_file):
        raise RuntimeError(f"Cannot find version file: {version_file}")

    with open(version_file, 'r', encoding='utf-8') as f:
        version_match = re.search(r"^__version__\s*=\s*['\"]([^'\"]*)['\"]",
                                  f.read(), re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError(f"Unable to find version string in {version_file}.")

def get_long_description(readme_file="README.md"):
    """Reads the README file."""
    try:
        with open(readme_file, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        print(f"Warning: {readme_file} not found. Long description will be empty.")
        return ""


PACKAGE_INSTALL_NAME = "narxpy" 
PACKAGE_CODE_DIR = "src/narxpy"

setup_args = dict(
    name=PACKAGE_INSTALL_NAME,
    version=get_version(PACKAGE_CODE_DIR), 
    author="Marco Pardini",
    author_email="marco.pardini@phd.unipi.it", 
    description="A PyTorch implementation of a NARX neural network.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/ilmarcopardo/NARX",
    license="MIT", 
    package_dir={"":"src"},
    packages=[PACKAGE_INSTALL_NAME],
    install_requires=[
        'torch>=1.8.0', 
    ],
    python_requires='>=3.8', 
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
)

# --- Run Setup ---
if __name__ == "__main__":
    setuptools.setup(**setup_args)