from pathlib import Path

from setuptools import find_namespace_packages, setup

setup(
    name="hydra-apptainer-launcher",
    version="1.0.0",
    author="J.H Garcia-Guzman",
    author_email="jhelg@ugr.es",
    description="Hydra Submitit Launcher with Apptainer container support for HPC clusters",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=["hydra_plugins.*"]),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 4 - Beta",
    ],
    install_requires=[
        "hydra-core>=1.1.0",
        "submitit>=1.3.3",
    ],
    include_package_data=True,
)
