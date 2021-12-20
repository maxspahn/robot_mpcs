import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="robot_mpcs",
    version="0.0.1",
    author="Max Spahn",
    author_email="m.spahn@tudelft.nl",
    description="Mpc code generation for various robots.",
    long_description=long_description,
    url="https://github.com/maxspahn/robot_mpcs",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "casadi", "shutil", "glob"],
    python_requires=">=3.6",
)
