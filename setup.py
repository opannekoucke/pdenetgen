import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


if __name__ == "__main__":
    setuptools.setup(
        name="pdenetgen",
        version="1.0.0",
        author="Olivier Pannekoucke ",
        author_email="olivier.pannekoucke@meteo.fr",
        description="Partial Differential Equation Network Generator",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: CeCILL License",
            "Operating System :: OS Independent",
        ],
    )
