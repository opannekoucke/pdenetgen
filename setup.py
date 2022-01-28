# Update from https://packaging.python.org/en/latest/tutorials/packaging-projects/
#
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pdenetgen",
    version="1.0.2",
    author="Olivier Pannekoucke ",
    author_email="olivier.pannekoucke@meteo.fr",
    description="Partial Differential Equation Network Generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/opannekoucke/pdenetgen",
    project_urls={
        "Bug Tracker": "https://github.com/opannekoucke/pdenetgen/issues",
    },                
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: CeCILL-B Free Software License Agreement (CECILL-B) ",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},        
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6"        
)
