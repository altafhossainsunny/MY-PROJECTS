from distutils.core import setup
from setuptools import find_packages, setup
from typing import List




HYPEN_E_DOT = "-e ."
def get_requirements(requirements_path:str) ->List[str]:
    requirements = []
    with open(requirements_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n"," ") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements
    



setup(
   name ="ML PROJECT",
   version = "0.0.1",
   author = "MD ALTAF HOSSAIN SUNNY",
   author_email = "www.altafhossainsunny1552@gmail.com",
   packages = find_packages(),
   install_requires = get_requirements(requirements_path="requirements.txt"),
)