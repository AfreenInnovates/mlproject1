from setuptools import find_packages, setup
from typing import List

def get_requirements(file_name:str) -> List[str]:
    """
        Returns list of requirements
    """
    requirements = []

    with open(file_name) as file:
        requirements = file.readlines()
        [requirement.strip() for requirement in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

setup(
    name="ML Project 1",
    version="0.0.1",
    author="Afreen",
    author_email="afreenhossain0000@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)