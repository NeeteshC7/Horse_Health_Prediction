from setuptools import find_packages,setup
from typing import List


def get_requirements_from_file(file_path: str) -> List[str]:
    """
    This function reads a file and returns a list of requirements.
    """
    requirements = []
    hyphen_e_dot = '-e .'

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)

    return requirements

setup(
name='Horse_Health_Prediction',
version='0.0.1',
author='Neetesh Chauhan',
author_email='neeteshchauhan3@gmail.com',
packages=find_packages(),
install_requires=get_requirements_from_file('requirements.txt')

)