from setuptools import find_packages,setup
from typing import List

var='-e .'
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","")  for req in requirements]
    if var in requirements:
        requirements.remove(var)
    return requirements

setup(
name='automl-pipeline-optimzer',
version='0.0.1',
author='Ramkumar',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)