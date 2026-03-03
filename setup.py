from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='jobable',
      description="package for project",
      packages=find_packages(), # NEW: find packages automatically
      install_requires=requirements) # NEW
