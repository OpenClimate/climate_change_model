import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "coscm",
    version = "0.0.1",
    author = "Brian Beckage",
    author_email = "brian.beckage@uvm.edu",
    description = ("Studying the effects of social interactions on climate "
                   "change"),
    license = "BSD",
    keywords = "climate change, social interactions",
    url = "https://github.com/OpenClimate/climate_change_model",
    packages=['coscm'],
    long_description=read('README.md'),
    classifiers=[
        "License :: OSI Approved :: BSD License",
    ],
)
