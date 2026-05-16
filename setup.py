
import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

setup(
    name='mav_baselines',
    version='0.0.1',
    author='Anonymous',
    author_email='anonymous@anonymous.org',
    description='A simulator for reinforcement learning',
    long_description='',
    packages=['mav_baselines'],
)
