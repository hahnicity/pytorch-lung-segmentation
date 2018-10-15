#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


setup(name='Pytorch Lung Segmentation',
      version="1.0",
      description='CXR Lung Segmentation Tools for Pytorch',
      packages=find_packages(exclude=[]),
      entry_points={
      },
      include_package_data=True,
      )
