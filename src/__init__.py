# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:07:39 2016

@author: pablo
"""

""" CREATING PYTHON plasma_plume PACKAGE """

""" IMPORTING GENERAL USAGE MODULES """ 

from scipy.interpolate import interp1d,interp2d,griddata #1D, and 2D interpolation libraries

from scipy.integrate import odeint # Ordinary Diffferential Equation (ODE) Solver

import numpy as np #Scientific and numerical general module

import math #Simbolic math library

import matplotlib.pyplot as plt #General Plotter library

import unittest #Testing library

""" IMPORT PARENT CLASS Hyperplume """

from .HYPERPLUME.hyperplume import Hyperplume

""" IMPORT SUBCLASSES (FOR PACKAGE TESTS ONLY) """

from .SSM.SSM_plume import SSM,type_parks,type_korsun,type_ashkenazy

from .AEM.AEM_plume import AEM

