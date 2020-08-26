#publicLibrary.py是一个公共包，import经常用到的库，避免在每个模块都大量import，个人偏好
import warnings as wn
wn.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
# import missingno
import sys

#__all__可以控制我们哪些变量可见
# __all__ = ['wn','np','pd','os','plt','sns','sys']