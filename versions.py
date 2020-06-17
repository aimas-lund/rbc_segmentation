import cv2 as cv
import matplotlib as m
import numpy as np
import pandas as pd
import scipy as sci
import seaborn as sn
import tensorflow as tf

versions = [
    tf.__version__,
    cv.__version__,
    np.__version__,
    pd.__version__,
    sn.__version__,
    m.__version__,
    sci.__version__
]

for i in range(len(versions)):
    print(versions[i])