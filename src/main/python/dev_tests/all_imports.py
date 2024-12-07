import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Dropout
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.layers import LSTM, GRU,SimpleRNN

import np_utils
# from keras.utils  import np_utils

from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
