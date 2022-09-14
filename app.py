import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from covid_time_series_prediction.ml_logic.preprocessor import train_test_set, scaler
from tensorflow.keras import models,utils
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.python.keras import utils
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import RMSprop
from covid_time_series_prediction.ml_logic import preprocessor
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

st.markdown(
    '''
    #        HELLO !
    '''
    '''
    #        Welcome to COVID-19 Prediction App !
    '''



    '''
    ##### Our COVID-19 prediction models were based on restriction indicators, stringency index, and vaccination campaigns



    They were performed by Alberto, Sumedha, Thomas, and Kim under supervision of Arnaud and TAs


    ''')



option=st.selectbox('PLEASE SELECT YOUR COUNTRY',

('AUSTRALIA', 'BRAZIL', 'FRANCE', 'INDIA', 'MEXICO', 'RUSSIA', 'USA', 'UNITED KINGDOM'))


st.write('YOU SELECTED:', option)

# find csv
# read csv
# preprocess csv with Sumedha process
# predict the X_predict(dataframe)
# plot the prediction
