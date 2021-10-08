"""
Originally coded by Jongwon Yoon
revised on 10 sep 2021 by CheolJ

licensed by Advanced Manufcaturing Laboratory (AML)
"""

# Base Library
from tcn import TCN  # pip install keras-tcn
import tensorflow.keras.backend as kb
from tensorflow.keras.layers import Conv2D, Conv1D, Conv2DTranspose, MaxPooling2D, BatchNormalization, concatenate, Input
from tensorflow.keras import Model
import os
import sys
from os import path

# Data process Library
import pandas as pd
import numpy as np
import datetime
import shap

# Custom data process Module
import data_preprocessor


# Model Library - tensorflow
import tensorflow as tf
tf.compat.v1.disalbe_v2_behavior()  # <- For shap value calculation


# Model Library - TCN

##################################################################################################################################

# data load
rail_file = 'rail_data.xlsx'
cloud_file = 'cloud_2018.csv'
pm_file = 'pm_data.csv'
rail_feature = ['date', 'air_temp', 'wind_speed', 'rain',
                'humidity', 'TSI', 'altitude', 'azimuth', 'rail_temp']
cloud_data_columns = ['num', 'name', 'date', 'cloud', 'height', 'sight']
pm_data_columns = ['loc', 'type', 'code', 'name', 'date',
                   'so2', 'co', 'o3', 'no2', 'pm10', 'pm25', 'loc2']

# Data Handling - rail
drop_column = ['wind_direction_unused', 'wind_direction']

rail_data = pd.read_excel('data/' + rail_file, sheet_name=0)  # Load rail data

rail_data['date'] = pd.to_datetime(
    rail_data['date'])  # change type -> to datetime
rail_data.drop(drop_column, axis=1, inplace=True)  # Remove Unused column
rail_data = rail_data.loc[rail_data.date.dt.date !=
                          datetime.date(2019, 5, 17)]  # Remove error data
rail_data.iloc[:, 1:] = rail_data.iloc[:, 1:].astype(
    'float')  # Change type -> to float (except date)
rail_data = data_preprocessor.input_data_organizer(
    rail_data)  # Calculating Solar Features

# Check all data types of rail-data
print('Data type check\n', rail_data.dtypes)
print('-----------------------------------------------------------------------')
print('Null Check\n', rail_data.isnull().sum())  # Null Check

# Dviding data by its orientation
data_1 = rail_data.loc[rail_data.rail_direction == 0, :]
data_2 = rail_data.loc[rail_data.rail_direction == 90, :]

# Reset Indexes
data_1.reset_index(drop=True, inplace=True)
data_2.reset_index(drop=True, inplace=True)

# drop rail_direction column
data_1.drop('rail_direction', axis=1, inplace=True)
data_2.drop('rail_direction', axis=1, inplace=True)


# Data Handling - Cloud data

cloud_data = pd.read_csv('data/' + cloud_file, skiprows=1,
                         names=cloud_data_columns, usecols=['date', 'cloud'])
# skip row - 원하는 행부터 불러오기
# names - 컬럼 명 지정
# usecols - 사용할 컬럼명 지정
# index_col - date로 지정후 리샘플링을 위해 준비
# names를 지정하면 첫번째 행부터 데이터로 불러오기 때문에, skiprows를 통해, 첫번째 행을 무시하고 데이터 로드

cloud_data['date'] = pd.to_datetime(
    cloud_data['date'])  # datetime으로 data_type 변경
# resampling(업샘플링)을 위해서 date를 index로 지정
cloud_data.set_index('date', inplace=True)
cloud_data = cloud_data.interpolate()  # 선형으로 cloud_data 보간
cloud_data = cloud_data.resample('10T').last()  # 10분 간격으로 데이터 업샘플링
cloud_data = cloud_data.interpolate()  # 업샘플링 이후 cloud_data 한번 더 보간
cloud_data.reset_index(drop=False, inplace=True)

# Check all data types of cloud_Data
print('Data type check\n', cloud_data.dtypes)
print('-----------------------------------------------------------------------')
print('Null Check\n', cloud_data.isnull().sum())  # Null Check

# Data handling - PM data

# Function - Change hour at 24 to 0
# Datetime이 24시로 표시된 걸 처리하지 못하기 때문에,
# 데이터상에 24시로 기록된걸 다음날 0시로 처리함
# 함수에 데이터를 넘겨줄 때, 날짜의 데이터값이 str로 설정되어야 함


def custom_to_datetime(date):
    # If the time is 24, set it to 0 and increment day by 1
    if date[8:10] == '24':
        return pd.to_datetime(date[:-2], format='%Y%m%d') + pd.Timedelta(days=1)
    else:
        return pd.to_datetime(date, format='%Y%m%d%H')


# Load Data
pm_data = pd.read_csv('data/' + pm_file, skiprows=1, names=pm_data_columns,
                      usecols=['date', 'so2', 'co', 'o3', 'no2', 'pm10', 'pm25'])
# skip row - 원하는 행부터 불러오기
# names - 컬럼 명 지정
# usecols - 사용할 컬럼명 지정
# index_col - date로 지정후 리샘플링을 위해 준비
# names를 지정하면 첫번째 행부터 데이터로 불러오기 때문에, skiprows를 통해, 첫번째 행을 무시하고 데이터 로드

# Change types - for custom_datetime_function
pm_data['date'] = pm_data['date'].astype('str')
pm_data['date'] = pm_data['date'].apply(
    custom_to_datetime)  # apply custom_datetime_function
pm_data.set_index('date', inplace=True)
pm_data = pm_data.resample('10T').last()
pm_data = pm_data.interpolate()
pm_data = pm_data.loc['2018-07-20 00:00:00':]
pm_data.reset_index(drop=False, inplace=True)

print('Data type check\n', pm_data.dtypes)  # Check all data types of PM_Data
print('-----------------------------------------------------------------------')
print('Null Check\n', pm_data.isnull().sum())  # Null Check

data_1 = pd.merge(data_1, cloud_data, on='date', how='outer')
data_1 = pd.merge(data_1, pm_data, on='date', how='outer')

data_2 = pd.merge(data_2, cloud_data, on='date', how='outer')
data_2 = pd.merge(data_2, pm_data, on='date', how='outer')

# Remove row with null-data
data_1.dropna(axis=0, inplace=True)
data_2.dropna(axis=0, inplace=True)
