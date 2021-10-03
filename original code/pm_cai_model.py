import pandas as pd
import sys 
import os 
from os import path
import numpy as np 
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv1D, Conv2DTranspose, MaxPooling2D, BatchNormalization, concatenate, Input
from tensorflow.keras import Model
import tensorflow.keras.backend as kb
#!pip install keras-tcn
from tcn import TCN
import datetime

#########################################################################################################

def input_spilt_date(date,tre_data):

    
    date_rail_temp = date.date()
    
    
    
    out_range_data = []
    data_inrange = [] 
    
    
    
    out_index = date_rail_temp
    
    out_range_data = list(filter(lambda x : x.index[-1].date()==out_index, tre_data))
    
    
    data_inrange = list(filter(lambda x : x.index[-1].date() != out_index, tre_data))    
    
    
    
    return data_inrange, out_range_data

def label_spilt_date(date,tre_data):
    
    
    date_rail_temp = date.date()
    
    
    
    out_range_data = []
    data_inrange = [] 
    
    
    
    out_index = date_rail_temp  
    
    out_range_data = list(filter(lambda x : x.name.date() == out_index, tre_data))
    
    
    
    data_inrange = list(filter(lambda x : x.name.date() != out_index, tre_data)) 
    
    
    return data_inrange, out_range_data



#######################################################################################################

def input_spilt_temp(data,feature,temp,tre_data):
    
    
    date_rail_temp = data['date'].loc[data[feature]>=temp].dt.date
    
    num = len(date_rail_temp.value_counts().index)
    
    out_range_data = []
    data_inrange = [] 
    
    np.random.seed(42)
    sample_size = int(np.round(num/2))
    
    idx = np.random.RandomState(seed=42).permutation(num)[:sample_size]
    
    
    out_index = date_rail_temp.value_counts().index[idx]
    
    out_range_data = list(filter(lambda x : x.index[-1].date() in out_index, tre_data))
    
    
    data_inrange = list(filter(lambda x : x.index[-1].date() not in out_index, tre_data))    
    
    
    
    return data_inrange, out_range_data

def label_spilt_temp(data,feature,temp,tre_data):
    
    
    date_rail_temp = data['date'].loc[data[feature]>=temp].dt.date
    
    num = len(date_rail_temp.value_counts().index)
    
    out_range_data = []
    data_inrange = []
    
    np.random.seed(42)
    sample_size = int(np.round(num/2))
    
    allid = range(num)
    idx = np.random.RandomState(seed=42).permutation(num)[:sample_size]
    
    out_index = date_rail_temp.value_counts().index[idx]    
    
    out_range_data = list(filter(lambda x : x.name.date() in out_index, tre_data))
    
    
    
    data_inrange = list(filter(lambda x : x.name.date() not in out_index, tre_data)) 
    
    
    return data_inrange, out_range_data



def data_spilt_temp(data,feature,temp):
    
    date_rail_temp = data['date'].loc[data[feature]>=temp].dt.date
    
    num = len(date_rail_temp.value_counts().index)
    
    out_range_data = pd.DataFrame()
    in_range_data = pd.DataFrame() 
    
    np.random.seed(42)
    sample_size = int(np.round(num/2))
    
    allid = range(num)
    idx = np.random.RandomState(seed=42).permutation(num)[:sample_size]
    
    
    for i in date_rail_temp.value_counts().index[idx]:
        a = data.loc[(data.date.dt.date == i)] 
        out_range_data = out_range_data.append(a)    
    
    #query 쓰면 100배 빠른듯.... 
    index_list = list(out_range_data.index)
    data_inrange = data.query("index not in @index_list")
    
    return data_inrange, out_range_data


########################################################################################################
def make_TS(wdata, pedata):
    data_TCN = []
    tmp_data = []
    label_data = []
    
    for i in range(0,len(wdata)-seq_len+1):
        timedelta = datetime.timedelta(minutes = (seq_len-1)*10)
        tmp_data = wdata.loc[wdata.index[i]:wdata.index[i]+timedelta,:]
        
        data_TCN.append(tmp_data)
        label_data.append(pedata.loc[:,['date','rail_temp']].loc[wdata.index[i]+timedelta])
        
        
        ################################################################################################
    
    drop_index_list = []    
    for j in range(len(data_TCN)):
        if pd.DataFrame(data_TCN[j]).isnull().sum().sum():
            drop_index_list.append(j)
    
    data_remove = data_TCN.copy()
    
    drop_index_list.reverse()
    
    
    for k in drop_index_list:        
        del data_remove[k]
        del label_data[k]
    
    
    drop_label_list=[]
    for q in range(len(label_data)):
        if pd.Series(label_data[q]).isnull().sum():
            drop_label_list.append(q)
    
    drop_label_list.reverse()
    
    for l in drop_label_list:
        del data_remove[l]
        del label_data[l]
    
    
    return data_remove, label_data
#################################################################################
#################################################################################
def select_date(date1, date2):
    if date1 >= date2:
        date = date1 
    if date1 < date2:
        date = date2
    return date


##################################################################################


file = 'data/rail_data.xlsx'
data = pd.read_excel(file)
data['date'] = pd.to_datetime(data['date'])

data.drop(['wind_direction_unused'],axis=1,inplace=True)

data = data.loc[data.date.dt.date != datetime.date(2019,5,17)]


data[data.columns[1:]] = data[data.columns[1:]].astype('float')


data_0 = data.loc[data['rail_direction']==0,:]
data_90 = data.loc[data['rail_direction']==90,:]

start_date_1 = data_0.iloc[0].date
start_date_2 = data_90.iloc[0].date

end_date_1 = data_0.iloc[-1].date
end_date_2 = data_90.iloc[-1].date

start_date = select_date(start_date_1,start_date_2)
end_date = select_date(end_date_1,end_date_2)



temp = pd.DataFrame()
temp['date'] = pd.date_range(start=start_date, end= end_date, freq='10T')

data_1 = pd.merge(temp,data_0,on='date',how='outer')
data_2 = pd.merge(temp,data_90,on='date',how='outer')
##################################################################################



cloud = 'data/cloud_2018.csv'

import data_preprocessor
data_1 = data_preprocessor.input_data_organizer(data_1)
data_2 = data_preprocessor.input_data_organizer(data_2)

data_1 = data_1[['date','air_temp','wind_speed','rain','humidity','TSI','altitude','azimuth','rail_temp']]
data_2 = data_2[['date','air_temp','wind_speed','rain','humidity','TSI','altitude','azimuth','rail_temp']]



#########################################################################################

cloud_data = pd.read_csv(cloud)
cloud_data.columns = ['num','name','date','cloud','height','sight']
cloud_data['date'] = pd.to_datetime(cloud_data['date'])
cloud_data = cloud_data.loc[(cloud_data.date >= start_date)&(cloud_data.date<=end_date)]


data_1 = pd.merge(data_1, cloud_data,on='date', how='outer')
data_2 = pd.merge(data_2, cloud_data,on='date', how='outer')


#####################################################################################
###################################################################################

data_1.drop(['num','name'],axis='columns', inplace=True)
data_1[['cloud','height','sight']] = data_1[['cloud','height','sight']].interpolate()

data_2.drop(['num','name'],axis='columns', inplace=True)
data_2[['cloud','height','sight']] = data_2[['cloud','height','sight']].interpolate()


################################################################################################

pm = 'data/2018_2019cnu.csv'
pm_data = pd.read_csv(pm)
pm_data.columns = ['loc','type','code','name','date','so2','co','o3','no2','pm10','pm25','loc2']

#값 인식이 안되서 문자열 자르기로 진행함
for i in range(len(pm_data.index)):
     tt = pm_data['date'].iloc[i]
     ty = int(str(tt)[0:4])
     tm = int(str(tt)[4:6])
     td = int(str(tt)[6:8])
     th = int(str(tt)[8:10])
     if th == 24:
         th = 0
         pm_data['date'].iloc[i] = datetime.datetime(ty,tm,td,th)+datetime.timedelta(days=1)
     else:
         
         pm_data['date'].iloc[i] = datetime.datetime(ty,tm,td,th)

pm_data['date'] = pd.to_datetime(pm_data['date'])
pm_data = pm_data.loc[(pm_data.date >= start_date)&(pm_data.date<=end_date)]     


data_1 = pd.merge(data_1, pm_data,on='date', how='outer')
data_2 = pd.merge(data_2, pm_data,on='date', how='outer')

data_1.drop(['loc','type','name','code','loc2'],axis='columns', inplace=True)
data_1[['so2','co','o3','no2','pm10','pm25']] = data_1[['so2','co','o3','no2','pm10','pm25']].interpolate()

data_2.drop(['loc','type','name','code','loc2'],axis='columns', inplace=True)
data_2[['so2','co','o3','no2','pm10','pm25']] = data_2[['so2','co','o3','no2','pm10','pm25']].interpolate()




###############################################################################################################

import CAI
data_1['avg_p10']=CAI.PM_10_avg(data_1['pm10'])
data_1['avg_p25']=CAI.PM_10_avg(data_1['pm25'])

data_2['avg_p10']=CAI.PM_10_avg(data_2['pm10'])
data_2['avg_p25']=CAI.PM_10_avg(data_2['pm25'])


I_LO = [0, 51, 101, 251]
I_HI = [50, 100, 250, 500]

datacai = {'sulfur_dioxide': [0, 0.02, 0.021, 0.05, 0.051, 0.15, 0.151, 1], 'carbon_monoxide': [0, 2, 2.1, 9, 9.1, 15, 15.1, 50], 
        'ozone': [0, 0.03, 0.031, 0.09, 0.091, 0.15, 0.151, 0.6], 'nitrogen_monoxide': [0, 0.03, 0.031, 0.06, 0.061, 0.2, 0.201, 2],
        'PM10': [0, 30, 31, 80, 81, 150, 151, 2000], 'PM2.5': [0, 15, 16, 35, 36, 75, 76, 1000]} #### 이 친구들이 PM10, PM2.5의 맥스 수치를 오기 해놓음, 그래서 수정함 (원본 : 600, 500 / 수정본 : 2000, 1000) ##
df=pd.DataFrame(datacai).T

def sulfur_dioxide(pol):
    if df.loc['sulfur_dioxide'][0] <= pol <= df.loc['sulfur_dioxide'][1] :
        I_p1 = (I_HI[0]-I_LO[0])/(df.loc['sulfur_dioxide'][1]-df.loc['sulfur_dioxide'][0]) * (pol-df.loc['sulfur_dioxide'][0]) + I_LO[0]
    elif df.loc['sulfur_dioxide'][1] < pol <= df.loc['sulfur_dioxide'][3] :
        I_p1 = (I_HI[1]-I_LO[1])/(df.loc['sulfur_dioxide'][3]-df.loc['sulfur_dioxide'][2]) * (pol-df.loc['sulfur_dioxide'][2]) + I_LO[1]
    elif df.loc['sulfur_dioxide'][3] < pol <= df.loc['sulfur_dioxide'][5] :
        I_p1 = (I_HI[2]-I_LO[2])/(df.loc['sulfur_dioxide'][5]-df.loc['sulfur_dioxide'][4]) * (pol-df.loc['sulfur_dioxide'][4]) + I_LO[2]
    elif df.loc['sulfur_dioxide'][5] < pol <= df.loc['sulfur_dioxide'][7] :
        I_p1 = (I_HI[3]-I_LO[3])/(df.loc['sulfur_dioxide'][7]-df.loc['sulfur_dioxide'][6]) * (pol-df.loc['sulfur_dioxide'][6]) + I_LO[3]
    return I_p1

def carbon_monoxide(pol):
    if df.loc['carbon_monoxide'][0] <= pol <= df.loc['carbon_monoxide'][1] :
        I_p2 = (I_HI[0]-I_LO[0])/(df.loc['carbon_monoxide'][1]-df.loc['carbon_monoxide'][0]) * (pol-df.loc['carbon_monoxide'][0]) + I_LO[0]
    elif df.loc['carbon_monoxide'][1] < pol <= df.loc['carbon_monoxide'][3] :
        I_p2 = (I_HI[1]-I_LO[1])/(df.loc['carbon_monoxide'][3]-df.loc['carbon_monoxide'][2]) * (pol-df.loc['carbon_monoxide'][2]) + I_LO[1]
    elif df.loc['carbon_monoxide'][3] < pol <= df.loc['carbon_monoxide'][5] :
        I_p2 = (I_HI[2]-I_LO[2])/(df.loc['carbon_monoxide'][5]-df.loc['carbon_monoxide'][4]) * (pol-df.loc['carbon_monoxide'][4]) + I_LO[2]
    elif df.loc['carbon_monoxide'][5] < pol <= df.loc['carbon_monoxide'][7] :
        I_p2 = (I_HI[3]-I_LO[3])/(df.loc['carbon_monoxide'][7]-df.loc['carbon_monoxide'][6]) * (pol-df.loc['carbon_monoxide'][6]) + I_LO[3]
    return I_p2

def ozone(pol):
    if df.loc['ozone'][0] <= pol <= df.loc['ozone'][1] :
        I_p3 = (I_HI[0]-I_LO[0])/(df.loc['ozone'][1]-df.loc['ozone'][0]) * (pol-df.loc['ozone'][0]) + I_LO[0]
    elif df.loc['ozone'][1] < pol <= df.loc['ozone'][3] :
        I_p3 = (I_HI[1]-I_LO[1])/(df.loc['ozone'][3]-df.loc['ozone'][2]) * (pol-df.loc['ozone'][2]) + I_LO[1]
    elif df.loc['ozone'][3] < pol <= df.loc['ozone'][5] :
        I_p3 = (I_HI[2]-I_LO[2])/(df.loc['ozone'][5]-df.loc['ozone'][4]) * (pol-df.loc['ozone'][4]) + I_LO[2]
    elif df.loc['ozone'][5] < pol <= df.loc['ozone'][7] :
        I_p3 = (I_HI[3]-I_LO[3])/(df.loc['ozone'][7]-df.loc['ozone'][6]) * (pol-df.loc['ozone'][6]) + I_LO[3]
    return I_p3

def nitrogen_monoxide(pol):
    global I_p4
    if df.loc['nitrogen_monoxide'][0] <= pol <= df.loc['nitrogen_monoxide'][1] :
        I_p4 = (I_HI[0]-I_LO[0])/(df.loc['nitrogen_monoxide'][1]-df.loc['nitrogen_monoxide'][0]) * (pol-df.loc['nitrogen_monoxide'][0]) + I_LO[0]
    elif df.loc['nitrogen_monoxide'][2] <= pol <= df.loc['nitrogen_monoxide'][3] :
        I_p4 = (I_HI[1]-I_LO[1])/(df.loc['nitrogen_monoxide'][3]-df.loc['nitrogen_monoxide'][2]) * (pol-df.loc['nitrogen_monoxide'][2]) + I_LO[1]
    elif df.loc['nitrogen_monoxide'][4] <= pol <= df.loc['nitrogen_monoxide'][5] :
        I_p4 = (I_HI[2]-I_LO[2])/(df.loc['nitrogen_monoxide'][5]-df.loc['nitrogen_monoxide'][4]) * (pol-df.loc['nitrogen_monoxide'][4]) + I_LO[2]
    elif df.loc['nitrogen_monoxide'][6] <= pol <= df.loc['nitrogen_monoxide'][7] :
        I_p4 = (I_HI[3]-I_LO[3])/(df.loc['nitrogen_monoxide'][7]-df.loc['nitrogen_monoxide'][6]) * (pol-df.loc['nitrogen_monoxide'][6]) + I_LO[3]
    return I_p4

def PM10(pol):
    if df.loc['PM10'][0] <= pol <= df.loc['PM10'][1] :
        I_p5 = (I_HI[0]-I_LO[0])/(df.loc['PM10'][1]-df.loc['PM10'][0]) * (pol-df.loc['PM10'][0]) + I_LO[0]
    elif df.loc['PM10'][2] <= pol <= df.loc['PM10'][3] :
        I_p5 = (I_HI[1]-I_LO[1])/(df.loc['PM10'][3]-df.loc['PM10'][2]) * (pol-df.loc['PM10'][2]) + I_LO[1]
    elif df.loc['PM10'][4] <= pol <= df.loc['PM10'][5] :
        I_p5 = (I_HI[2]-I_LO[2])/(df.loc['PM10'][5]-df.loc['PM10'][4]) * (pol-df.loc['PM10'][4]) + I_LO[2]
    elif df.loc['PM10'][6] <= pol <= df.loc['PM10'][7] :
        I_p5 = (I_HI[3]-I_LO[3])/(df.loc['PM10'][7]-df.loc['PM10'][6]) * (pol-df.loc['PM10'][6]) + I_LO[3]
    return I_p5

def PM25(pol):
    if df.loc['PM2.5'][0] <= pol <= df.loc['PM2.5'][1] :
        I_p6 = (I_HI[0]-I_LO[0])/(df.loc['PM2.5'][1]-df.loc['PM2.5'][0]) * (pol-df.loc['PM2.5'][0]) + I_LO[0]
    elif df.loc['PM2.5'][2] <= pol <= df.loc['PM2.5'][3] :
        I_p6 = (I_HI[1]-I_LO[1])/(df.loc['PM2.5'][3]-df.loc['PM2.5'][2]) * (pol-df.loc['PM2.5'][2]) + I_LO[1]
    elif df.loc['PM2.5'][4] <= pol <= df.loc['PM2.5'][5] :
        I_p6 = (I_HI[2]-I_LO[2])/(df.loc['PM2.5'][5]-df.loc['PM2.5'][4]) * (pol-df.loc['PM2.5'][4]) + I_LO[2]
    elif df.loc['PM2.5'][6] <= pol <= df.loc['PM2.5'][7] :
        I_p6 = (I_HI[3]-I_LO[3])/(df.loc['PM2.5'][7]-df.loc['PM2.5'][6]) * (pol-df.loc['PM2.5'][6]) + I_LO[3]
    return I_p6

def CAI(sd, cm, oz, nm, P10, P25):
    sulfur_dioxide_I =  sulfur_dioxide(sd)
    carbon_monoxide_I =  carbon_monoxide(cm)
    ozone_I =  ozone(oz)
    nitrogen_monoxide_I =  nitrogen_monoxide(nm)
    PM10_I =  PM10(P10)
    PM25_I =  PM25(P25)
    
    CAI_list = [sulfur_dioxide_I, carbon_monoxide_I, ozone_I, nitrogen_monoxide_I, PM10_I, PM25_I]
    
    Bad_CAT_list = [sulfur_dioxide_I >= 101, carbon_monoxide_I >= 101, ozone_I >= 101, nitrogen_monoxide_I >= 101, PM10_I >= 101, PM25_I >= 101]
    
    
    if Bad_CAT_list.count(True) == 2:
        CAI = max(CAI_list) + 50
    elif Bad_CAT_list.count(True) >= 3:
        CAI = max(CAI_list) + 75
    else : 
        CAI = max(CAI_list)
    
    # 중요 오염물질 계산
    CAI_imp = df.index[CAI_list.index(max(CAI_list))]  
    CAI = round(CAI)
    return CAI, CAI_list, CAI_imp



data_1['CAI']=data_1[['so2','co','o3','no2','avg_p10','avg_p25']].apply(lambda x : CAI(x[0],x[1],x[2],x[3],x[4],x[5])[0],axis=1)
data_1['CAI_value']=data_1[['so2','co','o3','no2','avg_p10','avg_p25']].apply(lambda x : CAI(x[0],x[1],x[2],x[3],x[4],x[5])[1],axis=1)
data_1['CAI_main_factor']=data_1[['so2','co','o3','no2','avg_p10','avg_p25']].apply(lambda x : CAI(x[0],x[1],x[2],x[3],x[4],x[5])[2],axis=1)

data_2['CAI']=data_2[['so2','co','o3','no2','avg_p10','avg_p25']].apply(lambda x : CAI(x[0],x[1],x[2],x[3],x[4],x[5])[0],axis=1)
data_2['CAI_value']=data_2[['so2','co','o3','no2','avg_p10','avg_p25']].apply(lambda x : CAI(x[0],x[1],x[2],x[3],x[4],x[5])[1],axis=1)
data_2['CAI_main_factor']=data_2[['so2','co','o3','no2','avg_p10','avg_p25']].apply(lambda x : CAI(x[0],x[1],x[2],x[3],x[4],x[5])[2],axis=1)



###############################################################################################################
data_1.index = data_1['date']
data_2.index = data_2['date']




data_3 = pd.concat([data_1,data_2])

data_TCN = []
tmp_data = pd.DataFrame()
seq_len = 18

#################################################################################
####################################################################################

valid = 8265*2+1168

scal_data =  data_spilt_temp(data_1,'CAI',temp = 139)[0]

scal_data.dropna(inplace=True)

from sklearn.preprocessing import RobustScaler
scal_feature = ['air_temp','wind_speed','rain','humidity','cloud','TSI','altitude','azimuth','so2','co','o3','no2','pm10','pm25','avg_p10','avg_p25','CAI']
remain_feature =[]
scaler = RobustScaler().fit(scal_data[scal_feature].iloc[:-valid])

#################################################################################
#################################################################################

data1_sc = scaler.transform(data_1[scal_feature])
data2_sc = scaler.transform(data_2[scal_feature])

data1_scpd = pd.DataFrame(data1_sc,index=data_1.index,columns=scal_feature)
data2_scpd = pd.DataFrame(data2_sc,index=data_2.index,columns=scal_feature)

data1_scpd = pd.merge(data1_scpd,data_1[remain_feature],right_on='date',left_on=data_1.index,how='inner')
data2_scpd = pd.merge(data2_scpd,data_2[remain_feature],right_on='date',left_on=data_2.index,how='inner')

data1_scpd.index = data1_scpd['date']
data2_scpd.index = data2_scpd['date']



data1_input, data1_label = make_TS(data1_scpd, data_1)
data2_input, data2_label = make_TS(data2_scpd, data_2)

#################################################################################

data1_input,test1_input = input_spilt_temp(data_1, feature='CAI', temp=139, tre_data = data1_input)
data1_label,test1_label = label_spilt_temp(data_1, feature='CAI', temp=139, tre_data = data1_label)



data2_input,test2_input = input_spilt_temp(data_3, feature='rail_temp', temp=50, tre_data = data2_input)
data2_label,test2_label = label_spilt_temp(data_3, feature='rail_temp', temp=50, tre_data = data2_label)



###############################################################################################################
'''
use0_index =[]
for j in range(len(data0_label)):
    if data0_label[j]['date'].hour>=9 :
        use0_index.append(j)

use90_index =[]
for j in range(len(data90_label)):
    if data90_label[j]['date'].hour>=9:
        use90_index.append(j)

#9-18 : 
#All  :  
'''
######################################################################################################

valid1 = int(np.round(len(data1_input) *0.2))
valid2 = int(np.round(len(data2_input) *0.2))



data1_input_use = np.array(data1_input)
data1_label_use = np.array(data1_label)

data2_input_use = np.array(data2_input)
data2_label_use = np.array(data2_label)

#################################################################################

data1_input_n = np.array(data1_input_use[:-valid1])[:,:,1:]
data1_label_n = np.array(data1_label_use[:-valid1])[:,1]

data1_input_v = np.array(data1_input_use[-valid1:])[:,:,1:]
data1_label_v = np.array(data1_label_use[-valid1:])[:,1]



data2_input_n = np.array(data2_input_use[:-valid2])[:,:,1:]
data2_label_n = np.array(data2_label_use[:-valid2])[:,1]

data2_input_v = np.array(data2_input_use[-valid2:])[:,:,1:]
data2_label_v = np.array(data2_label_use[-valid2:])[:,1]


###############################################################################
trainX1 = data1_input_n
trainY1 = data1_label_n
trainY1 = trainY1.astype(np.float64)
trainX1 = trainX1.astype(np.float64)


valX1 = data1_input_v
valY1 = data1_label_v
valY1 = valY1.astype(np.float64)
valX1 = valX1.astype(np.float64)
##############################################################################

###############################################################################
trainX2 = data2_input_n
trainY2 = data2_label_n
trainY2 = trainY2.astype(np.float64)
trainX2 = trainX2.astype(np.float64)


valX2 = data2_input_v
valY2 = data2_label_v
valY2 = valY2.astype(np.float64)
valX2 = valX2.astype(np.float64)
##############################################################################
# 3.2. TCN base model              
def base_model(input_layer):
    
    tcn1 = TCN(nb_filters=64, nb_stacks=2, use_batch_norm=True,return_sequences=False,dropout_rate= 0.2)(input_layer)
   # nom1 = TCN(nb_filters=64, nb_stacks=2, use_batch_norm=True)(tcn1)
   # tcn2 = TCN(nb_filters=7)(tcn1)        
    #flat =  tf.keras.layers.Flatten()(conv2)
    #output_layer =  tf.keras.layers.Dense(96, activation='linear')(tcn1)
    output_layer =  tf.keras.layers.Dense(1,activation='linear')(tcn1)
    
    return output_layer
def base_model(input_layer):
    
    tcn1 = TCN(nb_filters=64, nb_stacks=2, use_batch_norm=True,return_sequences=False,dropout_rate= 0.2)(input_layer)
   # nom1 = TCN(nb_filters=64, nb_stacks=2, use_batch_norm=True)(tcn1)
   # tcn2 = TCN(nb_filters=7)(tcn1)        
    #flat =  tf.keras.layers.Flatten()(conv2)
    #output_layer =  tf.keras.layers.Dense(96, activation='linear')(tcn1)
    output_layer =  tf.keras.layers.Dense(1,activation='linear')(tcn1)
    
    return output_layer

input_layer = Input((18,17))
output_layer = base_model(input_layer)

TCN_nodirection = Model(input_layer, output_layer)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

TCN_nodirection.compile(loss='mae', optimizer=optimizer, metrics=["mse"])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True,verbose=1)
history_warehouse = TCN_nodirection.fit(trainX1, trainY1, validation_data = (valX1,valY1), epochs = 3000, verbose=2, callbacks=[callback])
TCN_nodirection.save('2021_09_13_original_add_pm.h5')
test1_input_use = np.array(test1_input)[:,:,1:]
test1_label_use = np.array(test1_label)[:,1]
testX1 = test1_input_use
testY1 = test1_label_use
testY1 = testY1.astype(np.float64)
testX1 = testX1.astype(np.float64)


pre = TCN_nodirection.predict(testX1)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

plt.plot(testY1,pre,'k.')
plt.ylim(0,65)
plt.xlim(0,65)
plt.xlabel('cnu_original' )
plt.title('mse : %f' %mse(testY1,pre))
plt.show()