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
##################################################################################

import shap

model = tf.keras.models.load_model('2021_08_27_original_add_pm.h5',custom_objects={'TCN':TCN})


#shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
#tf.compat.v1.disable_v2_behavior() -> import tensorflow하자 마자 실행


num = len(trainX1)
np.random.seed(42)
idx = np.random.RandomState(seed=42).permutation(num)[:1000]
    

    


explainer = shap.DeepExplainer(model,trainX1[idx])

shap_value = explainer.shap_values(valX1[:100])


shap_value0 = explainer.shap_values(testX1[testY1>=50])


shap.image_plot(shap_value0,testX1[testY1>=50])



a = np.absolute(shap_value0)
a = a.reshape(41,18,17)

s = np.sum(a,axis=0)

plt.pcolor(shap_value[0][0])
plt.colorbar()
plt.show()













