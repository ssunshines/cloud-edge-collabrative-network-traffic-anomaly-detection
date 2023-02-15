# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 11:12:29 2021

@author: YT
"""

import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import csv
import statsmodels.api as sm
import math
from statsmodels.tsa.stattools import adfuller 
import time

f=open('path')
df=pd.DataFrame(pd.read_csv(f))
value=df['value']

train_num = 'ttrain'
period_series_num ="m" 
forecast_num = "tfc"
stationary_test = 'adf'
seasonal_diff = 'True'
trigger=[]



def train(train_dataa):

    mod = sm.tsa.statespace.SARIMAX(train_dataa,
                                    order=(2, 1, 4),
                                    seasonal_order=(1, 1, 1, 10),
                                    enforce_stationarity=True,
                                    enforce_invertibility=False)
    result = mod.fit()
    return result

def anomaly_detection_normal(pc_normal,low_anomaly_normal,high_anomaly_normal):
    
    flag=0
    j=0
    i=0
    for i in range(train_num-1):
        pc_normal+=1
        
        if pc_normal>=len(value)-1:
            return flag,pc_normal
        
        if value[pc_normal]<low_anomaly_normal[j] or value[pc_normal]>high_anomaly_normal[j]:
            trigger.append(pc_normal)
            flag=1
            return flag,pc_normal
        j+=1
        if j == "tfc":
            j=0
            
    return flag,pc_normal
        
    
    

def anomaly_detection(low_anomaly,high_anomaly,pc_anomaly):
    
    j=0
    
    while True:
        
        pc_anomaly+=1
        if pc_anomaly>=len(value)-1:
            return pc_anomaly
        
        
        if value[pc_anomaly]<low_anomaly[j] or value[pc_anomaly]>high_anomaly[j]:
            trigger.append(pc_anomaly)
            j+=1
            if j == "tfc":
                j=0
                
        else:
            flag,pc_anomaly=anomaly_detection_normal(pc_anomaly,low_anomaly,high_anomaly)
            
            if flag==0:
                return pc_anomaly
            
            if flag==1:
                j=0
                
            
            
i=0
pc=0

while True:
    start_time=time.time()
    online_flag=0
    train_data=value[pc:train_num+pc]
    test_data=value[train_num+pc:train_num+forecast_num+pc].tolist()
    results=train(train_data)
    pred = results.get_prediction(start = train_num+pc,end=train_num+forecast_num-1+pc,dynamic=False)
    pred_ci = pred.conf_int()
    print(pred_ci)
    predd = pred.predicted_mean

    low=pred_ci.iloc[:,0].tolist()
    high=pred_ci.iloc[:,1].tolist()

    for j in range(forecast_num):
        if test_data[j]>high[j] or test_data[j]<low[j]:
            trigger.append(j+pc+train_num)
            pc=anomaly_detection(low,high,j+pc+train_num)
            pc=pc-train_num
            online_flag=1
            break
    
    if online_flag==0:
        pc=pc+forecast_num
        
    if pc+train_num>=len(value)-1:
        break
    
end_time=time.time()

print((end_time-start_time)/len(value))
print(len(trigger))
    
    
    
    
    


