# import pages.Dataset_Batik as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import random as rdm
import json as js
import streamlit as st
import function as fc
import os

dt = __import__('pages.1_Dataset_Batik', fromlist=[''])

st.set_page_config('Testing Backpropagation',layout='wide')
st.title("Hasil Testing Backpropagation Berbagai Skenario")
#fungsi data batik konversi
def Conv(batik, arch):
    xinput, yinput=fc.convToDataProccess(batik, arch[0])
    dataproses = fc.normalization(xinput), fc.normalization(yinput)
    xtrain, ytrain, xtest, ytest = fc.dataSplitting(dataproses ,0.7)
    return xtrain, ytrain, xtest, ytest 

def ConvDate(batik, arch):
    dataproses=fc.convToDataProccess(batik, arch[0])
    xtrain, ytrain, xtest, ytest = fc.dataSplitting(dataproses ,0.7)
    return xtrain, ytrain, xtest, ytest 

model = os.listdir("model_4")
arch = [4,7,1], [6,6,1], [8,13,1]
learning_rate = 0.2, 0.02, 0.002
momentum = 0.6, 0.3, 0.9

# make a date
date = pd.read_csv("tanggal.csv")
date.columns = ["tanggal", "none"]
date = date.drop("none", axis=1)

# download data
data_batikcap, data_batiktulis, data_semitulis, data_semiwarna = dt.batikcap, dt.batiktulis, dt.semitulis, dt.semiwarna
# xtrain, ytrain, xtest, ytest  = dt.setTrainTest("batikcap",4,0.7)
batikcap, batiktulis, semitulis, semiwarna=st.tabs(["Batik Cap", "Batik Tulis", "Semi Tulis", "Semi Warna"])
#batikcap
#make a parameter
with batikcap:
    #sview on streamlit
    batik = "batikcap"
    arch_selectbc =st.selectbox("Arsitektur :",arch,key="archbc")
    lr_selectbc =st.selectbox("Learning Rate :", learning_rate,key="lrbc")
    momentum_selectbc =st.selectbox("Momentum :", momentum, key="momentbc")
    #make a model
    try:
        xtrain, ytrain, xtest, ytest = Conv(data_batikcap, arch_selectbc)
        modelBatikcap = fc.backpropagation(arch_selectbc, lr_selectbc, momentum_selectbc, 1000, 0.001)
        dateXtrain, dateYtrain, dateXtest, dateYtest = ConvDate(date, arch_selectbc)
    except: 
        arch_selectbc = [4,7,1]     
        lr_selectbc = 0.2
        momentum_selectbc = 0.6
        xtrain, ytrain, xtest, ytest = Conv(data_batikcap, arch_selectbc)
        dateXtrain, dateYtrain, dateXtest, dateYtest = ConvDate(date, arch_selectbc)
        modelBatikcap = fc.backpropagation(arch_selectbc, lr_selectbc, momentum_selectbc, 1000, 0.001)
    for path in model:
        if batik in path and str(arch_selectbc) in path and str(lr_selectbc) in path and str(momentum_selectbc) in path:
            path_select = path
            break
    modelBatikcap.loadModel("model4/"+path)
    modelBatikcap.test(xtest,ytest)
    grafbatikcap = fc.grafPrediksi(modelBatikcap.arrayofY, ytest)
    
    prediksi = np.trunc(fc.denormalisasi(modelBatikcap.arrayofY,np.array(data_batikcap).min(), np.array(data_batikcap).max())).astype(int)
    actual = np.trunc(fc.denormalisasi(ytest[:,0],np.array(data_batikcap).min(), np.array(data_batikcap).max())).astype(int)
    
    tabel = {"Tanggal": dateYtest[:,0],"Prediksi" : prediksi, "Actual" : actual, "error":np.absolute(prediksi-actual)}
    errbc = modelBatikcap.msetest
    st.write(f"ARSITEKTUR : {arch_selectbc}")
    st.write(f"Learning Rate : {lr_selectbc}, Momentum : {momentum_selectbc}, MSE : 0.001")
    st.write("MSE Test :", str(errbc))
    a=st.pyplot(grafbatikcap)
    st.table(tabel)
#batiktulis
with batiktulis:
    batik = "batiktulis"    
    arch_selectbt =st.selectbox("Arsitektur :",arch, key="archbt")
    lr_selectbt =st.selectbox("Learning Rate :", learning_rate, key="lrbt")
    momentum_selectbt =st.selectbox("Momentum :", momentum, key="momentbt")
    #make a model
    try:
        xtrain, ytrain, xtest, ytest = Conv(data_batiktulis, arch_selectbt)
        modelBatiktulis = fc.backpropagation(arch_selectbt, lr_selectbt, momentum_selectbt, 1000, 0.001)
        dateXtrain, dateYtrain, dateXtest, dateYtest = ConvDate(date, arch_selectbt)
    except:
        arch_selectbt = [4,7,1]
        lr_selectbt = 0.2
        momentum_selectbt = 0.6
        xtrain, ytrain, xtest, ytest = Conv(data_batiktulis, arch_selectbt)
        dateXtrain, dateYtrain, dateXtest, dateYtest = ConvDate(date, arch_selectbt)
        modelBatiktulis = fc.backpropagation(arch_selectbt, lr_selectbt, momentum_selectbt, 1000, 0.001)
    for path in model:
        if batik in path and str(arch_selectbt) in path and str(lr_selectbt) in path and str(momentum_selectbt) in path:
            path_select = path
            break
    modelBatiktulis.loadModel("model4/"+path)
    modelBatiktulis.test(xtest,ytest)
    grafbatiktulis = fc.grafPrediksi(modelBatiktulis.arrayofY, ytest)

    prediksibt = np.trunc(fc.denormalisasi(modelBatiktulis.arrayofY,np.array(data_batiktulis).min(), np.array(data_batiktulis).max())).astype(int)
    actualbt = np.trunc(fc.denormalisasi(ytest[:,0],np.array(data_batiktulis).min(), np.array(data_batiktulis).max())).astype(int)

    tabel = {"Tanggal": dateYtest[:,0],"Prediksi" : prediksibt, "Actual" : actualbt, "error":np.absolute(prediksibt-actualbt)}
    errbt = modelBatiktulis.msetest
    #sview on streamlit
    st.write(f"ARSITEKTUR : {arch_selectbt}")
    st.write(f"Learning Rate : {lr_selectbt}, Momentum : {momentum_selectbt}, MSE : 0.001")
    st.write("MSE  :", str(errbt))
    b=st.pyplot(grafbatiktulis)
    st.table(tabel)
with semitulis:
    batik = "semitulis"

    arch_selectst=st.selectbox("Arsitektur :",arch, key="archst")
    lr_selectst=st.selectbox("Learning Rate :", learning_rate, key="lrst")
    momentum_selectst=st.selectbox("Momentum :", momentum, key="momentst")
    #make a model
    try:
        xtrain, ytrain, xtest, ytest = Conv(data_semitulis, arch_selectst)
        modelsemitulis = fc.backpropagation(arch_selectst, lr_selectst, momentum_selectst, 1000, 0.001)
        dateXtrain, dateYtrain, dateXtest, dateYtest = ConvDate(date, arch_selectst)
    except:
        arch_selectst = [4,7,1]
        lr_selectst = 0.2
        momentum_selectst = 0.6
        xtrain, ytrain, xtest, ytest = Conv(data_semitulis, arch_selectst)
        modelsemitulis = fc.backpropagation(arch_selectst, lr_selectst, momentum_selectst, 1000, 0.001)
        dateXtrain, dateYtrain, dateXtest, dateYtest = ConvDate(date, arch_selectst)
    for path in model:
        if batik in path and str(arch_selectst) in path and str(lr_selectst) in path and str(momentum_selectst) in path:
            path_select = path
            break
    modelsemitulis.loadModel("model4/"+path)
    modelsemitulis.test(xtest, ytest)
    grafsemitulis = fc.grafPrediksi(modelsemitulis.arrayofY, ytest)

    prediksist = np.trunc(fc.denormalisasi(modelsemitulis.arrayofY,np.array(data_semitulis).min(), np.array(data_semitulis).max())).astype(int)
    actualst = np.trunc(fc.denormalisasi(ytest[:,0],np.array(data_semitulis).min(), np.array(data_semitulis).max())).astype(int)

    tabel = {"Tanggal": dateYtest[:,0],"Prediksi" : prediksist, "Actual" : actualst, "error":np.absolute(prediksist-actualst)}
    errst = modelsemitulis.msetest
    #sview on streamlit
    st.write(f"ARSITEKTUR : {arch_selectst}")
    st.write(f"Learning Rate : {lr_selectst}, Momentum : {momentum_selectst}, MSE : 0.001")
    st.write("MSE  :", str(errst))
    b=st.pyplot(grafsemitulis)
    st.table(tabel)
with semiwarna:
    batik = "semiwarna"
    arch_selectsw=st.selectbox("Arsitektur :",arch, key="archsw")
    lr_selectsw=st.selectbox("Learning Rate :", learning_rate, key="lrsw")
    momentum_selectsw=st.selectbox("Momentum :", momentum, key="momentsw")

    #make a model
    try:
        xtrain, ytrain, xtest, ytest = Conv(data_semiwarna, arch_selectsw)
        modelsemiwarna = fc.backpropagation(arch_selectsw, lr_selectsw, momentum_selectsw, 1000, 0.001)
        dateXtrain, dateYtrain, dateXtest, dateYtest = ConvDate(date, arch_selectsw)
    except:
        arch_selectsw = [4,7,1]
        lr_selectsw = 0.2
        momentum_selectsw = 0.6
        xtrain, ytrain, xtest, ytest = Conv(data_semiwarna, arch_selectsw)
        modelsemiwarna = fc.backpropagation(arch_selectsw, lr_selectsw, momentum_selectsw, 1000, 0.001)
        dateXtrain, dateYtrain, dateXtest, dateYtest = ConvDate(date, arch_selectsw)

    for path in model:
        if batik in path and str(arch_selectsw) in path and str(lr_selectsw) in path and str(momentum_selectsw) in path:
            path_select = path
            break
    modelsemiwarna.loadModel("model4/"+path)
    modelsemiwarna.test(xtest, ytest)
    grafsemiwarna = fc.grafPrediksi(modelsemiwarna.arrayofY, ytest)

    prediksisw = np.trunc(fc.denormalisasi(modelsemiwarna.arrayofY,np.array(data_semiwarna).min(), np.array(data_semiwarna).max())).astype(int)
    actualsw = np.trunc(fc.denormalisasi(ytest[:,0],np.array(data_semiwarna).min(), np.array(data_semiwarna).max())).astype(int)

    tabel = {"Tanggal": dateYtest[:,0],"Prediksi" : prediksisw, "Actual" : actualsw, "error":np.absolute(prediksisw-actualsw)}
    errsw = modelsemiwarna.msetest
    #sview on streamlit
    st.write(f"ARSITEKTUR : {arch_selectsw}")
    st.write(f"Learning Rate : {lr_selectsw}, Momentum : {momentum_selectsw}, MSE : 0.001")
    st.write("MSE :", str(errsw))
    b=st.pyplot(grafsemiwarna)
    st.table(tabel)

