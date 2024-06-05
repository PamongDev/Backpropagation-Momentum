# import pages.Testing_Backpropagation as test
# import pages.Dataset_Batik as dt 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import random as rdm
import json as js
import streamlit as st
import function as fc
import os
import math
test = __import__('pages.3_Testing_Backpropagation', fromlist=[''])
dt = __import__('pages.1_Dataset_Batik', fromlist=[''])

st.set_page_config('Prediksi Backpropagation',layout='wide')


st.title("Prediksi Backpropagation Data Tunggal")
# tbatikcap, tbatiktulis, tsemitulis, tsemiwarna=st.tabs(["Batik Cap", "Batik Tulis", "Semi Tulis", "Semi Warna"])


#fungsi data batik konversi
def Conv(batik, arch):
    xinput, yinput=fc.convToDataProccess(batik, arch[0])
    dataproses = fc.normalization(xinput), fc.normalization(yinput)
    xtrain, ytrain, xtest, ytest = fc.dataSplitting(dataproses ,0.7)
    return xtrain, ytrain, xtest, ytest 

model = os.listdir("training")
arch = [4,7,1], [6,6,1], [8,13,1]
learning_rate = 0.2, 0.02, 0.002
momentum = 0.6, 0.3, 0.9

data_batikcap, data_batiktulis, data_semitulis, data_semiwarna = dt.batikcap, dt.batiktulis, dt.semitulis, dt.semiwarna
batikcap, batiktulis, semitulis, semiwarna=st.tabs(["Batik Cap", "Batik Tulis", "Semi Tulis", "Semi Warna"])

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
    except: 
        arch_selectbc = [4,7,1]     
        lr_selectbc = 0.2
        momentum_selectbc = 0.6
        xtrain, ytrain, xtest, ytest = Conv(data_batikcap, arch_selectbc)
        modelBatikcap = fc.backpropagation(arch_selectbc, lr_selectbc, momentum_selectbc, 1000, 0.001)
    #load model
    for path in model:
        if batik in path and str(arch_selectbc) in path and str(lr_selectbc) in path and str(momentum_selectbc) in path:
            path_select = path
            break
    modelBatikcap.loadModel("training3/"+path)
    modelBatikcap.test(xtest,ytest)
    #make a form
    num_col = arch_selectbc[0]//2
    kolom = st.columns(num_col)
    colom = st.columns(num_col)
    listInputbc =[]
    for i in range(num_col):
        kolom[i]=kolom[i].text_input(f"Penjualan Hari ke-{i+1} :", key=f"inputkey{i}")
    for j in range(num_col):
        colom[j]=colom[j].text_input(f"Penjualan Hari ke-{i+j+2} :", key=f"inputkol{j}")    

    try:
        kolom1=np.array(kolom).reshape(-1,1).astype('int64')   
        colom1=np.array(colom).reshape(-1,1).astype('int64')   
        listInputbc=np.concatenate((kolom1,colom1),axis=0)
        hasilprediksi=modelBatikcap.predict(listInputbc.astype('int64'))
    except:
        listInputbc = []
        for j in range(arch_selectbc[0]):
            listInputbc.append(0)
        listInputbc=np.array(listInputbc).reshape(-1,1).astype('int64')    
        hasilprediksi=modelBatikcap.predict(listInputbc.astype('int64'))
    if np.any(listInputbc==0):
        st.write("Hasil Prediksi : ")
    else:
        # st.write(f"Hasil Prediksi :{listInput}")
        st.write(f" Hasil Prediksi : {round(fc.denormalisasi(hasilprediksi, np.array(data_batikcap).min(), np.array(data_batikcap).max())[0,0])}")
        
with batiktulis:
    #sview on streamlit
    batik = "batiktulis"
    arch_selectbt =st.selectbox("Arsitektur :",arch,key="archbt")
    lr_selectbt =st.selectbox("Learning Rate :", learning_rate,key="lrbt")
    momentum_selectbt =st.selectbox("Momentum :", momentum, key="momentbt")
    #make a model
    try:
        xtrain, ytrain, xtest, ytest = Conv(data_batiktulis, arch_selectbt)
        modelBatiktulis = fc.backpropagation(arch_selectbt, lr_selectbt, momentum_selectbt, 1000, 0.001)
    except: 
        arch_selectbt = [4,7,1]     
        lr_selectbt = 0.2
        momentum_selectbt = 0.6
        xtrain, ytrain, xtest, ytest = Conv(data_batiktulis, arch_selectbt)
        modelBatikcap = fc.backpropagation(arch_selectbt, lr_selectbt, momentum_selectbt, 1000, 0.001)
    #load model
    for path in model:
        if batik in path and str(arch_selectbt) in path and str(lr_selectbt) in path and str(momentum_selectbt) in path:
            path_select = path
            break
    modelBatiktulis.loadModel("training3/"+path)
    modelBatiktulis.test(xtest,ytest)
    #make a form
    num_col = arch_selectbt[0]//2
    kolombt = st.columns(num_col)
    colombt = st.columns(num_col)
    listInputbt =[]
    for i in range(num_col):
        kolombt[i]=kolombt[i].text_input(f"Penjualan Hari ke-{i+1} :", key=f"inputkeybt{i}")
    for j in range(num_col):
        colombt[j]=colombt[j].text_input(f"Penjualan Hari ke-{i+j+2} :", key=f"inputkolbt{j}")    

    try:
        kolom1=np.array(kolombt).reshape(-1,1).astype('int64')   
        colom1=np.array(colombt).reshape(-1,1).astype('int64')   
        listInputbt=np.concatenate((kolom1,colom1),axis=0)
        hasilprediksi=modelBatiktulis.predict(listInputbt.astype('int64'))
    except:
        listInputbt = []
        for j in range(arch_selectbt[0]):
            listInputbt.append(0)
        listInputbt=np.array(listInputbt).reshape(-1,1).astype('int64')    
        hasilprediksi=modelBatiktulis.predict(listInputbt.astype('int64'))
    if np.any(listInputbt==0):
        st.write("Hasil Prediksi : ")
    else:
        # st.write(f"Hasil Prediksi :{listInput}")
        st.write(f" Hasil Prediksi : {round(fc.denormalisasi(hasilprediksi, np.array(data_batiktulis).min(), np.array(data_batiktulis).max())[0,0])}")

with semitulis:
    #sview on streamlit
    batik = "semitulis"
    arch_selectst =st.selectbox("Arsitektur :",arch,key="archst")
    lr_selectst =st.selectbox("Learning Rate :", learning_rate,key="lrst")
    momentum_selectst =st.selectbox("Momentum :", momentum, key="momentst")
    #make a model
    try:
        xtrain, ytrain, xtest, ytest = Conv(data_semitulis, arch_selectst)
        modelsemitulis = fc.backpropagation(arch_selectst, lr_selectst, momentum_selectst, 1000, 0.001)
    except: 
        arch_selectst = [4,7,1]     
        lr_selectst = 0.2
        momentum_selectst = 0.6
        xtrain, ytrain, xtest, ytest = Conv(data_semitulis, arch_selectst)
        modelBatikcap = fc.backpropagation(arch_selectst, lr_selectst, momentum_selectst, 1000, 0.001)
    #load model
    for path in model:
        if batik in path and str(arch_selectst) in path and str(lr_selectst) in path and str(momentum_selectst) in path:
            path_select = path
            break
    modelsemitulis.loadModel("training3/"+path)
    modelsemitulis.test(xtest,ytest)
    #make a form
    num_col = arch_selectst[0]//2
    kolomst = st.columns(num_col)
    colomst = st.columns(num_col)
    listInputst =[]
    for i in range(num_col):
        kolomst[i]=kolomst[i].text_input(f"Penjualan Hari ke-{i+1} :", key=f"inputkeyst{i}")
    for j in range(num_col):
        colomst[j]=colomst[j].text_input(f"Penjualan Hari ke-{i+j+2} :", key=f"inputkolst{j}")    

    try:
        kolom1=np.array(kolomst).reshape(-1,1).astype('int64')   
        colom1=np.array(colomst).reshape(-1,1).astype('int64')   
        listInputst=np.concatenate((kolom1,colom1),axis=0)
        hasilprediksi=modelsemitulis.predict(listInputst.astype('int64'))
    except:
        listInputst = []
        for j in range(arch_selectst[0]):
            listInputst.append(0)
        listInputst=np.array(listInputst).reshape(-1,1).astype('int64')    
        hasilprediksi=modelsemitulis.predict(listInputst.astype('int64'))
    if np.any(listInputst==0):
        st.write("Hasil Prediksi : ")
    else:
        st.write(f"Hasil Prediksi :")
        st.write(f" Hasil Prediksi : {round(fc.denormalisasi(hasilprediksi, np.array(data_semitulis).min(), np.array(data_semitulis).max())[0,0])}")

with semiwarna:
    #sview on streamlit
    batik = "semiwarna"
    arch_selectsw =st.selectbox("Arsitektur :",arch,key="archsw")
    lr_selectsw =st.selectbox("Learning Rate :", learning_rate,key="lrsw")
    momentum_selectsw =st.selectbox("Momentum :", momentum, key="momentsw")
    #make a model
    try:
        xtrain, ytrain, xtest, ytest = Conv(data_semiwarna, arch_selectsw)
        modelsemiwarna = fc.backpropagation(arch_selectsw, lr_selectsw, momentum_selectsw, 1000, 0.001)
    except: 
        arch_selectsw = [4,7,1]     
        lr_selectsw = 0.2
        momentum_selectsw = 0.6
        xtrain, ytrain, xtest, ytest = Conv(data_semiwarna, arch_selectsw)
        modelsemiwarna = fc.backpropagation(arch_selectsw, lr_selectsw, momentum_selectsw, 1000, 0.001)
    #load model
    for path in model:
        if batik in path and str(arch_selectsw) in path and str(lr_selectsw) in path and str(momentum_selectsw) in path:
            path_select = path
            break
    modelsemiwarna.loadModel("training3/"+path)
    modelsemiwarna.test(xtest,ytest)
    #make a form
    num_col = arch_selectsw[0]//2
    kolomsw = st.columns(num_col)
    colomsw = st.columns(num_col)
    listInputst =[]
    for i in range(num_col):
        kolomsw[i]=kolomsw[i].text_input(f"Penjualan Hari ke-{i+1} :", key=f"inputkeysw{i}")
    for j in range(num_col):
        colomsw[j]=colomsw[j].text_input(f"Penjualan Hari ke-{i+j+2} :", key=f"inputkolsw{j}")    

    try:
        kolom1=np.array(kolomsw).reshape(-1,1).astype('int64')   
        colom1=np.array(colomsw).reshape(-1,1).astype('int64')   
        listInputsw=np.concatenate((kolom1,colom1),axis=0)
        hasilprediksi=modelsemiwarna.predict(listInputsw.astype('int64'))
    except:
        listInputsw = []
        for j in range(arch_selectsw[0]):
            listInputsw.append(0)
        listInputsw=np.array(listInputsw).reshape(-1,1).astype('int64')    
        hasilprediksi=modelsemiwarna.predict(listInputsw.astype('int64'))
    if np.any(listInputsw==0):
        st.write("Hasil Prediksi : ")
    else:
        st.write(f"Hasil Prediksi :")
        st.write(f" Hasil Prediksi : {round(fc.denormalisasi(hasilprediksi, np.array(data_semiwarna).min(), np.array(data_semiwarna).max())[0,0])}")

# #batiktulis
# with batiktulis:
#     batik = "batiktulis"    
#     arch_selectbt =st.selectbox("Arsitektur :",arch, key="archbt")
#     lr_selectbt =st.selectbox("Learning Rate :", learning_rate, key="lrbt")
#     momentum_selectbt =st.selectbox("Momentum :", momentum, key="momentbt")
#     #make a model
#     try:
#         xtrain, ytrain, xtest, ytest = Conv(data_batiktulis, arch_selectbt)
#         modelBatiktulis = fc.backpropagation(arch_selectbt, lr_selectbt, momentum_selectbt, 1000, 0.001)
#     except:
#         arch_selectbt = [4,7,1]
#         lr_selectbt = 0.2
#         momentum_selectbt = 0.6
#         xtrain, ytrain, xtest, ytest = Conv(data_batiktulis, arch_selectbt)
#         modelBatiktulis = fc.backpropagation(arch_selectbt, lr_selectbt, momentum_selectbt, 1000, 0.001)
#     for path in model:
#         if batik in path and str(arch_selectbt) in path and str(lr_selectbt) in path and str(momentum_selectbt) in path:
#             path_select = path
#             break
#     modelBatiktulis.loadModel("training/"+path)
#     modelBatiktulis.test(xtest,ytest)
#     grafbatiktulis = fc.grafPrediksi(modelBatiktulis.arrayofY, ytest)
#     tabel = {"Prediksi" : modelBatiktulis.arrayofY, "Actual" : ytest[:,0], "error":modelBatiktulis.arrayofY-ytest[:,0]}
#     errbt = modelBatiktulis.msetest
#     #sview on streamlit
#     st.write(f"ARSITEKTUR : {arch_selectbt}")
#     st.write(f"Learning Rate : {lr_selectbt}, Momentum : {momentum_selectbt}, MSE : 0.001")
#     st.write("MSE  :", str(errbt))
#     b=st.pyplot(grafbatiktulis)
#     st.table(tabel)
# with semitulis:
#     batik = "semitulis"

#     arch_selectst=st.selectbox("Arsitektur :",arch, key="archst")
#     lr_selectst=st.selectbox("Learning Rate :", learning_rate, key="lrst")
#     momentum_selectst=st.selectbox("Momentum :", momentum, key="momentst")
#     #make a model
#     try:
#         xtrain, ytrain, xtest, ytest = Conv(data_semitulis, arch_selectst)
#         modelsemitulis = fc.backpropagation(arch_selectst, lr_selectst, momentum_selectst, 1000, 0.001)
#     except:
#         arch_selectst = [4,7,1]
#         lr_selectst = 0.2
#         momentum_selectst = 0.6
#         xtrain, ytrain, xtest, ytest = Conv(data_semitulis, arch_selectst)
#         modelsemitulis = fc.backpropagation(arch_selectst, lr_selectst, momentum_selectst, 1000, 0.001)
#     for path in model:
#         if batik in path and str(arch_selectst) in path and str(lr_selectst) in path and str(momentum_selectst) in path:
#             path_select = path
#             break
#     modelsemitulis.loadModel("training/"+path)
#     modelsemitulis.test(xtest, ytest)
#     grafsemitulis = fc.grafPrediksi(modelsemitulis.arrayofY, ytest)
#     tabel = {"Prediksi" : modelsemitulis.arrayofY, "Actual" : ytest[:,0], "error":modelsemitulis.arrayofY-ytest[:,0]}
#     errst = modelsemitulis.msetest
#     #sview on streamlit
#     st.write(f"ARSITEKTUR : {arch_selectst}")
#     st.write(f"Learning Rate : {lr_selectst}, Momentum : {momentum_selectst}, MSE : 0.001")
#     st.write("MSE  :", str(errst))
#     b=st.pyplot(grafsemitulis)
#     st.table(tabel)
# with semiwarna:
#     batik = "semiwarna"
#     arch_selectsw=st.selectbox("Arsitektur :",arch, key="archsw")
#     lr_selectsw=st.selectbox("Learning Rate :", learning_rate, key="lrsw")
#     momentum_selectsw=st.selectbox("Momentum :", momentum, key="momentsw")

#     #make a model
#     try:
#         xtrain, ytrain, xtest, ytest = Conv(data_semiwarna, arch_selectsw)
#         modelsemiwarna = fc.backpropagation(arch_selectsw, lr_selectsw, momentum_selectsw, 1000, 0.001)
#     except:
#         arch_selectsw = [4,7,1]
#         lr_selectsw = 0.2
#         momentum_selectsw = 0.6
#         xtrain, ytrain, xtest, ytest = Conv(data_semiwarna, arch_selectsw)
#         modelsemiwarna = fc.backpropagation(arch_selectsw, lr_selectsw, momentum_selectsw, 1000, 0.001)

#     for path in model:
#         if batik in path and str(arch_selectsw) in path and str(lr_selectsw) in path and str(momentum_selectsw) in path:
#             path_select = path
#             break
#     modelsemiwarna.loadModel("training/"+path)
#     modelsemiwarna.test(xtest, ytest)
#     grafsemiwarna = fc.grafPrediksi(modelsemiwarna.arrayofY, ytest)
#     tabel = {"Prediksi" : modelsemiwarna.arrayofY, "Actual" : ytest[:,0], "error":modelsemiwarna.arrayofY-ytest[:,0]}
#     errsw = modelsemiwarna.msetest
#     #sview on streamlit
#     st.write(f"ARSITEKTUR : {arch_selectsw}")
#     st.write(f"Learning Rate : {lr_selectsw}, Momentum : {momentum_selectsw}, MSE : 0.001")
#     st.write("MSE :", str(errsw))
#     b=st.pyplot(grafsemiwarna)
#     st.table(tabel)

