# import pages.Dataset_Batik as dt 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import random as rdm
import json as js
import streamlit as st
import function as fc
import os
st.set_option('deprecation.showPyplotGlobalUse', False)
# dt = __import__('pages.1_Dataset_Batik')
st.set_page_config('Training Backpropagation',layout='wide')
st.title("Hasil Training Backpropagation Berbagai Skenario")
model = os.listdir("model_4")
arch = [4,7,1], [6,6,1], [8,13,1]
learning_rate = 0.2, 0.02, 0.002
momentum = 0.6, 0.3, 0.9

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
        modelBatikcap = fc.backpropagation(arch_selectbc, lr_selectbc, momentum_selectbc, 1000, 0.001)
    except: 
        arch_selectbc = [4,7,1]
        lr_selectbc = 0.2
        momentum_selectbc = 0.6
        modelBatikcap = fc.backpropagation(arch_selectbc, lr_selectbc, momentum_selectbc, 1000, 0.001)

    for path in model:
        if batik in path and str(arch_selectbc) in path and str(lr_selectbc) in path and str(momentum_selectbc) in path:
            path_select = path
            break
    modelBatikcap.loadModel("model4/"+path)
    grafbatikcap = modelBatikcap.graf()
    errbc = modelBatikcap.arrayError[-1]
    st.write(f"ARSITEKTUR : {arch_selectbc}")
    st.write(f"**Learning Rate** : {lr_selectbc}, **Momentum** : {momentum_selectbc}, **MSE** : 0.001")
    st.write("**MSE Epoch ke-1000** :", str(errbc))
    st.write("**time** : {:.3f}".format(modelBatikcap.waktu),'s')
    a=st.pyplot(grafbatikcap)
#batiktulis
with batiktulis:
    batik = "batiktulis"    
    arch_selectbt =st.selectbox("Arsitektur :",arch, key="archbt")
    lr_selectbt =st.selectbox("Learning Rate :", learning_rate, key="lrbt")
    momentum_selectbt =st.selectbox("Momentum :", momentum, key="momentbt")
    #make a model
    try:
        modelBatiktulis = fc.backpropagation(arch_selectbt, lr_selectbt, momentum_selectbt, 1000, 0.001)
    except:
        arch_selectbt = [4,7,1]
        lr_selectbt = 0.2
        momentum_selectbt = 0.6
        modelBatiktulis = fc.backpropagation(arch_selectbt, lr_selectbt, momentum_selectbt, 1000, 0.001)
    for path in model:
        if batik in path and str(arch_selectbt) in path and str(lr_selectbt) in path and str(momentum_selectbt) in path:
            path_select = path
            break
    modelBatiktulis.loadModel("model4/"+path)
    grafbatiktulis = modelBatiktulis.graf()
    errbt = modelBatiktulis.arrayError[-1]
    #sview on streamlit
    st.write(f"ARSITEKTUR : {arch_selectbt}")
    st.write(f"**Learning Rate** : {lr_selectbt}, **Momentum** : {momentum_selectbt}, **MSE** : 0.001")
    st.write("**MSE Epoch ke-1000** :", str(errbt))
    st.write("**time** : {:.3f}".format(modelBatiktulis.waktu),'s')
    b=st.pyplot(grafbatiktulis)
with semitulis:
    batik = "semitulis"

    arch_selectst=st.selectbox("Arsitektur :",arch, key="archst")
    lr_selectst=st.selectbox("Learning Rate :", learning_rate, key="lrst")
    momentum_selectst=st.selectbox("Momentum :", momentum, key="momentst")
    #make a model
    try:
        modelsemitulis = fc.backpropagation(arch_selectst, lr_selectst, momentum_selectst, 1000, 0.001)
    except:
        arch_selectst = [4,7,1]
        lr_selectst = 0.2
        momentum_selectst = 0.6
        modelsemitulis = fc.backpropagation(arch_selectst, lr_selectst, momentum_selectst, 1000, 0.001)
    for path in model:
        if batik in path and str(arch_selectst) in path and str(lr_selectst) in path and str(momentum_selectst) in path:
            path_select = path
            break
    modelsemitulis.loadModel("model4/"+path)
    grafsemitulis = modelsemitulis.graf()
    errst = modelsemitulis.arrayError[-1]
    #sview on streamlit
    st.write(f"ARSITEKTUR : {arch_selectst}")
    st.write(f"**Learning Rate** : {lr_selectst}, **Momentum** : {momentum_selectst}, **MSE** : 0.001")
    st.write("**MSE Epoch ke-1000** :", str(errst))
    st.write("**time** : {:.3f}".format(modelsemitulis.waktu),'s')
    b=st.pyplot(grafsemitulis)
with semiwarna:
    batik = "semiwarna"
    arch_selectsw=st.selectbox("Arsitektur :",arch, key="archsw")
    lr_selectsw=st.selectbox("Learning Rate :", learning_rate, key="lrsw")
    momentum_selectsw=st.selectbox("Momentum :", momentum, key="momentsw")

    #make a model
    try:
        modelsemiwarna = fc.backpropagation(arch_selectsw, lr_selectsw, momentum_selectsw, 1000, 0.001)
    except:
        arch_selectsw = [4,7,1]
        lr_selectsw = 0.2
        momentum_selectsw = 0.6
        modelsemiwarna = fc.backpropagation(arch_selectsw, lr_selectsw, momentum_selectsw, 1000, 0.001)

    for path in model:
        if batik in path and str(arch_selectsw) in path and str(lr_selectsw) in path and str(momentum_selectsw) in path:
            path_select = path
            break
    modelsemiwarna.loadModel("model4/"+path)
    grafsemiwarna = modelsemiwarna.graf()
    errsw = modelsemiwarna.arrayError[-1]
    #sview on streamlit
    st.write(f"ARSITEKTUR : {arch_selectsw}")
    st.write(f"**Learning Rate** : {lr_selectsw}, **Momentum** : {momentum_selectsw}, **MSE** : 0.001")
    st.write("**MSE Epoch ke-1000** :", str(errsw))
    st.write("**time** : {:.3f}".format(modelsemiwarna.waktu),'s')
    b=st.pyplot(grafsemiwarna)




