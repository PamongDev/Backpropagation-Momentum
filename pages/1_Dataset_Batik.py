import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import random as rdm
import json as js
import function as fc
# st.set_page_config(layout="wide")


st.set_page_config('Dataset Batik',layout='wide')
st.title("Dataset Batik 2001-2020")

file="datapenjualan_.csv"
data = pd.read_csv(file)
data = data.drop(data.columns[0], axis=1)
# data = data.drop([0])
kolom = ['batikcap', 'semiwarna', 'batiktulis','semitulis']
data.columns = kolom
data[['batikcap', 'semiwarna', 'batiktulis','semitulis']] = data[['batikcap', 'semiwarna', 'batiktulis','semitulis']].astype('int64')

date = pd.read_csv("tanggal.csv")
date.columns = ["tanggal", "none"]
date = date.drop("none", axis=1)
batikcap, semiwarna, batiktulis, semitulis = data[['batikcap']], data[['semiwarna']], data[['batiktulis']], data[['semitulis']]
# datainput = xinput, yinput=fc.convToDataProccess(semiwarna,5)

# st.dataframe(data, width=1000, height=500)
tab1, tab2, tab3=st.tabs(["Dataset","Data Normalization", "Pola Input dan Target"])
with tab1:
    st.dataframe(data,height=600, width=900)
with tab2 :
    st.dataframe(fc.normalization(data),height=600, width=900)
with tab3:
    datainput =xinput, yinput=fc.convToDataProccess(date,4)
    st.write("Data Input:")
    st.dataframe(fc.toPandas(xinput,"x"),height=600, width=900)
    st.write("Data Target:")
    st.dataframe(fc.toPandas(yinput,"y"),height=600, width=900)

