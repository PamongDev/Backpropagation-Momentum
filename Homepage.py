import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import random as rdm
import json as js
import streamlit as st
import function as fc
import time
from PIL import Image

st.set_page_config('Dashboard',layout='wide')
st.title("Backpropogation Prediksi Produksi Batik CV. Naraya")

dataset = "<p style='text-align: justify;'> Pada skripsi ini menggunakan dataset penjualan batik dari wawancara terhadap pemilik batik CV NARAYA kecamatan Tanjung Bumi. Dataset tersebut merupakan data mingguan penjualan batik sebanyak 1.000 data dimulai dari Januari 2001 hingga Maret 2020. Atribut yang digunakan pada skripsi ini yaitu penjualan batik cap, semi warna, tulis, dan semi tulis dengan tipe data numerik</p>"
backpro = "<p style='text-align: justify;'> Backpropagation adalah metode yang biasa digunakan pada berbagai bidang seperti peramalan, optimasi dan pengenalan pola karena metode ini termasuk pembelajaran yang terawasi (Supervised Learning). Metode Backpropagation memiliki tujuan agar memperoleh keseimbangan antara kemampuan jaringan untuk mengidentifikasi pola dipakai selama proses pelatihan (Training) beserta kemampuan jaringan untuk memberikan respon yang benar mengenai pola inputan yang berbeda dengan pola inputan pelatihan.</p>"
skenario = {"Parameter":["Skenario 1","Skenario 2","Skenario 3"], "Input Layer": [4,6,8], "Hidden Layer":[7,6,13], "Learning rate" : [0.2, 0.02,0.002], "Momentum" : [0.6, 0.3, 0.9]
}
skenario_pd = pd.DataFrame(skenario)
flowchart = Image.open("media/flow.png")
dokumentasi = Image.open("media/dokumentasi.jpg")
st.write("**Dataset**")
st.write(dataset, unsafe_allow_html=True)
st.write("**Backpropagation**")
st.write(backpro, unsafe_allow_html=True)
st.write("**Skenario**")
st.dataframe(skenario_pd)
st.write("**Diagram Alir**")
st.image(flowchart)
st.write("**Dokumentasi**")
st.image(dokumentasi,width=500)



