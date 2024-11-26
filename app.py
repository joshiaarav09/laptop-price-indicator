import streamlit as st
import pickle as pkl
import numpy as np


# Importing the Model
pipe = pkl.load(open('pipe.pkl','rb'))
df = pkl.load(open('df.pkl','rb'))

st.title('Laptop Price Predictor')

# Brand Selection
company = st.selectbox('Brand',df['Company'].unique())

# Type of laptop Selection
type = st.selectbox('Type', df['TypeName'].unique())

# Ram Selection
ram = st.selectbox('Ram(in GB)',[2,4,6,8,12,16,32,64])

# Weight Selection
weight = st.number_input('Weight of the Laptop')

# TouchScreen Selection
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS Display or not selection
ips = st.selectbox('IPSS',['No','Yes'])

# Screensize
screen_size = st.slider('Screen Size in inches', 10.0, 18.0, 13.0)

# Resolution Selection
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
                                                '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# CPU Selection
cpu = st.selectbox('CPU',df['Cpu Brand'].unique())

# HDD Selection
hdd = st.selectbox('HDD()in GB',[0,128,256,512,1024,2048])

# SSD Selection
ssd = st.selectbox('SSD(in GB)',[0,128,256,512,1024,2048])

# Gpu Selection
gpu = st.selectbox('GPU',df['Gpu Brand'].unique())

# OS Selection
os = st.selectbox('Operating System',df['Os'].unique())

if st.button('Predict Price'):
    # Making input query
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])

    ppi = ((X_res**2) + (Y_res ** 2))**0.5/screen_size
    # Converting values of touchscreen and ppi into 1 for Yes and 0 for No
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    st.title("The Price is " +  str(int(np.exp(pipe.predict(query)[0]))))



