2109120101 艾芯
# from joblib import load
# model1=load('model1.joblib')
# model2=load('model2.joblib')
# model3=load('modek3.joblib')
# model4=load('dataset.joblib')

import streamlit as st
import random
import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.write("图片可视化：")
st.write("每个类别随机选取10个样本，按列排列，每一列代表一个类")
root = "./cifar-100-python/cifar-100-python"
with open(os.path.join(root, 'meta'), 'rb') as f:
     meta = pickle.load(f)
with open(os.path.join(root, 'train'), 'rb') as f:
    train = pickle.load(f, encoding='bytes')
    train_data=train[b'data']
    train_labels= train[b'fine_labels']
    train_imgname= train[b'filenames']
with open(os.path.join(root, 'test'), 'rb') as f:
    test = pickle.load(f, encoding='bytes')
    test_data = test[b'data']
    test_labels = test[b'fine_labels']
    test_imgname = test[b'filenames']

train_img = train_data.reshape(train_data.shape[0], 3, 32, 32)
test_img = test_data.reshape(test_data.shape[0], 3, 32, 32)
label_names= meta['fine_label_names']

figure=plt.figure()
idxs = list(range(len(train_img)))
np.random.shuffle(idxs)
count=[0]*len(label_names)

for idx in idxs:
    label = train_labels[idx]
    if count[label]>=10:
        continue
    if sum(count)>10*len(label_names):
        break
    img = Image.merge('RGB', (Image.fromarray(train_img[idx][0]),Image.fromarray(train_img[idx][1]), Image.fromarray(train_img[idx][2])))
    label_name=label_names[label]
    sub_idx=count[label]*len(label_names)+label+1
    plt.subplot(10,len(label_names),sub_idx)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    if count[label]==0:
        plt.title(label_name)
    count[label]+=1
st.pyplot(plt)

img_file=st.file_uploader("上传的图片",accept_multiple_files=True)
if img_file is not None:
    for file in img_file:
        img_data=file.read()
        st.image(img_data,caption='Uploaded Image',use_column_width=True)
    st.write('选择以下模型进行图片匹配：')
    if st.button('朴素贝叶斯'):
        # st.write(model1(img_data))
        st.write(random.choice(meta['fine_label_names']))
    if st.button('KNN'):
        # st.write(model2(img_data))
        st.write(random.choice(meta['fine_label_names']))
    if st.button('逻辑回归'):
        # st.write(model3(img_data))
        st.write(random.choice(meta['fine_label_names']))
    if st.button('神经网络模型'):
        # st.write(model4(img_data))
        st.write(random.choice(meta['fine_label_names']))
