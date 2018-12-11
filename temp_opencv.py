# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:11:01 2018

@author: 資管碩一 李彥羲 7107029011
Using python + openCV;
To execute use shift + enter;
"""
# In[1] 灰階
import cv2;
import numpy as np;
#from matplotlib import pyplot as plt;

img = cv2.imread('lena_color.tiff');
#rgbIm = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); #轉換成灰階
cv2.imshow('Gray', gray)
cv2.waitKey(0);  # 按下任意鍵則關閉所有視窗
cv2.destroyWindow('Gray');
# In[2] 負片
negative = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
n = negative[:,:];
n = 255 - n;
negative[:,:] = n;
cv2.imshow('Negative', negative)
cv2.waitKey(0);  # 按下任意鍵則關閉所有視窗
cv2.destroyWindow('Negative');
    
# In[3] Gamma<1
gamma = 0.5;
def gammaCorrection():
    lookUpTable = np.empty((1,256), np.uint8);
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255); # array 限制在 0~255

    res = cv2.LUT(gray, lookUpTable);  #一對一映射
    return res;
    
res=gammaCorrection();  #呼叫gamma函式
img_gamma_contrast = cv2.hconcat([gray, res]);
cv2.imshow("Gamma correction(<1) (contrast)", img_gamma_contrast);
cv2.waitKey(0);  # 按下任意鍵則關閉所有視窗
cv2.destroyWindow('Gamma correction(<1) (contrast)');

# In[4] Gamma<1 --> Salt and pepper
from skimage.util import random_noise;
salt_pepper = random_noise(res, mode='s&p',amount=0.09,salt_vs_pepper=0.3); # salt_vs_pepper代表鹽與胡椒的比值[0,1]
cv2.imshow("Salt and pepper noise",salt_pepper);
cv2.waitKey(0);  # 按下任意鍵則關閉所有視窗
cv2.destroyWindow('Salt and pepper noise');
'''
for i in range(2000): #方法二
    temp_x = np.random.randint(0,gray.shape[0])
    temp_y = np.random.randint(0,gray.shape[1])
    gray[temp_x][temp_y] = 255    # salt
    gray[temp_x][temp_y] = 0    # pepper
'''
# In[5] Gamma<1 --> Salt and pepper --> 3x3 中值濾波器
from skimage import img_as_ubyte;
salt_pepper = img_as_ubyte(salt_pepper); # float轉unit8

blurred = cv2.medianBlur(salt_pepper, 3);
cv2.imshow("3x3 Median Filter",blurred);
cv2.waitKey(0);  # 按下任意鍵則關閉所有視窗
cv2.destroyWindow('3x3 Median Filter');


# In[6] 直方圖均化(對比拉開)
equ = cv2.equalizeHist(gray);
equ_contrast = cv2.hconcat([gray, equ]); # 與gray比較
cv2.imshow("Histogram Equalization (contrast)",equ_contrast);
cv2.waitKey(0);  # 按下任意鍵則關閉所有視窗
cv2.destroyWindow('Histogram Equalization (contrast)');
'''
#方法二 以numpy實現
lut = np.zeros(256, dtype = image.dtype )#空的查找表
 
hist,bins = np.histogram(image.flatten(),256,[0,256]) 
cdf = hist.cumsum() #累積直方圖
cdf_m = np.ma.masked_equal(cdf,0) #除去直方圖中的0值
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())#lut[i] = int(255.0 *p[i])
cdf = np.ma.filled(cdf_m,0).astype('uint8') #補0
#結果計算
NumPyLUT = cdf[image]
OpenCVLUT = cv2.LUT(image, cdf)
'''
# In[7] 直方圖均化 --> Laplacian邊緣偵測
laplacian = cv2.Laplacian(equ,cv2.CV_64F,ksize = 1); # cv2.BORDER_DEFAULT/ 1*1 filter
laplacian = cv2.convertScaleAbs(laplacian); # float64轉uint8

cv2.imshow("Laplacian",laplacian);
cv2.waitKey(0);  # 按下任意鍵則關閉所有視窗
cv2.destroyWindow('Laplacian');

# In[8] 直方圖均化 --> Laplacian邊緣偵測 --> 3x3 最大值濾波器
kernel = np.ones((3,3),np.uint8);
dilate = cv2.dilate(laplacian, kernel, iterations = 1);
dilate_contrast = cv2.hconcat([laplacian, dilate]);
cv2.imshow("3x3 Max Filter (contrast)",dilate_contrast);
cv2.waitKey(0);  # 按下任意鍵則關閉所有視窗
cv2.destroyWindow('3x3 Max Filter (contrast)'); 

# In[9] Gamma>1
gamma = 2;
def gammaCorrection():
    lookUpTable = np.empty((1,256), np.uint8);
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255); # array 限制在 0~255

    res = cv2.LUT(gray, lookUpTable);  #一對一映射
    return res;
   
res_2=gammaCorrection();  #呼叫gamma函式
img_gamma_contrast_2 = cv2.hconcat([gray, res_2]);
cv2.imshow("Gamma correction(>1) (contrast)", img_gamma_contrast_2);
cv2.waitKey(0);  # 按下任意鍵則關閉所有視窗
cv2.destroyWindow('Gamma correction(>1) (contrast)');

# In[10] Gamma>1 --> 二值化(用平均值當門檻值)
pixel_mean = res_2.mean(); # 計算像素平均值，當作門檻
(T, threshInv) = cv2.threshold(res_2, pixel_mean, 255, cv2.THRESH_BINARY); # 黑白反轉:THRESH_BINARY_INV

cv2.imshow("Thresholding", threshInv);
cv2.waitKey(0);  # 按下任意鍵則關閉所有視窗
cv2.destroyWindow('Thresholding');

