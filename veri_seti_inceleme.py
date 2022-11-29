import os #işletim sisteminden diske bağlanmak için
import matplotlib.pyplot as plt #fotolarımızı incelemek için grfk kütüphanesi
import cv2 #computer vision
from alg1_plaka_tespiti import plaka_konum_don

"""
#1. Alg veri inceleme
#---------------------
veri = os.listdir("veriseti") #bu klasördeki dosyalar
for image_url in veri:
    img = cv2.imread("veriseti/"+image_url)
    #"veriseti/1.jpg" #bgr değerlerine döndür
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#rgb renk uzayına dönüş
    img = cv2.resize(img, (500,500))
    plt.imshow(img)
    plt.show()
"""

#2. Alg veri inceleme
#---------------------
veri = os.listdir("veriseti") #bu klasördeki dosyalar
for image_url in veri:
    img = cv2.imread("veriseti/"+image_url)
    #"veriseti/1.jpg" #bgr değerlerine döndür
    img = cv2.resize(img, (500,500))
    plaka = plaka_konum_don(img) #x,y,w,h değerleri gelcek
    x,y,w,h = plaka
    if(w>h): #bazen plaka h w degerleri tam tersi gelebilir mesela plaka 90 derece dik
        plaka_bgr = img[y:y+h,x:x+w].copy()
    else:
        plaka_bgr = img[y:y+w,x:x+h].copy()
    img = cv2.cvtColor(plaka_bgr,cv2.COLOR_BGR2RGB)#rgb renk uzayına dönüş
    plt.imshow(img)
    plt.show()
