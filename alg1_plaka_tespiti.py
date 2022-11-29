import os
import cv2
import matplotlib.pyplot as plt
import numpy as np #liste işlemlerini kolaylıkla ve hızlı yapma, ramden tasarruf (500-600 kat daha hızlı), içinde bir çok fonks var

resim_adresler = sorted(os.listdir("veriseti"), key=lambda x: int(os.path.splitext(x)[0]))
print(resim_adresler)

img = cv2.imread("veriseti/"+resim_adresler[0])
img = cv2.resize(img, (500,500))
"""
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()

img_bgr = img
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray,cmap="gray")
plt.show()

# islem resmi ir_img

ir_img = cv2.medianBlur(img_gray,5) #5x5 boyutları olacak şekilde çalıştıracagız
ir_img = cv2.medianBlur(ir_img,5) #5x5 bir daha

plt.imshow(ir_img,cmap="gray")
plt.show()

medyan = np.median(ir_img)
low = 0.67*medyan
high = 1.33*medyan

kenarlik = cv2.Canny(ir_img,low,high)

plt.imshow(kenarlik,cmap="gray")
plt.show()

#np.ones((3,3),np.uint8) --> [[1,1,1],[1,1,1],[1,1,1]]
kenarlik = cv2.dilate(kenarlik,np.ones((3,3),np.uint8),iterations=1)

plt.imshow(kenarlik,cmap="gray")
plt.show()

# cv2.RETR_TREE --> hiyerarşik yapıyı anlatmaktadır
# CHAIN_APPROX_SIMPLE --> kosegen
cnt = cv2.findContours(kenarlik,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt = cnt[0]
cnt = sorted(cnt,key=cv2.contourArea,reverse=True)

H,W = 500,500
plaka = None

#hangisi plaka hangisi değil algılama
for c in cnt:
    rect = cv2.minAreaRect(c) #dikdörtgenleri tespit et
    (x,y),(w,h),r = rect
    if(w>h and w>h*2) or (h>w and h>w*2): #dörtgenin bir kenar boyutu iki katından daha yüksek
        box = cv2.boxPoints(rect) #[[12,13],[25,13],[20,13],[13,45]]
        box = np.int64(box)

        minx =np.min(box[:,0])
        miny =np.min(box[:,1])
        maxx =np.max(box[:,0])
        maxy =np.max(box[:,1])

        muh_plaka = img_gray[miny:maxy,minx:maxx].copy()
        muh_medyan = np.median(muh_plaka)

        kon1 = muh_medyan>84 and muh_medyan<200 #yogunluk kontrolu
        kon2 = h<50 and w<150 #sınır kontrolü
        kon3 = w<50 and h<150 #sınır kontrolü

        print(f"muh_plaka medyan:{muh_medyan} genislik:{w} yukseklik:{h}")

        plt.figure()
        kon=False
        if(kon1 and (kon2 or kon3)):
            #plakadır
            cv2.drawContours(img,[box],0,(0,255,0),2)
            plaka=[int(i) for i in [minx,miny,w,h]] #x,y,w,h
            
            plt.title("plaka tespit edildi!!")
            kon=True
        else:
            #plaka değildir
            cv2.drawContours(img,[box],0,(0,0,255),2)
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        plt.show()

        if(kon):
            break
        #plaka bulunmuştur!!
"""
def plaka_konum_don(img):
    ##plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    ##plt.show()

    img_bgr = img
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ##plt.imshow(img_gray,cmap="gray")
    ##plt.show()

    # islem resmi ir_img

    ir_img = cv2.medianBlur(img_gray,5) #5x5 boyutları olacak şekilde çalıştıracagız
    ir_img = cv2.medianBlur(ir_img,5) #5x5 bir daha

    ##plt.imshow(ir_img,cmap="gray")
    ##plt.show()

    medyan = np.median(ir_img)
    low = 0.67*medyan
    high = 1.33*medyan

    kenarlik = cv2.Canny(ir_img,low,high)

    ##plt.imshow(kenarlik,cmap="gray")
    ##plt.show()

    #np.ones((3,3),np.uint8) --> [[1,1,1],[1,1,1],[1,1,1]]
    kenarlik = cv2.dilate(kenarlik,np.ones((3,3),np.uint8),iterations=1)

    ##plt.imshow(kenarlik,cmap="gray")
    ##plt.show()

    # cv2.RETR_TREE --> hiyerarşik yapıyı anlatmaktadır
    # CHAIN_APPROX_SIMPLE --> kosegen
    cnt = cv2.findContours(kenarlik,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnt[0]
    cnt = sorted(cnt,key=cv2.contourArea,reverse=True)

    H,W = 500,500
    plaka = None

    #hangisi plaka hangisi değil algılama
    for c in cnt:
        rect = cv2.minAreaRect(c) #dikdörtgenleri tespit et
        (x,y),(w,h),r = rect
        if(w>h and w>h*2) or (h>w and h>w*2): #dörtgenin bir kenar boyutu iki katından daha yüksek
            box = cv2.boxPoints(rect) #[[12,13],[25,13],[20,13],[13,45]]
            box = np.int64(box)

            minx =np.min(box[:,0])
            miny =np.min(box[:,1])
            maxx =np.max(box[:,0])
            maxy =np.max(box[:,1])

            muh_plaka = img_gray[miny:maxy,minx:maxx].copy()
            muh_medyan = np.median(muh_plaka)

            kon1 = muh_medyan>84 and muh_medyan<200 #yogunluk kontrolu
            kon2 = h<50 and w<150 #sınır kontrolü
            kon3 = w<50 and h<150 #sınır kontrolü

            print(f"muh_plaka medyan:{muh_medyan} genislik:{w} yukseklik:{h}")

            ##plt.figure()
            kon=False
            if(kon1 and (kon2 or kon3)):
                #plakadır
                #cv2.drawContours(img,[box],0,(0,255,0),2)
                plaka=[int(i) for i in [minx,miny,w,h]] #x,y,w,h
                ##plt.title("plaka tespit edildi!!")
                kon=True
            #else:
                #plaka değildir
                #cv2.drawContours(img,[box],0,(0,0,255),2)
                
            ##plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            ##plt.show()

            if(kon):
                return plaka
    return[]


