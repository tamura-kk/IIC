from genericpath import exists
import cv2
import glob
import os
import sys

files_datas = glob.glob("C:/Users/nougata-share-pc/Desktop/Tamura/IIC/dataset/color_norm/*/*/*.jpg")
print(len(files_datas))


def get_ave_pix(im, h, w):
    b_ave, g_ave, r_ave = 0, 0, 0
    l = 0
    for i in range(h):
        for j in range(w):
            l+=1
            b_ave += im[i, j, 0]
            g_ave += im[i, j, 1]
            r_ave += im[i, j, 2]
    b_ave = b_ave/l
    g_ave = g_ave/l
    r_ave = r_ave/l
    return r_ave, g_ave, b_ave


#################患者ごと###################
for fn in files_datas:
    
    fn_name = os.path.basename(fn) #jpg
    fn_human = os.path.basename(os.path.dirname(fn)) #human
    fn_type = os.path.basename(os.path.dirname(os.path.dirname(fn))) #diabetes
    
    #出力のpath#####################################################################################
    dir_path = 'padding_img/' + fn_type + '/' + fn_human
    ##############################################################################################
    
    os.makedirs(dir_path,exist_ok=True)
    
    
    img = cv2.imread(fn)
    h, w = img.shape[:2]
    R, G, B = get_ave_pix(img, h, w)
    
    if h > w:
        yohaku = h-w
        right = yohaku//2
        left = yohaku-right
        new_img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(R, G, B))
        #new_img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)) #黒の余白
        cv2.imwrite(dir_path+'/'+fn_name, new_img)
    else:
        yohaku = w-h
        top = yohaku//2
        bottom = yohaku-top
        new_img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(R, G, B))
        #new_img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.imwrite(dir_path+'/'+fn_name, new_img)
        
    print(fn)
        
    