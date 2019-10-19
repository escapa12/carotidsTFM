####SCRIPT DATASET Patch -hemodine CREATOR####
##Author:arnau

import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.image as mpimg
from PIL import Image
import math
def flat_1channel_img(im):
    im = im[:,:,0]*255
    im=im.astype(np.uint8)
    return im.flatten().tolist(),im

def black_background(img_original,GT):
    for i in range(GT.shape[0]):
        for j in range(GT.shape[1]):
            if (GT[i,j]==0):
                img_original[i,j]=[0,0,0]
    return img_original.flatten().tolist()

def img_placa_patch(orig,GT):
    patch_size= [160,90]
    inx=np.nonzero(GT)
    x1,x2=min(inx[0]),max(inx[0]);
    y1,y2=min(inx[1]),max(inx[1]);
    patch=orig[x1:x2,y1:y2]
    patch= Image.fromarray(patch)
    final = patch.resize((patch_size[0], patch_size[1]))
    return np.array(final)


df=pd.read_excel('../data/dadesFinal.xlsx')
df.set_index('Códigobiobanco', inplace=True)

cols_clas_placa=['placas_cc_d','placas_cc_i','placas_cc_d_24M','placas_cc_i_24M',
                 'clasif_cc_d','clasif_cc_i','clasif_cc_d_24','clasif_cc_i_24',
                 'lipidos_cc_d','fibrosis_cc_d','calcio_cc_d',
                 'lipidos_cc_i','fibrosis_cc_i','calcio_cc_i',
                 'trasplante1','EventoCV_Si_No','muerte']
df=df.loc[:,cols_clas_placa]

GT_path='../data/GT_ACC_PL/'
original_path='../data/NEFRONA_crops/ACC/parelles_pl/original/'

Df = pd.DataFrame(columns=['ID','img_original', 'patch', 'target','eventCV','mortCV','hemodine'])

i=0
j=0
Incong=0
Incong24=0
for file_name in listdir(original_path):
    F=file_name.split('_')
    if(F[1]=='D'):
        isD=True
    elif(F[1]=='E'):
        isD=False
    else:
        print('Error: ni dreta ni esquerra')
    Id=F[0].split('-')
    if (Id[1]=='AC2'):
        isBasal=False
    elif(Id[1]=='AC' or Id[1]=='AC1'):
        isBasal=True
    else:
        print('Error: not AC AC1 or AC2')

    orig_flat,orig = flat_1channel_img(mpimg.imread(original_path+file_name)) ### read original image. take 1 channel
    try:
        GT= np.asarray(Image.open(GT_path+file_name)) ### trey to read GT. It may not exist
        #img_placa=black_background(orig,GT) ###uncommend if you want black background images
      #  patch=img_placa_patch(,GT)
        patch=img_placa_patch(orig,GT).flatten().tolist()
    except:
        j+=1
        patch=np.nan

    if (isD and isBasal):
        ID = file_name[:4]+'_D'
        if(df['placas_cc_d'][int(file_name[:4])]!=1):
            Incong+=1
        tar=df['clasif_cc_d'][int(file_name[:4])]
        hem=df.loc[int(file_name[:4]),['lipidos_cc_d', 'fibrosis_cc_d', 'calcio_cc_d']].values.astype(np.float)/100
        if math.isnan(hem[0]):
            hem=np.nan
        else:
            hem=hem.tolist()
    if (isD and not isBasal):
        if(df['placas_cc_d_24M'][int(file_name[:4])]!=1):
            Incong24+=1
        ID = file_name[:4]+'_D_24'
        tar=df['clasif_cc_d_24'][int(file_name[:4])]
        hem=np.nan
    if (not isD and isBasal):
        if(df['placas_cc_i'][int(file_name[:4])]!=1):
            Incong+=1
        ID=file_name[:4]+'_E'
        tar = df['clasif_cc_i'][int(file_name[:4])]
        hem=df.loc[int(file_name[:4]),['lipidos_cc_d', 'fibrosis_cc_d', 'calcio_cc_d']].values.astype(np.float)/100
        if math.isnan(hem[0]):
            hem=np.nan
        else:
            hem=hem.tolist()
    if (not isD and not isBasal):
        if(df['placas_cc_i_24M'][int(file_name[:4])]!=1):
            Incong24+=1
        ID=file_name[:4]+'_E_24'
        tar = df['clasif_cc_i_24'][int(file_name[:4])]
        hem=np.nan

    muerte=df['muerte'][int(file_name[:4])]
    df['muerte'][int(file_name[:4])]

    if (muerte=='CV death'):
        CV=1
    else:
        CV=0

    eventCV=df['EventoCV_Si_No'][int(file_name[:4])]
    tranplantament=df['trasplante1'][int(file_name[:4])]

    if (tranplantament==0): ###si tenen transplantament no l'afegim
        row=[ID,orig_flat,patch,tar,eventCV,CV,hem]
        Df.loc[i]=row
        i+=1
print(i)
print('Incongruencies(basals i seguiment):',Incong,Incong24)


Df.to_csv("../data/patch_no_transp.csv")
