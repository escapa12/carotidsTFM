import PIL.Image as pilimage
import numpy as np
import os

def im_adjust(path_in,path_out):
    im=pilimage.open(path_in)
    im=np.asarray(im)
    im=im[:,:,0] if len(im.shape)==3 else im
    #print(im.shape,im.min(),im.max())
    hist,num=np.histogram(im,bins=255)
    freq=np.cumsum(hist)/float(np.sum(hist)) 
    #print(hist)
    llindar=0.01
    max_argument=np.argmin(np.abs(freq-(1-llindar)))
    min_argument=np.argmin(np.abs(freq-llindar)) if freq[0]<llindar else 0
    #print(max_argument,min_argument)
    im2=im*(im<max_argument)+max_argument*(im>=max_argument)
    im2=im2*((im2-min_argument)>0)
    im2=im2-min_argument
    im2=(im2/float(max_argument-min_argument)*255)
    im2=im2.astype(dtype='uint8')
    im_out=pilimage.fromarray(im2)
    im_out=im_out.resize((884,825),resample=pilimage.BICUBIC)
    im_out.save(path_out)

path='/media/HDD3TB/data_carotid/NEFRONA_crops/ACC/parelles_imt/original/'
out_path='/media/HDD3TB/data_carotid/adjusted_ENRIC/parelles_imt/original_resolucio_enric/'

for imatge in os.listdir(path):
	im_adjust(path+imatge,out_path+imatge)
    
path='/media/HDD3TB/data_carotid/NEFRONA_crops/ACC/parelles_pl/original/'
out_path='/media/HDD3TB/data_carotid/adjusted_ENRIC/parelles_pl/original_resolucio_enric/'

for imatge in os.listdir(path):
	im_adjust(path+imatge,out_path+imatge)

