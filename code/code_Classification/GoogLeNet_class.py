import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.inception_v3 import InceptionV3,preprocess_input
#from keras.applications.xception import Xception, preprocess_input

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Flatten,Dropout,GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Input
from keras.optimizers import SGD,Nadam
from keras.callbacks import EarlyStopping, LearningRateScheduler


path_data  = '../data/RGB_orig.csv'
img_shape=[365, 391]
n_clas=3  #set to 3 o 5
GT_img=False


def to_categorical(y, num_classes=n_clas):
    """
    Converts a class vector of [1,2,3,4,5] to binary class matrix of 5 or 3 classes(1and2,3and4,5).
    """
    y = np.array(y, dtype='int').ravel()
    n = y.shape[0]
    y=y-1

    if (num_classes==3):
        for i in range(len(y)):
            k=y[i]
            if (k==0) or (k==1) :
                y[i]=0
            if (k==2) or (k==3) :
                y[i]=1
            if (k==4):
                y[i]=2
    elif (num_classes==5):
        y=y
    else:
        print('Wrong expected number of classes')
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def load_train_test(path_data = path_data,images_col='img_original',target_col='target', test_size=0.2,print_info=True):
    """Loads data"""

    df = pd.read_csv(path_data)

    if print_info:
        N=df.shape[0]
        target_miss=df[target_col].dropna().values.shape[0]
        img_miss=df[images_col].dropna().values.shape[0]
        print("Dataset number of samples:",N)
        print("Targets missing:",N-target_miss)
        print("Images missing:",N-img_miss)

    df = df[[images_col] + [target_col]]

    df = df.dropna()
    df[images_col] = df[images_col].apply(lambda im: np.fromstring(im[1:-1], sep=','))

    X = np.vstack(df[images_col].values).astype(np.float32)
    X = X.reshape(-1, 365, 391, 3)  #SHAPE IMG: (365, 391,3)
    X= preprocess_input(X) ##process input data in the same way than the pretrained network

    y =  df[target_col].values.astype(np.int16)
    y= to_categorical(y,num_classes=n_clas)   ### transform target classes into distribuiton probabilities

    X, y = shuffle(X, y,random_state=42)
    #X = np.moveaxis(X, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size, random_state=12)

    if print_info:
        print("\nTest size:",test_size)
        print("Number of training samples:",X_train.shape[0])
        print("Number of test samples:",X_test.shape[0])

    return X_train, X_test, y_train, y_test

###Data augmentation.  ###flipping Images horizontally
class FlippedImageDataGenerator(ImageDataGenerator):

    def next(self):
        X_batch, y_batch = super(FlippedImageDataGenerator, self).next()
        batch_size = X_batch.shape[0]
        indices1 = np.random.choice(batch_size, batch_size / 2, replace=False) ###we choose half of the img of this batch
        indices2 = np.random.choice(batch_size, batch_size / 2, replace=False) ###we choose half of the img of this batch
        X_batch[indices1] = X_batch[indices, :, :, ::-1] ##we flip x coordenates
        X_batch[indices2] = np.flip(X_batch[ind],1) ##we flip y coordenates
        return X_batch, y_batch

def train(fase,Model,name='GoogLeNet',GT_img=GT_img):

    if fase==1:
        start = 0.001 ###small because it is already trained
        stop = 0.0001
        nb_epoch = 10
        batch_size= 32


    if fase==2:
        start = 0.0001 ###small because it is already trained
        stop  = 0.00001
        nb_epoch = 10
        batch_size= 32

    pat=100
    decay_interval=5

    if (nb_epoch%decay_interval!=0):
       print("decay interval must divide number of epochs!!!")
       return 0
    #learning_rate = np.linspace(start, stop, nb_epoch)

    lr=np.linspace(start, stop, int(nb_epoch/decay_interval))
    learning_rate_decay=np.repeat(lr,decay_interval)

    opt_method= SGD(lr=start, momentum=0.9, nesterov=True)

    if(GT_img):
        img_col='img_GT'
    else:
        img_col='img_original'


    X_train, X_test, y_train, y_test = load_train_test(images_col=img_col)

    Model.compile(optimizer=opt_method, loss='categorical_crossentropy',metrics=['accuracy'])


    change_lr = LearningRateScheduler(lambda epoch: float(learning_rate_decay[epoch])) ###decay lr
    early_stop = EarlyStopping(patience=pat) #### early stop
    flipgen = FlippedImageDataGenerator()    #### data augmentation

    #hist=model.fit(X_train, y_train,batch_size=batch_size,epochs=nb_epoch,verbose=1,  callbacks=[change_lr, early_stop],validation_data=(X_test, y_test), shuffle=True)

    hist = Model.fit_generator(flipgen.flow(X_train, y_train),steps_per_epoch=int(X_train.shape[0]/batch_size),epochs=nb_epoch,validation_data=(X_test, y_test),verbose=1,
                             callbacks=[change_lr, early_stop])

    if (GT_img):
        gt='_gt_'
        print('Trained with Ground Truth img and '+str(n_clas)+' number of classes')
    else:
        gt='_orig_'
        print('Trained with original img and '+str(n_clas)+' number of classes')
    #model.save_weights('weights/'+name+gt+'fase_'+str(fase)+'_weights.h5', overwrite=True) ##save weights
    np.savetxt('results/'+name+gt+'fase_'+str(fase)+'_loss.csv', hist.history['loss'])
    np.savetxt('results/'+name+gt+'fase_'+str(fase)+'val_loss.csv', hist.history['val_loss'])
    np.savetxt('results/'+name+gt+'fase_'+str(fase)+'_acc.csv', hist.history['acc'])
    np.savetxt('results/'+name+gt+'fase_'+str(fase)+'_val_acc.csv', hist.history['val_acc'])

    y_pred = Model.predict(X_test)
    print('Predicted distributions for y validation small sample:\n',y_pred[:5])
    print('Classes of y validation\n:',y_test[:5])


input_tensor = Input(shape=(img_shape[0], img_shape[1],3))
 # this assumes kearas uses 'channels_first'. By defualt in tensorflow backend

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False,input_tensor=input_tensor)



#base_model= Xception(include_top=False, weights='imagenet', input_tensor=input_tensor)

# add a global spatial average pooling layer
x = base_model.output
x=Dropout(0.35)(x)
x = Flatten()(x)

x = Dense(256, activation='relu')(x)
x=Dropout(0.45)(x)
x = Dense(512, activation='relu')(x)
x=Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x=Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

###Set fase 1 trainable parameters:
for layer in model.layers[:-6]:
    layer.trainable = False

print('Fase 1:')
model.summary()
train(fase=1,Model=model)

###Set fase 2 parameters:
print('Fase 2:')
for layer in model.layers[-15:]:
    layer.trainable = True
model.summary()
model.compile(optimizer='Nadam', loss='categorical_crossentropy',metrics=['accuracy'])
X_train, X_test, y_train, y_test = load_train_test(images_col='img_original')
hist=model.fit(X_train, y_train,batch_size=8,epochs=10,verbose=1,validation_data=(X_test, y_test), shuffle=True)
