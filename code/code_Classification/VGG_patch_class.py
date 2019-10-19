import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import decode_predictions
from keras.models import Model
from keras.layers import Dense, Flatten,Dropout
from keras import backend as K
from keras.layers import Input
from keras.optimizers import SGD,Nadam
from keras.callbacks import EarlyStopping, LearningRateScheduler


###Per executar:
######activar environment: source /home/esarle/miniconda2/envs/envTFM/bin/activate envTFM
###### executar(desdel directori on es trova aquest fitxer):
# PYTHONPATH=/home/esarle/miniconda2/envs/envTFM/lib/python2.7/site-packages/theano:/home/esarle/miniconda2/envs/envTFM/lib/python2.7/site-packages/keras:$PYTHONPATH THEANO_FLAGS='device=cuda' python -u GPU_placa_class.py


path_data2  = '../data/full_patches.csv'
path_data='../data/patch_fem.csv'
img_shape=[90,160]
n_clas=3
useFem=True
###count the relative freq of classes
#df = pd.read_csv(path_data)
#y =df['target'].dropna().values.astype(np.int16)
#count=np.bincount(y)[1::]
#rel_freq=count/float(len(y))

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

def load_train_test(path_data = path_data,images_col='img_original',target_col='target', test_size=0.1,print_info=True):
    """Loads data"""

#     path_data = '../data/class_placa_train_RGB.csv'
    df= pd.read_csv(path_data)
    df = df[[images_col] + [target_col]]
    if (useFem):
        df2=pd.read_csv(path_data2)
        df2 = df2[[images_col] + [target_col]]
        df=pd.concat([df,df2])
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
    X =np.repeat(X.reshape(-1, img_shape[0], img_shape[1], 1), 3, axis=3)
    X = preprocess_input(X) ##process input data in the same way than the pretrained network

    y =  df[target_col].values.astype(np.int16)
    y= to_categorical(y,num_classes=n_clas)   ### transform target classes into distribuiton probabilities

    X, y = shuffle(X, y,random_state=42)
    #X = np.moveaxis(X, -1, 1) ###to use channels_first

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

def train(fase,model,name='VGG'):

    if fase==1:
        start = 0.001 ###small because it is already trained
        stop = 0.0001
        nb_epoch = 50
        batch_size= 16


    if fase==2:
        start = 0.0001 ###small because it is already trained
        stop  = 0.00001
        nb_epoch = 50
        batch_size= 16

    pat=100
    decay_interval=5

    if (nb_epoch%decay_interval!=0):
       print("decay interval must divide number of epochs!!!")
       return 0
    #learning_rate = np.linspace(start, stop, nb_epoch)

    lr=np.linspace(start, stop, int(nb_epoch/decay_interval))
    learning_rate_decay=np.repeat(lr,decay_interval)

    opt_method= SGD(lr=start, momentum=0.9, nesterov=True)



    X_train, X_test, y_train, y_test = load_train_test(images_col='patch')

    model.compile(optimizer='Nadam', loss='categorical_crossentropy',metrics=['accuracy'])


    change_lr = LearningRateScheduler(lambda epoch: float(learning_rate_decay[epoch])) ###decay lr
    early_stop = EarlyStopping(patience=pat) #### early stop
    flipgen = FlippedImageDataGenerator()    #### data augmentation

    #hist=model.fit(X_train, y_train,batch_size=batch_size,epochs=nb_epoch,verbose=1,  callbacks=[change_lr, early_stop],validation_data=(X_test, y_test), shuffle=True)

    hist = model.fit_generator(flipgen.flow(X_train, y_train),steps_per_epoch=int(X_train.shape[0]/batch_size),epochs=nb_epoch,validation_data=(X_test, y_test),verbose=1,
                             callbacks=[change_lr, early_stop])

    #uncomment to save weights:
    gt='_patch_'
    #model.save_weights('weights/'+name+gt+'fase_'+str(fase)+'_weights.h5', overwrite=True) ##save weights
    np.savetxt('results/'+name+gt+'fase_'+str(fase)+'_loss.csv', hist.history['loss'])
    np.savetxt('results/'+name+gt+'fase_'+str(fase)+'val_loss.csv', hist.history['val_loss'])
    np.savetxt('results/'+name+gt+'fase_'+str(fase)+'_acc.csv', hist.history['acc'])
    np.savetxt('results/'+name+gt+'fase_'+str(fase)+'_val_acc.csv', hist.history['val_acc'])

    y_pred = model.predict(X_test)
    print('Predicted distributions for y validation:\n',y_pred)
    print('Classes of y validation\n:',y_test)




input_tensor = Input(shape=(img_shape[0], img_shape[1],3))
 # this assumes kearas uses 'channels_first'. By defualt in tensorflow backend

# create the base pre-trained model
base_model =  VGG16(weights='imagenet', include_top=False,input_tensor=input_tensor)

init2='glorot_normal'
x = base_model.output
x=Dropout(0.35)(x)
x = Flatten()(x)
# let's add the top of the network. Some fully-connected layers and dropout
x = Dense(256, activation='relu',kernel_initializer=init2)(x)
x=Dropout(0.35)(x)
x = Dense(512, activation='relu',kernel_initializer=init2)(x)
x=Dropout(0.45)(x)
x = Dense(512, activation='relu',kernel_initializer=init2)(x)

# x=Dropout(0.45)(x)
# x = Dense(1028, activation='relu')(x)
x=Dropout(0.5)(x)
# and a logistic layer: we have 3 classes
predictions = Dense(n_clas, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# #per veure numero de llista  layer
# for i, layer in enumerate(model.layers):
#     print(i, layer.name)

###Set fase 1 trainable parameters:
for layer in model.layers[:-7]:
    layer.trainable = False

print('Fase 1:')
model.summary()
train(fase=1,model=model)

###Set fase 2 parameters:
print('Fase 2:')
for layer in model.layers[-14:]:
    layer.trainable = True
model.summary()
train(fase=2,model=model)
