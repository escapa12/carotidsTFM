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


#path_data  = '../data/class_placa_train_RGB.csv'
path_data  = '../data/RGB_orig.csv'
img_shape=[365, 391]
GT_img=False
###count the relative freq of classes
#df = pd.read_csv(path_data)
#y =df['target'].dropna().values.astype(np.int16)
#count=np.bincount(y)[1::]
#rel_freq=count/float(len(y))

def load_train_test(path_data = path_data,images_col='img_original',target_col='eventCV', test_size=0.2,print_info=True):
    """Loads data"""

#     path_data = '../data/class_placa_train_RGB.csv'

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

    X, y = shuffle(X, y,random_state=42)

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

def train(fase,model,name='VGG',GT_img=GT_img):

    if fase==1:
        start = 0.001 ###small because it is already trained
        stop = 0.0001
        nb_epoch = 150
        batch_size= 16


    if fase==2:
        start = 0.0001 ###small because it is already trained
        stop  = 0.00001
        nb_epoch = 150
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

    if(GT_img):
        img_col='img_GT'
    else:
        img_col='img_original'


    X_train, X_test, y_train, y_test = load_train_test(images_col=img_col)

    model.compile(optimizer=opt_method, loss='binary_crossentropy',metrics=['accuracy'])


    change_lr = LearningRateScheduler(lambda epoch: float(learning_rate_decay[epoch])) ###decay lr
    early_stop = EarlyStopping(patience=pat) #### early stop
    flipgen = FlippedImageDataGenerator()    #### data augmentation

    #hist=model.fit(X_train, y_train,batch_size=batch_size,epochs=nb_epoch,verbose=1,  callbacks=[change_lr, early_stop],validation_data=(X_test, y_test), shuffle=True)

    hist = model.fit_generator(flipgen.flow(X_train, y_train),steps_per_epoch=int(X_train.shape[0]/batch_size),epochs=nb_epoch,validation_data=(X_test, y_test),verbose=1,
                             callbacks=[change_lr, early_stop])

    #uncomment to save weights:

    if (GT_img):
        gt='_gt_'
        print('Trained with Ground Truth imgs')
    else:
        gt='_orig_'
        print('Trained with original img')
    #model.save_weights('weights/'+name+gt+'fase_'+str(fase)+'_weights.h5', overwrite=True) ##save weights
    np.savetxt('results/event_'+name+gt+'fase_'+str(fase)+'_loss.csv', hist.history['loss'])
    np.savetxt('results/event_'+name+gt+'fase_'+str(fase)+'val_loss.csv', hist.history['val_loss'])
    np.savetxt('results/event'+name+gt+'fase_'+str(fase)+'_acc.csv', hist.history['acc'])
    np.savetxt('results/event'+name+gt+'fase_'+str(fase)+'_val_acc.csv', hist.history['val_acc'])

    y_pred = model.predict(X_test)
    print('Predicted values y validation:\n',y_pred)
    print('Classes of y validation\n:',y_test)




input_tensor = Input(shape=(img_shape[0], img_shape[1],3))
 # this assumes kearas uses 'channels_first'. By defualt in tensorflow backend

# create the base pre-trained model
base_model =  VGG16(weights='imagenet', include_top=False,input_tensor=input_tensor)

x = base_model.output
x=Dropout(0.35)(x)
x = Flatten()(x)
# let's add the top of the network. Some fully-connected layers and dropout

x = Dense(512, activation='relu')(x)
x=Dropout(0.35)(x)
x = Dense(1024, activation='relu')(x)
x=Dropout(0.45)(x)
x = Dense(2048, activation='relu')(x)
x=Dropout(0.5)(x)

predictions = Dense(1, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)


for layer in model.layers[:-3]:
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
