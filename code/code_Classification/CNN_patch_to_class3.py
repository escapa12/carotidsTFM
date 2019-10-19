import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold,train_test_split

from keras.preprocessing import image

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Input
from keras.optimizers import SGD,Nadam
from keras.callbacks import EarlyStopping, LearningRateScheduler,ModelCheckpoint
from keras.initializers import glorot_normal,glorot_uniform

path_data  = '../data/patch_fem.csv'  ### artery territorty data path
path_data2='../data/full_patches.csv' ### additional territory data path

useFem=False                          ### use 2n territory
img_shape=[90,160]                    ### input shape
n_clas=3                              ### number of classes (3 or 5)

def to_categorical(y, num_classes=n_clas):
    """
    Converts a class vector of [1,2,3,4,5] to binary class matrix. Each row has 0's except for the i-th component which is a 1. i corresponds to the class of the sample
    If num_clsses= 3, some classes are joined: 1-2 and 3-4.
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

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy with weights.
    """
    weights = K.variable(weights)
    def loss(y_true, y_pred):
    # scale predictions so that the class probabilities of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss


def load_train_test(images_col='patch',target_col='target', test_size=0.1, print_info=True):
    """Loads data from one or more territories. Splits data intro train and test"""

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
    df = df.dropna()
    print("Number of useful samples:",df.shape[0])
    df[images_col] = df[images_col].apply(lambda im: np.fromstring(im[1:-1], sep=','))

    X = np.vstack(df[images_col].values).astype(np.float32)
    X = X.reshape(-1, img_shape[0], img_shape[1], 1)
    X= X/255. ##pre-process input data

    y =  df[target_col].values.astype(np.int16)
    y= to_categorical(y,num_classes=n_clas)   ### transform target classes into distribuiton probabilities
    print(y.shape)

    X, y = shuffle(X, y,random_state=23)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size, random_state=23)

    if print_info:
        print("\nTest size:",test_size)
        print("Number of training samples:",X_train.shape[0])
        print("Number of test samples:",X_test.shape[0])

    return X_train, X_test, y_train, y_test

###Data augmentation.  ###flipping Images horizontally and vertically
class FlippedImageDataGenerator(ImageDataGenerator):
"Fucntion that randomly flips half of the minibatch samples vertially and/or horitzontally"
    def next(self):
        X_batch, y_batch = super(FlippedImageDataGenerator, self).next()
        batch_size = X_batch.shape[0]
        indices1 = np.random.choice(batch_size, batch_size / 2, replace=False) ###we randomly choose half of the img of this batch
        indices2 = np.random.choice(batch_size, batch_size / 2, replace=False) ###we randomly  choose half of the img of this batch
        X_batch[indices1] = X_batch[indices, :, :, ::-1] ## we flip x coordenates
        X_batch[indices2] = np.flip(X_batch[ind],1) ## we flip y coordenates
        return X_batch, y_batch


def CNN_model(num_clas=n_clas):
" CNN architecure. VGG inpired with less parameters. Has 0-padding and Glorot normal initialization"
    init1='glorot_normal'
    init2='glorot_uniform'
    model = Sequential()
    model.add(Conv2D(32, (3, 3),padding='same', activation='relu', input_shape=(img_shape[0], img_shape[1], 1),kernel_initializer=init1))
    model.add(Dropout(0.1))
    model.add(Conv2D(32, (3, 3), padding='same',activation="relu",kernel_initializer=init1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding='same',activation="relu",kernel_initializer=init1))
    model.add(Dropout(0.1))
    model.add(Conv2D(32, (3, 3), padding='same', activation="relu",kernel_initializer=init1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same',activation="relu",kernel_initializer=init1))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (3, 3), padding='same',activation="relu",kernel_initializer=init2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Conv2D(64, (3, 3), padding='same',activation="relu",kernel_initializer=init1))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (3, 3), padding='same',activation="relu",kernel_initializer=init1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Flatten())

    model.add(Dense(256, activation='relu',kernel_initializer=init1),)
    model.add(Dropout(0.35))
    model.add(Dense(256, activation='relu',kernel_initializer=init1))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu',kernel_initializer=init1))
    model.add(Dropout(0.5))

    model.add(Dense(num_clas, activation='softmax'))
    return model

def train_model(CV,param=None,name='patch2classv2'+str(n_clas)+'CCA'):
    "Function that trains the CNN for a given set of parameters."
    if (param==None):
        p= {'epochs':40, 'batch_size':4, 'lr_start': 0.01, 'lr_stop': 0.0001,
            'decay_interval':5, 'loss_weights': np.array([1,1,1])}
    else:
        p=param
    start = p['lr_start']                   ### start lr
    stop = p['lr_stop']                     ### last lr
    nb_epoch = p['epochs']                  ### number of epochs
    batch_size= p['batch_size']             ### mini-batch_size
    loss_weights = p['loss_weights']        ### weighs of the categ.CE loss
    pat=max(50,nb_epoch/10 )                ### patience (for the Early stopping)
    decay_interval=p['decay_interval']      ### interval between decay
    if (nb_epoch%decay_interval!=0):
       print("decay interval must divide number of epochs!!!")
       return 0

    #learning_rate = np.linspace(start, stop, nb_epoch)
    lr=np.linspace(start, stop, int(nb_epoch/decay_interval))
    learning_rate_decay=np.repeat(lr,decay_interval)    ###defines learning_rate

    opt_method= SGD(lr=start, momentum=0.9, nesterov=True)          ###set opt method
    WCateCros=weighted_categorical_crossentropy(loss_weights)       ### set loss
    change_lr = LearningRateScheduler(lambda epoch: float(learning_rate_decay[epoch])) ###set lr
    early_stop = EarlyStopping(monitor='acc',patience=pat)          ### set early stop
    flipgen = FlippedImageDataGenerator()                           ### set data augmentation
    weights_path='weights/'+name+'_weights.h5'
    save_best_model=ModelCheckpoint(weights_path, monitor='val_loss',  ### set model choice (best val loss)
                                    verbose=0, save_best_only=True,
                                    save_weights_only=True,
                                    mode='auto', period=1)

    if(CV):    ### cross validation
        folds=8
        kf = KFold(n_splits=folds)
        print(str(folds)+"-fold crossvalidation, for each training:")
        print("Number of training samples:",int((folds-1)/folds*X_train.shape[0]))
        print("Number of validation samples:",int(1/folds*X_train.shape[0]))
        print('Using the following parameters:',p)
        fold=0
        acc_s=np.zeros(folds)
        for train_inx, val_inx in kf.split(X_train):
            print('Fold '+str(fold+1)+'/'+str(folds)+':')
            model= CNN_model()
            model.compile(optimizer=opt_method, loss=WCateCros,metrics=['acc'])
            model.fit_generator(flipgen.flow(X_train[train_inx], y_train[train_inx]),
                                steps_per_epoch=int((X_train.shape[0]*(folds-1)/folds)/batch_size),
                                epochs=nb_epoch,
                                validation_data=(X_train[val_inx], y_train[val_inx]),
                                verbose=0,
                                callbacks=[change_lr, early_stop, save_best_model])
            model.load_weights(weights_path)                   ###load best weights
            y_pred = np.array(model.predict(X_train[val_inx])) ### predictions distributions
            pred=np.array([x.argmax() for x in y_pred])        ### distributions2classes
            true=np.array([x.argmax() for x in y_train[val_inx]])
            acc=sum(pred==true)/len(pred)                      ### compute accuracy
            print('Accuracy:',acc)
            acc_s[fold]=acc
            fold+=1
        print('Accuracy mean and sd:',acc_s.mean(),np.sqrt(acc_s.var()))
        return acc_s.mean()

    else:   ### training with the hole training set
        print('Trainig with best parameters:')
        model= CNN_model()
        model.compile(optimizer=opt_method, loss=WCateCros,metrics=['accuracy'])
        hist = model.fit_generator(flipgen.flow(X_train, y_train),
                                   steps_per_epoch=int(X_train.shape[0]/batch_size),
                                   epochs=100,      ### more epochs when training for the best parameters
                                   validation_data=(X_test, y_test),
                                   verbose=1,
                                   callbacks=[change_lr, early_stop, save_best_model])

        np.savetxt('results/'+name+'_loss.csv', hist.history['loss'])
        np.savetxt('results/'+name+'val_loss.csv', hist.history['val_loss'])
        np.savetxt('results/'+name+'_acc.csv', hist.history['acc'])
        np.savetxt('results/'+name+'_val_acc.csv', hist.history['val_acc'])

        model.load_weights(weights_path) ###load best weights
        y_pred = np.array(model.predict(X_test))
        pred=np.array([x.argmax() for x in y_pred])
        true=np.array([x.argmax() for x in y_test])

        # def to3class(y):
        #     for i in range(len(y)):
        #         k=y[i]
        #         if (k==0) or (k==1) :
        #             y[i]=0
        #         if (k==2) or (k==3) :
        #             y[i]=1
        #         if (k==4):
        #             y[i]=2
        #     return y
        acc=sum(pred==true)/len(pred)

        print('Predicted distributions for y validation(10 samples):',y_pred)
        print('Classes of y validation(10 samples):',y_test[:10])
        np.savetxt('results/'+name+'_y_pred.csv', pred) ### save predicted classes
        np.savetxt('results/'+name+'_y_true.csv',true)  ### save true classes

        print('Accuracy:',acc)
        # if n_clas == 5:
        #     acc3=sum(to3class(pred)==to3class(true))/len(pred)
        #     print('Accuracy predicting 3 classes:',acc3)
        return acc



X_train, X_test, y_train, y_test = load_train_test(images_col='patch')

### Grid Search paramenters Note that the program will train
### k-fold*numb combinations models! Around 7 -10minuts per model.
### In this case k=8---> 8*3*2*2*3*min_model= 288* min x model.
epochs=[50]
batch_size=[4,8,16]
lr_start=[0.01,0.001]
lr_stop=[0.0001,0.000001]
decay_interval=[5]
loss_weights= [np.array([1,1,1]),np.array([1,1,5]),np.array([2,1,5])]

best_p=None
best_acc_val=0

###Train the model for each combination of parameters of the GS.
for a in epochs:
    for b in batch_size:
        for c in lr_start:
            for d in lr_stop:
                for e in decay_interval:
                    for f in loss_weights:
                        p= {'epochs':a, 'batch_size':b, 'lr_start': c, 'lr_stop': d,'decay_interval':e, 'loss_weights': f}
                        acc_val = train_model(CV=True,param=p)   ###train the model usint Cross Validation
                        if acc_val>best_acc_val :
                            best_acc_val = acc_val
                            best_p = p
print('Best Validation loss:',best_acc_val)
print('Best parameters:',best_p)

val_loss = train_model(CV=False,param=best_p)
