import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten

from keras.optimizers import SGD,Nadam
from keras.callbacks import EarlyStopping, LearningRateScheduler,ModelCheckpoint


df=pd.read_excel('../data/dadesFinal.xlsx')
df.set_index('Códigobiobanco', inplace=True)
df['mortCV']=df['muerte']=='CV death'
df['mortCV'].replace((True, False), (1, 0), inplace=True)

def create_hemodine_dataset(tipus):
    cols_clas_placa=['placas'+tipus,'lipidos'+tipus,'fibrosis'+tipus,'calcio'+tipus,
                     'clasif'+tipus,'EventoCV_Si_No','mortCV']
    DF=df.loc[:,cols_clas_placa]
    DF=DF.dropna()
    print('Number of sample for'+tipus+':', DF.shape[0])
    DF=DF.rename(index=str, columns={'lipidos'+tipus:'lipidos',
                                     'fibrosis'+tipus:'fibrosis',
                                     'calcio'+tipus:'calcio',
                                      'clasif'+tipus:'class'})
    return DF

def to3classes(y):
    for i in range(len(y)):
        k=y[i]
        if (k==0) or (k==1) :
            y[i]=0
        if (k==2) or (k==3) :
            y[i]=1
        if (k==4):
            y[i]=2
    return y

def to_categorical(y):
    """
    Converts a class vector of [1,2,3,4,5] to binary class matrix of 5 or 3 classes(1and2,3and4,5).
    """
    y = np.array(y, dtype='int').ravel()
    n = y.shape[0]
    categorical = np.zeros((n, 3))
    categorical[np.arange(n), y] = 1
    return categorical

def load_data(df,test_size=0.1):
    cols=['lipidos','fibrosis','calcio']
    X=df[cols].values
    print("Number of  samples:",X.shape[0])
    y=df['class'].values
    y=to3classes(y-1)
    y=to_categorical(y)
    X, y = shuffle(X, y,random_state=20)
    X = X - X.mean(axis=0, keepdims=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size, random_state=22)
    print("Number of training samples:",X_train.shape[0])
    print("Number of test samples:",X_test.shape[0])
    return X_train, X_test, y_train, y_test


def DNN():
    init1='glorot_uniform'
    init2='glorot_normal'
    model = Sequential()
    model.add(Dense(258, input_shape=(3,),activation='relu',kernel_initializer=init2))
    model.add(Dropout(0.15))
    model.add(Dense(258, activation='relu',kernel_initializer=init2))
    model.add(Dropout(0.15))
    model.add(Dense(258, activation='relu',kernel_initializer=init2))
    model.add(Dropout(0.15))
    model.add(Dense(512, activation='relu',kernel_initializer=init2))
    model.add(Dropout(0.15))
    model.add(Dense(512, activation='relu',kernel_initializer=init2))
    model.add(Dropout(0.15))
    model.add(Dense(512, activation='relu',kernel_initializer=init2))
    model.add(Dropout(0.15))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    return model

def train_model(X_train, X_test, y_train, y_test):
    nb_epoch = 200
    batch_size= 4
    pat=100
    decay_interval=5

    model.compile(optimizer='RMSprop', loss='categorical_crossentropy',metrics=['accuracy'])
    save_best_model=ModelCheckpoint('./weights/hemodine_to_class.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=5)
    change_lr = LearningRateScheduler(lambda epoch: float(learning_rate_decay[epoch])) ###decay lr
    early_stop = EarlyStopping(patience=pat) #### early stop

    hist=model.fit(X_train, y_train,batch_size=batch_size,epochs=nb_epoch,verbose=1,  validation_data=(X_test, y_test), shuffle=True,callbacks=[early_stop, save_best_model])
    return hist
# callbacks=[change_lr, early_stop]
# hist = model.fit_generator(flipgen.flow(X_train, y_train),steps_per_epoch=int(X_train.shape[0]/batch_size),epochs=nb_epoch,validation_data=(X_test, y_test),verbose=1)
                        # callbacks=[change_lr, early_stop])

#uncomment to save weights:
#model.save_weights('weights/'+name+'_weights.h5', overwrite=True) ##save weights
# np.savetxt('results/'+name+'_loss.csv', hist.history['loss'])
# np.savetxt('results/'+name+'val_loss.csv', hist.history['val_loss'])
# np.savetxt('results/'+name+'_acc.csv', hist.history['acc'])
# np.savetxt('results/'+name+'_val_acc.csv', hist.history['val_acc'])
def model_predict(X_test,y_test):
    y_pred = model.predict(X_test)
    pred=np.array([x.argmax() for x in y_pred])
    true=np.array([x.argmax() for x in y_test])
    acc=sum(pred==true)/len(pred)
    print('Predicted distributions for y validation(10 samples):',y_pred[:10])
    print('Classes of y validation(10 samples):',y_test[:10])
    print('Accuracy:',acc)
    return acc


#### train network for each region
tipus=['_cc_','_fem_com_','_fem_sup_','_med_bif_','_med_car_']
file = open('./results/hemodin2class_results.txt','w')
for x in tipus:
    print('\n\n Tipus:',x)
    df_d=create_hemodine_dataset(x+'d')
    df_i=create_hemodine_dataset(x+'i')
    DF=pd.concat([df_d,df_i])
    X_train, X_test, y_train, y_test= load_data(DF)
    model=DNN()
    hist=train_model(X_train, X_test, y_train, y_test)
    acc=model_predict(X_test,y_test)
    file.write(x+' , '+str(acc)+'\n')
####train joining al regions data
print('All classes:')
tipus=['_cc_','_fem_com_','_fem_sup_','_med_bif_','_med_car_']

DF = pd.DataFrame(columns=df_d.columns)
for x in tipus:
    df_d=create_hemodine_dataset(x+'d')
    df_i=create_hemodine_dataset(x+'i')
    DF=pd.concat([DF,df_d,df_i])
X_train, X_test, y_train, y_test= load_data(DF)
model=DNN()
hist=train_model(X_train, X_test, y_train, y_test)
acc=model_predict(X_test,y_test)
file.write('All clases '+' , '+str(acc)+'\n')
#### train joining both femorals regions(superior and common)
print('Both femorals:')
tipus=['_fem_com_','_fem_sup_']

DF = pd.DataFrame(columns=df_d.columns)
for x in tipus:
    df_d=create_hemodine_dataset(x+'d')
    df_i=create_hemodine_dataset(x+'i')
    DF=pd.concat([DF,df_d,df_i])
X_train, X_test, y_train, y_test= load_data(DF)
model=DNN()
hist=train_model(X_train, X_test, y_train, y_test)
acc=model_predict(X_test,y_test)
file.write('Femorals'+' , '+str(acc)+'\n')

file.close()
