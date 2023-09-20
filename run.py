import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Add,GlobalAveragePooling2D
#cifar10
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def Bottleneck_layer(inputs,filters,strides=(1,1),cardinality=32):
    shortcut=inputs
    output_channels=filters//cardinality
    x=Conv2D(filters,kernel_size=(1,1),strides=(1,1))(inputs)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)

    #3x3
    group_list=[]
    for i in range(cardinality):
        group=Conv2D(output_channels,kernel_size=(3,3),strides=strides,padding='same')(x)
        group=BatchNormalization()(group)
        group=Activation('relu')(group)
        group_list.append(group)

    x=tf.keras.layers.concatenate(group_list,axis=-1)
    #1x1
    x=Conv2D(filters*2,kernel_size=(1,1),strides=(1,1))(x)
    x=BatchNormalization()(x)

    if strides!=1 or shortcut.shape[-1]!=x.shape[-1]:
        shortcut=Conv2D(filters*2,kernel_size=(1,1),strides=strides)(shortcut)
        shortcut=BatchNormalization()(x)
    
    #adding shortcut
    x=tf.keras.layers.concatenate([x,shortcut],axis=-1)
    x=Activation('relu')(x)
    return x
def Resnext_50(inputs,C=32):
    inputs=tf.keras.Input(shape=(32,32,3))
    x=Conv2D(64,kernel_size=(7,7),strides=(2,2))(inputs)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
    for i in range(3):
        if i==2:
            x=Bottleneck_layer(x,128,strides=(2,2),cardinality=C)
        else:
            x=Bottleneck_layer(x,128,cardinality=C)
    for i in range(4):
        if i==3:
            x=Bottleneck_layer(x,256,strides=(2,2),cardinality=C)
        else:
            x=Bottleneck_layer(x,256,cardinality=C)
    for i in range(6):
        if i==5:
            x=Bottleneck_layer(x,512,strides=(2,2),cardinality=C)
        else:
            x=Bottleneck_layer(x,512,cardinality=C)
    for i in range(3):
        x=Bottleneck_layer(x,1024,cardinality=C)
    x=GlobalAveragePooling2D()(x)
    x=Flatten()(x)
    x=Dense(3,activation='softmax')(x)
    model=tf.keras.Model(inputs=inputs,outputs=x)
    return model
#load data
# Define the classes you want to keep (e.g., 'airplane', 'automobile', 'bird')
selected_classes = ['airplane', 'automobile', 'bird']

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# all dataset
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
#categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
#normalize  
x_train=x_train/255.0
x_test=x_test/255.0








model=Resnext_50(x_train,C=32)
#weight decay
weight_decay=1e-4
#momentum
momentum=0.9
#SGD
sgd=tf.keras.optimizers.SGD(lr=0.1,momentum=momentum,nesterov=True)
#train

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
print(model.summary())
#learning rate decay
def lr_schedule(epoch):
    if epoch<50:
        return 0.1
    elif epoch<75:
        return 0.01
    else:
        return 0.001
lr_decay=tf.keras.callbacks.LearningRateScheduler(lr_schedule)
#train
model.fit(x_train,y_train,batch_size=128,epochs=100,validation_data=(x_test,y_test),shuffle=True,callbacks=[lr_decay])
#evaluate
_,acc=model.evaluate(x_test,y_test,verbose=1)
print('Accuracy:%.3f'%(acc*100.0))






