from collections import OrderedDict
import numpy as np
import keras
from keras import layers
from keras.models import Model
from keras.layers import Input, Activation
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers import ZeroPadding2D
from keras.layers import BatchNormalization,concatenate
from models.instance_normalization import InstanceNormalization
import tensorflow.keras.backend as K
import tensorflow as tf
import os
from tensorflow.python.keras.utils.data_utils import get_file

def loss(y_true, y_pred):
    void_label = -1.
    y_pred = K.reshape(y_pred, [-1])
    y_true = K.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def acc(y_true, y_pred):
    void_label = -1.
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

class Net(object):
    
    def __init__(self, lr, img_shape):
        self.lr = lr
        self.img_shape = img_shape
        
    def decoder(self, x,a): 
        
        d4 = self.UNetUp(x, [], 256,'d1', 2, batch_norm=True,upsample=False,mix=False)
        d3 = self.UNetUp(d4, a, 128,'d2', 2, batch_norm=True)
        d2 = self.UNetUp(d3, [], 64,'d3', 2, batch_norm=True,mix=False)
        d1 = self.UNetUp(d2, [], 32,'d4', 2, batch_norm=True,upsample=False,mix=False)
        out = Conv2D(1, (1,1), strides=1, padding='same', activation='sigmoid',name='d5_convT')(d1)
        return out
    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):      #Convolutional Block for ResNet architecture
         
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        x = Conv2D(filters1, (1, 1), strides=strides,
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters2, kernel_size, padding='same',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
        
        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
        
        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x 
    def identity_block(self, input_tensor, kernel_size, filters, stage, block):                 #Identity Block for ResNet architecture

        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
        return x
    def resnet50(self, x):
                                  #Low-level feature extraction for decoder 
        x = ZeroPadding2D((3, 3))(x)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=3, name='bn_conv1')(x)
        a = Activation('relu')(x)
                                    #Low-level feature extraction for decoder 
            

        x = self.conv_block(a, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        return x,a
    
    def UNetUp(self,x,skip,ch,block,num_rep,batch_norm=False,upsample=True,mix=True):
        
        if upsample:
            x = UpSampling2D(size=(2, 2))(x)
        if mix:
            x = concatenate([x,skip], axis=-1, name=block+'cat')
        x = Conv2D(ch, (3, 3), activation='relu', padding='same', name=block+'_convT')(x)
        if batch_norm:
            x = InstanceNormalization(name=block+'_instwT')(x)

        for k in range(num_rep):
            i = ord('a')
            i+=k
            c = chr(i)
            x = Conv2D(ch, (3, 3), activation='relu', padding='same', name=block+'_conv'+c+'T')(x)
            if batch_norm:
                x = InstanceNormalization(name=block+'_inst'+c+'T')(x)

        return x

    def initModel(self):
        h, w, d = self.img_shape
        self.resnet50_weights_path='resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        if not os.path.exists(self.resnet50_weights_path):
            WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
            self.resnet50_weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        WEIGHTS_PATH_NO_TOP, cache_subdir='models',
                                        file_hash='6d6bbae143d832006294945121d1f1fc')
        net_input = Input(shape=(h, w, d), name='net_input')
        resnet50_output = self.resnet50(net_input)
        model = Model(inputs=net_input, outputs=resnet50_output, name='model')
        
        po = model.trainable_variables
        do = OrderedDict()
        for i in range(len(po)):
            do.update({po[i].name:np.asarray(po[i])})

        model.load_weights(self.resnet50_weights_path, by_name=True)

        
        pn = model.trainable_variables
        dn = OrderedDict()
        for i in range(len(pn)):
            dn.update({pn[i].name:np.asarray(pn[i])})


        for layer in model.layers:            
            layer.trainable = False
                
        x,a = model.output
        x = self.decoder(x, a)
        
        
        vision_model = Model(inputs=net_input, outputs=x, name='vision_model')
        opt = tf.keras.optimizers.RMSprop(lr = self.lr, rho=0.9, epsilon=1e-08, decay=0.)
        
        c_loss = loss
        c_acc = acc
        
        vision_model.compile(loss=c_loss, optimizer=opt, metrics=[c_acc])
        return vision_model


