import numpy as np
import random as rn
import os,sys
from keras import backend as k
from keras import losses
import numpy as np
import tensorflow as tf
from math import sqrt
import keras, glob
from keras.preprocessing import image as kImage
from models.Net_S import Net
import gc
from PIL import Image
cur_dir = os.getcwd()
os.chdir(cur_dir)
sys.path.append(cur_dir)
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from tensorflow.python.keras import backend as K
tf.compat.v1.set_random_seed(1234)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)



def getfiles(dataset):
       
    Y_list = []
    X_list = []
    basepath = '../'


    if dataset == 'YOVOS':
        pathf = basepath+'datasets/train/JPEGImages/base_foreground_segmentation/yovos'
        tasks = os.listdir(pathf)
        pathf = basepath+'datasets/train/'
        for task in tasks:
            Y_list.append(glob.glob(os.path.join(pathf,'Annotations/base_foreground_segmentation/yovos', task, '*.png')))
            X_list.append(glob.glob(os.path.join(pathf,'JPEGImages/base_foreground_segmentation/yovos', task,'*.jpg')))

    elif dataset == 'SEGTRACK':
        pathf = basepath+'datasets/segtrackv2/JPEGImages'
        tasks = os.listdir(pathf)
        pathf = basepath+'datasets/segtrackv2'
        for task in tasks:
            Y_list.append(glob.glob(os.path.join(pathf,'GroundTruth',task, '*.png')))
            X_list.append(glob.glob(os.path.join(pathf,'JPEGImages',task, 'input','*.png')))


        for k in range(len(Y_list)):
           Y_list_temp = []
           E_list_temp = []
           for i in range(len(X_list[k])):
               X_name = os.path.basename(X_list[k][i])
               X_name = X_name.split('.')[0]
               
               for j in range(len(Y_list[k])):
                   Y_name = os.path.basename(Y_list[k][j])
                   Y_name = Y_name.split('.')[0]
                   if (Y_name == X_name):
                       Y_list_temp.append(Y_list[k][j])
                       break
                   

           Y_list[k] = Y_list_temp

           Y_list[k] = Y_list_temp
    elif dataset == 'DAVIS16':
        pathf = basepath+'datasets/DAVIS/480p'
        tasks = os.listdir(pathf)
        pathf = basepath+'datasets/DAVIS'
        for task in tasks:
            Y_list.append(glob.glob(os.path.join(pathf,'480pY',task, '*.png')))
            X_list.append(glob.glob(os.path.join(pathf,'480p', task,'input','*.jpg')))


        for k in range(len(Y_list)):
           Y_list_temp = []
           E_list_temp = []
           for i in range(len(X_list[k])):
               X_name = os.path.basename(X_list[k][i])
               X_name = X_name.split('.')[0]
               
               for j in range(len(Y_list[k])):
                   Y_name = os.path.basename(Y_list[k][j])
                   Y_name = Y_name.split('.')[0]
                   if (Y_name == X_name):
                       Y_list_temp.append(Y_list[k][j])
                       break
                   

           Y_list[k] = Y_list_temp
    xlist = []
    ylist = []
    for k in range(len(X_list)):
        for i in range(len(Y_list[k])):
            xlist.append(X_list[k][i])
            ylist.append(Y_list[k][i])
            
    
    X_list = xlist
    Y_list = ylist
    
    X_list = np.array(X_list)
    Y_list = np.array(Y_list)
    idx = list(range(X_list.shape[0]))
    np.random.shuffle(idx)
    np.random.shuffle(idx)
    X_list = X_list[idx]
    Y_list = Y_list[idx]

    return X_list,Y_list

def getImgs(X_list,Y_list):

    # load training data
    num_imgs = len(X_list)
    X = np.zeros((num_imgs,240,320,3),dtype="float32")
    Y = np.zeros((num_imgs,240,320,1),dtype="float32")
    for i in range(len(X_list)):
        x = kImage.load_img(X_list[i],target_size = [240,320,3])
        x = kImage.img_to_array(x)
        X[i,:,:,:] = x
        
        x = kImage.load_img(Y_list[i], grayscale = True,target_size = [240,320])
        x = kImage.img_to_array(x)
        x /= 255.0
        x = np.floor(x)
        Y[i,:,:,0] = np.reshape(x,(240,320))
        
        
        
    idx = list(range(X.shape[0]))
    np.random.shuffle(idx)
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]

    
    return X, Y

### training function    
def train(X, Y, data_name):
    
    lr = 1e-4
    max_epoch = 20
    batch_size = 50
    
    model = Net(lr,(240,320,3))
    model = model.initModel()
    
    sub_dataset = int(len(X)/batch_size)
    for epoch in range(max_epoch):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step in range(batch_size):
            
            y = Y[step*sub_dataset:(step+1)*sub_dataset]
            x = X[step*sub_dataset:(step+1)*sub_dataset]
            
            cx,cy = getImgs(x,y)
            
            model.fit(cx, cy,
                epochs=1, batch_size=1, verbose=2, shuffle = False)
            del cx,cy
            gc.collect()

        weights = model.get_weights()
        a = np.array(weights)
        np.save('../weights/base_weights/'+data_name+'_weights_'+str(epoch)+'.npy', a)
            
    del model


# =============================================================================
# Main
# =============================================================================
data_name = 'YOVOS'
x,y = getfiles(data_name)
train(x,y,data_name)

from models.DGT import DGTree

tree = DGTree(lr)
tree.loadPre('../weights/base_weights/'+data_name+'_weights_19.npy')

for t in range(tasks):
    print('task'+str(t))
    tree.root.tasks.append(t)
    Tx = X[t]
    Ty = Y[t]
    sub_dataset = int(len(Tx)/batch_size)
    x = Tx[:sub_dataset]
    y = Ty[:sub_dataset]
    cx,cy = getImgs(x,y)
    fscore = fmeasure(cx,cy,tree)
    tree.root.taskFmMap.update({t:fscore})

tree.save('../weights/tree_weights/tree_weights_'+data_name+'_'+str(0)+'_'+str(0)+'.npy')

del tree
gc.collect()