import numpy as np
import tensorflow as tf
import random as rn
import os,sys

from tensorflow.python.keras import backend as k
from tensorflow.python.keras import losses
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import isnan, sqrt

from PIL import Image

import tensorflow.python.keras, glob
from keras.preprocessing import image as kImage
from models.DGT import DGTree
import gc
from sklearn.metrics import precision_score,recall_score,f1_score
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



def fmeasure(cx,cy,tree,th=0.2):
    yhat = np.zeros((len(cx),240,320,1))
    for ii in range(len(cx)):
        yhat[ii,:,:,:] = tree.predict(cx[ii:ii+1,:,:,:])
    
    actualy = cy
    yp = np.zeros((len(yhat),240,320,1))
    yp[yhat >= th] = 1
    yp[yhat < th] = 0
    
    pred = (yp == 1)
    gt = (actualy == 1)
    
    pred = np.reshape(pred,(len(yhat),240,320))
    gt = np.reshape(gt,(len(yhat),240,320))

    gtflat = gt.flatten()
    predflat = pred.flatten()
    
    fscore = f1_score(gtflat, predflat, average='binary')
    return fscore

def mse(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.square(np.subtract(actual,pred)).mean() 

def generateData(X_list,Y_list):
    # read images
    
    T = np.zeros([len(X_list),240,320,3],dtype="float32")
    for i in range(0, len(X_list)):
        img = kImage.load_img(X_list[i],target_size=[240,320,3])
        x = kImage.img_to_array(img)
        T[i,:,:,:3] = x
    
    del img, x

    YT = np.zeros([len(Y_list),240,320,1],dtype="float32")
    for i in range(0, len(Y_list)):
        x = kImage.load_img(Y_list[i], grayscale = True,color_mode='grayscale',target_size = [240,320,1])
        x = kImage.img_to_array(x)
        shape = x.shape
        x /= 255.0
        x = x.reshape(-1)
        idx = np.where(np.logical_and(x>0.25, x<0.8))[0] # find non-ROI
        if (len(idx)>0):
            x[idx] = void_label
        x = x.reshape(shape)
        x = np.floor(x)

        YT[i,:,:,:1] = x
    
    return T,YT

def getfiles(dataset):
    
    Y_list = []
    X_list = []
    basepath = '../'


    if dataset == 'YOVOS':
        pathf = basepath+'datasets/train/JPEGImages/base_foreground_segmentation/yovos'
        tasks = os.listdir(pathf)
        pathf = basepath+'datasets/train/'
        for task in tasks:
            Y_list.append([])
            X_list.append([])
            Y_list[-1].append(glob.glob(os.path.join(pathf,'Annotations/base_foreground_segmentation/yovos', task, '*.png')))
            X_list[-1].append(glob.glob(os.path.join(pathf,'JPEGImages/base_foreground_segmentation/yovos', task,'*.jpg')))
    
    elif dataset == 'SEGTRACK':
        pathf = basepath+'datasets/segtrackv2/JPEGImages'
        tasks = os.listdir(pathf)
        pathf = basepath+'datasets/segtrackv2'
        for task in tasks:
            Y_list.append([])
            X_list.append([])
            Y_list[-1].append(glob.glob(os.path.join(pathf,'GroundTruth',task, '*.png')))
            X_list[-1].append(glob.glob(os.path.join(pathf,'JPEGImages',task, 'input','*.png')))


        for k in range(len(Y_list)):
           Y_list_temp = []
           for i in range(len(X_list[k][0])):
               X_name = os.path.basename(X_list[k][0][i])
               X_name = X_name.split('.')[0]
               
               for j in range(len(Y_list[k][0])):
                   Y_name = os.path.basename(Y_list[k][0][j])
                   Y_name = Y_name.split('.')[0]
                   if (Y_name == X_name):
                       Y_list_temp.append(Y_list[k][0][j])
                       break

           Y_list[k][0] = Y_list_temp

    
    elif dataset == 'DAVIS16':
        pathf = basepath+'datasets/DAVIS/480p'
        tasks = os.listdir(pathf)
        pathf = basepath+'datasets/DAVIS'
        for task in tasks:
            Y_list.append([])
            X_list.append([])
            Y_list[-1].append(glob.glob(os.path.join(pathf,'480pY',task, '*.png')))
            X_list[-1].append(glob.glob(os.path.join(pathf,'480p', task,'input','*.jpg')))


        for k in range(len(Y_list)):
           Y_list_temp = []
           for i in range(len(X_list[k][0])):
               X_name = os.path.basename(X_list[k][0][i])
               X_name = X_name.split('.')[0]
               
               for j in range(len(Y_list[k][0])):
                   Y_name = os.path.basename(Y_list[k][0][j])
                   Y_name = Y_name.split('.')[0]
                   if (Y_name == X_name):
                       Y_list_temp.append(Y_list[k][0][j])
                       break

           Y_list[k][0] = Y_list_temp
    xlist = []
    ylist = []
    for j in range(len(X_list)):
        xlist.append([])
        ylist.append([])
        for k in range(len(X_list[j])):
            for i in range(len(Y_list[j][k])):
                xlist[-1].append(X_list[j][k][i])
                ylist[-1].append(Y_list[j][k][i])
                
    
    X_list = xlist
    Y_list = ylist

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
        
        x = kImage.load_img(Y_list[i], grayscale = True,color_mode='grayscale',target_size = [240,320])
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
def train(X, Y,data_name):
    
    lr = 1e-4
    max_epochs = [10,10,10,10,10]
    max_epoch = 5
    batch_size = 80
    batch_size = 10
    batch_size = 1
    max_store_num = 8
    
    tree = DGTree(lr)
    tree.loadPre('../weights/base_weights/'+data_name+'_weights_19.npy')
    tree.load('../weights/tree_weights/tree_weights_'+data_name+'_'+str(0)+'_'+str(0)+'.npy')
    
    
    epsilon = 1e-6

    tasks = len(X)
    max_depth = tree.getMaxDepth()
    
    th = [0.2,0.1,0.05,0.025,0.012]
    minfm = 0.9
    for depth in range(max_depth):
        
        tree.goToNextDepth()
        for t in range(tasks):
            print('\n\ntask'+str(t)+'\n\n')

            Tx = X[t]
            Ty = Y[t]
            sub_dataset = int(len(Tx)/batch_size)
            x = Tx[:sub_dataset]
            y = Ty[:sub_dataset]
            cx,cy = getImgs(x,y)

            nodeName = 'null'
            node = tree.findByTask(t,tree.root)

            isNewNode = False
            if node != None:
                if abs(node.depth-tree.currentDepth)>1:
                    continue
                elif abs(node.depth-tree.currentDepth)==0:
                    tree.loadNodeWeights(root=node)
                    nodeName = node.name
                else:
                    fm = node.taskFmMap[t]
                    if fm > minfm:
                        continue
                    if len(cx) > 5:
                        nodeName,isNewNode = tree.createOrReuseNodeFM(cx[:5,:,:,:],cy[:5,:,:,:],t,th=th[depth])
                    else:
                        nodeName,isNewNode = tree.createOrReuseNodeFM(cx,cy[:,:,:,:],t,th=th[depth])
               

            else:
                if len(cx) > 5:
                    nodeName,isNewNode = tree.createOrReuseNodeFM(cx[:5,:,:,:],cy[:5,:,:,:],t,th=th[depth])
                else:
                    nodeName,isNewNode = tree.createOrReuseNodeFM(cx,cy[:,:,:,:],t,th=th[depth])
               
            
            node = tree.nodesMapping[nodeName]
            if not (t in node.tasks):
                tree.assignTaskToNode(t,nodeName)
            Ylist = glob.glob(os.path.join('../stored_examples/stored_examples/'+data_name+'/'+nodeName+'/gt', '*.png'))
            Xlist = glob.glob(os.path.join('../stored_examples/stored_examples/'+data_name+'/'+nodeName+'/images', '*.png'))
            if len(Xlist) >= max_store_num:
                mintask = 99999
                minIndex = 0
                for tt in range(len(Xlist)):
                    tasknum = int(Xlist[tt].split('uu')[1].split('.')[0])
                    if tasknum < mintask:
                        mintask = tasknum
                        minIndex = tt
                os.remove(Xlist[minIndex])
                os.remove(Ylist[minIndex])
            
            
            if not os.path.exists('../stored_examples/stored_examples/'+data_name+'/'+nodeName+'/images'):
                os.makedirs('../stored_examples/stored_examples/'+data_name+'/'+nodeName+'/images')
            if not os.path.exists('../stored_examples/stored_examples/'+data_name+'/'+nodeName+'/gt'):
                os.makedirs('../stored_examples/stored_examples/'+data_name+'/'+nodeName+'/gt')

                
            yd = cx[0,:,:,:3]
            d = np.uint8(yd)
            im = Image.fromarray(d)
            im.save('../stored_examples/stored_examples/'+data_name+'/'+nodeName+'/images/'+'tuu'+str(t)+'.png')
            
            d = np.zeros((240,320,3))
            yd = cy[0,:,:,:1]
            yd = np.reshape(yd, (240,320))
            d[:,:,0] = yd*255
            d[:,:,1] = yd*255
            d[:,:,2] = yd*255
            d = np.uint8(d)
            
            im = Image.fromarray(d)
            im.save('../stored_examples/stored_examples/'+data_name+'/'+nodeName+'/gt/'+'tuu'+str(t)+'.png')
            
            
            Ylist = glob.glob(os.path.join('../stored_examples/stored_examples/'+data_name+'/'+nodeName+'/gt', '*.png'))
            Xlist = glob.glob(os.path.join('../stored_examples/stored_examples/'+data_name+'/'+nodeName+'/images', '*.png'))
            Ylist=np.asarray(Ylist)
            Xlist=np.asarray(Xlist)
            
            mask = []
            if (not isNewNode) and (len(Xlist)!=0):
                stored_x,stored_y = generateData(Xlist,Ylist)
                mask = tree.ComputeFisherMatrix(stored_x,stored_y)
                
            max_epoch = max_epochs[depth]
            for epoch in range(max_epoch):
                print("\nDepth %d, Task %d and epoch %d" % (depth,t,epoch,))
                # Iterate over the batches of the dataset.
                for step in range(batch_size+1):
                    if step == batch_size:
                        x = Tx[step*sub_dataset:]
                        y = Ty[step*sub_dataset:]
                    
                    else:
                        x = Tx[step*sub_dataset:(step+1)*sub_dataset]
                        y = Ty[step*sub_dataset:(step+1)*sub_dataset]
                    
                    cx,cy = getImgs(x,y)

                    if len(cx) == 0:
                        continue

                    if len(mask) == 0:
                        tree.fit(cx,cy)
                    else:
                        tree.fitWithFIM(cx,cy,stored_x,stored_y,mask)
                    

                    del cx,cy
               
            
            x = Tx[:sub_dataset]
            y = Ty[:sub_dataset]
            cx,cy = getImgs(x,y)
            
            fscore = fmeasure(cx,cy,tree)

            previousFM = tree.currentNode.parent.taskFmMap[t]
            if fscore > previousFM:
                tree.saveFM(t,fscore)
                tree.storeNodeWeights()
            else:
                tree.deleteTaskFromNode(t,tree.currentNode.name)
                taskid = 'tuu'+str(t)+'.png'
                Ylist = glob.glob(os.path.join('../stored_examples/stored_examples/'+data_name+'/'+nodeName+'/gt', '*.png'))
                Xlist = glob.glob(os.path.join('../stored_examples/stored_examples/'+data_name+'/'+nodeName+'/images', '*.png'))
                for tt in range(len(Xlist)):
                    if taskid in Xlist[tt]: 
                        os.remove(Xlist[tt])
                        os.remove(Ylist[tt])
                        break
                    
            tree.save('../weights/tree_weights/tree_weights_'+data_name+'_'+str(depth)+'_'+str(t)+'.npy')
    

    list_of_deleted_nodes = []
    for k,v in tree.nodesMapping.items():
        if(tree.isNodeUseless(k)):
            list_of_deleted_nodes.append(k)
    for k in list_of_deleted_nodes:
        tree.deleteNode(k)

    tree.save('../weights/tree_weights/tree_weights_'+data_name+'_final.npy')       
    


# =============================================================================
# Main
# =============================================================================
x,y = getfiles('YOVOS')
train(x,y,'YOVOS')
gc.collect()