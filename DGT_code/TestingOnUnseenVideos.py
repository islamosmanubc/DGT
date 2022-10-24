import numpy as np
import tensorflow as tf
import random as rn
import os,sys

from keras import backend as k
from keras import losses
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import isnan, sqrt

from sklearn.metrics import precision_score,recall_score,f1_score
from PIL import Image
import time
import keras, glob
from keras.preprocessing import image as kImage
from sklearn.utils import compute_class_weight
from models.DGT import DGTree
import gc
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
        x = kImage.load_img(Y_list[i], grayscale = True,target_size = [240,320,1])
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

def getfiles(dataset,split):
    
    Y_list = []
    X_list = []
    T_list = []
    basepath = '../'

    
    taskcounter = -1
    
    if dataset == 'YOVOS':
        if split == 'train':
            pathf = basepath+'datasets/train/JPEGImages/base_foreground_segmentation/yovos'
            tasks = os.listdir(pathf)
            pathf = basepath+'datasets/train/'
        else:
            pathf = basepath+'datasets/test/JPEGImages/base_foreground_segmentation/yovos'
            tasks = os.listdir(pathf)
            pathf = basepath+'datasets/test/'
        for task in tasks:
            taskcounter = taskcounter+1
            Y_list.append([])
            X_list.append([])
            T_list.append([])
            T_list[-1].append([])
            Y_list[-1].append(glob.glob(os.path.join(pathf,'Annotations/base_foreground_segmentation/yovos', task, '*.png')))
            X_list[-1].append(glob.glob(os.path.join(pathf,'JPEGImages/base_foreground_segmentation/yovos', task,'*.jpg')))
            T_list[-1][-1].append(taskcounter)
    if dataset == 'CDNet':
        if split == 'train':
            pathf = basepath+'datasets/train/JPEGImages/continual_foreground_segmentation/cdnet'
            tasks = os.listdir(pathf)
            pathf = basepath+'datasets/train/'
        else:
            pathf = basepath+'datasets/test/JPEGImages/continual_foreground_segmentation/cdnet'
            tasks = os.listdir(pathf)
            pathf = basepath+'datasets/test/'
        for task in tasks:
            taskcounter = taskcounter+1
            Y_list.append([])
            X_list.append([])
            T_list.append([])
            T_list[-1].append([])
            Y_list[-1].append(glob.glob(os.path.join(pathf,'Annotations/continual_foreground_segmentation/cdnet', task, '*.png')))
            X_list[-1].append(glob.glob(os.path.join(pathf,'JPEGImages/continual_foreground_segmentation/cdnet', task,'*.jpg')))
            T_list[-1][-1].append(taskcounter)
    if dataset == 'DAVIS17':
        if split == 'train':
            pathf = basepath+'datasets/train/JPEGImages/fewshot_foreground_segmentation/davis'
            tasks = os.listdir(pathf)
            pathf = basepath+'datasets/train/'
        else:
            pathf = basepath+'datasets/test/JPEGImages/fewshot_foreground_segmentation/davis'
            tasks = os.listdir(pathf)
            pathf = basepath+'datasets/test/'
        for task in tasks:
            taskcounter = taskcounter+1
            Y_list.append([])
            X_list.append([])
            T_list.append([])
            T_list[-1].append([])
            if split == 'train':
                Y_list[-1].append(glob.glob(os.path.join(pathf,'Annotations/fewshot_foreground_segmentation/davis_5shot', task, '*.png')))
                X_list[-1].append(glob.glob(os.path.join(pathf,'JPEGImages/fewshot_foreground_segmentation/davis_5shot', task,'*.jpg')))
            if split == 'test':
                Y_list[-1].append(glob.glob(os.path.join(pathf,'Annotations/fewshot_foreground_segmentation/davis', task, '*.png')))
                X_list[-1].append(glob.glob(os.path.join(pathf,'JPEGImages/fewshot_foreground_segmentation/davis', task,'*.jpg')))
            T_list[-1][-1].append(taskcounter)

    xlist = []
    ylist = []
    tlist = []
    for j in range(len(X_list)):
        xlist.append([])
        ylist.append([])
        tlist.append([])
        for k in range(len(X_list[j])):
            for i in range(len(Y_list[j][k])):
                xlist[-1].append(X_list[j][k][i])
                ylist[-1].append(Y_list[j][k][i])
                tlist[-1].append(T_list[j][0][0])    
                
    
    X_list = xlist
    Y_list = ylist
    T_list = tlist

    return X_list,Y_list,T_list

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
def train(X, Y,Tl,Xe,Ye,Tle,data_name):
    
    lr = 1e-4
    max_epoch = 20
    batch_size = 1
    max_store_num = 8
    
    tree = DGTree(lr)

    tree.model.summary()

    tree.loadPre('../weights/base_weights/base_weights_19.npy')
    tree.load('../weights/growing_tree_weights/tree_weights_'+data_name+'_final.npy')
    tasks = len(X)
    
    ths = [0.2,0.4,0.6,0.8]
    epsilon = 1e-6
    iou_all = {}
    fm_all = {}
    for th in ths:
        iou_all.update({th:[]})
        fm_all.update({th:[]})

    Tls = []
    for i in range(len(Tle)):
        for j in range(len(Tle[i])):
            Tls.append(Tle[i][j])
    actual_tasks = len(list(set(Tls)))
    iou_task_based = []
    fm_task_based = []
    for i in range(actual_tasks):
        iou_task_based.append({})
        fm_task_based.append({})
        for th in ths:
            iou_task_based[-1].update({th:[]})
            fm_task_based[-1].update({th:[]})

            
    f = open('../results/'+data_name+'_results.txt','a')
    for t in range(tasks):
        print("task " + str(t) + " is in progress!")
        iou_t = {}
        fm_t = {}
        for th in ths:
            iou_t.update({th:[]})
            fm_t.update({th:[]})
            
        
            
        #test task T
        
        Te_x = Xe[t]
        Te_y = Ye[t]
        Te_t = Tle[t]
        cx,cy = getImgs(Te_x[0:1],Te_y[0:1])
        tree.loadSuitableNodeGreedy(cx,cy)
        
        
        sub_dataset = int(len(Te_x)/batch_size)
        
        for step in range(batch_size+1):
            if step == batch_size:
                x = Te_x[step*sub_dataset:]
                y = Te_y[step*sub_dataset:]
            
            else:
                x = Te_x[step*sub_dataset:(step+1)*sub_dataset]
                y = Te_y[step*sub_dataset:(step+1)*sub_dataset]
            
            cx,cy = getImgs(x,y)

            if len(cx) == 0:
                continue
            
            yhat = tree.predict(cx)
            actualy = cy
            max_iou = 0
            max_th = 0
            max_fscore = 0
            yp = np.zeros((len(yhat),240,320,1))

            for th in ths:
                yp[yhat >= th] = 1
                yp[yhat < th] = 0
            
                pred = (yp == 1)
                gt = (actualy == 1)
            
                pred = np.reshape(pred,(len(yhat),240,320))
                gt = np.reshape(gt,(len(yhat),240,320))

                gtflat = gt.flatten()
                predflat = pred.flatten()

                rscore = recall_score(gtflat, predflat, average='binary')
                pscore = precision_score(gtflat, predflat, average='binary')
                fscore = f1_score(gtflat, predflat, average='binary')

                intersection = np.sum((pred*gt),axis=1)
                intersection = np.sum(intersection,axis=1)
                union = np.sum(((pred+gt)>0),axis=1)
                union = np.sum(union,axis=1)

                iou = np.mean((intersection+epsilon)/(union+epsilon))
                iou_t[th].append(iou)
                fm_t[th].append(fscore)
                iou_all[th].append(iou)
                fm_all[th].append(fscore)

                if iou > max_iou:
                    max_iou = iou
                    max_th = th
                    max_fscore = fscore

            del cx,cy

        fml = []
        ioul = []

        actualtask = Tle[t][0]
        for th in ths:
            iou = np.mean(np.array(iou_t[th]))
            fm = np.mean(np.array(fm_t[th]))
            ioul.append(iou)
            fml.append(fm)

            
            iou_task_based[actualtask][th].append(iou)
            fm_task_based[actualtask][th].append(fm)
            

    #write to file all fmeasures and IoUs
    for tt in range(actual_tasks):
        f.write('Task = ' +str(tt) + '\n')
        for th in ths:
            iou = np.mean(np.array(iou_task_based[tt][th]))
            fm = np.mean(np.array(fm_task_based[tt][th]))
            ioul.append(iou)
            fml.append(fm)
            f.write('Threshold = ' +str(th) + '\t')
            f.write('iou = ' +str(iou) + '\t')
            f.write('fm = ' +str(fm) + '\n')
        f.write('===================\n')

    f.write('All dataset: \n')
    for th in ths:
        iou = np.mean(np.array(iou_all[th]))
        fm = np.mean(np.array(fm_all[th]))
        f.write('Threshold = ' +str(th) + '\t')
        f.write('iou = ' +str(iou) + '\t')
        f.write('fm = ' +str(fm) + '\n')
    f.close()

    
    np.save('../results/lll_iou_task_'+data_name+'_results.npy', iou_task_based)
    np.save('../results/lll_fm_task_'+data_name+'_results.npy', fm_task_based)
# =============================================================================
# Main 
# =============================================================================

current_dataset = 'CDNet'
te_x,te_y,te_t = getfiles(current_dataset,'test')
train(tr_x,tr_y,tr_t,te_x,te_y,te_t,current_dataset)
gc.collect()