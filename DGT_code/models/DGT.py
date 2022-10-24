import tensorflow as tf
from collections import OrderedDict
from models.Net_S import Net
import numpy as np
import torch
import tensorflow.keras.backend as K
from sklearn.metrics import f1_score

def mse(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.square(np.subtract(actual,pred)).mean() 

class DecoderTreeNode:
  def __init__(self,parameters=None):
    self.parameters = parameters
    self.tasks = []
    self.parent = None
    self.depth = 0
    self.children = []
    self.trainable = True
    self.parentTasks = []
    self.name = '0_0'
    self.taskFmMap = {}


class DGTree(object):
    
    def __init__(self, lr):
        self.CreateModel(lr)
        self.weights = self.model.get_weights()
        self.names = [weight.name for layer in self.model.layers for weight in layer.weights]
        self.weightsName = OrderedDict()
        for name, weight in zip(self.names, self.weights):
            self.weightsName.update({name: weight})
        self.currentDepth = 0
        self.getDecoderParameters()
        self.root = DecoderTreeNode(self.parameters)
        self.currentNode = self.root
        self.nodesMapping = OrderedDict()
        self.nodesMapping.update({self.root.name:self.root})
        self.datasetMapping = OrderedDict()
        
    
    def CreateModel(self,lr):
        self.model = L3TNetwork(lr,(240,320,3))
        self.model = self.model.initModel() 

    def getDecoderParameters(self):
        self.parameters = OrderedDict()
        layers_of_parameters = []
        for k,v in self.weightsName.items():
            ks = k.split('_')[0]
            if ks[0]=='d':
                layers_of_parameters.append(ks)
                self.parameters.update({k:v})

        self.unique_layers = list(set(layers_of_parameters))
        self.max_depth = len(self.unique_layers)
        return self.parameters,self.unique_layers,self.max_depth

    def getMaxDepth(self):
        return self.max_depth

    def goToNextDepth(self):
        self.currentDepth = self.currentDepth+1
        self.createDeeperNodes(self.root)

    def createDeeperNodes(self,root):
        if len(root.children) == 0:
            if len(root.tasks) > 1:
                self.cloneSingleNode(root)
        else:
            for child in root.children:
                if len(child.children) == 0:
                    if len(child.tasks) > 1:
                        self.cloneSingleNode(child)
                else:
                    self.createDeeperNodes(child)

    def cloneSingleNode(self,root,initialization=True):
        newParameters = self.getDeeperNodeParameters(root.parameters, root.depth)
        newNode = DecoderTreeNode(newParameters)
        newNode.depth = root.depth+1
        newNode.parent = root
        newNode.parentTasks = root.tasks
        newNode.name = root.name + 'I' + str(newNode.depth) + '_' + str(len(root.children)+1)
        if initialization:
            #minFM = 9999
            mintask = root.tasks[0]
            #for k,v in root.taskFmMap.items():
            #    if root.taskFmMap[k] < minFM:
            #        minFM = root.taskFmMap[k]
            #        mintask = k
            newNode.tasks.append(mintask)
        self.currentNode = newNode
        root.children.append(newNode)
        root.trainable = False
        
        self.nodesMapping.update({newNode.name:newNode})
        return newNode.name

    
    def getDeeperNodeParameters(self,parameters,depth):
        new_parameters = OrderedDict()
        layers_of_parameters = []
        for k,v in self.parameters.items():
            ks = k.split('_')[0]
            if ks[0]=='d':
                layer_depth = int(ks[1:])
                if layer_depth > depth:
                    new_parameters.update({k:v})
        return new_parameters

    def addNewNode(self,root = None, name = 'null'):
        if root == None and name == 'null':
            root = self.currentNode.parent
        elif name != 'null':
            root = self.nodesMapping[name]
        return self.cloneSingleNode(root,False)
       

    def loadNodeWeights(self, T = -1, root = None):
        if T == -1 and root == None:
            self.getWeights(self.currentNode)
        elif root == None:
            root = self.findByTask(T,self.root)
            self.getWeights(root)
        elif T == -1:
            self.getWeights(root)
        

    def getWeights(self,root):
        self.currentNode = root
        parameters = root.parameters
        if root.parent != None:
            parameters.update(self.getPreviousNodeParameters(root.parent))
        i = 0
        for name, weight in zip(self.names, self.weights):
            for kt,vt in parameters.items():
                if name == kt:
                    self.weights[i]=vt
            i = i + 1
        self.model.set_weights(self.weights)
        self.changeTrainabeLayers(root.depth)


    def getPreviousNodeParameters(self,root):
        if root.parent == None:
            return self.getAncestorNodeParameters(root.parameters,root.depth)
        else:
            parameters = self.getAncestorNodeParameters(root.parameters,root.depth)
            parameters.update(self.getPreviousNodeParameters(root.parent))
            return parameters

     
    def getAncestorNodeParameters(self,parameters,depth):
        new_parameters = OrderedDict()
        layers_of_parameters = []
        for k,v in parameters.items():
            ks = k.split('_')[0]
            if ks[0]=='d':
                layer_depth = int(ks[1:])
                if layer_depth <= depth:
                    new_parameters.update({k:v})
        return new_parameters

    def findByTask(self,T,root):
        if T in root.tasks and len(root.children) == 0:
            return root

        for child in root.children:
            if T in child.tasks:
                return self.findByTask(T,child)
        return root

    #def findByName(self,name,root):
    #    if name == root.name:
    #        self.loadNodeWeights(root)
    #    for child in root.children:
    #        self.findByName(name,child)

    
    def changeTrainabeLayers(self,depth):
        for layer in self.model.layers:
            for weight in layer.weights:
                if weight.name[0] == 'd':
                    decoder_layer = int(weight.name.split('_')[0][1:])
                    if decoder_layer < depth:
                        layer.trainable = False
                    else:
                        layer.trainable = True
                    break

    def storeNodeWeights(self,params = None):
        parameters = self.currentNode.parameters
        if params == None:
            self.weights = self.model.get_weights()
        else:
            self.weights = params
        i=0
        for name, weight in zip(self.names, self.weights):
            for kt,vt in parameters.items():
                if name == kt:
                    parameters.update({kt:self.weights[i]})
            i = i + 1
        self.currentNode.parameters = parameters
        if len(self.currentNode.children) > 0:
            for i in range(len(self.currentNode.children)):
                newParameters = self.getDeeperNodeParameters(self.currentNode.parameters, self.currentNode.depth)
                self.currentNode.children[i].parameters = newParameters

    def getNodesAtDepth(self,D):
        nodes = {}
        for k,v in self.nodesMapping.items():
            ks = k.split('I')
            if len(ks)-1 == D:
                nodes.update({k:v})
        return nodes

    def getCurrentWeights(self):
        return self.model.get_weights()
    def getParameters(self):
        return self.currentNode.parameters
    def setParameters(self,parameters):
        self.currentNode.parameters = parameters

    def predict(self,X):
        return self.model.predict(X)

    def predictTask(self,X,T):
        self.loadNodeWeights(T)
        return self.model.predict(X)

    def predictNode(self,X,node = None,name = 'null'):
        if name == 'null':
            self.loadNodeWeights(root=node)
        else:
            if name in self.nodesMapping:
                self.loadNodeWeights(root=self.nodesMapping[name])
                #self.findByName(name,self.root)
        return self.model.predict(X)

    def predictFullTree(self,X):
        predictions = OrderedDict()
        return self.traverseTree(X,predictions,self.root)

    
    def traverseTree(self,X,predictions,root):
        predictions.update({root.name:self.predictNode(X,root)})
        for i in range(len(root.children)):
            self.traverseTree(X,predictions,root.children[i])
        return predictions
    def greedyTraverseTree(self,X,Y,root,parent_loss):
        
        if len(root.children) == 0:
            return root
        loss = []
        for i in range(len(root.children)):
            yhat=self.predictNode(X,root.children[i])
            loss.append(mse(yhat,Y))
        t = np.argmin(loss)
        if loss[t] < parent_loss:
            selected_node = root.children[t]
            selected_node = self.greedyTraverseTree(X,Y,selected_node,loss[t])
        else:
            selected_node = root
        
        return selected_node

    def loadSuitableNodeGreedy(self,X,Y):
        yhat = self.predictNode(X,self.root)
        parent_loss = mse(yhat,Y)
        self.currentNode = self.greedyTraverseTree(X,Y,self.root,parent_loss)
        self.loadNodeWeights(root=self.currentNode)

    def loadSuitableNode(self,X,Y):
        predictions = OrderedDict()
        predictions.update(self.predictFullTree(X))
        losses = []
        names = []
        for name, Yhat in predictions.items():
            losses.append(mse(Yhat, Y))
            names.append(name)

        t = np.argmin(losses)
        selected_node = names[t]
        self.currentNode = self.nodesMapping[selected_node]
        self.loadNodeWeights(root=self.nodesMapping[selected_node])
       
        

    def createOrReuseNode(self,X,Y,T,fullTree=False):
        predictions = OrderedDict()

        if fullTree:
            predictions.update(self.predictFullTree(X))
        else:
            parent = self.findByDepth(T,self.root,self.currentDepth-1)
            predictions.update({parent.name:self.predictNode(X,parent)})
            for child in parent.children:
                predictions.update({child.name:self.predictNode(X,child)})
        
        losses = []
        names = []
        for name, Yhat in predictions.items():
            losses.append(mse(Yhat, Y))
            names.append(name)

        t = np.argmin(losses)
        selected_node = names[t]
        newNodeCreated = False

        if fullTree:
            currentNode = self.nodesMapping[selected_node]
            if len(currentNode.children)==0:
                #reuse
                self.loadNodeWeights(root=self.nodesMapping[selected_node])
                self.assignTaskToNode(T,selected_node)
                newNodeCreated = False
            else:
                #create_node
                new_node_name = self.addNewNode(currentNode)
                self.loadNodeWeights(root=self.nodesMapping[new_node_name])
                self.assignTaskToNode(T,new_node_name)
                selected_node = new_node_name
                newNodeCreated = True

        else:
            if selected_node == parent.name:
                #create node
                new_node_name = self.addNewNode(parent)
                self.loadNodeWeights(root=self.nodesMapping[new_node_name])
                self.assignTaskToNode(T,new_node_name)
                selected_node = new_node_name
                newNodeCreated = True
            else:
                #reuse node
                self.loadNodeWeights(root=self.nodesMapping[selected_node])
                self.assignTaskToNode(T,selected_node)
                newNodeCreated = False

        return selected_node,newNodeCreated

    def checkParentAndChild(self,X,Y,nodename):
        node = self.nodesMapping[nodename]
        
        self.loadNodeWeights(node.parent)
    def createOrReuseNodeMSE(self,X,Y,T,fullTree=False,th=0.2):
        predictions = OrderedDict()

        if fullTree:
            predictions.update(self.predictFullTree(X))
        else:
            parent = self.findByDepth(T,self.root,self.currentDepth-1)
            predictions.update({parent.name:self.predictNode(X,parent)})
            for child in parent.children:
                predictions.update({child.name:self.predictNode(X,child)})
        
        losses = []
        names = []
        for name, Yhat in predictions.items():
            losses.append(mse(Yhat, Y))
            names.append(name)

        t = np.argmin(losses)
        bestloss = losses[t]

        selected_node = names[t]
        losses.pop(t)
        names.pop(t)

        if len(losses) == 0:
            return selected_node,False

        secondt = np.argmin(losses)

        secondbestloss = losses[secondt]
        bestchild = names[secondt]
        newNodeCreated = False

        if fullTree:
            currentNode = self.nodesMapping[selected_node]
            if len(currentNode.children)==0:
                #reuse
                self.loadNodeWeights(root=self.nodesMapping[selected_node])
                self.assignTaskToNode(T,selected_node)
                newNodeCreated = False
            else:
                
                #create_node
                new_node_name = self.addNewNode(currentNode)
                self.loadNodeWeights(root=self.nodesMapping[new_node_name])
                self.assignTaskToNode(T,new_node_name)
                selected_node = new_node_name
                newNodeCreated = True

        else:
            if selected_node == parent.name:
                #create node
                new_node_name = self.addNewNode(parent)
                self.loadNodeWeights(root=self.nodesMapping[new_node_name])
                self.assignTaskToNode(T,new_node_name)
                selected_node = new_node_name
                newNodeCreated = True
            else:
                #reuse node
                self.loadNodeWeights(root=self.nodesMapping[selected_node])
                self.assignTaskToNode(T,selected_node)
                newNodeCreated = False

        return selected_node,newNodeCreated
    def createOrReuseNodeFM(self,X,Y,T,fullTree=False,th=0.2):
        predictions = OrderedDict()

        if fullTree:
            predictions.update(self.predictFullTree(X))
        else:
            parent = self.findByDepth(T,self.root,self.currentDepth-1)
            predictions.update({parent.name:self.predictNode(X,parent)})
            for child in parent.children:
                predictions.update({child.name:self.predictNode(X,child)})
        
        losses = []
        names = []
        gt = (Y == 1)
        gt = np.reshape(gt,(len(Y),240,320))
        for name, Yhat in predictions.items():
            yp = np.zeros((len(Yhat),240,320,1))
            yp[Yhat >= th] = 1
            yp[Yhat < th] = 0
            
            pred = (yp == 1)
            pred = np.reshape(pred,(len(Yhat),240,320))
            
            gtflat = gt.flatten()
            predflat = pred.flatten()
            fscore = f1_score(gtflat, predflat, average='binary')

            losses.append(fscore)
            names.append(name)

        t = np.argmax(losses)
        bestloss = losses[t]

        selected_node = names[t]
        losses.pop(t)
        names.pop(t)

        if len(losses) == 0:
            return selected_node,False

        secondt = np.argmax(losses)

        secondbestloss = losses[secondt]
        bestchild = names[secondt]
        newNodeCreated = False

        if fullTree:
            currentNode = self.nodesMapping[selected_node]
            if len(currentNode.children)==0:
                #reuse
                self.loadNodeWeights(root=self.nodesMapping[selected_node])
                self.assignTaskToNode(T,selected_node)
                newNodeCreated = False
            else:
                if (bestloss - secondbestloss) < th:
                    if (self.nodesMapping[bestchild].children) == 0:
                        #reuse best child
                        self.loadNodeWeights(root=self.nodesMapping[bestchild])
                        self.assignTaskToNode(T,bestchild)
                        selected_node = bestchild
                        newNodeCreated = False
                    else:
                        #create_node
                        new_node_name = self.addNewNode(currentNode)
                        self.loadNodeWeights(root=self.nodesMapping[new_node_name])
                        self.assignTaskToNode(T,new_node_name)
                        selected_node = new_node_name
                        newNodeCreated = True
                else:
                    #create_node
                    new_node_name = self.addNewNode(currentNode)
                    self.loadNodeWeights(root=self.nodesMapping[new_node_name])
                    self.assignTaskToNode(T,new_node_name)
                    selected_node = new_node_name
                    newNodeCreated = True

        else:
            if selected_node == parent.name:
                if (bestloss - secondbestloss) < th:
                    #reuse best child
                    self.loadNodeWeights(root=self.nodesMapping[bestchild])
                    self.assignTaskToNode(T,bestchild)
                    selected_node = bestchild
                    newNodeCreated = False

                else:
                    #create node
                    new_node_name = self.addNewNode(parent)
                    self.loadNodeWeights(root=self.nodesMapping[new_node_name])
                    self.assignTaskToNode(T,new_node_name)
                    selected_node = new_node_name
                    newNodeCreated = True
            else:
                #reuse node
                self.loadNodeWeights(root=self.nodesMapping[selected_node])
                self.assignTaskToNode(T,selected_node)
                newNodeCreated = False

        return selected_node,newNodeCreated
        
    def findByDepth(self,T,root,depth):
        if root.depth == depth:
            if T in root.tasks:
                return root
        for i in range(len(root.children)):
            if T in root.children[i].tasks:
                return self.findByDepth(T,root.children[i],depth)
    
    def taskToParents(self,root,T):
        if root == None:
            return
        if T in root.tasks:
            return
        else:
            root.tasks.append(T)
            if root.parent == None:
                return
            self.taskToParents(root.parent,T)

    def assignTaskToNode(self,T,name):
        root = self.nodesMapping[name]
        root.tasks.append(T)
        root.taskFmMap.update({T:0})
        self.taskToParents(root.parent,T)
        #self.taskToNode(name,self.root,T)
    def isNodeUseless(self,name):
        root = self.nodesMapping[name]
        if len(root.tasks) == 0:
            return True
        return False
    def deleteNode(self,name):
        root = self.nodesMapping[name]
        parent = root.parent
        selectednode = -1
        for i in range(len(parent.children)):
            if parent.children[i].name == name:
                selectednode = i
                break
        parent.children.pop(i)
        self.nodesMapping.pop(name)
    def deleteTaskFromNode(self,T,name):
        root = self.nodesMapping[name]
        root.tasks.remove(T)
        root.taskFmMap.update({T:9999})

    def fit(self,X,Y,epochs=1):
        self.model.fit(X, Y, epochs=epochs, batch_size=1, verbose=2, shuffle = False)

    def fitWithFIM(self,X,Y,sx,sy,mask,epochs=1):
        #mask = self.ComputeFisherMatrix(sx,sy)
        #mask = self.ComputeFisherMatrix(X,Y)
        po = self.model.trainable_variables
        do = OrderedDict()
        for i in range(len(po)):
            do.update({po[i].name:np.asarray(po[i])})
        
        X = np.append(X,sx,axis=0)
        Y = np.append(Y,sy,axis=0)

        self.model.fit(X, Y, epochs=epochs, batch_size=1, verbose=2, shuffle = False)

        pn = self.model.trainable_variables
        dn = OrderedDict()
        for i in range(len(pn)):
            dn.update({pn[i].name:np.asarray(pn[i])})
        c = 0
        for name, weight in zip(self.names, self.weights):
            if do.__contains__(name):
                oldweight = do[name]
                newweight = dn[name]
                fimask = mask[name]

                newweight = oldweight + (newweight-oldweight)*(1-fimask)
                self.weights[c] = newweight
            c=c+1
        self.model.set_weights(self.weights)

    def ComputeFisherMatrix(self,xx,yy):
        w = self.model.trainable_variables
        varnames = []
        for i in range(len(w)):
            varnames.append(w[i].name)

        d = OrderedDict()
        for i in range(len(w)):
	        s = w[i].shape
	        d.update({i:np.zeros(s)})

        b,h,w,c = xx.shape
        for i in range(xx.shape[0]):
	        x = np.reshape(xx[i],(1,h,w,3))
	        y = np.reshape(yy[i],(1,h,w,1))
	        with tf.GradientTape() as tape:
		        logits = self.model(x)
		        loss =  K.binary_crossentropy(y, logits)

	        grads = tape.gradient(loss, self.model.trainable_variables)
	        for k,v in d.items():
		        d.update({k:v+np.asarray(grads[k])})

        sizes = {}
        tensors = []
        classifier_size = 0
        all_params_size = 0
        keep_ratio = 0.25
        classifier_mask_dict = {}
        for k, v in d.items():
            sizes[k] = v.shape
            tensors.append(torch.tensor(v).view(-1))
            all_params_size += torch.prod(torch.tensor(v.shape)).item()
        tensors = torch.cat(tensors, 0)

        mx = torch.max(tensors)
        mn = torch.min(tensors)
        tensors = (tensors-mn)/(mx-mn)

        keep_num = int(all_params_size * keep_ratio)
        top_pos = torch.topk(tensors, keep_num)[1]
        masks = torch.zeros_like(tensors)
        masks[top_pos] = 1
        mask_dict = {}

        now_idx = 0
        for k, v in sizes.items():
            end_idx = now_idx + torch.prod(torch.tensor(v))
            mask_dict[k] = tensors[now_idx: end_idx].reshape(v)
            now_idx = end_idx
	
        fishermask = {}
        for k,v in mask_dict.items():
	        fishermask[k] = np.asarray(v)

        fisherinfo = OrderedDict()
        for k in range(len(varnames)):
            fisherinfo.update({varnames[k]:fishermask[k]})
        return fisherinfo

    def saveFM(self,t,fm):
        self.currentNode.taskFmMap[t] = fm

    def loadPre(self,name):
        weights = np.load(name, allow_pickle=True, encoding="latin1").tolist()
        a = np.array(self.model.get_weights())
        for k in range(len(weights)):
            a[k] = weights[k]
        self.model.set_weights(a)
        self.weights = weights
        for name, weight in zip(self.names, self.weights):
            self.weightsName.update({name: weight})
        self.getDecoderParameters()
        self.root = DecoderTreeNode(self.parameters)
        self.currentNode = self.root
        self.nodesMapping = OrderedDict()
        self.nodesMapping.update({self.root.name:self.root})

    def save(self,name,dataset = '',startIndex = -1):
        weights = self.model.get_weights()
        DictOfData = OrderedDict()
        DictOfData.update({'currentDepth':self.currentDepth})
        DictOfData.update({'tree':self.root})
        DictOfData.update({'mapping':self.nodesMapping})
        if dataset != '':
            self.datasetMapping.update({dataset:startIndex})
        DictOfData.update({'datamapping':self.datasetMapping})
        np.save(name, DictOfData)


    def load(self,name):
        ckpt = np.load(name, allow_pickle=True, encoding="latin1").tolist()
        self.currentDepth = ckpt['currentDepth']
        self.root = ckpt['tree']
        self.nodesMapping = ckpt['mapping']
        self.loadNodeWeights(root = self.root)
        self.datasetMapping = ckpt['datamapping']






