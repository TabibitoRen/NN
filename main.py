from random import random
import math


def rmatdot(r,c):
    m1 = r
    m2 = []
    p = []
    ##  make a new array from c for m2 to for loop the maxtrix multiplication
    for x in range(0,len(c[0])):
        m2.append([])
        for y in range(0,len(c)):
            m2[x].append(c[y][x])
##    print(m2)
    for x in range(0,len(m2)):
        t = 0
        for y in range(0,len(m2[x])):
            t += m1[y]*m2[x][y]
        p.append(t)
    return p

def matdot(m1,m2):
    if len(m1[0]) != len(m2):
        return False
    p = []
    for x in range(0,len(m1)):
        p.append(rmatdot(m1[x],m2))
    return p

def sigmoid(x):
    return 1/(1+math.exp(-x))

#  Define the hyperparameters
#  takes an array which specify the number of nodes for each layer and creates an Artificial Neural Network
#  starting weights are random
#  model[layers][nodes]
#  model[layers][nodes]['a'] activation
# model[layers][nodes]['w'][1,2,3] weights

def buildANN(nodes = []):
    model = []
    layers = len(nodes)
    weights = layers-1

    for x in range(0,layers):
        model.append([])
        for y in range(0,nodes[x]):
            model[x].append([])
            model[x][y] = {}
            model[x][y]['a'] = 0
            if x < weights:
                model[x][y]['w'] = []
                for z in range(0,nodes[x+1]):
                    model[x][y]['w'].append(int(random()*100)/100)
    return model

def setInputs(model,inputs = []):
    if len(model[0]) != len(inputs):
        return False
    for x in range(0,len(inputs)):
        model[0][x]['a'] = inputs[x]
    return model

def getOutputs(model):
    return model[len(model)-1]
#  calculate the next layer's activation function based on the previous layers, and should return an array for l2 next activation function
#  build a row vector, then column vectors for the weights and then perform matrix multiplication
def forward(model):
    layers = len(model)
    for x in range(0,layers-1):
        a1 = []
        w = []
        for y in range(0,len(model[x])):
            a1.append(model[x][y]['a'])
            w.append(model[x][y]['w'])
            a2 = rmatdot(a1,w)
            for z in range(0,len(a2)):
                model[x+1][z]['a'] = sigmoid(a2[z])
    return model

# accepts an array of the differences, set them to 1 for completely active and 0 shouldn't be active
def getCost(model,diff = []):
    cost = 0
    outputs  = getOutputs(model)
    if len(diff) != len(outputs): return False

    for x in diff:
        d = (outputs[x]['a']-diff[x])
        cost += d*d
    return cost

#print("Running ANN")
#ann = buildANN([20,15,20,10])

#ann = setInputs(ann,[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5])
#if ann == False : exit("Wrong number of inputs")
#ann = forward(ann)
#print(*getOutputs(ann))
#cost = getCost(ann,[1,0,0,0,1,0,0,0,0,0])
#if cost == False : exit("Wrong number of differences")
#print(cost)

#ann = buildANN([2,3,10])

#ann = setInputs(ann,[1,2])
#if ann == False : exit("Wrong number of inputs")
#ann = forward(ann)
#print(getCost(ann,[1,0]))
