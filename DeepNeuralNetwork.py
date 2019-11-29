f=open("/home/jovyan/demo/data/patients.txt")
h=f.readline()
lines=f.readlines()
print(len(lines))

x=[]
y=[]
for line in lines:
    w=line.strip().lower().split(",")
    ins=[float(v) for v in w[1:-1]]
    x.append(ins)
    if(w[-1]=="yes"):
        y.append(1)
    else:
        y.append(0)
print(x)
print(y)

import numpy as np
ones=np.ones(len(lines))
x=np.c_[ones,x]
y=np.c_[y]
print(x)
print(y)

#network Building
#1 input layer, 4 neurons
#3 hidden layers, hidden1-6 neurons,hidden2-9 neurons, hidden3-6 neurons
#1 output layer, 1 neuron
np.random.seed(100)
i1=x.shape[1]
h1=int(i1*1.5)
h2=int(h1*1.5)
h3=h1
o1=1
w1=2*np.random.random((i1,h1))-1
w2=2*np.random.random((h1,h2))-1
w3=2*np.random.random((h2,h3))-1
w4=2*np.random.random((h3,o1))-1
print(w1.shape)
print(w2.shape)
print(w3.shape)
print(w4.shape)

#Training DNN
def sigmoid(x):
    return 1/(1+np.exp(-x))
def mse(y,ycap):
    return ((y-ycap)**2).mean()
def derivative(x):
    return x*(1-x)
ploss=0
flag=0
conv=1e-8
for i in range(100000):
    l1=sigmoid(x.dot(w1))
    l2=sigmoid(l1.dot(w2))
    l3=sigmoid(l2.dot(w3))
    l4=sigmoid(l3.dot(w4))
    e4=y-l4
    closs=mse(y,l4)
    if abs(ploss-closs)<=1e-8:
        print("Training completed after",i+1,"iterations")
        flag=1
        break
    if i%1000==0:
        print("loss at iteration",i+1,"is ",closs)
    d4=e4*derivative(l4)
    e3=d4.dot(w4.T)
    d3=e3*derivative(l3)
    e2=d3.dot(w3.T)
    d2=e2*derivative(l2)
    e1=d2.dot(w2.T)
    d1=e1*derivative(l1)
    w1+=x.T.dot(d1)
    w2+=l1.T.dot(d2)
    w3+=l2.T.dot(d3)
    w4+=l3.T.dot(d4)
    ploss=closs
if(flag==0):
    print("training not completed run few more iterations")
def predict(x,w):
    r=x
    for i in w:
        r=sigmoid(r.dot(i))
    return r
ycap=predict(x,[w1,w2,w3,w4])
