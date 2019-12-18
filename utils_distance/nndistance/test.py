import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.nnd import NNDModule

dist =  NNDModule()

p1 = torch.ones(2,100,3)*0.2
p2 = torch.ones(2,150,3)*0.3
p1[0,0,0] = 1
p2[1,1,1] = -1
#points1 = Variable(p1,requires_grad = True)
#points2 = Variable(p2)
#dist1, idx1, dist2, idx2 = dist(points1, points2)
#print(dist1, dist2)
#loss = torch.sum(dist1)
#print(loss)
#loss.backward()
#print(points1.grad, points2.grad)


points1 = Variable(p1.cuda(), requires_grad = True)
points2 = Variable(p2.cuda())
dist1, dist2, idx1, idx2 = dist(points1, points2)
#print(dist1, dist2)
print(idx1, idx2)
loss = torch.mean(dist1) + torch.mean(dist2)
print(loss)
loss.backward()
print(points1.grad, points2.grad)