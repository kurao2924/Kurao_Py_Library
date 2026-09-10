import sys
import os
import string
from array import array
sys.setrecursionlimit(3*10**7)

Alp_low=list(string.ascii_lowercase)
Alp_up=list(string.ascii_uppercase)
dij=[[0,1],[1,0],[0,-1],[-1,0]]
def nin():
    return list(map(int,input().split()))
def deq(x):
    return [i-1 for i in x]

if not os.path.isfile('sample.txt'):
    save=list(range(1000))
    ar=array('i', save)
    with open('sample.txt', 'wb') as f:
        ar.tofile(f)
else:
    load=array('i')
    with open('sample.txt', 'rb') as f:
        load.fromfile(f, 1000)
    print(load[-2])