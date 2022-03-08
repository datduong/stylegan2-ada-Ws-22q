
import os,sys,re,pickle 
import numpy as np 
import pandas as pd 

# ! rename into WS,22q,... instead of indexing

newpath = '/data/duongdb/WS22qOther_08102021/Classify/WS+22q11DS+Control+Normal+kimg10+target0.6+blankcenterM1T0.6Ave/RenameF0X1'
path = '/data/duongdb/WS22qOther_08102021/Classify/WS+22q11DS+Control+Normal+kimg10+target0.6+blankcenterM1T0.6Ave/F0X1'

img = os.listdir(path)

img = [i for i in img if 'WS' in i]

if not os.path.exists(newpath): 
  os.mkdir(newpath)

#

pattern1 = ['C3,4C0,4',
            'C3,5C0,5',
            'C3,6C0,6',
            'C3,7C0,7',
            'C3,8C0,8']

pattern2 = ['C3,4C1,4',
            'C3,5C1,5',
            'C3,6C1,6',
            'C3,7C1,7',
            'C3,8C1,8']

pattern3 = ['C3,4C2,4',
            'C3,5C2,5',
            'C3,6C2,6',
            'C3,7C2,7',
            'C3,8C2,8']

#

for i in img: 
  for p in pattern1: 
    if p in i: 
      newname = re.sub(p,'WS-Notmix',i)
      os.system('scp ' + os.path.join(path,i) + ' ' + os.path.join(newpath,newname))
  #
  for p in pattern2: 
    if p in i: 
      newname = re.sub(p,'WS-Notmix',i)
      os.system('scp ' + os.path.join(path,i) + ' ' + os.path.join(newpath,newname))
  #
  for p in pattern3: 
    if p in i: 
      newname = re.sub(p,'WS-Notmix',i)
      os.system('scp ' + os.path.join(path,i) + ' ' + os.path.join(newpath,newname))


