
# ! make a json 

# "people": [{"name": "Scott", "website": "stackabuse.com", "from": "Nebraska"}, {"name": "Larry", "website": "google.com", "from": "Michigan"}, {

import os,sys,re,pickle
import json
import pandas as pd 
import numpy as np 

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--image_dir', type=str, default=None)
parser.add_argument('--outjson', type=str, default=None)
parser.add_argument('--normal_as_0', type=int, default=0)
parser.add_argument('--labels', type=str, default='WS,22q11DS,Controls')


args = parser.parse_args()

label_json = {}
label_json['labels'] = []

images = sorted ( os.listdir(args.image_dir) ) 
images = [ i for i in images if i.endswith('png') ]

if args.normal_as_0 == 1: 
  labelset1 = sorted(args.labels.split(',')) 
else: 
  args.labels = args.labels+',Normal' # ! add normal faces
  labelset1 = sorted(args.labels.split(','))
#
labelset1 = {v.strip():i for i,v in enumerate(labelset1)}

# '22q11DS2y,22q11DSadolescence,22q11DSolderadult,22q11DSyoungadult,22q11DSyoungchild,Controls2y,Controlsadolescence,Controlsolderadult,Controlsyoungadult,Controlsyoungchild,WS2y,WSadolescence,WSolderadult,WSyoungadult,WSyoungchild'

labelset2 = sorted('2y,adolescence,olderadult,youngadult,youngchild'.split(','))
labelset2 = {v.strip():i for i,v in enumerate(labelset2)}

# ! we may want to use just WS or 22q, nothing else.

for index,filename in enumerate(images):  
  condition1 = [0] * len(labelset1) # replicate this
  filename_temp = filename.split('_')[-1]
  found = 0 
  for l in labelset1: 
    if l in filename_temp: # ! set normal as default 0 ?? or each label has its own 1-hot ?? 
      condition1[labelset1[l]] = 1 # 1hot
      found = 1 # ! we may want to just do WS + controls for now? 
  if found == 0: # ! we may want to just do WS + controls for now? 
    continue # ! skip 
  #
  condition2 = [0] * len(labelset2) # replicate this
  for l in labelset2: 
    if l in filename_temp: 
      condition2[labelset2[l]] = 1 # 1hot
  #
  # ! use [0]*3 to pad the normal cases ?
  label = condition1 + condition2
  #
  label_json['labels'].append([filename,label]) # {img1:[0], img2:[1] ...}


#

with open(os.path.join(args.image_dir,args.outjson), 'w') as outfile: # ! we fix @dataset_tool.py to take json
  json.dump(label_json, outfile)

#

print (len(label_json['labels']))


# {'22q11DS_earlySlide89.png', 'WS_lateSlide125.png', 'Controls_earlySlide10.png'}

# python3 predict.py --csv $inputname --output $outputname --detected_faces $detectfaces                         
# using CUDA?: True                                                                                              
# ---0/997---                                                                                                    
# Sorry, there were no faces found in '/data/duongdb/WS22qOther_06012021/Align512/Controls_earlySlide10.png'    
# Sorry, there were no faces found in '/data/duongdb/WS22qOther_06012021/Align512/22q11DS_earlySlide89.png'     
# Sorry, there were no faces found in '/data/duongdb/WS22qOther_06012021/Align512/WS_lateSlide125.png'          
# detected faces are saved at  /data/duongdb/WS22qOther_06012021/Classify/LabelFromFairFace/FairFaceDetectedFaces
# Predicting... 0/994                                                                                            
# saved results at  /data/duongdb/WS22qOther_06012021/Classify/LabelFromFairFace/FairFaceOutput.csv              
# Segmentation fault                                                                                             
