
import os,sys,re,pickle
import numpy as np 
import pandas as pd 

import argparse

parser = argparse.ArgumentParser(description='make normal face csv in gan')

parser.add_argument('--foldcsv', type=str, default=None)
parser.add_argument('--fold', type=int, default=None)
parser.add_argument('--foutpath', type=str, default=None)
parser.add_argument('--select_num', type=int, default=500)
parser.add_argument('--foutpathimg', type=str, default=None)
parser.add_argument('--ourimgpath', type=str, default=None)
parser.add_argument('--ourimgtocopy', type=str, default=None)


args = parser.parse_args()

# ! select some number of normal, train with diseases in GAN
images_in_train = pd.read_csv(args.foldcsv)
images_in_train = images_in_train[images_in_train['fold']!=args.fold] # ! remove this fold, because fold=1 means we train on "not 1" and validate on "1"
images_in_train = images_in_train[images_in_train['fold']!=5] # ! remove this fold 5 which is "test" in our naming convention 

print ('fold {}'.format(args.fold))
print (images_in_train.shape)

# ! 
label_set = sorted ( ['Normal'+i for i in '2y,youngchild,adolescence,youngadult,olderadult'.split(',')] ) 

# ! 
df_final = None
for this_label in label_set: 
  df = images_in_train[images_in_train['label'].str.contains(this_label)]
  print (this_label)
  print (df.shape)
  df = df.sample(n = args.select_num, replace = False, random_state=args.fold*200+1) # ! set seed
  if df_final is None: 
    df_final = df 
  else: 
    df_final = pd.concat([df_final, df]) # ! save a csv of all "normal" images selected 

#
df_final.to_csv(args.foutpath,index=False)

# ! move files into separate folder ? ... yes
# ! we need to replicate the "disease" folder, and then move these normal faces in. 
# ! to create tfrecord, we need a folder of images, and the json.
if args.ourimgtocopy is None: 
  os.system('scp -r '+ args.ourimgpath+'/*png' + ' ' + args.foutpathimg)
else: 
  label_set = []
  for a in args.ourimgtocopy.split(','): # ! copy just these images, should take in "WS,Controls" or something like this
    a = a.strip()
    label_set = label_set + sorted ( [a+i for i in '2y,youngchild,adolescence,youngadult,olderadult'.split(',')] ) 
  # 
  label_set = sorted (label_set)
  for l in label_set: 
    os.system('scp -r ' + args.ourimgpath+'/*'+l+'.png' + ' ' + args.foutpathimg) # ! copy just these images

for index,row in df_final.iterrows(): # name,path,label,fold,is_ext
  os.system ('scp ' + row['path'] + ' ' + args.foutpathimg )

