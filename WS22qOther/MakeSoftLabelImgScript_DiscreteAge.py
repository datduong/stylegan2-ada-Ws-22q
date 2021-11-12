import re,sys,os,pickle

# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=12g --cpus-per-task=12
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:p100:2 --mem=24g --cpus-per-task=24 

script="""#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ! model path
# path='/data/duongdb/WSq22_04202021/Model/00001-ImgTf256-paper256-ada-target0.7-resumeffhq256'
# model=$path/network-snapshot-002406.pkl


fold=FOLD 

if [ $fold == 0 ]; then
  # model=/data/duongdb/WS22qOther_08102021/Model/00000-TfStyleganImg+Os5+WS+22q11DS+Control+500Normal+Fold0-paper256-kimg10000-ada-target0.8-resumeffhq256-divlabel4/network-snapshot-001485.pkl
  # model=/data/duongdb/WS22qOther_08102021/Model/00005-TfStyleganImg+Os5+blankcenter+WS+22q11DS+Control+500Normal+Fold0-paper256-kimg20000-ada-target0.8-resumeffhq256-divlabel4/network-snapshot-001741.pkl
  # !
  # model=/data/duongdb/WS22qOther_08102021/Model/00015-TfStyleganImg+Os5+WS+22q11DS+Control+500Normal+Fold0-paper256-kimg10000-ada-target0.6-resumeffhq256-divlabel4/network-snapshot-001177.pkl 
  # ! blank background 
  model=/data/duongdb/WS22qOther_08102021/Model/00009-TfStyleganImg+Os5+blankcenter+WS+22q11DS+Control+500Normal+Fold0-paper256-kimg10000-ada-target0.6-resumeffhq256-divlabel4/network-snapshot-001382.pkl
fi

if [ $fold == 1 ]; then
  # model=/data/duongdb/WS22qOther_08102021/Model/00001-TfStyleganImg+Os5+WS+22q11DS+Control+500Normal+Fold1-paper256-kimg10000-ada-target0.8-resumeffhq256-divlabel4/network-snapshot-000819.pkl
  # model=/data/duongdb/WS22qOther_08102021/Model/00006-TfStyleganImg+Os5+blankcenter+WS+22q11DS+Control+500Normal+Fold1-paper256-kimg20000-ada-target0.8-resumeffhq256-divlabel4/network-snapshot-001638.pkl
  # model=/data/duongdb/WS22qOther_08102021/Model/00014-TfStyleganImg+Os5+WS+22q11DS+Control+500Normal+Fold1-paper256-kimg10000-ada-target0.6-resumeffhq256-divlabel4/network-snapshot-001331.pkl 
  # !
  model=/data/duongdb/WS22qOther_08102021/Model/00010-TfStyleganImg+Os5+blankcenter+WS+22q11DS+Control+500Normal+Fold1-paper256-kimg10000-ada-target0.6-resumeffhq256-divlabel4/network-snapshot-001177.pkl
fi

if [ $fold == 2 ]; then
  # model=/data/duongdb/WS22qOther_08102021/Model/00002-TfStyleganImg+Os5+WS+22q11DS+Control+500Normal+Fold2-paper256-kimg10000-ada-target0.8-resumeffhq256-divlabel4/network-snapshot-000870.pkl
  # model=/data/duongdb/WS22qOther_08102021/Model/00006-TfStyleganImg+Os5+blankcenter+WS+22q11DS+Control+500Normal+Fold2-paper256-kimg20000-ada-target0.8-resumeffhq256-divlabel4/network-snapshot-001331.pkl
  # model=/data/duongdb/WS22qOther_08102021/Model/00016-TfStyleganImg+Os5+WS+22q11DS+Control+500Normal+Fold2-paper256-kimg10000-ada-target0.6-resumeffhq256-divlabel4/network-snapshot-001485.pkl 
  # !
  model=/data/duongdb/WS22qOther_08102021/Model/00011-TfStyleganImg+Os5+blankcenter+WS+22q11DS+Control+500Normal+Fold2-paper256-kimg10000-ada-target0.6-resumeffhq256-divlabel4/network-snapshot-001433.pkl
fi

if [ $fold == 3 ]; then
  # model=/data/duongdb/WS22qOther_08102021/Model/00003-TfStyleganImg+Os5+WS+22q11DS+Control+500Normal+Fold3-paper256-kimg10000-ada-target0.8-resumeffhq256-divlabel4/network-snapshot-001075.pkl
  # model=/data/duongdb/WS22qOther_08102021/Model/00007-TfStyleganImg+Os5+blankcenter+WS+22q11DS+Control+500Normal+Fold3-paper256-kimg20000-ada-target0.8-resumeffhq256-divlabel4/network-snapshot-001741.pkl
  # model=/data/duongdb/WS22qOther_08102021/Model/00019-TfStyleganImg+Os5+WS+22q11DS+Control+500Normal+Fold3-paper256-kimg10000-ada-target0.6-resumeffhq256-divlabel4/network-snapshot-001433.pkl 
  # !
  model=/data/duongdb/WS22qOther_08102021/Model/00012-TfStyleganImg+Os5+blankcenter+WS+22q11DS+Control+500Normal+Fold3-paper256-kimg10000-ada-target0.6-resumeffhq256-divlabel4/network-snapshot-001485.pkl
fi

if [ $fold == 4 ]; then
  # model=/data/duongdb/WS22qOther_08102021/Model/00004-TfStyleganImg+Os5+WS+22q11DS+Control+500Normal+Fold4-paper256-kimg10000-ada-target0.8-resumeffhq256-divlabel4/network-snapshot-001331.pkl
  # model=/data/duongdb/WS22qOther_08102021/Model/00008-TfStyleganImg+Os5+blankcenter+WS+22q11DS+Control+500Normal+Fold4-paper256-kimg20000-ada-target0.8-resumeffhq256-divlabel4/network-snapshot-001331.pkl
  # model=/data/duongdb/WS22qOther_08102021/Model/00018-TfStyleganImg+Os5+WS+22q11DS+Control+500Normal+Fold4-paper256-kimg10000-ada-target0.6-resumeffhq256-divlabel4/network-snapshot-001126.pkl 
  # !
  model=/data/duongdb/WS22qOther_08102021/Model/00013-TfStyleganImg+Os5+blankcenter+WS+22q11DS+Control+500Normal+Fold4-paper256-kimg10000-ada-target0.6-resumeffhq256-divlabel4/network-snapshot-001331.pkl
fi


truncationpsi=TRUNCATION # @trunc=0.7 is recommended on their face dataset, looks like 0.5 works pretty well. 

cd /data/duongdb/stylegan2-ada-Ws-22q 

outdir=OUTPUTDIR
mkdir $outdir

class='CLASS'
classnext=NEXT 

label=LABEL

for age_label in 4 5 6 7 8
do

  thisclass=$class','$age_label
  python3 generate.py --outdir=$outdir --trunc=$truncationpsi --seeds=SEED --network $model --savew 0 --suffix 'F'$fold'C'$thisclass'T'$truncationpsi$label --class=$thisclass 

done 


"""

import time
from datetime import datetime
import numpy as np 
import pandas as pd 

now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y%H%M%S")

os.chdir('/data/duongdb/stylegan2-ada-Ws-22q')

# ------------------------------------------------------------------------------------------

name_option = 'WS+22q11DS+Control+Normal+kimg10+target0.6+DiscA+blankcenter' # ! name of how stylegan2 was trained
TRAINCSV = '/data/duongdb/WS22qOther_08102021/Classify/train+blankcenter+WS+22q11DS+Control+Normal+Split.csv' # ! blank background center ?? +blankcenter
TRAINCSV = pd.read_csv ( TRAINCSV ) 

TRUNCATION = .6

# ------------------------------------------------------------------------------------------


labelset_head = sorted('WS,22q11DS,Controls,Normal'.split(','))
labelset_head_in_gan = {v.strip():i for i,v in enumerate(labelset_head)}

labelset_tail = sorted('2y,adolescence,olderadult,youngadult,youngchild'.split(','))
labelset_tail_in_gan = {v.strip():(i+len(labelset_head)) for i,v in enumerate(labelset_tail)} # ! shift because label takes the first spots

def make_label_vec (thisname,labelset_head_in_gan,labelset_tail_in_gan): 
  vec = []
  for head in labelset_head_in_gan: 
    if head in thisname: 
      vec.append ( labelset_head_in_gan[head] ) 
      break 
  for tail in labelset_tail_in_gan: 
    if tail in thisname: 
      vec.append( labelset_tail_in_gan[tail] )
      break 
  print ('label is {} vector is {}'.format(thisname,vec))
  return ','.join(str(v) for v in vec)


import itertools
def make_label_pairs (name_array1,name_array2,labelset_head_in_gan,labelset_tail_in_gan): 
  pairs = { }
  for n1 in name_array1: # should be age
    for n2 in list(itertools.combinations(name_array2, 2)): # @n2 is array of tuple [(1,2)...]
      t1 = make_label_vec(n2[0]+n1, labelset_head_in_gan,labelset_tail_in_gan)
      t2 = make_label_vec(n2[1]+n1, labelset_head_in_gan,labelset_tail_in_gan)
      pairs[n2[0]+n1+'_'+n2[1]+n1] = [t1,t2]
      pairs[n2[1]+n1+'_'+n2[0]+n1] = [t2,t1]
  return (pairs)

#
label_pair = make_label_pairs(labelset_tail, labelset_head,labelset_head_in_gan,labelset_tail_in_gan)
print (label_pair)


# labelset=sorted('22q11DS2y,22q11DSadolescence,22q11DSolderadult,22q11DSyoungadult,22q11DSyoungchild,Controls2y,Controlsadolescence,Controlsolderadult,Controlsyoungadult,Controlsyoungchild,WS2y,WSadolescence,WSolderadult,WSyoungadult,WSyoungchild'.split(','))
# label_seed = {}
# for i,l in enumerate(labelset): 
#   label_seed[l] = i * 500

labelset=sorted('WS,22q11DS,Controls'.split(','))
label_seed = {}
for i,l in enumerate(labelset): 
  label_seed[l] = i * 2500 # ! to match the seed used with age. 
  
# ------------------------------------------------------------------------------------------

#
rootout = '/data/duongdb/WS22qOther_08102021/Classify/'+name_option+'T'+str(TRUNCATION) # +'M'+str(MIXRATIO)+'T'+str(TRUNCATION) # 
if not os.path.exists(rootout): 
  os.mkdir(rootout)

# 

MULTIPLY_BY = 1 # ! how many times we do style mix ?

fold_seed = {i : i*10000+1 for i in range(5)}

label_pair_key = sorted(list(label_pair.keys()))

counter = 1
for fold in [0,1,2,3,4]: 
  print ('fold {}'.format(fold))
  OUTPUTDIR=os.path.join(rootout,'F'+str(fold)+'X'+str(MULTIPLY_BY))
  if not os.path.exists (OUTPUTDIR):  
    os.mkdir(OUTPUTDIR)
  for label1 in labelset: 
    if 'Normal' in label1: 
      continue # ! dont need to make more normal
    #
    class1 = str(labelset_head_in_gan[label1]) 
    #
    classifier_train_csv = TRAINCSV[TRAINCSV['fold'] != fold ] 
    classifier_train_csv = classifier_train_csv[classifier_train_csv["label"].str.contains(label1)] # ! get all ages for 22q or WS
    # ! compute average number, then make 5 age groups 
    size = int(classifier_train_csv.shape[0]/5) # ! how many fake images? 
    print ('img size for this pair {} {}'.format(label1,size))
    # ! make seed based on @size
    seed_start = fold_seed[fold] + label_seed[label1] # ! off set the label seed by the fold seed
    seed_end = seed_start + size
    #
    newscript = re.sub('TRUNCATION',str(TRUNCATION),script)
    newscript = re.sub('FOLD',str(fold),newscript)
    newscript = re.sub('CLASS',str(class1),newscript)
    #
    newscript = re.sub('LABEL',label1,newscript)
    newscript = re.sub('OUTPUTDIR',OUTPUTDIR,newscript)
    newscript = re.sub('SEED',str(seed_start)+'-'+str(seed_end),newscript) # ! //2 half female/male
    fname = 'script'+str(counter+1)+date_time+'.sh'
    fout = open(fname,'w')
    fout.write(newscript)
    fout.close()
    counter = counter + 1
    time.sleep(1)
    os.system('sbatch --partition=gpu --time=00:35:00 --gres=gpu:k80:1 --mem=6g --cpus-per-task=4 '+fname)




