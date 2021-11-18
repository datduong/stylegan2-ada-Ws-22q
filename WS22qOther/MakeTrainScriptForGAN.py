import re,sys,os,pickle

script="""#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

cd /data/duongdb/stylegan2-ada-Ws-22q

# ! doing training

outputdir=/data/duongdb/WS22qOther_08102021/

# resolution=512

resolution=256

fold=FOLD

# ! OVERSAMPLE IS DONE AHEAD IN TFRECORD, not on-the-fly in @dataset pytorch call
oversample_time=5 

# ! dataset to train, use each fold, SPLIT LABELS
imagedata=$outputdir/Classify/TfStyleganImg+Os$oversample_time'+WS+22q11DS+Control+'$select_num'500Normal+Fold'$fold #+blankcenter

# ! resume ? 
resume=ffhq$resolution

cd /data/duongdb/stylegan2-ada-Ws-22q
python3 train_with_labels.py \
--data=$imagedata \
--gpus=2 --target=0.6 \
--aug=ada \
--outdir=$outputdir/Model \
--resume=$resume \
--cfg=paper$resolution \
--snap=10 \
--oversample_prob=0 \
--mix_labels=0 \
--metrics=fid3500_full \
--kimg 10000 \
--split_label_emb_at 4


"""


import time
from datetime import datetime

now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y%H%M%S")

os.chdir('/data/duongdb/stylegan2-ada-Ws-22q')
counter = 1


for fold in [0,1,2,3,4]: 
  newscript = re.sub('FOLD',str(fold),script)
  fname = 'script'+str(counter+1)+date_time+'.sh'
  fout = open(fname,'w')
  fout.write(newscript)
  fout.close()
  counter = counter + 1
  time.sleep(5)
  # sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:p100:2 --mem=24g --cpus-per-task=24 
  # sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:v100x:2 --mem=24g --cpus-per-task=24 
  os.system('sbatch --partition=gpu --time=16:00:00 --gres=gpu:v100x:2 --mem=24g --cpus-per-task=24 '+fname)

