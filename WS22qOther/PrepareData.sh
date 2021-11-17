#!/bin/bash

# sbatch --partition=gpu --time=2:30:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=8 
# sinteractive --time=3:30:00 --gres=gpu:p100:1 --mem=4g --cpus-per-task=4

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ----------------------------------------------------------------------------------------------------


# ! cut out white space. 
codepath=$datadir/HAM10000_dataset/EnhanceQuality # ! borrow this code # we can reuse same code we had for NF1 skin 
mkdir /data/duongdb/WS22qOther_08102021/TrimImg # ! all images will be in same folder, we need to run the @extract_code 
cd $codepath
for type in WS_early WS_late WS_inter 22q11DS_early 22q11DS_late 22q11DS_inter Controls_early Controls_inter Controls_late  
do 
  datapath=/data/duongdb/WS22qOther_08102021/$type
  python3 CropWhiteSpaceCenter.py $datapath 0 png 'dummy' > $type'trim_log.txt' # ! @png is probably best for ffhq style @'dummy' is a hack
done 
cd /data/duongdb/WS22qOther_08102021/TrimImg


# ----------------------------------------------------------------------------------------------------


# ! align images into ffhq format # this has to be done so we can greatly leverage transfer-ability of ffhq
resolution=512
datapath=/data/duongdb/WS22qOther_08102021
datadir=/data/duongdb # ! needed if run sbatch call
cd $datadir/stylegan2-ada-Ws-22q/WS22qOther # ! use styleflow... 
python3 AlignImage.py $datapath/TrimImg $datapath/Align$resolution'BlankBackgroundCenter' $resolution > $datapath/align_log_background.txt


# ----------------------------------------------------------------------------------------------------


# ! align images for survey 
resolution=512 
datapath=/data/duongdb/WS22qOther_08102021

cd $datadir/stylegan2-ada-Ws-22q/WS22qOther # ! use styleflow... 
python3 AlignImage.py $datapath/QualtricImg/WS $datapath/QualtricImg/WSAlign$resolution $resolution > $datapath/WSQualtric_align_log.txt

cd $datadir/stylegan2-ada-Ws-22q/WS22qOther # ! use styleflow... 
python3 AlignImage.py $datapath/QualtricImg/22q11DS $datapath/QualtricImg/22q11DSAlign$resolution $resolution > $datapath/22q11DSQualtric_align_log.txt

# ----------------------------------------------------------------------------------------------------

# ! make tfrecord data.
# ! extract WS + 22q + controls + random-normal-faces -- 5 folds
# ! OVERSAMPLE BY 5X ?? we need to do this because we have 5x more normal vs. diseases 

cd /data/duongdb/stylegan2-ada-Ws-22q

oversample_time=5 # ! OVERSAMPLE BY 5X, so that we have roughly equal chance of selecting affected vs normal people
select_num=500

classifier_train_csv='/data/duongdb/WS22qOther_08102021/Classify/train+blankcenter+WS+22q11DS+Control+Normal+Split.csv' 
datapath=/data/duongdb/WS22qOther_08102021
for fold in 0 1 2 3 4 
do 

  outjson=dataset_ws_22q_control_randnormal_divlabel.json

  # ! create json as label input
  python3 WS22qOther/CreateLabelJsonDivLabelAddNormal.py --image_dir $datapath/Classify/StyleganImg+blankcenter+WS+22q11DS+Control+$select_num'Normal+Fold'$fold --outjson $outjson
  
  # ! create tfrecords
  for resolution in 256  
  do 
  outdir=$datapath/Classify/TfStyleganImg+Os$oversample_time'+blankcenter+WS+22q11DS+Control+'$select_num'Normal+Fold'$fold

  python3 dataset_tool.py create_from_images_with_labels_fromjson $outdir $datapath/Classify/StyleganImg+blankcenter+WS+22q11DS+Control+$select_num'Normal+Fold'$fold --resolution $resolution --labeljson_name $outjson --use_these_label '22q11DS2y,22q11DSadolescence,22q11DSolderadult,22q11DSyoungadult,22q11DSyoungchild,Controls2y,Controlsadolescence,Controlsolderadult,Controlsyoungadult,Controlsyoungchild,WS2y,WSadolescence,WSolderadult,WSyoungadult,WSyoungchild,Normal2y,Normaladolescence,Normalolderadult,Normalyoungadult,Normalyoungchild' --classifier_train_csv $classifier_train_csv --fold $fold --oversample_time $oversample_time --oversample_label '22q11DS2y,22q11DSadolescence,22q11DSolderadult,22q11DSyoungadult,22q11DSyoungchild,Controls2y,Controlsadolescence,Controlsolderadult,Controlsyoungadult,Controlsyoungchild,WS2y,WSadolescence,WSolderadult,WSyoungadult,WSyoungchild'

  done 
done 

# [2646. 3210. 2500. 2298. 1814. 2222. 1100. 1682. 3836.]

# ------------------------------------------------------------------------------------------------------------------------------------------------
