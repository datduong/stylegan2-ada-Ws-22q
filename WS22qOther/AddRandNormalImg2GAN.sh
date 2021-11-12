
cd /data/duongdb/stylegan2-ada-Ws-22q/WS22qOther

maindir=/data/duongdb/WS22qOther_08102021/Classify

foldcsv=$maindir/train+blankcenter+WS+22q11DS+Control+Normal+Split.csv # ! which train.csv ? 

ourimgpath=/data/duongdb/WS22qOther_08102021/Align512BlankBackgroundCenter # Align512 # ! our source WS q22 etc..

ourimgtocopy='WS,22q11DS,Controls'

select_num=500 # ! how many fake normal to be used ?? 

for fold in 0 1 2 3 4 
do 

foutpathimg=$maindir/StyleganImg+blankcenter+WS+22q11DS+Control+$select_num'Normal+Fold'$fold

rm -rf $foutpathimg # ! just to be safe
mkdir $foutpathimg

foutpath=$maindir/'normal-img-gan-fold'$fold.csv
python3 AddRandNormalImg2GAN.py --select_num $select_num --foutpath $foutpath --foldcsv $foldcsv --fold $fold --ourimgpath $ourimgpath --foutpathimg $foutpathimg --ourimgtocopy $ourimgtocopy

done

cd $maindir

# fold 0
# (42804, 5)
# Normal2y
# (557, 5)
# Normaladolescence
# (3242, 5)
# Normalolderadult
# (12938, 5)
# Normalyoungadult
# (19201, 5)
# Normalyoungchild
# (5496, 5)