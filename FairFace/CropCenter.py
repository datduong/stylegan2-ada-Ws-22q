
import os,sys,re,pickle 
import numpy as np 
import pandas as pd
import PIL.Image
from tqdm import tqdm 

os.chdir('/data/duongdb/FairFace')

# ! take what we already aligned, crop/center ?? 

# why 2 folders don't match ?? 

old_path_ff = '/data/duongdb/FairFace/FairFace-aligned-60k-agegroup-06012021/'
new_path_ff = '/data/duongdb/FairFace/FairFace-aligned-60k-agegroup-06012021-BlankBackgroundCenter/'


# ! need to rename fairface ??
img_list = os.listdir('FairFace-aligned-60k-agegroup-06012021-BlankBackgroundCenter')
for img in img_list: 
  new_name = re.sub(r'_01.png',r'.png', img)
  # break 
  os.system ( 'mv ' + os.path.join(new_path_ff, img) + ' ' + os.path.join(new_path_ff, new_name) )


# 
old_img_list = os.listdir(old_path_ff)
new_img_list = os.listdir(new_path_ff)

centerface='25,25,487,487'
centerface = [ int(i) for i in centerface.split(',') ]

missing = sorted( list (set (old_img_list) - set(new_img_list)) )

for img in tqdm (missing): 
  src_file = os.path.join(old_path_ff,img)
  dst_file = os.path.join(new_path_ff,img)
  img = PIL.Image.open(src_file).convert('RGBA').convert('RGB')
  img = img.resize((512, 512), PIL.Image.ANTIALIAS)
  img = img.crop(centerface)  # coordinates of the crop im1 = im.crop((left, top, right, bottom))
  img.save(dst_file, 'PNG')
  
  

