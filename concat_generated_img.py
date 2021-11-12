

import sys,os,re,pickle
from PIL import Image
import numpy as np 

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, default=None)
parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--folder_prefix", type=str, default=None)
parser.add_argument("--interval", type=str, default=None)
parser.add_argument("--seed_range", type=str, default='0,100')
parser.add_argument("--spec_folder_order", type=str, default=None) # ! specific ordering

args = parser.parse_args()

args.seed_range = [int(i) for i in args.seed_range.split(',')]

if args.outdir is None: 
  args.outdir = os.path.join(args.path,args.folder_prefix)


if not os.path.exists(args.outdir): 
  os.makedirs(args.outdir)


if args.interval is not None: 
  args.interval = args.interval.strip().split() 
  print (args.interval)


if args.spec_folder_order is not None: 
  args.spec_folder_order = [f.strip() for f in args.spec_folder_order.split('-')]

  
LABEL_MAP = {'0':'22q11DS', '1':'Controls', '2':'Normal', '3':'WS'} # WS+22q11DS+Control
GROUP_MAP = {'4':'2y', '5':'adolescence', '6':'olderadult', '7':'youngadult', '8':'youngchild' }

def GetLabelFromName (foldername): 
  # ! wants 3,82,8T0.5M1_WSyoungchild_
  # 3,82,8T0.5M1_WSyoungchild_
  # 3,82,8T0.5M1_WSyoungchild_seed00000021.png
  label1 = LABEL_MAP [ foldername[0] ] # 0,42,4T0.5M.8
  label2 = LABEL_MAP [ foldername[3] ]
  group1 = GROUP_MAP [ foldername[2] ]
  group2 = GROUP_MAP [ foldername[5] ]
  mix = float ( foldername.split('M')[1] ) 
  if mix >= 0.5: 
    label = label1 + group1
  else: 
    label = label2 + group2
  name_ret = foldername+'_'+label+'_'
  return name_ret
  

for seed in np.arange(args.seed_range[0],args.seed_range[1]): 
  if args.spec_folder_order is None:  # ! add in name? or just use seed ?? 
    image_list = [os.path.join(args.path,args.folder_prefix+'M'+str(m), f'seed{seed:08d}.png') for m in args.interval ]
  else: 
    image_list = [os.path.join(args.path,f, f'seed{seed:08d}.png') for f in args.spec_folder_order ]
  #
  images = [Image.open(x) for x in image_list]
  widths, heights = zip(*(i.size for i in images))
  total_width = sum(widths)
  max_height = max(heights)
  new_im = Image.new('RGB', (total_width, max_height))
  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]
  #
  new_im.save(os.path.join(args.outdir,f'seed{seed:04d}.png'))


