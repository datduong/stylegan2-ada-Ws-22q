# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import argparse
import os
import pickle
import re

import numpy as np
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib

from copy import deepcopy

#----------------------------------------------------------------------------

def generate_images(network_pkl, seeds, truncation_psi, outdir, class_idx, dlatents_npz, mix_ratio=None, class_idx_next=None, suffix='', savew=0, special_mix_ratio=None, prefix='', dpi=0):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    os.makedirs(outdir, exist_ok=True)

    # Render images for a given dlatent vector.
    if dlatents_npz is not None:
        print(f'Generating images from dlatents file "{dlatents_npz}"') # ! here we take in the W from z-->linear-->W 
        dlatents = np.load(dlatents_npz)['dlatents']
        assert dlatents.shape[1:] == (18, 512) # [N, 18, 512] # ! 9 for the downward and 9 for the upward Unet-style
        imgs = Gs.components.synthesis.run(dlatents, output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))
        for i, img in enumerate(imgs):
            fname = f'{outdir}/dlatent{i:02d}.png'
            print (f'Saved {fname}')
            PIL.Image.fromarray(img, 'RGB').save(fname,dpi=(dpi,dpi))
        return

    # Render images for dlatents initialized from random seeds.
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
    }
    if truncation_psi is not None:
        Gs_kwargs['truncation_psi'] = truncation_psi

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    label = np.zeros([1] + Gs.input_shapes[1][1:])

    assert class_idx is not None
    class_idx = [int(s.strip()) for s in class_idx.split(',')]
    label [:, class_idx ] = 1
        
    if class_idx_next is not None: 
        if ',' in mix_ratio: # ! mix when a 2nd label exists
            mix_ratio = np.fromstring(mix_ratio, dtype=float, sep=',').reshape((1,label.shape[1])) # ! broadcast and multiply later with @label
        else: 
            mix_ratio = float(mix_ratio)
        #
        class_idx_next = [int(s.strip()) for s in class_idx_next.split(',')]
        label_next = np.zeros([1] + Gs.input_shapes[1][1:])
        label_next[:, class_idx_next] = 1 # new label 
        if special_mix_ratio == 0: 
            label = mix_ratio*label + (1-mix_ratio)*label_next # linear interpolation
        else: 
            label = mix_ratio*label + special_mix_ratio*label_next
            label[:, class_idx_next[3]] = 1 # ! we take 4 diseases, the last position is "age", set same age for all diseases

    print ('see label')
    print (label)

    if savew: 
        all_w = []
        w_avg = Gs.get_var('dlatent_avg') # [component]
        Gs_syn_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
        }

    # tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width] # ! noise
    
    for seed_idx, seed in enumerate(seeds):

        if seed_idx % 25 == 0: 
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component] # ! random vec Z

        if savew: # if save W for later classifier? 
            tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
            w = Gs.components.mapping.run(z, label)
            w = w_avg + (w - w_avg) * truncation_psi # [minibatch, layer, component]
            images = Gs.components.synthesis.run(w, **Gs_syn_kwargs)
            all_w.append(w) 
        else: 
            tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
            images = Gs.run(z, label, **Gs_kwargs) # [minibatch, height, width, channel] # ! @label is used here
        
        PIL.Image.fromarray(images[0], 'RGB').save(f'{outdir}/'+prefix+f'seed{seed:08d}'+suffix+'.png',dpi=(dpi,dpi))

    if savew:
        w_dict = {seed: w for seed, w in zip(seeds, list(all_w))} # [layer, component]
        pickle.dump(w_dict,open(f'{outdir}/all_w'+suffix+'.pickle','wb'))

#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

_examples = '''examples:

  # Generate curated MetFaces images without truncation (Fig.10 left)
  python %(prog)s --outdir=out --trunc=1 --seeds=85,265,297,849 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

  # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
  python %(prog)s --outdir=out --trunc=0.7 --seeds=600-605 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

  # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
  python %(prog)s --outdir=out --trunc=1 --seeds=0-35 --class=1 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/cifar10.pkl

  # Render image from projected latent vector
  python %(prog)s --outdir=out --dlatents=out/dlatents.npz \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate images using pretrained network pickle.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--seeds', type=_parse_num_range, help='List of random seeds')
    g.add_argument('--dlatents', dest='dlatents_npz', help='Generate images for saved dlatents')
    parser.add_argument('--trunc', dest='truncation_psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser.add_argument('--class', dest='class_idx', type=str, default=None, help='Class label (default: unconditional)')
    parser.add_argument('--class_next', dest='class_idx_next', type=str, default=None, help='next label to morph')
    parser.add_argument('--outdir', help='Where to save the output images', required=True, metavar='DIR')
    parser.add_argument('--mix_ratio', type=str, default='0.25', help='Mix before/after labels, we will take in an array, use string 1,2,3')
    parser.add_argument('--special_mix_ratio', type=float, default=0, help='meant for split label emb')
    parser.add_argument('--suffix', type=str, default='', help='Add extra name')
    parser.add_argument('--prefix', type=str, default='', help='Add extra name')
    parser.add_argument('--savew', type=int, default=0, help='save w?')
    parser.add_argument('--dpi', type=int, default=300, help='dpi')
    
    args = parser.parse_args()
    generate_images(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
