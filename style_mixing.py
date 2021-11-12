# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate style mixing image matrix using pretrained network pickle."""

import argparse
import os,sys
import pickle
import re

import numpy as np
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib

#----------------------------------------------------------------------------

def style_mixing_example(network_pkl, row_seeds, col_seeds, truncation_psi, col_styles, outdir, minibatch_size=4, class_idx=None, num_labels=0, dlatents_np=None, suffix=None, save_individual=0, grid=1):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    w_avg = Gs.get_var('dlatent_avg') # [component]
    Gs_syn_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False,
        'minibatch_size': minibatch_size
    }

    all_labels = None
    if dlatents_np is None:
        print('Generating W vectors...') # ! by default, looks like @labels are not used. 
        all_seeds = list(set(row_seeds + col_seeds))
        all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
        # ! add in labels 
        if class_idx is not None: 
            all_labels = np.zeros((len(all_seeds), num_labels)) # [minibatch, component] for example, this is 16xnum_labels (1-hot)
            all_labels[:, class_idx] = 1
        #
        all_w = Gs.components.mapping.run(all_z, all_labels) # [minibatch, layer, component]
        all_w = w_avg + (all_w - w_avg) * truncation_psi # [minibatch, layer, component]
    
    else: 
        # ! load W
        print(f'load dlatents file "{dlatents_np}"')
        files = [f for f in sorted(os.listdir(dlatents_np)) if 'npz' in f]
        all_seeds = list(set(row_seeds + col_seeds)) # ! seeds are just dummy holder in this case. 
        print ('row seed images {}'.format(np.array(files)[row_seeds]))
        print ('col seed images {}'.format(np.array(files)[col_seeds]))
        all_w = []
        for f in files: 
            w = np.load(os.path.join(dlatents_np,f))['dlatents'] # ! should be 16 x 512 (or something like that)
            # print (w.shape)
            # ! note: @w is "optim all layers" so we don't have 1 single w that is replicated and then passed to generator
            # w = np.expand_dims(w, axis=0) # ! make it 1x16x512
            all_w.append(w) 
        # 
        all_w = np.concatenate(all_w) # [minibatch, layer, component]
        # all_w = w_avg + (all_w - w_avg) * truncation_psi # [minibatch, layer, component] # ! should we do this ? 

    # all w
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))} # [layer, component]

    print('Generating images...')
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs) # [minibatch, height, width, channel]
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    if grid == 0: 
        print('Generating style-mixed images FOR PAIRS...') # ! do not need whole grid
        for row_seed,col_seed in zip(row_seeds,col_seeds):
            w = w_dict[row_seed].copy() # ! pick a row
            w[col_styles] = w_dict[col_seed][col_styles] # ! copy the value of the column into row
            w = w_avg + (w - w_avg) * truncation_psi # [minibatch, layer, component] # ! results can look bad without truncation
            image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
            image_dict[(row_seed, col_seed)] = image
    else:     
        print('Generating style-mixed images...')
        for row_seed in row_seeds:
            for col_seed in col_seeds:
                w = w_dict[row_seed].copy() # ! pick a row
                w[col_styles] = w_dict[col_seed][col_styles] # ! copy the value of the column into row
                w = w_avg + (w - w_avg) * truncation_psi # [minibatch, layer, component] # ! results can look bad without truncation
                image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
                image_dict[(row_seed, col_seed)] = image

    # ! suffix 
    suffix = '' if suffix is None else suffix

    if save_individual == 1: 
        print('Saving images...')
        os.makedirs(outdir, exist_ok=True)
        for (row_seed, col_seed), image in image_dict.items():
            if row_seed != col_seed: 
                PIL.Image.fromarray(image, 'RGB').save(f'{outdir}/{row_seed}-{col_seed}-{suffix}.png') # could be useful

    if grid == 0: 
        sys.exit('not do grid')

    print('Saving image grid...')
    _N, _C, H, W = Gs.output_shape
    canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([None] + row_seeds):
        for col_idx, col_seed in enumerate([None] + col_seeds):
            if row_seed is None and col_seed is None:
                continue
            key = (row_seed, col_seed)
            if row_seed is None:
                key = (col_seed, col_seed)
            if col_seed is None:
                key = (row_seed, row_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
    #
    canvas.save(f'{outdir}/grid{suffix}.png')

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

  python %(prog)s --outdir=out --trunc=1 --rows=85,100,75,458,1500 --cols=55,821,1789,293 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate style mixing image matrix using pretrained network pickle.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser.add_argument('--rows', dest='row_seeds', type=_parse_num_range, help='Random seeds to use for image rows', required=True)
    parser.add_argument('--cols', dest='col_seeds', type=_parse_num_range, help='Random seeds to use for image columns', required=True)
    parser.add_argument('--styles', dest='col_styles', type=_parse_num_range, help='Style layer range (default: %(default)s)', default='0-6')
    parser.add_argument('--trunc', dest='truncation_psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser.add_argument('--outdir', help='Where to save the output images', required=True, metavar='DIR')
    parser.add_argument('--class', dest='class_idx', type=int, help='Class index (default: %(default)s)', default=None)
    parser.add_argument('--num_labels', dest='num_labels', type=int, default=0)
    parser.add_argument('--dlatents_np', dest='dlatents_np', type=str, default=None)
    parser.add_argument('--suffix', dest='suffix', type=str, default=None)
    parser.add_argument('--grid', dest='grid', type=int, default=1)
    parser.add_argument('--save_individual', dest='save_individual', type=int, default=0)
    
    
    args = parser.parse_args()
    style_mixing_example(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
