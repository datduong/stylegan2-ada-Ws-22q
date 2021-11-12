# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import numpy as np
import dnnlib
import dnnlib.tflib as tflib

import os
import re
import sys
import pickle

import PIL.Image

import projector
import pretrained_networks
from training import dataset
from training import misc

import pickle 
from datetime import datetime

#----------------------------------------------------------------------------

def project_image(proj, targets, png_prefix, num_snapshots,save_img=False):
    # snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))
    if save_img: 
        misc.save_image_grid(targets, png_prefix + 'target.png', drange=[-1,1])
    proj.start(targets)
    while proj.cur_step < proj.num_steps: # get_cur_step --> cur_step 
        dist, loss = proj.step()
        print('\r%d / %d ... %.4f %.2f' % (proj.cur_step, proj.num_steps, dist[0], loss), end='', flush=True)
        # if proj.cur_step in snapshot_steps:
        #     misc.save_image_grid(proj.get_images(), png_prefix + 'step%04d.png' % proj.cur_step, drange=[-1,1])
    print('\r%-30s\r' % '', end='', flush=True)
    print('end loss %.4f %.2f' % (dist[0], loss))

    # Save results.
    if save_img:
        PIL.Image.fromarray(proj.images_uint8[0], 'RGB').save(png_prefix + 'step%04d.png' % proj.cur_step)
    np.savez(png_prefix+'dlatents.npz', dlatents=proj.dlatents)


#----------------------------------------------------------------------------

def project_generated_images(network_pkl, seeds, num_snapshots, truncation_psi):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    proj = projector.Projector()
    proj.set_network(Gs)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print('Projecting seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
        images = Gs.run(z, None, **Gs_kwargs) # ! we create an image from Z, and then we project this image to get the W
        project_image(proj, targets=images, png_prefix=dnnlib.make_run_dir_path('seed%04d-' % seed), num_snapshots=num_snapshots)

#----------------------------------------------------------------------------

def project_real_images(network_pkl, data_dir, num_images, num_snapshots, classid=None, tiled=1, num_steps=100, img_range=None, save_img=0):

    tiled = False if tiled==0 else True
    save_img = False if save_img==0 else True

    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)  
    proj = projector.Projector(num_steps=num_steps, classid=classid, tiled=tiled) # ! for 1 class id
    proj.set_network(Gs)

    print('Loading images from "%s"...' % data_dir)
    # path=None, resolution=None, max_images=None, max_label_size=0, mirror_augment=False, repeat=True, shuffle=True, seed=None
    dataset_obj = dataset.load_dataset(path=data_dir, max_label_size=0, repeat=False, shuffle=False)
    assert dataset_obj.shape == Gs.output_shape[1:]

    print ('num img {}'.format(num_images))
    todo = np.array( range(num_images) )
    
    if img_range is not None: 
        img_range = [int(i.strip()) for i in img_range.split(',')]
        if img_range[1] > len(todo): img_range[1] = len(todo) # just safety
        todo_range = todo[img_range[0]:img_range[1]] # ! just do subset
        print ('will just do {}'.format(len(todo_range)))
    else: 
        todo_range = np.array( range(num_images) )

    for image_idx in todo: # iter over all ranges
        images, _labels = dataset_obj.get_minibatch_np(1) # ! MINIBATCH RETURNS ONE OBSERVATION. so we have to read in each time. 
        if image_idx not in todo_range: # ! only do what is needed
            continue 
        png_prefix=dnnlib.make_run_dir_path('image%04d-' % image_idx)
        if os.path.exists(png_prefix+'dlatents.npz') : 
            print('Skip, because exists, image %d/%d ...' % (image_idx, num_images))
        else: 
            print('Projecting image %d/%d ...' % (image_idx, num_images))
            images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
            # ! projecting real image into W space, so we actually don't know what is the z and label vec. 
            project_image(proj, targets=images, png_prefix=png_prefix, num_snapshots=num_snapshots, save_img=save_img)

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

  # Project generated images
  python %(prog)s project-generated-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=0,1,5

  # Project real images
  python %(prog)s project-real-images --network=gdrive:networks/stylegan2-car-config-f.pkl --dataset=car --data-dir=~/datasets

'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 projector.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    project_generated_images_parser = subparsers.add_parser('project-generated-images', help='Project generated images')
    project_generated_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_generated_images_parser.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', default=range(3))
    project_generated_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=5)
    project_generated_images_parser.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=1.0)
    project_generated_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    project_real_images_parser = subparsers.add_parser('project-real-images', help='Project real images')
    project_real_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_real_images_parser.add_argument('--data-dir', help='TF record data path', required=True) # 'Dataset root directory'
    # project_real_images_parser.add_argument('--dataset', help='Training dataset', dest='dataset_name', required=True)
    project_real_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=5)
    project_real_images_parser.add_argument('--num-images', type=int, help='Number of images to project (default: %(default)s)', default=3)
    project_real_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    project_real_images_parser.add_argument('--classid', type=str, help='index in format 1,2,3,...', default=None)
    project_real_images_parser.add_argument('--tiled', type=int, help='tiled', default=1) # ! if true, then use W(1,*)
    project_real_images_parser.add_argument('--num_steps', type=int, help='num_steps', default=100)
    project_real_images_parser.add_argument('--img_range', type=str, help='img_range', default=None)
    project_real_images_parser.add_argument('--save_img', type=int, help='save img', default=0)
    
    
    args = parser.parse_args()
    subcmd = args.command
    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    kwargs = vars(args)
    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = kwargs.pop('command')

    func_name_map = {
        'project-generated-images': 'run_projector.project_generated_images',
        'project-real-images': 'run_projector.project_real_images'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
