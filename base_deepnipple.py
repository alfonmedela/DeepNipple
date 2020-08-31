'''
DEEPNIPPLE MAIN SCRIPT
alfonsomedela
alfonmedela@gmail.com
alfonsomedela.com
'''

import argparse

from utils.code.deep_nipple import DeepNipple

parser = argparse.ArgumentParser(description='DeepNipple algorithm')
parser.add_argument('--img_path', type=str, help='path to the input image')
parser.add_argument('--mode', type=str, default='seg', help='seg or bbox mode')
parser.add_argument('--show', type=bool, default=True, help='show the output')


if __name__ == '__main__':

    print('Running DeepNipple...')

    args = parser.parse_args()
    image_path = args.img_path
    alg_mode = args.mode
    show = args.show

    output = DeepNipple(image_path, alg_mode, show)



