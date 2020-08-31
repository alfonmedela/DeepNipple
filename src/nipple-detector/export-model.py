'''
EXPORT BEST MODEL
alfonsomedela
alfonmedela@gmail.com
alfonsomedela.com
'''

from fastai.vision import *
import argparse

parser = argparse.ArgumentParser(
    description='Training function')
parser.add_argument('--device', type=int)
parser.add_argument('--path', type=str, help='Path to dataset')

parser.add_argument('--model_dir', type=str, default='models/')


if __name__ == '__main__':

    args = parser.parse_args()

    # CUDA device
    device = args.device

    if device == 0:
        torch.cuda.set_device(device)

    bs = 2
    size = 512

    val_pct = 0.2

    path = args.path

    path_img = path + 'images/'
    codes = ['Background',
             'Nipple',
             'Areola']

    path_label = path + 'labels/'
    get_y_fn = lambda x: path_label + f'{x.stem}.png'

    # create valid.txt
    src = (SegmentationItemList.from_folder(path_img)
           .split_by_rand_pct(val_pct, seed=666)
           .label_from_func(get_y_fn, classes=codes))


    data = (src.transform(get_transforms(max_zoom=1.3, max_lighting=0.4, max_warp=0.4, p_affine=1., p_lighting=1.),
                          size=size, tfm_y=True)
            .databunch(bs=bs, num_workers=0)
            .normalize(imagenet_stats))

    print(data)
    print(data.classes)
    print(len(data.classes))

    # define metrics
    metrics = [partial(dice, iou=True), dice]

    wd = 1e-2
    learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)
    learn.model_dir = 'MODEL_DIR'

    learn.load('MODEL_NAME')
    learn.export()





















