'''
TRAIN
alfonsomedela
alfonmedela@gmail.com
alfonsomedela.com
'''

from fastai.vision import *
import argparse

parser = argparse.ArgumentParser(
    description='Training function')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--bs', type=int,default=8)
parser.add_argument('--lr_find', type=bool,default=False)
parser.add_argument('--lr', type=float,default=0)
parser.add_argument('--val_pct', type=float,default=0.2)
parser.add_argument('--path', type=str)
parser.add_argument('--model_dir', type=str, default='models/')


if __name__ == '__main__':

    args = parser.parse_args()

    # CUDA device
    device = args.device

    torch.cuda.set_device(device)

    bs = args.bs
    size = 512

    val_pct = args.val_pct

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

    lr_find = args.lr_find
    lr_def = args.lr

    print('LR FIND: ', lr_find)

    # define metrics
    metrics = [partial(dice, iou=True), dice]

    wd = 1e-2
    learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)
    learn.model_dir = args.model_dir

    if lr_find:
        learn.lr_find()
        fig = learn.recorder.plot(return_fig=True)
        fig.savefig('lr_figure_freezed.png')
    else:
        print('training...')
        if lr_def == 0:
            lr = 1e-3
        else:
            lr = lr_def

        learn.fit_one_cycle(10, slice(lr), pct_start=0.9)
        learn.save('stage_1')
        print('Finished training head')

        learn.unfreeze()
        lrs = slice(lr / 400, lr / 4)
        learn.fit_one_cycle(12, lrs, pct_start=0.8)
        learn.save('stage_2')
        print('DONE')







