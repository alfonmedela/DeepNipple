from fastai.vision import *
import argparse
import cv2
import glob
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score

def mIOU(label, pred, num_classes=3):

    pred = torch.argmax(pred, dim=1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)


parser = argparse.ArgumentParser(
    description='Training function')
parser.add_argument('--device', type=int,
                    help='path to a test image or folder of images',
                    default=1)
parser.add_argument('--bs', type=int,
                    help='path to a test image or folder of images',
                    default=16)
parser.add_argument('--lr_find', type=bool,
                    help='path to a test image or folder of images',
                    default=False)
parser.add_argument('--lr', type=float,
                    help='path to a test image or folder of images',
                    default=0)
parser.add_argument('--val_pct', type=float,
                    help='path to a test image or folder of images',
                    default=0.2)
parser.add_argument('--path', type=str,
                    default='/Users/alfonsomedela/Desktop/Personal/datasets/TheNippleProject/1-Originals/base_model/train/')

parser.add_argument('--model_dir', type=str, default='models/')

def seg2bbox(img, mask, show = False):

    mask[mask != 0] = 1

    mask = mask * 255

    #plt.imshow(mask)
    #plt.show()

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    mask = mask.astype(np.uint8)

    print(img.shape, mask.shape, np.max(img), np.max(mask))

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    blobs = 0
    spacing = 0.4
    coordinates = []
    for c in cnts:
        x1, x2 = np.min(c[:, 0, 0]), np.max(c[:, 0, 0])
        y1, y2 = np.min(c[:, 0, 1]), np.max(c[:, 0, 1])

        dx = x2 - x1
        dy = y2 - y1

        diff = min(dx, dy)

        x1 = int(x1 - spacing * diff / 2.0)
        x2 = int(x2 + spacing * diff / 2.0)
        y1 = int(y1 - spacing * diff / 2.0)
        y2 = int(y2 + spacing * diff / 2.0)

        coordinates.append([y1, y2, x1, x2])

        cv2.rectangle(img, (x1, y1), (x2, y2),(36,255,12), 2, -1)
        blobs += 1

    #print('blobs/boobs:', blobs)
    if show:
        plt.imshow(img)
        plt.show()
    return blobs


if __name__ == '__main__':

    args = parser.parse_args()

    # CUDA device
    device = args.device

    if device == 0:
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
    learn.model_dir = '/Users/alfonsomedela/Desktop/Personal/TheNippleProject/nipple-detector/base_model/base_models/'

    learn.load('stage_2')

    test_paths = glob.glob('/Users/alfonsomedela/Desktop/Personal/datasets/TheNippleProject/1-Originals/base_model/test/images/*')
    path_lab = '/Users/alfonsomedela/Desktop/Personal/datasets/TheNippleProject/1-Originals/base_model/test/labels/'

    y_nipple = []
    y_predicted = []
    for i in range(len(test_paths)):

        img = PIL.Image.open(test_paths[i])
        original_image = np.array(img)
        a = original_image.shape[0]
        b = original_image.shape[1]
        if a > b:
            original_image = cv2.resize(original_image, (b, b))
        else:
            original_image = cv2.resize(original_image, (a, a))

        img = PIL.Image.fromarray(original_image).convert('RGB')
        img = pil2tensor(img, np.float32)
        img = img.div_(255)
        img = Image(img)

        _, _, mask = learn.predict(img)

        mask = mask.detach().numpy()
        c1 = mask[0, :, :, np.newaxis]
        c2 = mask[1, :, :, np.newaxis]
        c3 = mask[2, :, :, np.newaxis]
        mask = np.concatenate((c1, c2, c3), axis=-1)

        original_image = cv2.resize(original_image, (b, a))
        mask = cv2.resize(mask, (b, a))
        mask = np.argmax(mask, axis=-1)

        detected = seg2bbox(original_image, mask)

        # LABEL
        path_to_annotation = path_lab + test_paths[i].split('/')[-1].split('.')[0] + '.png'
        label = cv2.imread(path_to_annotation)
        real = seg2bbox(original_image, label)

        print(real, detected)

        y_nipple.append(1)
        if detected != 0:
            y_predicted.append(1)
        else:
            y_predicted.append(0)

    no_nipple_images = glob.glob('/Users/alfonsomedela/Desktop/Personal/DeepNipple/data/other/no-nipple/*')
    for i in range(len(no_nipple_images)):

        img = PIL.Image.open(no_nipple_images[i])
        original_image = np.array(img)
        a = original_image.shape[0]
        b = original_image.shape[1]
        if a > b:
            original_image = cv2.resize(original_image, (b, b))
        else:
            original_image = cv2.resize(original_image, (a, a))

        img = PIL.Image.fromarray(original_image).convert('RGB')
        img = pil2tensor(img, np.float32)
        img = img.div_(255)
        img = Image(img)

        _, _, mask = learn.predict(img)

        mask = mask.detach().numpy()
        c1 = mask[0, :, :, np.newaxis]
        c2 = mask[1, :, :, np.newaxis]
        c3 = mask[2, :, :, np.newaxis]
        mask = np.concatenate((c1, c2, c3), axis=-1)

        original_image = cv2.resize(original_image, (b, a))
        mask = cv2.resize(mask, (b, a))
        mask = np.argmax(mask, axis=-1)

        detected = seg2bbox(original_image, mask)
        print(detected)

        y_nipple.append(0)

        if detected != 0:
            y_predicted.append(1)
        else:
            y_predicted.append(0)

    cm = confusion_matrix(y_nipple, y_predicted)
    print(cm)



















