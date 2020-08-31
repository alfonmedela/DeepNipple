'''
DEEPNIPPLE SEG TEST
alfonsomedela
alfonmedela@gmail.com
alfonsomedela.com
'''


from fastai.vision import *
import cv2
import glob
from sklearn.metrics import classification_report
from utils.code.metrics import mIOU

def predict(path, test_mask_path):

    img = PIL.Image.open(path)
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

    mask = cv2.resize(mask, (b, a))

    mask_flatten_int = mask.reshape((a * b, mask.shape[-1]))
    mask_flatten = np.argmax(mask_flatten_int, -1)

    path_to_annotation = test_mask_path + path.split('/')[-1].split('.')[0] + '.png'
    label = cv2.imread(path_to_annotation, -1)
    y = label.reshape(label.shape[0] * label.shape[1])
    return y, mask_flatten, mask_flatten_int

def obtain_metrics(test_paths, test_mask_path):

    PRECISION_0, RECALL_0, F1_0 = [], [], []
    PRECISION_1, RECALL_1, F1_1 = [], [], []
    PRECISION_2, RECALL_2, F1_2 = [], [], []
    for i in range(len(test_paths)):

        y, mask_flatten, mask_flatten_int = predict(test_paths[i], test_mask_path)

        # report
        rp = classification_report(y, mask_flatten, output_dict=True)
        background_p = rp['0']['precision']
        background_recall = rp['0']['recall']
        background_f1 = rp['0']['f1-score']
        PRECISION_0.append(background_p)
        RECALL_0.append(background_recall)
        F1_0.append(background_f1)

        nipple_p = rp['1']['precision']
        nipple_recall = rp['1']['recall']
        nipple_f1 = rp['1']['f1-score']
        PRECISION_1.append(nipple_p)
        RECALL_1.append(nipple_recall)
        F1_1.append(nipple_f1)

        areola_p = rp['2']['precision']
        areola_recall = rp['2']['recall']
        areola_f1 = rp['2']['f1-score']
        PRECISION_2.append(areola_p)
        RECALL_2.append(areola_recall)
        F1_2.append(areola_f1)

        # meanIOU
        y = torch.from_numpy(y)
        y_pred = torch.from_numpy(mask_flatten_int)
        res_iou = mIOU(y, y_pred)
        print(res_iou)

    PRECISION_0, RECALL_0, F1_0 = np.asarray(PRECISION_0), np.asarray(RECALL_0), np.asarray(F1_0)
    PRECISION_1, RECALL_1, F1_1 = np.asarray(PRECISION_1), np.asarray(RECALL_1), np.asarray(F1_1)
    PRECISION_2, RECALL_2, F1_2 = np.asarray(PRECISION_2), np.asarray(RECALL_2), np.asarray(F1_2)

    print(np.mean(PRECISION_0), np.std(PRECISION_0))
    print(np.mean(RECALL_0), np.std(RECALL_0))
    print(np.mean(F1_0), np.std(F1_0))

    print(' ')
    print(np.mean(PRECISION_1), np.std(PRECISION_1))
    print(np.mean(RECALL_1), np.std(RECALL_1))
    print(np.mean(F1_1), np.std(F1_1))

    print(' ')
    print(np.mean(PRECISION_2), np.std(PRECISION_2))
    print(np.mean(RECALL_2), np.std(RECALL_2))
    print(np.mean(F1_2), np.std(F1_2))


if __name__ == '__main__':

    # load pyrtoch model
    learner_path = 'utils/models/base-model/'
    learn = load_learner(learner_path)

    # path to test images
    test_img_path = 'IMAGES-PATH'
    test_paths = glob.glob(test_img_path + '*')

    # path to test labels
    test_mask_path = 'LABELS-PATH'

    obtain_metrics(test_paths, test_mask_path)



















