'''
DEEPNIPPLE DET TEST
alfonsomedela
alfonmedela@gmail.com
alfonsomedela.com
'''

from fastai.vision import *
import cv2
import glob
from sklearn.metrics import confusion_matrix


def seg2bbox(img, mask, show=False):

    mask[mask != 0] = 1
    mask = mask * 255

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    mask = mask.astype(np.uint8)

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

    if show:
        plt.imshow(img)
        plt.show()
    return blobs

def predict(input_img, y_nipple, y_predicted, cl):

    img = PIL.Image.open(input_img)
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

    y_nipple.append(cl)

    if detected != 0:
        y_predicted.append(1)
    else:
        y_predicted.append(0)

    return y_nipple, y_predicted

def obtain_metrics(test_paths, no_nipple_images):

    y_nipple = []
    y_predicted = []
    for i in range(len(test_paths)):
        y_nipple, y_predicted = predict(no_nipple_images[i], y_nipple, y_predicted, cl=1)

    for i in range(len(no_nipple_images)):
        y_nipple, y_predicted = predict(no_nipple_images[i], y_nipple, y_predicted, cl=0)

    cm = confusion_matrix(y_nipple, y_predicted)
    return cm

if __name__ == '__main__':

    # load pytorch model
    learner_path = 'utils/models/base-model/'
    learn = load_learner(learner_path)

    test_imgs_path = '/Users/alfonsomedela/Desktop/Personal/datasets/TheNippleProject/1-Originals/base_model/test/images/*'
    test_paths = glob.glob(test_imgs_path)

    no_nipple_imgs_path = '/Users/alfonsomedela/Desktop/Personal/DeepNipple/data/other/no-nipple/*'
    no_nipple_images = glob.glob(no_nipple_imgs_path)

    # get confusion matrix to calculate sensitivity and specificity
    cm = obtain_metrics(test_paths, no_nipple_images)



















