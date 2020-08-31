'''
DEEPNIPPLE MAIN SCRIPT
alfonsomedela
alfonmedela@gmail.com
alfonsomedela.com
'''

from fastai.vision import *
import argparse
import cv2

parser = argparse.ArgumentParser(description='DeepNipple algorithm')
parser.add_argument('--img_path', type=str, help='path to the input image')
parser.add_argument('--mode', type=str, default='seg', help='seg or bbox mode')
parser.add_argument('--show', type=bool, default=True, help='show the output')

def seg2bbox(mask):

    mask = np.argmax(mask, axis=-1)
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

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(mask.shape[0]-1, x2)
        y2 = min(mask.shape[1] - 1, y2)

        coordinates.append([y1, y2, x1, x2])
        blobs += 1
    return coordinates


def predict(img_path, learner):

    img = PIL.Image.open(img_path)
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

    _, _, mask = learner.predict(img)

    mask = mask.detach().numpy()
    c1 = mask[0, :, :, np.newaxis]
    c2 = mask[1, :, :, np.newaxis]
    c3 = mask[2, :, :, np.newaxis]
    mask = np.concatenate((c1, c2, c3), axis=-1)

    original_image = cv2.resize(original_image, (b, a))
    mask = cv2.resize(mask, (b, a))

    return original_image, mask

def DeepNipple(image_path, alg_mode, show=True):
    '''
    :param image_path: input image absolute path
    :param alg_mode: seg/bbox
    :param device: cpu or gpu number
    :return: segmentation mask / bounding boxes
    '''

    # load pyrtoch model
    learner_path = 'utils/models/base-model/'
    learner = load_learner(learner_path)

    image, mask = predict(image_path, learner)
    print(image.shape, mask.shape)

    if alg_mode == 'seg':

        output = mask

        plt.subplot(121)
        plt.imshow(image)
        plt.title('Original image')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(image)
        plt.axis('off')
        plt.imshow(mask[:, :, 1], alpha=0.6, interpolation='bilinear', cmap='magma')
        plt.axis('off')
        plt.imshow(mask[:, :, 2], alpha=0.6, interpolation='bilinear', cmap='afmhot')
        plt.axis('off')
        plt.title('NAC segmentation')
        plt.show()

    else:
        coords = seg2bbox(mask)
        output = coords

        if show:
            for coor in coords:
                y1, y2, x1, x2 = coor[0], coor[1], coor[2], coor[3]
                cv2.rectangle(image, (x1, y1), (x2, y2), (36, 255, 12), 2, -1)

            plt.imshow(image)
            plt.show()

    return output

if __name__ == '__main__':

    print('Running DeepNipple...')

    args = parser.parse_args()
    image_path = args.img_path
    alg_mode = args.mode
    show = args.show

    output = DeepNipple(image_path, alg_mode, show)



