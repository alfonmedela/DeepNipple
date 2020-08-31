'''
SEG2BBOX
alfonsomedela
alfonmedela@gmail.com
alfonsomedela.com
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt


def seg2bbox(img, mask, show, save):

    mask[mask != 0] = 1

    mask = mask * 255
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

    if save:
        cv2.imwrite('seg2bbox.png', mask)




if __name__ == '__main__':

    image_path = 'PATH TO IMAGES'
    mask_path = 'PATH TO MASK'

    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    seg2bbox(img, mask, show=True, save=False)
