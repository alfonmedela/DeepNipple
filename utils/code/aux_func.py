from fastai.vision import *
import cv2

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