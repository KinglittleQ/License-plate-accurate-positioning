import cv2
import numpy as np
import os


def transform(images_dir, landmarks_path, save_path, h=200, w=400):
    images_path = []
    for i in range(3000):
        path = os.path.join(images_dir, '{:0>4d}.jpg'.format(i))
        if os.path.exists(path):
            images_path.append(path)

    landmarks = []
    with open(landmarks_path) as f:
        for line in f.readlines():
            toks = line.strip().split()[2:]
            toks = list(map(int, toks))
            landmarks.append(toks)
    landmarks = np.array(landmarks)

    for i in range(landmarks.shape[0]):
        p = landmarks[i]
        img = cv2.imread(images_path[i])
        pts1 = np.float32(
            [
                [p[0], p[1]],
                [p[2], p[3]],
                [p[4], p[5]],
                [p[6], p[7]]
            ])

        pts2 = np.float32([[w, 0], [w, h], [0, h], [0, 0]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        res = cv2.warpPerspective(img, M, (w, h))

        cv2.imwrite(os.path.join(save_path, '{}.jpg'.format(i)), res)


if __name__ == '__main__':
    save_path = '/home/deng/Documents/recognization/resnet18/result'
    images_dir = '/home/deng/Documents/recognization/data/test'
    landmarks_path = '/home/deng/Documents/recognization/resnet18/landmarks.txt'
    transform(images_dir, landmarks_path, save_path)

