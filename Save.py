import torch
import cv2
from torch.autograd import Variable
import os


def save_fig(net, loader, images_dir):
    for batch in loader:
        # batch = next(loader)
        inputs = Variable(batch['image'])
        labels = Variable(batch['landmark'])
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        dtype = torch.FloatTensor

        outputs = net(inputs)
        landmarks = outputs.data.type(dtype).numpy()

        with open('landmarks.txt', 'w') as f:
            for i in range(landmarks.shape[0]):
                m = landmarks[i]
                s = str(i) + ' *'
                for j in range(8):
                    s += ' ' + str(m[j])
                s += '\n'
                f.write(s)

        images_path = []
        for i in range(200):
            path = os.path.join(images_dir, '{:0>4d}.jpg'.format(i))
            img = cv2.imread(path)
            if img is not None:
                images_path.append(path)

        for i in range(0, len(images_path)):
            img = cv2.imread(images_path[i])
            height, width, _ = img.shape
            landmarks[i, [0, 2, 4, 6]] *= width
            landmarks[i, [1, 3, 5, 7]] *= height

            p = landmarks[i]
            p1 = (p[0], p[1])
            p2 = (p[2], p[3])
            p3 = (p[4], p[5])
            p4 = (p[6], p[7])
            color = (0, 0, 255)
            cv2.line(img, p1, p2, color, 3)
            cv2.line(img, p2, p3, color, 3)
            cv2.line(img, p3, p4, color, 3)
            cv2.line(img, p4, p1, color, 3)

            cv2.imwrite('data_save/{}.jpg'.format(i), img)

        with open('landmarks.txt', 'w') as f:
            for i in range(landmarks.shape[0]):
                m = landmarks[i]
                s = str(i) + ' *'
                for j in range(8):
                    s += ' ' + str(int(m[j]))
                s += '\n'
                f.write(s)
