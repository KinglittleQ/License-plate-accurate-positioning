from LoadData import *
from Save import save_fig
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import copy
from time import time


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        n = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(n, 8)

    def forward(self, x):
        out = self.model(x)
        out = torch.nn.functional.sigmoid(out)
        return out


def main():
    train_images_dir = 'data/train'
    train_landmarks_path = 'data/train.txt'
    test_images_dir = 'data/test'
    test_landmarks_path = 'data/test.txt'

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    train_transform = transforms.Compose([Rescale(224), ToTensor(), Normalize(mean, std)])
    train_dataset = PlateNumberDataset(train_images_dir, train_landmarks_path, train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, num_workers=0)

    test_transform = transforms.Compose([Rescale(224), ToTensor(), Normalize(mean, std)])
    test_dataset = PlateNumberDataset(test_images_dir, test_landmarks_path, test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False, num_workers=0)

    net = Net()
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        net = net.cuda()

    lr = 0.001
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    print('start training...')

    epochs = 30
    min_loss = 100.0
    best_epoch = 0
    best_state = None
    for i in range(epochs):
        beg = time()

    #     j = 0;
        net.train(True)
        for j, batch in enumerate(train_loader):
            inputs = Variable(batch['image'])
            labels = Variable(batch['landmark'])
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            plt.plot(i * 90 + j, loss.data[0], 'bo')

        epoch_loss = 0.0

        net.train(False)
        for batch in test_loader:
            inputs = Variable(batch['image'])
            labels = Variable(batch['landmark'])
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.data[0]

        print('epoch loss:', epoch_loss)

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            best_epoch = i
            best_state = copy.deepcopy(net.state_dict())

        end = time()
        print('epoch {}/{} complete in {}s'.format(i + 1, epochs, end - beg))

    print('best epoch:', best_epoch)
    print('min loss:', min_loss)
    net.load_state_dict(best_state)

    save_fig(net, test_loader, test_images_dir)


if __name__ == '__main__':
    main()
