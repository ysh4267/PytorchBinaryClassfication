# -*- coding: utf-8 -*-

# 패키지를 임포트한다.
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary


# 모델을 설계
class Net(nn.Module):

    # 파이썬 클래스 초기화 함수에서 구조를 잡고
    def __init__(self):
        super(Net, self).__init__()

        # 1) filter 연산 네트워크
        self.conv1 = nn.Conv2d(3, 64, 5)  # 3채널을 64채널로
        self.conv2 = nn.Conv2d(64, 32, 5)  # 64채널을 32채널로
        self.pool = nn.MaxPool2d(2, 2)

        # 2) full connected layer 네트워크
        self.fc1 = nn.Linear(32 * 5 * 5, 128)  # 32채널에 5by5이미지를 눌러 출력이 128개인 FC를 구성

        self.fc2 = nn.Linear(128, 64)  # FC를 128개에서 64개로
        self.fc3 = nn.Linear(64, 2)  # FC를 64개에서 2개로 출력

    # forward에서 적용되는 순서를 정한다.
    def forward(self, x):
        # 구성한 conv layer를 relu 함수에 넣음
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # 원하는 행 수는 모르지만 열 수는 확신하는 상황
        x = x.view(-1, 32 * 5 * 5)


        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    # ===========================================================
    # 필수 코드 (1) 데이터 로드 & 가공

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # ===============================================================
    # 1) 데이터 로드 함수
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # imageFolder로 이미지 읽어들이기
    trainset = torchvision.datasets.ImageFolder("./data/nomask", transform=transform)

    # 2) 로드한 데이터를 iterator로 변환
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1)

    # ===============================================================
    # 테스트 시킬 데이터 셋
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 테스트 셋도 마찬가지
    testset = torchvision.datasets.ImageFolder("./data/face_testset", transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=1)

    classes = ('0', '1')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)

    # 2) 네트워크 생성
    net = Net().to(device)
    # net = Net()

    print(net)
    # summary(net, (3, 32, 32))

    # 3) loss 함수
    criterion = nn.CrossEntropyLoss()

    # 4) activation functoin 함수 momentum=0.9
    # 테스트셋과 트레인셋이 같은경우 가장 빠르게 학습함
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 5) 학습 코드
    for epoch in range(20):  # 데이터셋 전체 순회를 n번 반복
        running_loss = 0.0
        #데이터 셋 전체를 순회
        for i, data in enumerate(trainloader, 0):
            #inputs는 이미지 labels가 이미지가 들어있는 폴더 이름
            inputs, labels = data[0].to(device), data[1].to(device)

            print("input:{}".format(inputs.shape))
            print("label:{}".format(labels.shape))
            print("label value:{}".format(labels.item()))
            if labels.item() == 0:
                print("정답 : 여자")
            else:
                print("정답 : 남자")

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            print("output:{}".format(outputs.shape))
            print("output value:{}".format(outputs))
            if outputs[0][0] > outputs[0][1]:
                print("추측 : 여자")
            else:
                print("추측 : 남자")

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print("Here")

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # 6) 모델 저장
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    # 테스트 코드
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    # imshow(torchvision.utils.make_grid(images))

    net = Net()
    net.load_state_dict(torch.load(PATH))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the %d test images: %d %%' % (total,
            100 * correct / total))

    del dataiter


if __name__ == '__main__':
    main()

