import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
import time
import model

if __name__ == "__main__":
    #读取训练集和测试集并转换成pytorch所需要的格式
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    data_train = datasets.MNIST(root='./data/', transform=transform, train=True, download=True)
    data_test = datasets.MNIST(root='./data/', transform=transform, train=False)
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=128, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=128, shuffle=True)
    
    #定义训练所使用的设备、模型、损失函数和优化器（梯度下降算法）
    device = torch.device("cuda:0")
    #model = LeNet().to(device)
    model = model.Model().to(device)
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    #定义训练的轮次
    n_epochs = 10
    for epoch in range(n_epochs):
        time_start = time.time() #开始计时
        model.train()
        running_loss = 0.0
        running_correct = 0
        print("epoch {}/{}".format(epoch, n_epochs))
        print("-" * 10)
        #训练的批次
        for batch, (data, target) in enumerate(data_loader_train):
            X_train, y_train = data, target
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            X_train, y_train = torch.autograd.Variable(X_train), torch.autograd.Variable(y_train)
            outputs = model(X_train)
            _, pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = cost(outputs, y_train)

            loss.backward()
            optimizer.step()

            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch * len(data), len(data_loader_train.dataset), 100. * batch / len(data_loader_train), loss.data.item()))
        model.eval()
        testing_correct = 0
        testing_all=0
        for data in data_loader_test:
            X_test, y_test = data
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            X_test, y_test = torch.autograd.Variable(X_test), torch.autograd.Variable(y_test)
            outputs = model(X_test)
            _, pred = torch.max(outputs.data, 1)
            testing_all+=len(y_test.data)
            testing_correct += torch.sum(pred == y_test.data)
        time_end = time.time()    #结束计时
        time_c= time_end - time_start   #运行所花时间
        print('testing_correct rate:',round(testing_correct.item()/testing_all,4),',  time cose: ',round(time_c,2),'s')
    #将tensor的值保存为cpu格式，再保存训练权重
    '''
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = v.cpu()
    '''
    torch.save(model, "./model/mnist_model_"+str(n_epochs)+".pt")
