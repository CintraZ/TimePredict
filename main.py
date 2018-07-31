import os
import utils
import torch
import argparse
import TimeDataSet
import LossFunction
import Net.TopModule
import matplotlib
matplotlib.use('Agg')  # if this line is missing, program maybe exit abnormally
import torch.nn as nn
import matplotlib.pyplot as plt
from LossFunction import L2Loss
from LossFunction import RMSELoss
from LossFunction import MAPELoss
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, help='can be [train] or [test] or [trainval]')
parser.add_argument('--batch_size', type=int, default=100, help='batch_size, default: 100')
parser.add_argument('--epochs', type=int, default=40, help='epochs, default: 40')

parser.add_argument('--paramfile', type=str, default='None', help='when [train] or [trainval]: weights save path, '
                                                                  'when [test]: weights read path')
parser.add_argument('--save_loss', type=bool, default=False, help='be True if want to save test loss')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default: 0.001')

args = parser.parse_args()

trainDir = 'D:\\WorkFile\\PyTorch\\WorkSpace\\timedata\\train'
testDir = 'D:\\WorkFile\\PyTorch\\WorkSpace\\timedata\\test'
valDir = 'D:\\WorkFile\\PyTorch\\WorkSpace\\timedata\\val'

embeds_dims = [('lineNo', 479, 9), ('busNo', 6366, 13), ('upNo', 2, 1), ('nextSNo', 89, 7),
               ('weekNo', 7, 3), ('timeNo', 24 * 60, 11)]

# attr_list = ['line', 'terminals', 'up', 'station', 'week_id', 'time_id', 'distance']
# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper Parameter
num_epochs = args.epochs
num_classes = 1
batch_size = args.batch_size
learning_rate = args.lr
param_file = args.paramfile


def mkdir(dir):
    dir = dir.strip()
    dir = dir.rstrip('\\')
    if not os.path.exists(dir):
        os.makedirs(dir)
        # print(dir + ' create successful...')
        return True
    else:
        return False


# Input is a list
def input_to_device(Input):
    for i in range(len(Input)):
        Input[i] = Input[i].to(device)


# Method parameter
####################
# model: neural net model
# param_file: is to save train parameter
# validate: be True if need to validate model
def train(model, param_file, validate=True):

    train_loss_list = []
    val_loss_list = []
    epoch_list = []

    loss = 0

    plt.figure()
    plt.plot(epoch_list, train_loss_list, color='green', label='train')
    if validate:
        plt.plot(epoch_list, val_loss_list, color='red', label='val')
    plt.legend()

    for epoch in range(num_epochs):

        model.train()

        # Define var for print runtime out
        runtime_out = 0
        time_label = 0
        runtime_in = 0

        for root, dirs, files in os.walk(trainDir):
            for n, file in enumerate(files):
                print('train on file: {}'.format(root + '\\' + file))
                trainDataSet = TimeDataSet.TimeDataSet(file, root, predict=False)
                trainLoader = DataLoader(trainDataSet, batch_size=batch_size, shuffle=True)
                for i, (x1, x2, x3, x4, x5, x6, x7, time) in enumerate(trainLoader):
                    Input = [x1, x2, x3, x4, x5, x6, x7]

                    input_to_device(Input)
                    runtime_in = Input

                    time = time.to(device)
                    time_label = time

                    # Forward pass
                    outputs = model(Input)
                    runtime_out = outputs.view(1, -1)
                    loss = criterion(runtime_out[0], time.float())

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        print('Epoch {}/{}, loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        train_loss_list.append(loss.item())
        epoch_list.append(epoch + 1)

        if loss.item() > 1:
            # mkdir(param_file)
            # torch.save(model.state_dict(), param_file + '/params' + str(epoch + 1) + '.pkl')
            print(runtime_in)
            for i in range(time_label.size()[0]):
                print('predicted: {}, label: {}'.format(runtime_out[0][i], time_label[i]))
            print('abnormal weights:')
            print_weights(model)

        # Save model weights
        if param_file != 'None':
            mkdir(param_file)
            torch.save(model.state_dict(), param_file + '/params'+str(epoch+1)+'.pkl')

        if validate:
            # Validate model
            model.eval()
            with torch.no_grad():
                out = 0
                out_time = 0
                val_loss = 0
                v_criterion = LossFunction.MAPELoss.MAPELoss()
                for root, dirs, files in os.walk(valDir):
                    for file in files:
                        print('Validate on file: {}'.format(root + '\\' + file))
                        valDataSet = TimeDataSet.TimeDataSet(file, root, predict=False)
                        valLoader = DataLoader(valDataSet, batch_size=batch_size, shuffle=True)
                        for i, (x1, x2, x3, x4, x5, x6, x7, time) in enumerate(valLoader):
                            Input = [x1, x2, x3, x4, x5, x6, x7]
                            input_to_device(Input)
                            time = time.to(device)

                            # Forward pass
                            outputs = model(Input)
                            out = outputs.view(1, -1)
                            out_time = time

                            val_loss = v_criterion(outputs.view(1, -1)[0], time.float())
                print('Epoch: {}/{} validate loss: {:.4f}'.format(epoch + 1, num_epochs, val_loss.item()))
                # val_loss_file.write('Epoch: {}/{} validate loss: {}'.format(epoch+1, num_epochs, val_loss))
                val_loss_list.append(val_loss.item())

                # unnor_time = utils.unnormalize(out_time)
                # unnor_out = utils.unnormalize(out)
                # for i in range(out_time.size()[0]):
                #     print('time: {}, pred: {}'.format(unnor_time[i], unnor_out[0][i]))

        # Draw the loss line
        plt.plot(epoch_list, train_loss_list, color='green')
        if validate:
            plt.plot(epoch_list, val_loss_list, color='red')
        plt.pause(0.05)
    plt.savefig('loss256_big_RMSE.png')
    # plt.show()


#  Test
def test(model, param_file, save_loss=False):
    # Load model parameter
    model.load_state_dict(torch.load(param_file))
    print('Model parameter load from : {}'.format(param_file))

    model.eval()
    t_criterion = LossFunction.MAPELoss.MAPELoss()
    with torch.no_grad():
        test_loss = 0
        test_input = []
        for root, dirs, files in os.walk(testDir):
            for file in files:
                print('test on file: {}'.format(root + '\\' + file))
                testDataSet = TimeDataSet.TimeDataSet(file, root, predict=False)
                testLoader = DataLoader(testDataSet, batch_size=batch_size, shuffle=True)
                for i, (x1, x2, x3, x4, x5, x6, x7, time) in enumerate(testLoader):
                    Input = [x1, x2, x3, x4, x5, x6, x7]
                    test_input = Input
                    input_to_device(Input)

                    time = time.to(device)

                    outputs = model(Input)
                    test_loss = t_criterion(outputs.view(1, -1)[0], time.float())
        print('test loss: {:.4f}'.format(test_loss.item()))

        fp = open('in_out_weight.txt', 'a')
        fp.write('input: {}\n'.format(test_input))
        fp.close()
        # Save loss
        if save_loss:
            loss_file = open('./loss_file.txt')
            loss_file.write('Parameter file: {}, loss: {}'.format(param_file, test_loss.item()))


# 打印权值，如果更改了网络结构，则可能权值的名字也需要改变
def print_weights(model):
    params = model.state_dict()
    print('embedding weights')
    for attr, in_dim, out_dim in embeds_dims:
        print('  {}_em weight : {}'.format(attr, params['embed.'+attr+'_em.weight'].mean()))

    print('fc1 weight: {}'.format(params['layer.fc1.weight'].mean()))
    print('fc1 bias: {}'.format(params['layer.fc1.bias'].mean()))
    # print('bn1 weight: {}'.format(params['net.fc1.weight'].mean()))
    # print('bn1 bias mean: {}'.format(params['net.bn1.running_mean'].mean()))
    # print('fc2 weight: {}'.format(params['layer.fc2.weight'].mean()))
    # print('fc2 bias: {}'.format(params['layer.fc2.bias'].mean()))
    print('fc3 weight: {}'.format(params['layer.fc3.weight'].mean()))
    print('fc3 bias: {}'.format(params['layer.fc3.bias'].mean()))
    print('fc4 weight: {}'.format(params['layer.fc4.weight'].mean()))
    print('fc4 bias: {}'.format(params['layer.fc4.bias'].mean()))
    print('fc5 weight: {}'.format(params['layer.fc5.weight'].mean()))
    print('fc5 bias: {}'.format(params['layer.fc5.bias'].mean()))


if __name__ == '__main__':
    model = Net.TopModule.TimePredict(model='fc', num_classes=num_classes).to(device)
    print('init weights:')
    print_weights(model)

    # loss and optimizer
    criterion = LossFunction.L2Loss.L2Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(model)
    if args.task == 'train':
        train(model, param_file, validate=False)
    elif args.task == 'trainval':
        train(model, param_file, validate=True)
    elif args.task == 'test':
        # 此时的param_file是读取的参数文件
        test(model, param_file=param_file, save_loss=args.save_loss)

