import os
import utils
import L2Loss
import torch
import argparse
import TimeDataSet
import LossFunction
import Net.TimePredict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, help='can be [train] or [test] or [trainval]')
parser.add_argument('--batch_size', type=int, default=100, help='batch_size, default: 100')
parser.add_argument('--epochs', type=int, default=40, help='epochs, default: 40')

parser.add_argument('--param_file', type=str, help='when [train] or [trainval]: weights save path, '
                                                   'when [test]: weights read path')
parser.add_argument('--save_loss', type=bool, default=False, help='be True if want to save test loss')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default: 0.001')

args = parser.parse_args()

trainDir = 'D:\\WorkFile\\PyTorch\\WorkSpace\\timedata\\train'
testDir = 'D:\\WorkFile\\PyTorch\\WorkSpace\\timedata\\ShuffleData\\test'
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


def mkdir(dir):
    dir = dir.strip()
    dir = dir.rstrip('\\')
    if not os.path.exists(dir):
        os.makedirs(dir)
        # print(dir + ' create successful...')
        return True
    else:
        return False


# Put input to GPU/CPU
def input_to_device(Input):
    for i in range(len(Input)):
        Input[i] = Input[i].to(device)


# Method parameter:
# model: neural net model
# param_file: is to save train parameter
# validate: be True if need to validate model
def train(model, param_file,validate=True):

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
        for root, dirs, files in os.walk(trainDir):
            for n, file in enumerate(files):
                # print('train on file: {}'.format(root + '\\' + file))
                trainDataSet = TimeDataSet.TimeDataSet(file, root, predict=False)
                trainLoader = DataLoader(trainDataSet, batch_size=batch_size, shuffle=True)
                for i, (x1, x2, x3, x4, x5, x6, x7, time) in enumerate(trainLoader):
                    # for i, Input in enumerate(trainLoader):
                    Input = [x1, x2, x3, x4, x5, x6, x7]
                    # for key in Input:
                    #     Input[key].to(device)
                    input_to_device(Input)

                    # time = dt['label']
                    time = time.to(device)

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

        # Save model weights
        mkdir(param_file)
        torch.save(model.state_dict(), param_file + '/params'+str(epoch+1)+'.pkl')

        if validate:
            # Validate model
            model.eval()
            with torch.no_grad():
                out = 0
                out_time = 0
                val_loss = 0
                v_criterion = LossFunction.LossFunction()
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

                unnor_time = utils.unnormalize(out_time)
                unnor_out = utils.unnormalize(out)
                for i in range(out_time.size()[0]):
                    print('time: {}, pred: {}'.format(unnor_time[i], unnor_out[0][i]))

        # Draw the loss line
        # if 'loss_image' in globals():
        #     loss_image.remove()

        plt.plot(epoch_list, train_loss_list, color='green')
        if validate:
            plt.plot(epoch_list, val_loss_list, color='red')
        # plt.ioff()
        plt.pause(0.05)
    # plt.savefig('loss.png')
    # plt.show()

    print('after 1 epoch:')
    params = model.state_dict()
    print('embedding weights')
    for attr, in_dim, out_dim in embeds_dims:
        print('  {}_em weight : {}'.format(attr, params[attr+'_em.weight']))

    print('fc1 weight: {}'.format(params['net.fc1.weight']))
    print('fc1 bias: {}'.format(params['net.fc1.bias']))
    # print('bn1 weight: {}'.format(params['net.fc1.weight']))
    # print('bn1 bias mean: {}'.format(params['net.bn1.running_mean']))
    print('fc2 weight: {}'.format(params['net.fc2.weight']))
    print('fc2 bias: {}'.format(params['net.fc2.bias']))
    print('fc3 weight: {}'.format(params['net.fc3.weight']))
    print('fc3 bias: {}'.format(params['net.fc3.bias']))
    print('fc4 weight: {}'.format(params['net.fc4.weight']))
    print('fc4 bias: {}'.format(params['net.fc4.bias']))


#  Test
def test(model, param_file, save_loss=False):
    # Load model parameter
    model.load_state_dict(torch.load(param_file))
    print('Model parameter load from : {}'.format(param_file))

    model.eval()
    t_criterion = LossFunction.LossFunction()
    with torch.no_grad():
        test_loss = 0
        for root, dirs, files in os.walk(testDir):
            for file in files:
                print('test on file: {}'.format(root + '\\' + file))
                testDataSet = TimeDataSet.TimeDataSet(file, root, predict=False)
                testLoader = DataLoader(testDataSet, batch_size=batch_size, shuffle=True)
                for i, (x1, x2, x3, x4, x5, x6, x7, time) in enumerate(testLoader):
                    Input = [x1, x2, x3, x4, x5, x6, x7]
                    input_to_device(Input)
                    time = time.to(device)

                    outputs = model(Input)
                    test_loss = t_criterion(outputs.view(1, -1)[0], time.float())
        print('test loss: {:.4f}'.format(test_loss.item()))

        # Save loss
        if save_loss:
            loss_file = open('./loss_file.txt')
            loss_file.write('Parameter file: {}, loss: {}'.format(param_file, test_loss.item()))


if __name__ == '__main__':
    model = Net.TimePredict.Net(num_classes).to(device)

    # Print initial weights
    print('init weights:')
    params = model.state_dict()
    print('embedding weights')
    for attr, in_dim, out_dim in embeds_dims:
        print('  {}_em weight : {}'.format(attr, params[attr+'_em.weight']))

    print('fc1 weight: {}'.format(params['net.fc1.weight']))
    print('fc1 bias: {}'.format(params['net.fc1.bias']))
    # print('bn1 weight: {}'.format(params['net.fc1.weight']))
    # print('bn1 bias mean: {}'.format(params['net.bn1.running_mean']))
    print('fc2 weight: {}'.format(params['net.fc2.weight']))
    print('fc2 bias: {}'.format(params['net.fc2.bias']))
    print('fc3 weight: {}'.format(params['net.fc3.weight']))
    print('fc3 bias: {}'.format(params['net.fc3.bias']))
    print('fc4 weight: {}'.format(params['net.fc4.weight']))
    print('fc4 bias: {}'.format(params['net.fc4.bias']))

    # loss and optimizer
    # criterion = LossFunction.LossFunction()

    criterion = L2Loss.L2Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train(model, './ParamWahaha', validate=False)
    print(model)
    if args.task == 'train':
        train(model, args.param_file, validate=False)
    elif args.task == 'trainval':
        train(model, args.param_file, validate=True)
    elif args.task == 'test':
        test(model, param_file=args.param_file, save_loss=args.save_loss)

