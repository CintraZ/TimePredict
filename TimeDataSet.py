import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TimeDataSet(Dataset):
    # Parameter: filename, file dir, to predict?
    # if predict data: has no columns(distance, time)
    def __init__(self, csv_file, root_dir, predict=False):
        # construct read file
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.predict = predict

        self.df = pd.read_csv(self.root_dir + '/' + self.csv_file)
        # print('total len:{}'.format(len(self.df)))

        if not self.predict:
            # let column(distance) be the last column
            disColumn = self.df.pop('distance')
            self.df['distance'] = disColumn
            # print(self.df.columns)
            # print('df: ')
            # print(self.df)

        self.data = self.df.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if not self.predict:
            rt_time = self.data[index][5]
            rt_data = self.data[index]
            rt_data = np.delete(rt_data, [0, 5])
            rt_data = torch.from_numpy(rt_data)
            # print(rt_data.size())
            return rt_data[0], rt_data[1], rt_data[2], rt_data[3], rt_data[4], rt_data[5], rt_data[6], rt_time
        else:
            rt_data = np.delete(self.data, [0])
            rt_data = torch.from_numpy(rt_data)
            rt_data = torch.squeeze(rt_data)
            return rt_data


"""
# Function test
timeDataSet = TimeDataSet(csv_file='train_1_901530.csv',
                          root_dir='D:\\WorkFile\\PyTorch\\WorkSpace\\timedata',
                          predict=False)

timeLoader = DataLoader(dataset=timeDataSet, batch_size=1, shuffle=False)

total_step = len(timeLoader)
print('total:{}'.format(total_step))
# for i in range (total_step):
# for i, (data, time), in enumerate(time_dataloader):
#     print('i: {}, time: {}'.format(i + 1, time))
#     print('data: {}'.format(data))
#     if(i >= 5):
#         pass

for i, (x1,x2,x3,x4,x5,x6,x7, time) in enumerate(timeLoader):
    print('i:{}'.format(i + 1))
    # print('data size: {}, time size: {}'.format(data.size(), time.size()))
    print(x1,x2,x3,x4,x5,x6,x7,time)
    print('data:{}, time: {}'.format(data, time))

"""
