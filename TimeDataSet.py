import torch
import utils
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# attr_list = [('lineNo', 479, 9), ('busNo', 6366, 13), ('upNo', 2, 1), ('nextSNo', 89, 7),
#                ('weekNo', 7, 3), ('timeNo', 24 * 60, 11)]

dictList = utils.GetBusLineSDict()


class TimeDataSet(Dataset):
    # Parameter: filename, file dir, to predict?
    # if predict data: has no columns(distance, time)
    def __init__(self, csv_file, root_dir,predict=False):
        self.datadict = dict()
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

            # self.df = self.df[self.df['distance'] != 0]

        self.data = self.df.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if not self.predict:
            rt_time = self.data[index][4]
            rt_data = self.data[index]
            rt_data = np.delete(rt_data, [4])
            for i, dt in enumerate(rt_data):
                if i == 0 or i == 1:
                    rt_data[i] = dictList[i][rt_data[i]]
                elif i == 3:
                    rt_data[i] = dictList[2][rt_data[i]]
            rt_data = torch.from_numpy(rt_data)
            # print(rt_data.size())
            # for i in range(len(rt_data)):
            #     self.datadict[attr_list[i]] = rt_data[i]
            # self.datadict['label'] = rt_time
            # return self.datadict
            return rt_data[0], rt_data[1], rt_data[2], rt_data[3], rt_data[4],rt_data[5],rt_data[6], rt_time
        else:
            rt_data = np.delete(self.data, [0])
            rt_data = torch.from_numpy(rt_data)
            rt_data = torch.squeeze(rt_data)
            return rt_data



