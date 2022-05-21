import json

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from human_motion_analysis_with_gru_test import MMD_NCA_Net
from human_motion_analysis_with_gru_test import Net


use_cuda = torch.cuda.is_available()
# A = np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
#               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
#               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
class MMD_NCA_loss(nn.Module):
    def __init__(self):
        super(MMD_NCA_loss, self).__init__()

    def kernel_function(self, x1, x2):
        k1 = torch.exp(-torch.pow((x1 - x2), 2) / 2)
        k2 = torch.exp(-torch.pow((x1 - x2), 2) / 8)
        k4 = torch.exp(-torch.pow((x1 - x2), 2) / 32)
        k8 = torch.exp(-torch.pow((x1 - x2), 2) / 128)
        k16 = torch.exp(-torch.pow((x1 - x2), 2) / 512)
        k_sum = k1 + k2 + k4 + k8 + k16
        return k_sum

    def MMD(self, x, x_IID, y, y_IID):
        # x, x_IID's dimension: m*25,  y, y_IID's dimension: n*25
        m = x.size()[0]
        n = y.size()[0]
        x = x.view(m, 1, -1)
        # print(x.shape)
        x_square = x.repeat(1, m, 1)
        # print(x_square)
        # x_IID = x_IID.view(1, -1, m)
        x_IID = x_IID.view(-1, m, 1)
        # print(x_IID.shape)
        x_IID_square = x_IID.repeat(m, 1, 1)
        # print(x_IID_square.shape)
        value_1 = torch.sum(self.kernel_function(x_square, x_IID_square)) / (m ** 2)
        y = y.view(1, n, -1)
        y_square = y.repeat(n, 1, 1)
        value_2 = torch.sum(self.kernel_function(x_square, y_square)) / (m * n)
        y_IID = y_IID.view(n, 1, -1)
        y_IID_square = y_IID.repeat(1, n, 1)
        value_3 = torch.sum(self.kernel_function(y_IID_square, y_square)) / (n ** 2)
        print(value_1 - 2 * value_2 + value_3)
        return value_1 - 2 * value_2 + value_3

    def forward(self, x):
        # print(x[0].shape)
        x = x.view(7, 1)
        # print(x[0], x[1])
        # numerator = torch.exp(-self.MMD(x[0], x[1], x[2], x[3]))
        numerator = torch.exp(-self.MMD(x[0], x[0], x[1], x[1]))
        print(numerator)
        # numerator = torch.exp(self.MMD(x[0], x[1], x[2], x[3]))
        # print(self.MMD(x[0], x[1], x[2], x[3]))
        # calculate the denominator in MMD NCA loss, only use 3 negative catogories
        #         value_1 = torch.exp(-self.MMD(x[0], x[1], x[5], x[6]))
        #         value_2 = torch.exp(-self.MMD(x[0], x[2], x[7], x[8]))
        #         value_3 = torch.exp(-self.MMD(x[0], x[1], x[9], x[10]))
        value_1 = torch.exp(-self.MMD(x[0], x[0], x[2], x[2]))
        value_2 = torch.exp(-self.MMD(x[0], x[0], x[3], x[3]))
        value_3 = torch.exp(-self.MMD(x[0], x[0], x[4], x[4]))
        value_4 = torch.exp(-self.MMD(x[0], x[0], x[5], x[5]))
        value_5 = torch.exp(-self.MMD(x[0], x[0], x[6], x[6]))
        print(value_1)
        print(value_2)
        print(value_3)
        print(value_4)
        print(value_5)
        #         value_1 = torch.exp(self.MMD(x[0], x[1], x[5], x[6]))
        #         value_2 = torch.exp(self.MMD(x[0], x[2], x[7], x[8]))
        #         value_3 = torch.exp(self.MMD(x[0], x[1], x[9], x[10]))
        # print(value_1, value_2, value_3)
        # denominator = value_1 + value_2 + value_3 + value_4 + value_5
        # print(numerator, denominator)
        # mmd_nca = torch.exp(- numerator / denominator)
        #        return numerator / denominator
        return 0
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_to_json(dic, target_dir):
    dumped = json.dumps(dic, cls=NumpyEncoder)
    file = open(target_dir, 'w')
    json.dump(dumped, file)
    file.close()


def read_from_json(target_dir):
    f = open(target_dir, 'r')
    data = json.load(f)
    data = json.loads(data)
    f.close()
    return data


class MMD_NCA_Dataset(Dataset):
    def __init__(self, json_name):
        # datafile
        self.df = read_from_json(json_name)
        for key in self.df:
            self.df[key] = np.asarray(self.df[key])
            # # # print(self.df[key].shape)
            # self.df[key] = self.df[key].reshape(-1, 17)
            # # print(self.df[key].shape)
            # self.df_new = self.df[key][-1, -1]
            # self.df[key] = np.dot(self.df[key], A)
            # # print(self.df[key].shape)
            # self.df[key] = self.df[key].reshape(-1, 50, 2, 17)
            # # print(self.df[key].shape)    # 单一舞蹈数据形状
        self.training_MMD_NCA_Groups = self.generate_MMD_NCA_Dataset(self.df)


    @staticmethod
    def generate_MMD_NCA_Dataset(df):

        MMD_NCA_Groups = []
        g = []
        for key in df:

            arr = np.arange(df[key].shape[0])

            for i in range(len(arr)):
                if i == 0:
                    MMD_NCA_Group = df[key][arr[i]]
                else:
                    MMD_NCA_Group = np.concatenate((MMD_NCA_Group, df[key][arr[i]]), axis=0)

            MMD_NCA_Group = MMD_NCA_Group.reshape(-1, 50, 2, 17)
            MMD_NCA_Groups.append(MMD_NCA_Group)

        g.append(MMD_NCA_Groups)

        return g

    def __getitem__(self, index):
        # key stands for dictionary key, index stands for index for one group of MMD_NCA_Dataset
        return self.training_MMD_NCA_Groups[index]

    def __len__(self):
        return len(self.training_MMD_NCA_Groups)


def test(model, test_loader, mmd):
    model.eval()
    with torch.no_grad():
        for batch_idx, test_data in enumerate(test_loader):
            anchor = test_data[0][0][1]
            anchor = Variable(anchor).type(torch.cuda.DoubleTensor).squeeze().view(-1, 50, 34) \
                   .permute(1, 0, 2)
            anchor_output = model(anchor)
            ned = anchor_output[0]
            pos = test_data[0][0][0]
            pos = Variable(pos).type(torch.cuda.DoubleTensor).squeeze().view(-1, 50, 34) \
                .permute(1, 0, 2)
            pos_output = model(pos)
            ned = torch.cat((ned, pos_output[0]), dim=0)
            neg0 = test_data[1][0][0]
            neg0 = Variable(neg0).type(torch.cuda.DoubleTensor).squeeze().view(-1, 50, 34) \
                .permute(1, 0, 2)
            neg_output0 = model(neg0)
            ned = torch.cat((ned, neg_output0[0]), dim=0)
            neg1 = test_data[1][0][1]
            neg1 = Variable(neg1).type(torch.cuda.DoubleTensor).squeeze().view(-1, 50, 34) \
                .permute(1, 0, 2)
            neg_output1 = model(neg1)
            ned = torch.cat((ned, neg_output1[0]), dim=0)
            neg2 = test_data[1][0][2]
            neg2 = Variable(neg2).type(torch.cuda.DoubleTensor).squeeze().view(-1, 50, 34) \
                .permute(1, 0, 2)
            neg_output2 = model(neg2)
            ned = torch.cat((ned, neg_output2[0]), dim=0)
            neg3 = test_data[1][0][3]
            neg3 = Variable(neg3).type(torch.cuda.DoubleTensor).squeeze().view(-1, 50, 34) \
                .permute(1, 0, 2)
            neg_output3 = model(neg3)
            ned = torch.cat((ned, neg_output3[0]), dim=0)
            neg4 = test_data[1][0][4]
            neg4 = Variable(neg4).type(torch.cuda.DoubleTensor).squeeze().view(-1, 50, 34) \
                .permute(1, 0, 2)
            neg_output4 = model(neg4)
            ned = torch.cat((ned, neg_output4[0]), dim=0)
            mmd(ned)



criterion = MMD_NCA_loss()
model = MMD_NCA_Net().cuda().double()
# generate testing data
test_data = MMD_NCA_Dataset('./dataset_mmd.json')
model.load_state_dict(torch.load('./log_new/model_new_4999.pth'))
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
test(model, test_loader, criterion)

