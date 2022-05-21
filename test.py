import json
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from human_motion_analysis_with_gru_test import MMD_NCA_Net
from human_motion_analysis_with_gru_test import Net

from visdom import Visdom

# wind = Visdom()
# wind.line([[0., 0., 0.]], [0.], win='test', opts=dict(title='Acc & FPR & F1score', legend=['Acc', 'FPR', 'F1score']))
use_cuda = torch.cuda.is_available()
A = np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])

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

            # # print(self.df[key].shape)
            # self.df[key] = self.df[key].reshape(-1, 17)
            # # print(self.df[key].shape)
            # self.df_new = self.df[key][-1, -1]
            # self.df[key] = np.dot(self.df[key], A)
            # # print(self.df[key].shape)
            # self.df[key] = self.df[key].reshape(-1, 50, 2, 17)
            # # print(self.df[key].shape)    # 单一舞蹈数据形状
        self.training_MMD_NCA_Groups = self.generate_MMD_NCA_Dataset(self.df)[0]
        self.labels = self.generate_MMD_NCA_Dataset(self.df)[1]


    @staticmethod
    def generate_MMD_NCA_Dataset(df):

        MMD_NCA_Groups = []
        g = []
        q = []
        labels = []
        label = []
        for key in df:

            arr = np.arange(df[key].shape[0])
            np.random.shuffle(arr)

            for i in range(len(arr)):
                if i == 0:
                    MMD_NCA_Group = df[key][arr[i]]
                    label = np.asarray(int(key)).reshape(1, 1)
                else:
                    MMD_NCA_Group = np.concatenate((MMD_NCA_Group, df[key][arr[i]]), axis=0)
                    label = np.concatenate((label, np.asarray(int(key)).reshape(1, 1)), axis=0)

            MMD_NCA_Group = MMD_NCA_Group.reshape(-1, 50, 2, 17)
            MMD_NCA_Groups.append(MMD_NCA_Group)
            labels.append(label)

        g.append(MMD_NCA_Groups)
        q.append(labels)

        return g, q

    def __getitem__(self, index):
        # key stands for dictionary key, index stands for index for one group of MMD_NCA_Dataset
        return self.training_MMD_NCA_Groups[index], self.labels[index]

    def __len__(self):
        return len(self.training_MMD_NCA_Groups)


def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        t = 0
        c = 0
        htp = 0
        hfn = 0
        htn = 0
        hfp = 0
        a = 0
        hfpr = 0
        hfscore = 0
        for batch_idx, test_data in enumerate(test_loader):
            for x in range(16):
                correct = 0
                total = 0
                # y = np.random.choice()
                anchor = test_data[0][x][0][0]
                anchor = Variable(anchor).type(torch.cuda.DoubleTensor).squeeze().view(-1, 50, 34) \
                    .permute(1, 0, 2)
                anchor_output = model(anchor)
                ned = anchor_output[0]
                # print(anchor_output[1])
                # print(anchor_output[1].shape)
                # print(type(anchor_output[1]))
                # print(ned)
                w = anchor_output[1].squeeze(0)
                # print(w.shape)
                # anchor_predicted = torch.argmax(anchor_output[1].data, 1)
                # print(anchor_predicted)
                for i in range(16):
                    if i == x:
                        inputs = Variable(test_data[0][i]).type(torch.cuda.DoubleTensor).squeeze().view(-1, 50, 34) \
                            .permute(1, 0, 2)
                        input_lables = Variable(test_data[1][i]).type(torch.cuda.DoubleTensor).squeeze().view(-1)
                        outputs = model(inputs)
                        new = outputs[1]
                        nned = outputs[0]
                        # print(nned)
                        # print(outputs)
                        for k in range(new.size(0)):
                            if k == 0:
                                inpz = torch.cat((w, new[k]), dim=0)
                            else:
                                iinpz = torch.cat((w, new[k]), dim=0)
                                inpz = torch.cat((inpz, iinpz), dim=0)
                        inpz = inpz.view(-1, 256)
                        outp = net(inpz)
                        predicted = torch.argmax(outp.data, 1)
                        p = inpz.size(0)
                        total += inpz.size(0)
                        correct += (predicted == 1).sum().item()
                        tp = (predicted == 1).sum().item()
                        fn = p - tp
                        t += total
                        c += correct
                        htp += tp
                        hfn += fn


                    else:
                        inputs = Variable(test_data[0][i]).type(torch.cuda.DoubleTensor).squeeze().view(-1, 50, 34) \
                            .permute(1, 0, 2)
                        input_lables = Variable(test_data[1][i]).type(torch.cuda.DoubleTensor).squeeze().view(-1)
                        outputs = model(inputs)
                        news = outputs[1]
                        nnned = outputs[0]
                        # print(nnned)
                        for k in range(news.size(0)):
                            if k == 0:
                                inpz = torch.cat((w, news[k]), dim=0)
                            else:
                                iinpz = torch.cat((w, news[k]), dim=0)
                                inpz = torch.cat((inpz, iinpz), dim=0)
                        inpz = inpz.view(-1, 256)
                        outp = net(inpz)
                        predicted = torch.argmax(outp.data, 1)
                        u = inpz.size(0)
                        total += inpz.size(0)
                        correct += (predicted == 0).sum().item()
                        tn = (predicted == 0).sum().item()
                        fp = u - tn
                        t += total
                        c += correct
                        htn += tn
                        hfp += fp

                    # fpr = fp / (fp + tn)
                    # precison = tp / (tp + fp)
                    # recall = tp / (tp + fn)
                    # fscore = 2 * (precison * recall / (precison + recall))
                    # acc = 100 * correct / total
                    # print('Accuracy of dataset_test {} : {} %\tFPR : {}\tRecall : {}\tFscore : {}'.format(x, acc, 100
                    #                                                                                       * fpr,
                    #                                                                                       100 * recall,
                    #                                                                                       100 * fscore))
                # a = 100 * c / t
                # hfpr = 100 * hfp / (hfp + htn)
                # hprecison = 100 * htp / (htp + hfp)
                # hrecall = 100 * htp / (htp + hfn)
                # hfscore = 2 * hprecison * hrecall / (hprecison + hrecall)
                # # wind.line([[a, hfpr, hfscore]], [x], win='test', update='append')

            hfpr = hfp / (hfp + htn)
            hprecison = htp / (htp + hfp)
            hrecall = htp / (htp + hfn)
            hfscore = 2 * hprecison * hrecall / (hprecison + hrecall)
            a = 100 * c / t
            print('\n\n\nAccuracy of dataset_test : {} %\tFPR : {}\tRecall : {}\tFscore : {}'.format(a, 100 * hfpr, 100
                  * hrecall, 100 * hfscore))



start = time.time()
model = MMD_NCA_Net().cuda().double()
net = Net().cuda().double()
model.load_state_dict(torch.load('./log_new/model_new_4999.pth'))
net.load_state_dict(torch.load("./log_new_cross/net_new_4999.pth"))
# generate testing data
test_data = MMD_NCA_Dataset('./dataset_test.json')
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
test(model, test_loader)
end = time.time()
print(end-start)