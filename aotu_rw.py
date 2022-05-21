import os
import json
import numpy as np


code = [1584400, 4312900, 6218600, 3088900, 4054500,
        5120400, 2998800, 4005200, 3775700, 6063900,
        4273800, 7682300, 4797400, 4516900, 3889600, 4142900]

def read_from_json(target_dir):
    f = open(target_dir, 'r')
    data = json.load(f)
    data = json.loads(data)
    f.close()
    return data


# 读取路径并排序（与小到大）
dataset_train_path = os.listdir(r'./dataset_json')
dataset_test_path = os.listdir(r'./dataset_json')
dataset_train_path.sort(key=lambda x: int(x.split('.')[0]))
dataset_test_path.sort(key=lambda x: int(x.split('.')[0]))


# 读取dataset_json并取80%写入dataset_train
i = 0
for info in dataset_train_path:
    domain = os.path.abspath(r'./dataset_json')
    info = os.path.join(domain, info)
    print(info)
    read = read_from_json(info)
    i = i + 1
    print(i)

    # 加头部json格式字符
    with open("./dataset_train/{}_train.json".format(i-1), 'w') as load_f:
        load_f = load_f.write('{}\\"{}{}'.format('"{', i-1, '\\": '))

    for key in read:
        read[key] = np.asarray(read[key])
        read[key] = read[key].reshape(-1)
        write = read[key][0:code[i-1]]
        write = write.reshape(-1, 50, 2, 17)
        write = write.tolist()
        write = str(write)
    # 加尾部json字符
        with open("./dataset_train/{}_train.json".format(i-1), 'a') as load_f:
            load_dict = json.dumps(load_f.write(write + '"'))
        print("Write End!")
        load_f.close()


# 读取dataset_json并取20%写入dataset_test
i = 0
for info in dataset_test_path:
    domain = os.path.abspath(r'./dataset_json')
    info = os.path.join(domain, info)
    print(info)
    read = read_from_json(info)
    i = i + 1
    print(i)

    with open("./dataset_test/{}_test.json".format(i-1), 'w') as load_f:
        load_f = load_f.write('{}\\"{}{}'.format('"{', i-1, '\\": '))

    for key in read:
        read[key] = np.asarray(read[key])
        read[key] = read[key].reshape(-1)
        write = read[key][code[i-1]:]
        write = write.reshape(-1, 50, 2, 17)
        write = write.tolist()
        write = str(write)
        with open("./dataset_test/{}_test.json".format(i-1), 'a') as load_f:
            load_dict = json.dumps(load_f.write(write + '"'))
        print("Write End!")
        load_f.close()