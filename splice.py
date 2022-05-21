import os


def read_from_txt(target_dir):
    f = open(target_dir, 'r')
    data = f.read()
    f.close()
    return data


# 读取路径并排序（与小到大）
dataset_train_path = os.listdir(r'./dataset_train_txt')
dataset_test_path = os.listdir(r'./dataset_test_txt')
dataset_train_path.sort(key=lambda x: int(x.split('_')[0]))
dataset_test_path.sort(key=lambda x: int(x.split('_')[0]))


# 读取dataset_train并合为dataset_train.json文件
# 加头部json格式字符
with open("./dataset_train.txt", 'w') as load_f:
    head = load_f.write('{}'.format('"{'))
    load_f.close()


# 中间拼接
i = -1
for info in dataset_train_path:
    domain = os.path.abspath(r'./dataset_train_txt')
    info = os.path.join(domain, info)
    read = read_from_txt(info)
    i = i + 1
    if i == 15:
        with open('./dataset_train.txt', 'a') as f:
            f_new = f.write(read[2:-1])
            f.close()
    else:
        with open('./dataset_train.txt', 'a') as f:
            f_new = f.write(read[2:-1] + ', ')
            f.close()


# 加尾部json字符
with open("./dataset_train.txt", 'a') as load_f:
    end = load_f.write('{}'.format('}"'))
    load_f.close()
    print("Dataset_train Write End!")


# 读取dataset_test并合为dataset_test.json文件
# 加头部json格式字符
with open("./dataset_test.txt", 'w') as load_f:
    head = load_f.write('{}'.format('"{'))
    load_f.close()


# 中间拼接
i = -1
for info in dataset_test_path:
    domain = os.path.abspath(r'./dataset_test_txt')
    info = os.path.join(domain, info)
    read = read_from_txt(info)
    i = i + 1
    if i == 15:
        with open('./dataset_test.txt', 'a') as f:
            f_new = f.write(read[2:-1])
            f.close()
    else:
        with open('./dataset_test.txt', 'a') as f:
            f_new = f.write(read[2:-1] + ', ')
            f.close()

# 加尾部json字符
with open("./dataset_test.txt", 'a') as load_f:
    end = load_f.write('{}'.format('}"'))
    load_f.close()
    print("Dataset_test Write End!")