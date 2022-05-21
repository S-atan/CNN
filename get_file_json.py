import os
import json
import numpy as np

path = "./add_speed/speed_data.txt"


def txtToJson(path):  # 把字符串写好再附给字典？
    data = np.loadtxt(path, dtype=np.float32, delimiter=',')
    num = 0
    cnt = 0
    c = '\"{'
    for i in range(43750):
        if num == 50:
            num = 0
            cnt = cnt + 1
        if num == 0:
            c += "\\"+"\"" + str(cnt) + "\\" + "\"" + ": [["
        x = "[["
        y = '['
        z = '['
        for j in range(0, 48, 3):
            if j == 45:
                if num == 49:
                    sep = "]]"
                    if i != 43749:
                        sep += ', '
                else:
                    sep = ", "
                x += str(data[i][j]) + ']' + ", "
                y += str(data[i][j + 1]) + ']' + ", "
                z += str(data[i][j + 2]) + "]]" + sep
            else:
                x += str(data[i][j]) + ", "
                y += str(data[i][j + 1]) + ", "
                z += str(data[i][j + 2]) + ", "
        c = c + x + y + z
        num = num + 1
    c += "}\""
    return c


def save_json(my_data):
    print(type(my_data))
    with open('./add_speed/.data.json', 'w')as f:
        f.write(my_data)
        f.close()


data = txtToJson(path)
save_json(data)



