import json
import matplotlib.pyplot as plt

s = []
def getdate():
    with open('./dataset_mmd.json','r') as f :
        date = json.load(f)
    tt = eval(date)
    #print(type(tt))
    for x in tt:
    #    print(tt[x])
        for xx in tt[x] :
    #       print(xx)
            s.append(xx)

def creat_gra(xx, start, end) :
    sx = [xx[0][start], xx[0][end]]
    sy = [xx[1][start], xx[1][end]]
    plt.plot(sx, sy, color='r')
    plt.scatter(sx, sy, color='b')

def draw_pic() :
#    plt.figure(figsize=(100, 10))
    creat_gra(xx, 0, 1)
    creat_gra(xx, 1, 3)
    creat_gra(xx, 5, 6)
    creat_gra(xx, 5, 11)
    creat_gra(xx, 11, 12)
    creat_gra(xx, 6, 12)
    creat_gra(xx, 0, 2)
    creat_gra(xx, 2, 4)
    creat_gra(xx, 5, 7)
    creat_gra(xx, 7, 9)
    creat_gra(xx, 6, 8)
    creat_gra(xx, 8, 10)
    creat_gra(xx, 11, 13)
    creat_gra(xx, 13, 15)
    creat_gra(xx, 12, 14)
    creat_gra(xx, 14, 16)
cnt = 0
getdate()
ii = 0
for dx in s :  #对于s中的每一个样本
    for xx in dx :   #对于每一个样本中的每一帧图
#        print(len(xx))
        draw_pic()
        ax = plt.gca()
        ax.invert_yaxis()
        ax.set_aspect(1)
        plt.axis('off')
#        plt.show()
        plt.savefig("./14/" + str(ii) + ".png", transparent=True, bbox_inches='tight', pad_inches=0.0)
        plt.cla()
        ii = ii + 1