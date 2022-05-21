'''邻接矩阵法完成图的表示'''

#创建图，输入图的顶点个数、顶点、以及创建邻接表和存储顶点的数组
class Graph(object):

    def __init__(self):
        self._count = int(input('输入图的顶点的个数:'))
        self._adjlist = [[None for i in range(self._count)] for i in range(self._count)]
        # 存储顶点
        self._peak_list = []
        for i in range(self._count):
            self._peak = input('输入顶点:')
            # 将顶点添加到数组中
            self._peak_list.append(self._peak)

    #顶点之间的关系
    def ad_relationship(self):
        print('输入顶点之间的关系')
        for i in range(self._count):
            # 顶点自身无连通，赋值为0
            self._adjlist[i][i] = 0
            for j in range(self._count):
                while self._adjlist[i][j] == None:
                    # 输入各个顶点之间的关系
                    msg = input('输入顶点%s--%s之间的关系(0表示无连通，1表示有连通)' % (self._peak_list[i],self._peak_list[j]))
                    if msg == '0' or msg == '1':
                        # 将输入的只填入邻接矩阵中
                        self._adjlist[i][j] = int(msg)
                        self._adjlist[j][i] = self._adjlist[i][j]
                    else:
                        print('输入有误....')
        #输出
        for k in range(self._count):
            print(self._adjlist[k])

if __name__ == '__main__':
    s = Graph()
    s.ad_relationship()



