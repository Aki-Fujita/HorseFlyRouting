def nearest_n(data, datalen, root, ex_root):

    root.append(0)
    ex_root.remove(0)

    for i in range(datalen - 1):
        min_len = 0
        min_Num = 0
        for j in range(len(ex_root)):
            if j == 0 or min_len > np.linalg.norm([data[root[i]] - data[ex_root[j]]]):
                #print (ex_root[j])
                min_len = np.linalg.norm([data[root[i]] - data[ex_root[j]]])
                min_Num = ex_root[j]
        root.append(min_Num)
        ex_root.remove(min_Num)

    return root

#2-opt法
def readDistance(distDict, i,j):
    if i > j :
        a = distDict[(j , i)]
    elif i == j:
        a = 0
    else:
        a = distDict[(i,j)]
    return a

def readDistance2(distDict, i,j):
    try:
        a = distDict[(i,j)]
    except KeyError:
        if i ==j:
            a = 0
        else:
            a = distDict[(j,i)]
    return a

def opt_2(data, datalen, root):
    #print("data=",data)
    total = 0
    times = 0
    while True:
        count = 0
        for i in range(datalen - 2):
            i1 = i + 1
            for j in range(i + 2, datalen):
                if j == datalen - 1:
                    j1 = 0
                else:
                    j1 = j + 1
                if i != 0 or j1 != 0:
                    l1 = readDistance2(data, root[i] , root[i1])
                    l2 = readDistance2(data, root[j] , root[j1])
                    l3 = readDistance2(data, root[i] , root[j])
                    l4 = readDistance2(data, root[i1] , root[j1])
                    if l1 + l2 > l3 + l4:
                        #print("更新")
                        new_root = root[i1:j+1]
                        root[i1:j+1] = new_root[::-1]
                        count += 1
        total += count
        times += 1
        #print(times)
        #print (root) # rootは解のこと
        if count == 0: break
        #if times >= 1000: break

    return root

def cal_totalcost(data, route):
    totalcost = 0
    for i in range(len(route)):
        totalcost += readDistance(data, route[i] , route[i - 1])
    return totalcost