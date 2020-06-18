import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import networkx as nx
import sys
import pdb

import pickle
import algorithms_for_prepatation as algs

def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data
    
def crt_latice_network(n):
    G=nx.Graph()
    edgeList = np.zeros((0,3))
    """for i in range(n):
        for j in range(n):
            G.add_node((i,j))"""
    #nodeMatrix = np.zeros((n**2, n**2))
    pos = {}
    
    for y in range(n):
        for x in range(n):
            count = y * n + x
            right = y*n + x+1
            above = (y+1)*n + x
            pos[int(count)] = (x,y)
            if x < n - 1:
                a = np.random.rand()*1e-3
                G.add_edge((x,y),(x+1,y))
                edgeInfo = np.array([[count, right, a],[right, count, a]])
                edgeList = np.append(edgeList, edgeInfo, axis = 0)
                #nodeMatrix[count, right ] = 1 + np.random.rand()*1e-3

            if y < n - 1:
                b = np.random.rand()*1e-3
                G.add_edge((x,y),(x,y + 1))
                edgeInfo = np.array([[count, above, b],[above, count, b]])
                edgeList = np.append(edgeList, edgeInfo, axis = 0)

    print((edgeList))
    print(pos)
    #print(list(G.nodes))
    
    """plt.figure(figsize=((6,6)))
    pos = {n: (n[1], n[0]) for n in G.nodes()}
    nx.draw_networkx_nodes(G, pos, node_size=30, alpha=1, node_color='blue')
    nx.draw_networkx_edges(G, pos, font_size=20, label=1, edge_color="black", width=2)
    plt.show()
    plt.close()"""
    return pos, edgeList


def RenamePoints(df, pos):
    global pos2, G2, posRename, edges
    df1, df2, df3 = df["START_NODE"], df["END_NODE"], df["LENGTH"]
    #range(len(list(pos)))
    df2 = df2.replace(list(pos),range(len(list(pos))))
    df1 = df1.replace(list(pos),range(len(list(pos))))
    edges = np.concatenate([np.array([df1]), np.array([df2]), np.array([df3])], axis = 0).T
    np.savetxt("DelivInfoFiles3/edge_weight.txt", edges, fmt = '%d') #Renameした辺に関する情報
    G2 = nx.read_weighted_edgelist("DelivInfoFiles3/edge_weight.txt", nodetype = int) #これで読むと勝手にweight属性追加される
    #print("G=",G2.nodes())
    coords = df.values[:,1:3]
    posRename = {}
    # ndarrayの情報を「頂点:(x,y)」という値(辞書)に変換する
    for i in range(len(coords) ):
        if i != 0 and (coords[i][0], coords[i][1]) == (coords[i-1][0], coords[i-1][1]):
            pass
        else:
            posRename[int(df1[i])] = (coords[i][0], coords[i][1])
    #posRename
    pos2 = posRename.copy()
    #print(pos2)
    pickle_dump(pos2, "DelivInfoFiles3/positionAllNodes.pickle")

    return G2, posRename
    
def selectDelivPlaces(basePlace, DELIV_NUM, posRename):
    #global DELIV_PLACES
    basePlace = 270 # 基地の場所
    import random
    #DELIV_NUM = 40
    DELIV_PLACES = {}
    posList = list(posRename) # 辞書はリストじゃなくてセットオブジェクトだからindexing とかできない！！ので、こうやってkeyを取り出す必要あり
    random.shuffle(posList) # こいつは何も返さないからこの書き片しないといけない
    for i in range(DELIV_NUM):
        node = posList.pop()
        cor = posRename[node]
        DELIV_PLACES[node] = cor
    #DELIV_PLACES
    G1 = nx.empty_graph(0)
    G1.add_nodes_from(list(DELIV_PLACES))
    return DELIV_PLACES
    
def cluster(droneNum, carNum, driver_deliv, DELIV_PLACES):
    #global orderedNames, coors, labels, N_CLUST
    if driver_deliv:
        agentNum = carNum + droneNum
    else:
        agentNum = droneNum
    agentNum = int(agentNum / carNum)
    #nx.draw(G1, DELIV_PLACES, node_size = 10, node_color = "r")
    orderedNames = list(DELIV_PLACES.keys())
    N_CLUST = int(len(orderedNames) / agentNum)
    coors =np.array([list(DELIV_PLACES[i]) for i in orderedNames])
    
    from samesizeKmeans import EqualGroupsKMeans
    clf = EqualGroupsKMeans(n_clusters = N_CLUST)
    clf.fit(coors)
    labels = clf.labels_
    return orderedNames, coors, labels, N_CLUST

def dist(a,b):
    return np.sqrt(np.dot(b-a, b-a))

def crt_delivInfo(driver_deliv, G2, basePlace, orderedNames, coors, labels, N_CLUST, pos2):
    #global DelivInfo, centerNodes, centroids, success, DelivInfo2
    #cmap = plt.get_cmap("tab10") 
    #cl = ["r", "b", "g","orange", "brown", "violet", "black"]
    centerNodes = np.zeros(0)
    cords = np.array([pos2[key] for key in pos2.keys()])
    DelivInfo = {}
    centroids = np.zeros((0,2)) # 重心の座標集
    for cluster in range(N_CLUST):
        data_ = coors[np.where(labels == cluster)]
        Names_ = np.array(orderedNames)[np.where(labels == cluster)]
        centroid_ = np.mean(data_, axis = 0)#重心の座標
        centroids = np.append(centroids, np.array([centroid_]), axis = 0)
        if driver_deliv:
            distBox = [dist(data_[j,:], centroid_) for j in range(int(len(data_[:,0])))] # クラスター内の点に対して重心との距離を計算
            dist_ = np.array(distBox)
            centerNodes = np.append(centerNodes, Names_[np.argmin(dist_)] )
            DelivInfo[Names_[np.argmin(dist_)]] = Names_
            #print(cluster)
        else:
            distBox = [dist(cords[j,:], centroid_) for j in range(int(len(cords[:,0])))] # 全点に対して重心との距離を計算
            dist_ = np.array(distBox)
            centerNodes = np.append(centerNodes, np.argmin(dist_) )
            centerNodeCor = pos2[np.argmin(dist_)]
            DelivInfo[np.argmin(dist_)] = Names_
    #-----上述の作業によってDelivInfoというファイルに基本的なクラスタリングから代表点・クラスターのファイルができた-------------#
    ###-----以下、トラックが通る場所を決める----------####
    DelivInfo2 = {} # delivInfoとはkeyが変わってる
    centerNodes = np.zeros(0)
    for key in DelivInfo.keys():
        points = DelivInfo[key]
        ifConnected = [nx.has_path(G2, basePlace, point) for point in points]
        if np.any(ifConnected): ##少なくとも一つは結ばれてるやつがいた場合
            points_ = points[ifConnected]
            distance = [nx.shortest_path_length(G2, basePlace, goal, weight = "weight") for goal in points_]
            represent = points_[np.argmin(distance)] #クラスタリングした結果その組で一番近いやつを代表点
            DelivInfo2[represent] = points
            centerNodes = np.append(centerNodes, represent)
        else: #もし結ばれてるやつが一つもなかったら点の選択からやり直す。　→それとも辿れる点から適当に取る？→それだとクラスタリングもう一回する手間
            success = False
            break
    if len(DelivInfo2.keys() ) == len(DelivInfo.keys() ):
        success = True
    print("info=",DelivInfo2)
    #print("G=",G2.nodes())
    return success, DelivInfo2, centroids
    
        
    
def calArg(a,central): #centerから見たときのaの偏角を計算
    import cmath
    comp_ = np.dot(a - central, np.array([1,1j]) )
    return cmath.phase(comp_)
    
def decideOrderMultiple(carNum, basePlace, DelivInfo2, centroids): # 複数台車があったときのルート設計をする。→扇形分割スキーム
    global sortRoute, centerNode_per_car, extra, carDict
    baseCor = pos2[basePlace]
    centerNodes = list(DelivInfo2.keys())
    PlacesForOneCar = int(len(centerNodes) / carNum)  # 17点5台だったら3,3,3,4,4なので3が格納される
    extra = int(len(centerNodes) % carNum)
    #print(PlacesForOneCar)
    carDict = {}
    ###-------とりあえず図示-------------###
    nx.draw(G2, posRename, node_size = 0, node_color = "b", edge_color="gray", width=1,)
    plt.scatter(baseCor[0], baseCor[1], marker = "*",  c = "green", s = 300)
    plt.scatter(centroids[:,0], centroids[:,1], marker = "x",  c = "red", s = 10)
    ###-------まずは基地を中心に車の数でセクション分け-------------###
    ######-----その上でまずはcenterNodesに角度情報をつける-----#####
    #argList = [calArg(np.array(centroid), basePlace) for centroid in zip(centerNodes)]
    argList = [calArg(np.array(pos2[int(centroid)]), baseCor) for centroid in centerNodes]
    routes = np.stack((centerNodes, np.array(argList)), axis = 1) # ベクトル同士の足し算はこれがいい！
    sortRoute = routes[np.argsort(routes[:, 1])]
    ######-----その上でまずはcenterNodesに角度情報をつける-----#####
    begin = 0
    end = 0
    for car in range(carNum):
        #print(car)
        if car <= extra - 1:
            begin = car * (PlacesForOneCar + 1)
            end = begin + (PlacesForOneCar + 1)
            carDict[car] = sortRoute[begin:end, 0]
        elif car >= extra:
            begin = end
            end = begin + PlacesForOneCar
            carDict[car] = sortRoute[begin:end, 0]
        #print(carDict)
    centerNode_per_car = carDict.copy()
    return sortRoute, centerNode_per_car, extra, carDict
    

def find_minimum(array, excludeRange):
    thresh = min(array) * excludeRange
    candidate = np.where(array <= thresh)[0] # np.whereだけだとタプルが返ってくるのでタプルのどこを取ってくるのかを書いておかないといけない！！
    #print(candidate)
    min_ = np.random.choice(candidate)
    #print(min_)
    return min_
    
#find_minimum(np.array([3,2,1,2,1]), 1.02)    

def DijkstraAki(g, s, t, excludeRange): # 自分でいじる方　
    global result
    d = np.array([np.inf] * g.number_of_nodes()) # step 1
    b = np.array([False] * g.number_of_nodes()) # step 1
    d[s] = 0 # step 2
    routeFrom = np.zeros(g.number_of_nodes())
    count_ = 0
    while not b[t]: # step 3
        i = (d + (b*1e100)).argmin() # step 4  # false に数をかけると、0として扱われる 
        if np.isinf(d[i]): break # step4
        b[i] = True # step 4
        for j in g.neighbors(i): # step 4
            d_tent = d[j]
            d[j] = min(d[j], d[i] + g.adj[i][j].get('weight', 1)) # step 4
            if d_tent > d[i] + g.adj[i][j].get('weight', 1): #今までの値より早いパスが見つかった場合
                routeFrom[j] = i
            count_ += 1
            if count_ % 5000 == 0:
                #print("count=",count_)
                print("Dijkstra Failed.")
                result = False
                break
                #sys.exit()
        else:
            continue
        break
        
    route = []
    goal = t
    count = 0
    #print(routeFrom[163])
    if b[t]:
        while goal != s or (count > g.number_of_nodes()):
            route.append(routeFrom[goal])
            goal = int(routeFrom[goal])
            count += 1
            if count % 1000 == 0:
                print("count=",count)
            #print(route)
        result = True
    return d[t], route, result   

###-------続いて各車の辞書をもとに各車の配送コースをダイクストラで計算-------------###
def pathPlan(basePlace, carDict):
    global carRouteDict
    carRouteDict = {}
    for car in carDict.keys():
        route = carDict[car]
        #route = np.insert(route, 0, basePlace, axis = 0 )
        #route = np.append(route, basePlace )
        #print(route)
        path = np.zeros(0)
        for i in range(int(len(route) - 1)):
            s, t = int(route[i]), int(route[i + 1])
            d, routeList = DijkstraAki(G2, s, t, 1.00)[:2] # （グラフ、出発地、目的地, 範囲）
            path = np.append(path, np.array([routeList[::-1]]))
        path = np.append(path, basePlace)
        #print(path)
        carRouteDict[car] = path

# # 続いて車がRoute上を走ってるアニメを描く。
def drawGraph(basePlace, carDict,DelivInfo2):
    global data_
    #cmap = plt.get_cmap("tab10") 
    cl = ["r", "magenta", "g","orange", "brown", "violet", "black", "cyan", "mistyrose"]
    centroids = np.zeros((0,2))
    nx.draw(G2, pos2, node_size = 0, node_color = "b", edge_color="gray", width=1,)
    count = 0
    #DelivInfo2 = pickle_load("DelivInfoFiles2/delivInfo.pickle")
    for delivGroup in DelivInfo2.keys():
        group = DelivInfo2[delivGroup]
        data_ =  np.array([ pos2[num]  for num in group])
        #print(data_)
        plt.scatter(data_[:,0], data_[:,1], marker = ".", c =cl[int(count%9)], s = 40)
        plt.scatter(pos2[delivGroup][0], pos2[delivGroup][1], marker = "*", c =cl[int(count%9)], s = 80, alpha = 0.6)
        count += 1
        
    # とりあえず、上のグラフに最適ルート(仮)を書き込むためのコードを作る
    cl_car = ["blue", "red"]
    for car in carDict.keys():
        route = carDict[car]
        posOpt = { points: pos2[points] for points in route}
        EdgeOpts = [(route[i], route[i+1]) for i in range(len(route) - 1)]
        Gopt = nx.Graph()
        Gopt.add_edges_from(EdgeOpts)
        #nx.draw(Gopt, posOpt, node_size = 8, node_color = cl_car[car], edge_color=cl_car[car], width=3,)
        base = pos2[basePlace]
    plt.scatter(pos2[basePlace][0], pos2[basePlace][1], marker = "*",  c = "green", s = 300)
    plt.show()
    plt.close()


def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)

def saveFiles(DelivInfo2, basePlace):  
    """import os
    if os.path.exists("DelivInfoFiles2"):
        import shutil
        shutil.rmtree("DelivInfoFiles2")
    os.mkdir("DelivInfoFiles2")  """    
    for car in list(centerNode_per_car.keys()):
        route_ = centerNode_per_car[car]
        route_ = route_[~(route_ == basePlace )]
        centerNode_per_car[car] = route_
    pickle_dump(pos2, "DelivInfoFiles3/positionAllNodes.pickle")
    pickle_dump(centerNode_per_car, "DelivInfoFiles3/centerNode_per_car.pickle")
    pickle_dump(DelivInfo2, "DelivInfoFiles3/delivInfo.pickle")
    pickle_dump(carRouteDict, "DelivInfoFiles3/carDict.pickle")
    np.savetxt("DelivInfoFiles3/sort_route.csv", sortRoute, delimiter = ",")
    #np.savetxt("DelivInfoFiles2/route.csv", route, delimiter = ",")
    np.savetxt("DelivInfoFiles3/edge_weight.txt", edges, fmt = '%d')
    


def test(base, carNum, centerNode_per_car, distDict):
    #global distDict, centerNode_per_car
    #distDict = pickle_load("DelivInfoFiles3/distanceWeight.pickle")
    #print(distDict)
    #centerNode_per_car = pickle_load("DelivInfoFiles2/centerNode_per_car.pickle")
    #print(distDict[(10,888)])
    #datalen = print(len(list(pos2.keys())))
    for car in range(carNum):
        routeTent = centerNode_per_car[car] #とりあえずの解を作る
        #print("carNum:",car,"route:",routeTent)
        routeTent = np.insert(routeTent,0, base)
        routeTent = np.insert(routeTent, len(routeTent), base)
        datalen = len(routeTent)
        #print("routeTent=",routeTent)
        #print("cost=",cal_totalcost(distDict, routeTent))
        route = algs.opt_2(distDict, datalen, routeTent)
        centerNode_per_car[car] = route
 

def main(basePlace, Num_of_places, driver_deliv, droneNum, carNum):
    df = pd.read_csv("networkData.csv")
    pos = pickle_load("positionDict.pickle")
    G2, posRename = RenamePoints(df, pos)
    ###--------DelivPlaceを選んだ際、連結されてない点をトラックに辿らせちゃいけない--------###
    success = False
    count = 0
    #pdb.set_trace()
    while not success:
        count += 1
        selectDelivPlaces(basePlace, Num_of_places, posRename) # 引数は基地の場所
        cluster(droneNum, carNum, driver_deliv )
        success = crt_delivInfo(driver_deliv, G2, basePlace)
        print("{}回目のinitialize".format(count))
    #decideOrder()"""
    decideOrderMultiple(carNum, basePlace)
    test(basePlace, carNum)
    pathPlan(basePlace, centerNode_per_car)
    drawGraph(basePlace, carRouteDict)
    saveFiles()
    return result

def crt_dist_dict():
    df = pd.read_csv("networkData.csv")
    pos = pickle_load("DelivInfoFiles3/positionAllNodes.pickle")
    #print(pos)
    G2 = RenamePoints(df, pos)[0]
    #print(list(G2.nodes)[:1000])
    sortedNodes = list(G2.nodes).copy()
    sortedNodes.sort()
    distDict = {}
    import itertools
    for i, j in itertools.combinations(sortedNodes,2):

    #for i, j in itertools.combinations(list(G2.nodes),2):
        print(i,j)
        if nx.has_path(G2, i, j):
            a = nx.shortest_path_length(G2, source = i, target = j, weight="weight")
        else:
            a = 0
        distDict[(i,j)] = a
        #print("dist=",a)
    pickle_dump(distDict, "DelivInfoFiles3/distanceWeight.pickle")
    

def scatter_part(car_num, drone_num, basePlace, Num_of_places):
    df = pd.read_csv("networkData.csv") #jupyterNotebookで作る 
    pos = pickle_load("positionDict.pickle") #jupyterNBで作る
    G2, pos2 = RenamePoints(df, pos)
    #print(pos2)
    ###--------DelivPlaceを選んだ際、連結されてない点をトラックに辿らせちゃいけない--------###
    success = False
    count = 0
    #pdb.set_trace()
    while not success:
        count += 1
        DELIV_PLACES = selectDelivPlaces(basePlace, Num_of_places, pos2) # 引数は基地の場所
        orderedNames, coors, labels, N_CLUST = cluster(drone_num, car_num, True, DELIV_PLACES )
        success, DelivInfo2, centroids= crt_delivInfo(True, G2, basePlace, orderedNames, coors, labels, N_CLUST, pos2)
        print("{}回目のinitialize".format(count))
    order = np.array(orderedNames)
    #print(DELIV_PLACES)
    pickle_dump(DELIV_PLACES, "DelivInfoFiles3/DELIV_PLACES.pickle")
    np.savetxt("DelivInfoFiles3/orderPlaces.csv", order, delimiter = ",")
    #crt_dist_dict() #点数が少なかったらこれをここにいれてもいいけど、、って感じ
    #routingPart(car_num, drone_num, basePlace, DelivInfo2, centroids)
    return success, DelivInfo2, centroids

    
def routingPart(carNum, droneNum, basePlace, DelivInfo2, centroids):
    global DELIV_PLACES, orderedNames, pos2, G2, edges, success
    orderedNames = np.loadtxt("DelivInfoFiles3/orderPlaces.csv", delimiter = ",")
    DELIV_PLACES = pickle_load("DelivInfoFiles3/DELIV_PLACES.pickle")
    G2 = nx.read_weighted_edgelist("DelivInfoFiles3/edge_weight.txt", nodetype = int)
    pos2 = pickle_load("DelivInfoFiles3/positionAllNodes.pickle")
    edges = np.loadtxt("DelivInfoFiles3/edge_weight.txt")
    distDictMat = pickle_load("DelivInfoFiles3/distanceWeight.pickle")
    orderedNames, coors, labels, N_CLUST = cluster(droneNum, carNum, True , DELIV_PLACES)
    success, DelivInfo2, centroids = crt_delivInfo(True, G2, basePlace, orderedNames, coors, labels, N_CLUST, pos2)
    sortRoute, centerNode_per_car, extra, carDict=decideOrderMultiple(carNum, basePlace, DelivInfo2, centroids)
    test(basePlace, carNum, centerNode_per_car, distDictMat)
    pathPlan(basePlace, centerNode_per_car)
    drawGraph(basePlace, carRouteDict, DelivInfo2)
    saveFiles(DelivInfo2, basePlace)
    
def itertest():
    global DELIV_PLACES
    #import itertools
    #for i, j in itertools.combinations(np.arange(0,5,1),2):
     #   print(i,j)
    DELIV_PLACES = pickle_load("DelivInfoFiles3/distanceWeight.pickle")
    #print(DELIV_PLACES.keys())

    #print(DELIV_PLACES[(0,1)])
    #print(distDict[(0,1)])


        
    

if __name__ == "__main__":
    #result = main(554, 60, True, 4, 2) # 基地の位置, 配送点数, 運転手は運ぶか, 全部のドローン台数, トラック台数
    #print(result)
    #crt_latice_network(3)
    #RenamePoints

    #crt_dist_dict()
    #itertest()
    scatter_part(1, 1, 100, 30)
    #routingPart(2, 0, 554, 60)
    #routingPart(1, 0, 554)
    """test(554)
    pathPlan(554, centerNode_per_car )
    drawGraph(554, carRouteDict)
    for car in list(centerNode_per_car.keys()):
        route_ = centerNode_per_car[car]
        route_ = route_[~(route_ == 554 )]
        centerNode_per_car[car] = route_
    pickle_dump(centerNode_per_car, "DelivInfoFiles2/centerNode_per_car.pickle")
    #pickle_dump(DelivInfo2, "DelivInfoFiles2/delivInfo.pickle")
    pickle_dump(carRouteDict, "DelivInfoFiles2/carDict.pickle")"""
    #np.savetxt("DelivInfoFiles2/sort_route.csv", sortRoute, delimiter = ",")
    
    print("Done")