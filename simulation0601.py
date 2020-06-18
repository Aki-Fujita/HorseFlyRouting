
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import prepatation0602 as prep
import pdb
import os
import cv2

from matplotlib.offsetbox import OffsetImage, AnnotationBbox 


import pandas as pd
import pickle
def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data

###----------Preparation Fileから必要なデータのインポート--------------------########
def importFiles():
    global delivInfo, routeList, centerNode_per_car, pos2, G2, orderedNames
    orderedNames = np.loadtxt("DelivInfoFiles3/orderPlaces.csv", delimiter = ",")
    delivInfo = pickle_load("DelivInfoFiles3/delivInfo.pickle")
    posAllNodes = pickle_load("DelivInfoFiles3/positionAllNodes.pickle")
    routeList = pickle_load("DelivInfoFiles3/carDict.pickle")
    #sortRoute_ = np.loadtxt("DelivInfoFiles2/sort_route.csv", delimiter = ",")
    G2 = nx.read_weighted_edgelist("DelivInfoFiles3/edge_weight.txt", nodetype = int)
    centerNode_per_car = pickle_load("DelivInfoFiles3/centerNode_per_car.pickle")
    
    df = pd.read_csv("networkData.csv")
    coords = df.values[:,1:3]
    
    pos2 = posAllNodes.copy()
    nx.draw(G2, pos2, node_size = 0, node_color = "b", edge_color="gray", width=1)
    
def drawFirst():
    cl = ["r", "magenta", "g","orange", "brown", "violet", "black", "cyan", "mistyrose"]
    cluster = 0
    for centNum in delivInfo.keys():
        cluster += 1
        placesGroup = delivInfo[centNum] # グループに対してそのグループの配送点
        data_ = np.array([pos2[places] for places in placesGroup])
        centerNodes = pos2[centNum]
        plt.scatter(data_[:,0], data_[:,1], marker = ".", c =cl[int(cluster%9)], s = 40)
        plt.scatter(centerNodes[0], centerNodes[1], marker = "*", c =cl[int(cluster%9)], s = 80, alpha = 0.6)
    

####-------Simulationの始まり---------------####
def simulation_initialize(base, num_of_places, car_deliv, drone_num, car_num):
    result = False
    count = 0
    while not result:
        #print("a")
        count += 1
        #pdb.set_trace()
        result = prep.main(base, num_of_places, car_deliv, drone_num, car_num)
    print(result)
    print("prepare回数:",count)

class drone:
    global car_num
    def __init__(self, id_, dep_time, speed, pos, car_num): #startPoint, destination, time_outというのを後でagent変数として加える
        self.id_ = id_
        self.pos = pos
        #self.car_on = int( id_ / car_num )
        self.car_on = int( id_ % car_num )
        self.routeList = routeList[self.car_on] # ここの書き方がおかしかった！！
        self.depNum = self.routeList[0]
        self.dep_corr = self.pos[self.depNum]
        self.destNum = None
        self.dest_corr = None
        self.position = self.dep_corr
        self.speed = speed
        self.delivered = False
        self.working = False # 車に戻ったらこれを切る
        self.flightDistance = 0
        
    def procede(self):
        dist = np.linalg.norm(self.dest_corr - self.position)

        if dist < self.speed:
            self.position = self.dest_corr
            self.flightDistance += dist
            #print("drone_pos=", self.position)
            #print("Delivered")
            
            if self.delivered and (np.linalg.norm(np.array(car_corr[self.car_on]) - self.position) < self.speed): # 車に着いたら、という意味
                self.working = False
                #print("Drone_"+str(self.id_)+" arrived at car")
                #print("a")
            if not self.delivered:
                self.delivered = True
                #print("Drone_"+str(self.id_)+" Delivered")
            
        else:
            vector = self.dest_corr - self.position
            e = vector / np.linalg.norm(vector + 1e-10)
            self.position = self.position + e * self.speed
            self.flightDistance += self.speed
            #print("drone_pos=", self.position)
    def work(self):
        if self.delivered:
            #print("test")
            self.dest_corr = car_corr[self.car_on]
            #print("destination=", self.dest_corr)
        self.procede()
            
    def action(self):
        if self.working:
            self.work()

        elif GoSign[self.car_on]: #車がcentroidを超えていたらドローンに目的地を与える。
            self.destNum = deliverPlaces[self.car_on].pop(0)
            #print("POPPED! deliverPlaces", deliverPlaces)
            self.dest_corr = np.array(self.pos[self.destNum])
            self.working = True
            self.delivered = False
            if len(deliverPlaces[self.car_on]) == 0:
                GoSign[self.car_on] = False
            #print("dest=", self.dest_corr)
            #print("OFF")
        else:
            self.position = car_corr[self.car_on]
    
class car:
    def __init__(self, id_, dep_time, speed, routeList, centroids, pos, num_my_drone): #
        self.id_ = id_
        self.routeList = routeList
        self.centroids = centroids
        self.deliveryList = list(self.centroids)
        self.pos = pos
        self.depNum = 0 # この番号はあくまでもルートリストの中のindexであることに注意！！
        self.destNum = 1  # この番号はあくまでもルートリストの中のindexであることに注意！！
        self.dep_corr = self.pos[routeList[0]]
        self.dest_corr = self.pos[routeList[1]]
        self.position = self.dep_corr
        self.speed = speed
        self.visitNodes = len(self.routeList)
        self.centroidNum = 0
        self.finished = False
        self.changeDeliv = False
        self.driveDistance = 0
        self.num_my_drone = num_my_drone
        
    def procede(self):
        #print("destは",self.dest_corr)
        dist = np.linalg.norm(np.array(self.dest_corr) - self.position)
        #print(dist)
        if dist < self.speed: # 次のステップで近づけたら、という意味
            self.position = self.dest_corr
            self.changeDirection()
            self.driveDistance += dist
            #print("car arrived")
        else:
            vector = np.array(self.dest_corr) - self.position # まだ目標まで遠いときは、の意味
            e = vector / np.linalg.norm(vector + 1e-10)
            self.position = self.position + e * self.speed
            self.driveDistance += self.speed
            #print("car_pos=", self.position)
        
        if self.changeDeliv:
            deliverPlaces[self.id_] = delivInfo[int(self.routeList[self.depNum] )].tolist()
            deliverPlaces[self.id_].remove(self.routeList[self.depNum])
            self.changeDeliv = False
            #print("time=", time)
            #print("car_{0}のdelivListは現在{1}、目的地番号{2}".format(self.id_, deliverPlaces[self.id_], self.depNum))
            
            #print()
    
    def count(self): #次の重心についてしまったけどドローンが全員帰ってきてないときは待つ
        if self.num_my_drone >= 1:
            count_ = sum([1 for drone in droneSet if drone.working if drone.car_on == self.id_]) #countは飛んでるドローンの数
        else:
            count_ = 0
        return count_
    
    def changeDirection(self): # Centroidまでは何も考えず進む、centroidでは必ずdroneが全部来るまで待つ
        self.depNum = self.destNum # 出発点を変更
        if self.routeList[self.depNum] in self.centroids and self.routeList[self.depNum] in self.deliveryList and self.count() != 0: # まだ帰ってきてないドローンがいる場合
            #print("Waiting for Drone")
            pass
            
        elif self.destNum >= len(self.routeList) - 1:
            self.finished = True
        
        elif self.routeList[self.depNum] in self.centroids and self.routeList[self.depNum] in self.deliveryList and self.count() == 0: # これが0だったら全員車に戻ってる
            GoSign[self.id_] = True
            self.changeDeliv = True
            #print("switched: depNum=", self.depNum)
            self.deliveryList.remove(self.routeList[self.depNum])
            self.destNum += 1
            self.dest_corr = self.pos[self.routeList[self.destNum]]
                
        else: # centroids以外における次の目的地設定
            self.destNum += 1
            self.dest_corr = self.pos[self.routeList[self.destNum]]
            #print("car_dest=",self.dest_corr, "destNum=",self.destNum)
            
def imscatter(x, y, image, ax=None, zoom=1): 
    if ax is None: 
        #print("a")
        ax = plt.gcf() 
    try: 
        #print("a")
        image = plt.imread(image) 
    except TypeError: 
     # Likely already an array... 
        pass 
    im = OffsetImage(image, zoom=zoom) 
    x, y = np.atleast_1d(x, y) 
    artists = [] 
    for x0, y0 in zip(x, y): 
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False) 
        artists.append(ax.add_artist(ab)) 
    ax.update_datalim(np.column_stack([x, y])) 
    ax.autoscale() 
    return artists 


def simulation(drone_num, car_num, routeList, centerNode_per_car, basePlace):
    global droneSet, delivList, deliverPlaces, time, GoSign, car_corr
    num_my_drone = int(drone_num / car_num)
    car_trajectory = np.zeros((0, int(2 * car_num) ))
    carSet = [car(i, 0, 1e-4,  routeList[i], centerNode_per_car[i], pos2, num_my_drone) for i in range(car_num)]
    if drone_num >= 1:
        drone_trajectory = np.zeros((0, int(2 * drone_num) ))
        droneSet = [drone(i, 0, 2e-4,  pos2, car_num) for i in range(drone_num)]
    ##-----Simulationの初期設定------------
    GoSign = np.array([False for i in range(car_num)])
    #print(GoSign)
    deliverPlaces = {}
    for car_ in range(car_num):
        startPlace = centerNode_per_car[car_][0]
        deliverPlaces[car_] = delivInfo[int(startPlace)].tolist()
    #print("deliverPlaces=", deliverPlaces)
    cl_drone = [ "orange", "orange","green","green", "brown", "brown"]
    cl_car = ["blue", "red"]
    ##------開始----------##
    for time in range(4000):
        #if time % 20 == 0:
         #   print("time=", time)
        #print("deliverPlaces=", deliverPlaces)
        #print(np.any([drone.working for drone in droneSet]))
        if time == 0:
            pass
        elif np.all([car.finished for car in carSet]):
            print("time=", time)
            print("finished")
            break
        else:  
            #print("t=",time)
            [car.procede() for car in carSet]
            car_corr = [car.position for car in carSet]
            [drone.action() for drone in droneSet]
            #GoSign = car_1.GoSign
        #print(np.array([[car.position for car in carSet]]).flatten() )
        car_trajectory = np.append(car_trajectory, np.array([np.array([[car.position for car in carSet]]).flatten() ]), axis = 0)
        a = np.array([droneSet[i].position for i in range(drone_num)]).flatten() # 2の部分はドローンの台数
        #print("dronePos=",a)
        drone_trajectory = np.append(drone_trajectory, np.array([a]), axis = 0)
        #print(drone_trajectory)
        #print()
        
        ##-----------動画用---------------##
        
        """if time % 2 == 0:
            img = cv2.imread('realmappp.png')
            height = img.shape[0]
            width = img.shape[1]
            fig, ax = plt.subplots(figsize=(12, 9), sharex=True, sharey=True) 
            nx.draw(G2, pos2, node_size = 0, node_color = "b", edge_color="gray", width=0.1)
            drawFirst()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            img1=img[int(0.01*height):int(0.98*height),int(width*0.011):int(width*0.985)]
            ax.imshow(img1, extent=[*xlim, *ylim], aspect='auto', alpha=0.6)

            #plt.scatter(posOpt[212][:,0],posOpt[212][1], marker = "h",c = "green", )"""
    fig, ax = plt.subplots(figsize=(8, 6), sharex=True, sharey=True) 
    nx.draw(G2, pos2, node_size = 0, node_color = "b", edge_color="gray", width=0.1)
    for droneID in range(drone_num):
        ax.plot(drone_trajectory[:,int(droneID * 2)], drone_trajectory[:,int(droneID * 2 + 1)], c = cl_drone[droneID], linewidth = 1)
            #imscatter([drone_trajectory[time,int(droneID * 2)]], [drone_trajectory[time,int(droneID * 2 +1)]], "drone.png", ax=ax,  zoom=0.04) 
    for carID in range(car_num):
        ax.plot(car_trajectory[:,int(carID * 2)], car_trajectory[:,int(carID * 2 + 1)], c = cl_car[carID], linewidth = 3)
            #imscatter([car_trajectory[time,int(carID * 2)]], [car_trajectory[time,int(carID * 2 +1)]], "ambulance.png", ax=ax,  zoom=0.05) 
            #ax.scatter(car_trajectory[time,int(carID * 2)], car_trajectory[time,int(carID * 2 + 1)], marker = "*", s=300,c = cl_car[carID])

    ax.scatter(pos2[basePlace][0], pos2[basePlace][1], marker = "*",  c = "black", s = 300)
        #print(drone_trajectory)
    #plt.savefig("forMovieMultiple/time="+str(time)+"_test.png")
    plt.show()
        #plt.close()
        
            #plt.show()"""
    
    ###-------------諸々の値を計算する-----------------###
    drone_distance = np.sum(np.array([droneSet[i].flightDistance for i in range(drone_num)]))
    car_distance = np.sum([car.driveDistance for car in carSet])
    
    print("Total Car distance = ", car_distance)
    print("Total Drone distance = ", drone_distance)
    print("Total Time = ", time)
    
    result_ = np.array([drone_num, car_num,car_distance, drone_distance, time])
    return result_
    
def simulation_withoutdrone(drone_num, car_num, routeList, centerNode_per_car, basePlace):
    global delivList, deliverPlaces, time, GoSign, car_corr
    car_trajectory = np.zeros((0, int(2 * car_num) ))
    num_my_drone = int(drone_num / car_num)
    carSet = [car(i, 0, 1e-4,  routeList[i], centerNode_per_car[i], pos2, num_my_drone) for i in range(car_num)]
   
    GoSign = np.array([False for i in range(car_num)])
    #print(GoSign)
    deliverPlaces = {}
    for car_ in range(car_num):
        startPlace = centerNode_per_car[car_][0]
        print(delivInfo)
        
        deliverPlaces[car_] = delivInfo[int(startPlace)].tolist()
        #deliverPlaces[car_] = [basePlace]

    #print("deliverPlaces=", deliverPlaces)
    
    cl_car = ["blue", "red"]
    ##------開始----------##
    for time in range(5000):
        #print("time=", time)
        if time == 0:
            pass
        elif np.all([car.finished for car in carSet]):
            print("time=", time)
            print("finished")
            break
        else:  
            #print("t=",time)
            [car.procede() for car in carSet]
            car_corr = [car.position for car in carSet]

        car_trajectory = np.append(car_trajectory, np.array([np.array([[car.position for car in carSet]]).flatten() ]), axis = 0)

    #print(car_trajectory)
    #plt.scatter(posOpt[212][:,0],posOpt[212][1], marker = "h",c = "green", )
    for carID in range(car_num):
        plt.plot(car_trajectory[:,int(carID * 2)], car_trajectory[:,int(carID * 2 + 1)], c = cl_car[carID], linewidth = 3)
    plt.scatter(pos2[basePlace][0], pos2[basePlace][1], marker = "*",  c = "black", s = 300)
    
    plt.show()
    plt.close()
    
    ###-------------諸々の値を計算する-----------------###
    car_distance = np.sum([car.driveDistance for car in carSet])
    drone_distance = 0
    
    print("Total Car distance = ", car_distance)
    print("Total Time = ", time)
    result_ = np.array([drone_num, car_num, car_distance, drone_distance, time])
    return result_

def test(basePlace):
    plt.scatter(pos2[basePlace][0], pos2[basePlace][1], marker = "*",  c = "black", s = 300)


def main_multiple_params(car_num, droneMax, basePlace, Num_of_places):
    success, DelivInfo2, centroids = prep.scatter_part(car_num, 0, basePlace, Num_of_places)
    print(success)
    resultBox = np.zeros((0,6))
    for droneNum in range(droneMax):
        print()
        print("*****Simulation********")
        print("DroneNum = ", droneNum)
        prep.routingPart(car_num, droneNum, basePlace, DelivInfo2, centroids)
        importFiles()
        if droneNum == 0:
            result = simulation_withoutdrone(droneNum, car_num, routeList, centerNode_per_car, basePlace)
        elif droneNum>=1:
            result = simulation(droneNum, car_num, routeList, centerNode_per_car, basePlace)
        #result = np.append(result, Num_of_places)
        #resultBox = np.append(resultBox, np.array([result]), axis = 0)
    print("*****Simulation********")
    """print("CarNum = ", 2)
    prep.routingPart(2, 0, basePlace, DelivInfo2, centroids)
    importFiles()
    result = simulation_withoutdrone(0, 2, routeList, centerNode_per_car, basePlace)
    result = np.append(result, Num_of_places)
    resultBox = np.append(resultBox, np.array([result]), axis = 0)
    return resultBox"""
    #print(resultBox)
    
def simulationMulti(times, car_num, droneMax, basePlace, Num_of_places):
    resultBoxTot = np.zeros((0,7))
    for i in range(times):
        result_ = main_multiple_params(car_num, droneMax, basePlace, Num_of_places)
        #result_ = np.insert(result_, 0, i, axis = 1)
        print(result_)
        #resultBoxTot = np.append(resultBoxTot, result_, axis = 0)
    """print(resultBoxTot)
    df = pd.DataFrame(resultBoxTot)
    df.columns = ["ExpNum","droneNum","CarNum","carDistance" ,"droneDistance", "time", "NumPlaces" ]
    filename = "droneSimulationResult"
    #os.mkdir(filename)
    #------ まずはデータフレームと実験条件の保存 -----------
    df.to_csv(filename+"/droneSimulation_all_{0}placesAnd{1}times.csv".format(Num_of_places, times))"""

if __name__ == "__main__":
    #main_multiple_params(1 , 4, 554, 60) #basePlaceは通し名でよい
    times=1
    car_num=1
    droneMax=3
    basePlace=280
    Num_of_places=25
    simulationMulti(times, car_num, droneMax, basePlace, Num_of_places) # 554 の点の選び方を考えなおす。このシミュレーションをもっと使いやすく
    #main(1, 1, 554, 60)
    
    print("Done")



