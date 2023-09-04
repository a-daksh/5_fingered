import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional
import pybullet as p
import pybullet_data
import time
import pywt
import random
import math
import csv
import warnings
warnings.filterwarnings("ignore")
 
class GraspEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None):

        super(GraspEnv, self).__init__()

        low1=[0,0,-300,0,0,-200,0,0,-200,0,0,-300,0,0,-300]
        high1=[300,300,0,300,300,0,300,300,0,300,300,0,300,300,0]
        self.action_space = spaces.Box(low=np.array(low1),high=np.array(high1),dtype=np.float32)

        low2=[-1 for i in range(15)]+[-50 for i in range(15)]+[0,0,0,0,0]
        high2=[1.57 for i in range(15)]+[50 for i in range(15)]+ [50,50,50,50,50]# 15 angles +15 velocities
        self.observation_space = spaces.Box (np.array(low2),np.array(high2), dtype=np.float32 )

        self.render_mode = render_mode
        self.state = [0 for i in range(30)]
        # force array contains forces of tips : link number-6,9,12,15,18(fore,middle,ring,little,thumb)
        self.forceprev=[0,0,0,0,0]
        self.forces=[0,0,0,0,0]
        self.points=[[] for i in range(5)]
        self.force_log=[[0,0,0,0,0] for i in range(200)]
        self.time_prev=time.time()
        self.reward=0
        self.Terminated=False
        self.Truncated=False

        self.slip=0
        self.deformation=0

        if self.render_mode=="human":
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)

        self.planeId = p.loadURDF("plane.urdf")
        self.tableId = p.loadURDF("urdf/cube.urdf",[0.1,0.65,2.5],[0,0,0,1],globalScaling=10)
        

        self.rigidId = p.loadSoftBody("tinker.obj",[4.3,-6.9,5.5],mass = 0.75,scale=0.7,useMassSpring=True,
                    springElasticStiffness=500,springDampingStiffness=10,
                    springDampingAllDirections=1000,frictionCoeff=0.5)

        self.handId = p.loadURDF("urdf/hand_T42.urdf",[0,0,-0.25], [0,0,0.707,0.707],globalScaling=10)
        p.stepSimulation()

        meshData = p.getMeshData(self.rigidId)
        vertices=np.asarray(meshData)[1]
        self.initial_volume = 0.0

        for i in range(0, len(vertices), 3):
            
            A, B, C = vertices[i:i+3]
            D = [0, 0, 0]
            signed_volume = self.calculate_signed_volume(A, B, C, D)
            self.initial_volume += signed_volume

    def reset(self, seed=None, options=None):

        # with open("DR_0-75_0.5_NR.csv", "a", newline="") as file:

        #     writer = csv.writer(file)
        #     writer.writerow([self.slip, self.deformation])

        with open("DR_forces.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([999,999,999,999,999])

        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)
        
        self.planeId = p.loadURDF("plane.urdf")
        self.tableId = p.loadURDF("urdf/cube.urdf",[0.1,0.65,2.5],[0,0,0,1],globalScaling=10)

        self.rigidId = p.loadSoftBody("tinker.obj",[4.3,-6.9,5.5],mass = 0.75,scale=0.7,useMassSpring=True,
                    springElasticStiffness=500,springDampingStiffness=10,
                    springDampingAllDirections=1000,frictionCoeff=0.5)
        self.handId = p.loadURDF("urdf/hand_T42.urdf",[0,0,-0.25], [0,0,0.707,0.707],globalScaling=10)
        p.stepSimulation()

        self.state = self.get_state()
        self.reward=0
        # force array contains forces of tips : link number-6,9,12,15,18(fore,middle,ring,little,thumb)
        self.forceprev=[0,0,0,0,0]
        self.forces=[0,0,0,0,0]
        self.points=[[] for i in range(5)]
        self.force_log=[[0,0,0,0,0] for i in range(200)]
        self.Terminated=False
        self.Truncated=False
        self.time_prev=time.time()

        self.slip=0
        self.deformation=0

        meshData = p.getMeshData(self.rigidId)  
        vertices=np.asarray(meshData)[1]
        self.initial_volume = 0.0

        for i in range(0, len(vertices), 3):

            A, B, C = vertices[i:i+3]
            D = [0, 0, 0]
            signed_volume = self.calculate_signed_volume(A, B, C, D)
            self.initial_volume += signed_volume

        return np.array(self.state, dtype=np.float32),{}

    def step(self,action):

        time_now=time.time()
        self.get_force()
        del_t=time_now-self.time_prev


        if del_t<=5:

            p.stepSimulation()
            self.execute_action(action)
            self.state=self.get_state()
            self.reward=self.get_reward()
            # print(self.reward)

            self.Terminated=False
            self.Truncated=False
            if del_t>=4.5:
                with open("DR_forces.csv", "a", newline="") as file:

                    writer = csv.writer(file)
                    writer.writerow([self.forces,self.points])

        elif del_t>5 and del_t<=10:   
            # Replicating jerks at each second
            # if del_t==6 or del_t==7 or del_t==8 or del_t==9 or del_t==10:
            #     mass_random=random.uniform(0.75, 1.25)
            #     cof=random.uniform(0.04,1)
            #     p.changeDynamics(self.rigidId, -1, mass=mass_random, lateralFriction=cof)    

            p.stepSimulation()
            p.setJointMotorControl2(self.handId,2,p.POSITION_CONTROL,targetPosition=2,force = 2000,velocityGain=10,positionGain=1)
            self.execute_action(action)
            self.state=self.get_state()
            self.reward=self.get_reward()
            # print(self.reward)
            self.Terminated=False
            self.Truncated=False
            if del_t<=5.5:
                with open("DR_forces.csv", "a", newline="") as file:

                    writer = csv.writer(file)
                    writer.writerow([self.forces,self.points])
        
        else:
            self.Truncated=True

        return np.array(self.state, dtype=np.float32),self.reward,self.Terminated,self.Truncated, {}

    def render(self):
        if self.render_mode=="human":
            print(self.reward)
            print("##########")
        else:
            self.state=self.get_state()
            
    def close(self):
        p.disconnect(self.physicsClient)

    def get_state(self):

        statepos=[]
        statevel=[]

        for j in range(4,19):
            
            a=p.getJointState(self.handId,j)
            statepos.append(a[0])
            statevel.append(a[1])

        return(statepos+statevel+self.forces)

    def execute_action(self,action):

        time.sleep(1./240.)
        p.setJointMotorControlArray(self.handId,[i for i in range(4,19)],
                                p.TORQUE_CONTROL, forces=[action[i] for i in range(15)])

    def get_force(self):

        self.forceprev = self.forces

        self.force_log.append(self.forceprev) 
        if len(self.force_log)>200:
            self.force_log.remove(self.force_log[0])

        self.forces = [0,0,0,0,0]

        myDict = {0:6,1:9,2:12,3:15,4:18}
        for i in range(5):
            nor=0
            poi=[0,0,0]
            L=p.getContactPoints(bodyA = self.handId,bodyB = self.rigidId, linkIndexA = myDict[i])
            for con in L:
                nor+=con[9]
                poi=con[5]
            self.forces[i] = nor
            self.points[i] = poi
        # print(self.forces)
        return

    def calculate_signed_volume(self,A, B, C, D):

        AB = np.array(B) - np.array(A)
        AC = np.array(C) - np.array(A)
        AD = np.array(D) - np.array(A)
        volume = np.dot(AB, np.cross(AC, AD)) / 6

        return volume
    
    def chk_slip(self):

        lis_1=[0 for i in range(200)]
        lis_2=[0 for i in range(200)]
        lis_3=[0 for i in range(200)]
        lis_4=[0 for i in range(200)]
        lis_5=[0 for i in range(200)]
        
        slips=[False,False,False,False,False]

        for i in range(200):
            lis_1[i]=self.force_log[i][0]
            lis_2[i]=self.force_log[i][1]
            lis_3[i]=self.force_log[i][2]
            lis_4[i]=self.force_log[i][3]
            lis_5[i]=self.force_log[i][4]

        coefficients_1 = pywt.wavedec(lis_1, 'haar', level=5)
        approximation_1 = coefficients_1[0]
        data_reconstructed_1 = pywt.waverec([approximation_1] + [None] * 5, 'haar')
        for i in range(len(data_reconstructed_1)-1):
            if data_reconstructed_1[i]-data_reconstructed_1[i+1]>2:
                slips[0]=True

        coefficients_2 = pywt.wavedec(lis_2, 'haar', level=5)
        approximation_2 = coefficients_2[0]
        data_reconstructed_2 = pywt.waverec([approximation_2] + [None] * 5, 'haar')
        for i in range(len(data_reconstructed_2)-1):
            if data_reconstructed_2[i]-data_reconstructed_2[i+1]>2:
                slips[1]=True

        coefficients_3 = pywt.wavedec(lis_3, 'haar', level=5)
        approximation_3 = coefficients_3[0]
        data_reconstructed_3 = pywt.waverec([approximation_3] + [None] * 5, 'haar')
        for i in range(len(data_reconstructed_3)-1):
            if data_reconstructed_3[i]-data_reconstructed_3[i+1]>2:
                slips[2]=True

        coefficients_4 = pywt.wavedec(lis_4, 'haar', level=5)
        approximation_4 = coefficients_4[0]
        data_reconstructed_4 = pywt.waverec([approximation_4] + [None] * 5, 'haar')
        for i in range(len(data_reconstructed_4)-1):
            if data_reconstructed_4[i]-data_reconstructed_4[i+1]>2:
                slips[3]=True

        coefficients_5 = pywt.wavedec(lis_5, 'haar', level=5)
        approximation_5 = coefficients_5[0]
        data_reconstructed_5 = pywt.waverec([approximation_5] + [None] * 5, 'haar')
        for i in range(len(data_reconstructed_5)-1):
            if data_reconstructed_5[i]-data_reconstructed_5[i+1]>2:
                slips[4]=True

        return slips

    def get_distance(self):

        object_position,_=p.getBasePositionAndOrientation(self.rigidId)
        link_position=[0 for i in range(5)]
        for i in {6,9,12,15,18}:
            link_state=p.getLinkState(self.handId,i)
            link_position[int((i/3)-2)]=link_state[0]

        distance=[0 for i in range(5)]
        for i in range(5):
            distance[i]=np.linalg.norm(np.array(object_position)-np.array(link_position[i]))
        return distance
    
    def get_reward(self):

        # if touches +10 per finger
        R1_=[0,0,0,0,0]
        for index, value in enumerate(self.forces):
            if value >= 1:
                R1_[index] = 10 
        R1=sum(R1_)

        # if does not slip +10 per finger
        check=self.chk_slip()
        R2_=check.count(False)
        R2=10*R2_
        self.slip+=(5-R2_)

        # deformation subtracted 
        R3=0
        meshData = p.getMeshData(self.rigidId)  
        vertices=np.asarray(meshData)[1]
        volume = 0.0
        for i in range(0, len(vertices), 3):
            A, B, C = vertices[i:i+3]
            D = [0, 0, 0]
            signed_volume = self.calculate_signed_volume(A, B, C, D)
            volume += signed_volume
        self.deformation=self.initial_volume-volume
        R3=((self.initial_volume-volume)/self.initial_volume)*50

        # if distance 0, +10 per finger
        distance=self.get_distance()
        log_values=[1/math.log(x+1.1051) for x in distance]
        R4=sum(log_values)

        # print("Contact Reward: ", R1)
        # print("Slip Reward: ", R2)
        # print("Deformation Penalisation: ", R3)
        # print("Closeness Reward: ", R4)
        # print("##############")

        return (R1+R2-R3+R4) 

# from stable_baselines3.common.env_checker import check_env

# env = GraspEnv()
# check_env(env)
