import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional
import pybullet as p
import pybullet_data
import time
import random
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

        low2=[-1 for i in range(15)]+[-50 for i in range(15)]
        high2=[1.57 for i in range(15)]+[50 for i in range(15)] # 15 angles +15 velocities
        self.observation_space = spaces.Box (np.array(low2),np.array(high2), dtype=np.float32 )

        self.render_mode = render_mode
        self.state = [0 for i in range(30)]
        # force array contains forces of tips : link number-6,9,12,15,18(fore,middle,ring,little,thumb)
        self.forceprev=[0,0,0,0,0]
        self.forces=[0,0,0,0,0]
        self.force_log=[[0,0,0,0,0] for i in range(100)]
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
        
        # mass_random=random.uniform(0.75, 1.25)
        # mass_random=1
        mass_random=0.75
        # cof=random.uniform(0.04, 1)
        # cof=1
        cof=1

        self.rigidId = p.loadSoftBody("tinker.obj",[4.3,-6.9,5.5],mass = mass_random,scale=0.7,useMassSpring=True,
                    springElasticStiffness=500,springDampingStiffness=10,
                    springDampingAllDirections=1000,frictionCoeff=cof)

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

        with open("Custom_Logs_DR.csv", "a", newline="") as file:

            writer = csv.writer(file)
            writer.writerow([self.slip, self.deformation])

        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)
        
        self.planeId = p.loadURDF("plane.urdf")
        self.tableId = p.loadURDF("urdf/cube.urdf",[0.1,0.65,2.5],[0,0,0,1],globalScaling=10)

        # mass_random=random.uniform(0.75, 1.25)
        mass_random=0.75
        # mass_random=0.75
        # cof=random.uniform(0.04, 1)
        cof=1
        # cof=0.3

        self.rigidId = p.loadSoftBody("tinker.obj",[4.3,-6.9,5.5],mass = mass_random,scale=0.7,useMassSpring=True,
                    springElasticStiffness=500,springDampingStiffness=10,
                    springDampingAllDirections=1000,frictionCoeff=cof)
        self.handId = p.loadURDF("urdf/hand_T42.urdf",[0,0,-0.25], [0,0,0.707,0.707],globalScaling=10)
        p.stepSimulation()

        self.state = self.get_state()
        self.reward=0
        # force array contains forces of tips : link number-6,9,12,15,18(fore,middle,ring,little,thumb)
        self.forceprev=[0,0,0,0,0]
        self.forces=[0,0,0,0,0]
        self.force_log=[[0,0,0,0,0] for i in range(100)]
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

        elif del_t>5 and del_t<=10:

            p.stepSimulation()
            p.setJointMotorControl2(self.handId,2,p.POSITION_CONTROL,targetPosition=2,force = 2000,velocityGain=10,positionGain=1)
            self.execute_action(action)
            self.state=self.get_state()
            self.reward=self.get_reward()
            # print(self.reward)

            self.Terminated=False
            self.Truncated=False
        
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

        return(statepos+statevel)

    def execute_action(self,action):

        time.sleep(1./240.)
        p.setJointMotorControlArray(self.handId,[i for i in range(4,19)],
                                p.TORQUE_CONTROL, forces=[action[i] for i in range(15)])

    def get_force(self):

        self.forceprev = self.forces

        # self.force_log.append(self.forceprev) 
        # if len(self.force_log)>100:
        #     self.force_log.remove(self.force_log[0])

        self.forces = [0,0,0,0,0]

        myDict = {0:6,1:9,2:12,3:15,4:18}
        for i in range(5):
            nor=0
            L=p.getContactPoints(bodyA = self.handId,bodyB = self.rigidId, linkIndexA = myDict[i])
            for con in L:
                nor+=con[9]
            self.forces[i] = nor
        return

    def calculate_signed_volume(self,A, B, C, D):

        AB = np.array(B) - np.array(A)
        AC = np.array(C) - np.array(A)
        AD = np.array(D) - np.array(A)
        volume = np.dot(AB, np.cross(AC, AD)) / 6
        return volume

    def get_reward(self):

        R1_=[0,0,0,0,0]
        R2_=[0,0,0,0,0]

        for index, value in enumerate(self.forces):
            
            if value >= 1:
                
                R1_[index] = 20 
                R2_[index] = 10 

                if value-self.forceprev[index]<0:
                    self.slip += 1
                    R2_[index]=0

        R1=sum(R1_)
        R2=sum(R2_)
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

        # if self.get_deformation()>0:
        #     R3=self.get_deformation()

        # print("R1: ", R1)
        # print("R2: ", R2)
        # print("R3: ", R3)

        return (R1+R2-R3) 

# from stable_baselines3.common.env_checker import check_env

# env = GraspEnv()
# check_env(env)
