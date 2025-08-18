from typing import List
import numpy as np
from utils import Particle

### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 1
MAX_ERROR = 50000
K = 0.5
#UNIFORM_PARTION = 0.1
NOISE =0.1
NOISE_THETA =0.1

def is_valid(x,y,walls):#保证不会撞到墙
    max_x = np.max(walls[:,0])
    min_x = np.min(walls[:,0])
    max_y = np.max(walls[:,1])
    min_y = np.min(walls[:,1])
    if not (x<=max_x and x>=min_x and y>=min_y and y<=max_y):
        return False
    for wall in walls:
        wall_left = wall[0]-0.5
        wall_right = wall[0]+0.5
        wall_up = wall[1]-0.5
        wall_down = wall[1]+0.5
        if(x>=wall_left and x<=wall_right and y<=wall_down and y>=wall_up):
            return False
    return True
def noise(particle,walls):
    new_position = particle.position+np.random.normal(0,NOISE,size=particle.position.shape)
    new_theta = particle.theta+np.random.normal(0,NOISE_THETA)
    if(is_valid(new_position[0],new_position[1],walls)):
        return Particle(new_position[0],new_position[1],new_theta,particle.weight)
    return particle
### 可以在这里写下一些你需要的变量和函数 ###


def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    all_particles: List[Particle] = []
    max_x = np.max(walls[:,0])
    min_x = np.min(walls[:,0])
    max_y = np.max(walls[:,1])
    min_y = np.min(walls[:,1])
    num=0
    while num<N:
        #all_particles.append(Particle(1.0, 1.0, 1.0, 0.0))
    ### 你的代码 ###
        x = np.random.uniform(min_x,max_x)
        y = np.random.uniform(min_y,max_y)
        if is_valid(x,y,walls):
            num+=1
            theta = np.random.uniform(-np.pi,np.pi)
            all_particles.append(Particle(x,y,theta,1.0/N))
    ### 你的代码 ###
    return all_particles


def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    weight = 1.0
    ### 你的代码 ###
    weight = np.exp(-K*np.linalg.norm((estimated-gt)))#防止weight太小
    ### 你的代码 ###
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles: List[Particle] = []
    N = len(particles)
    ### 你的代码 ###
    weights = np.array([particle.weight for particle in particles])
    sample_N = np.floor(weights*N*0.9).astype(int)
    resample_N = 0
    for i,particle in enumerate(particles):
        resample_N += sample_N[i]
        for j in range(sample_N[i]):
            resampled_particles.append(noise(particle,walls))
    uniform_N = N-resample_N
    resampled_particles += generate_uniform_particles(walls,uniform_N)
    ### 你的代码 ###
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    ### 你的代码 ###
    p.position[0] += traveled_distance*np.cos(p.theta)
    p.position[1] += traveled_distance*np.sin(p.theta)
    p.theta += dtheta
    ### 你的代码 ###
    return p

def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    final_result = Particle()
    ### 你的代码 ###
    weights = np.array([particle.weight for particle in particles])
    final_result = particles[np.argmax(weights)]
    ### 你的代码 ###
    return final_result