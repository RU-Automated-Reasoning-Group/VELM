from environments.acc import ACC
from environments.car_racing import CarRacing

# gymnasium env
from environments.gymnasium_acc import Gymnasium_ACC
from environments.gymnasium_obstacle import GymnasiumObstacle
from environments.gymnasium_obstacle_mid import GymnasiumObstacleMid
from environments.gymnasium_car_racing import GymnasiumCarRacing
from environments.gymnasium_cartpole import GymnasiumCartPole
from environments.gymnasium_pendulum import GymnasiumPendulum
from environments.gymnasium_road import GymnasiumRoad
from environments.gymnasium_road_2d import GymnasiumRoad2D
from environments.gymnasium_noisy_road_2d import GymnasiumNoisyRoad2D
from environments.gymnasium_cartpole_move import GymnasiumCartPoleMove
from environments.gymnasium_cartpole_swing import GymnasiumCartPoleSwing
from environments.gymnasium_tora import Gymnasium_Tora
from environments.gymnasium_lalo import Gymnasium_Lalo

# simulated env
from environments.gymnasium_obstacle_simulate import GymnasiumObstacleSimulate
from environments.gymnasium_obstacle_mid_simulate import GymnasiumObstacleMidSimulate
from environments.gymnasium_acc_simulate import Gymnasium_ACCSimulate
from environments.gymnasium_car_racing_simulate import GymnasiumCarRacingSimulate
from environments.gymnasium_cartpole_simulate import GymnasiumCartPoleSimulate
from environments.gymnasium_pendulum_simulate import GymnasiumPendulumSimulate
from environments.gymnasium_road_simulate import GymnasiumRoadSimulate
from environments.gymnasium_road_2d_simulate import GymnasiumRoad2DSimulate
from environments.gymnasium_noisy_road_2d_simulate import GymnasiumNoisyRoad2DSimulate
from environments.gymnasium_cartpole_move_simulate import GymnasiumCartPoleMoveSimulate
from environments.gymnasium_cartpole_swing_simulate import GymnasiumCartPoleSwingSimulate
from environments.gymnasium_tora_simulate import Gymnasium_ToraSimulate
from environments.gymnasium_lalo_simulate import Gymnasium_LaloSimulate


from environments.mountain_car import MountainCar
from environments.noisy_road import NoisyRoad
from environments.noisy_road_2d import NoisyRoad2D
from environments.road_2d import Road2D


def get_env(env_name: str):
    # import pdb 
    # pdb.set_trace()
    simulated_env_info = None
    if env_name == "cartpole":
        env_info = GymnasiumCartPole()
        simulated_env_info = GymnasiumCartPoleSimulate()
    elif env_name == "obstacle":
        env_info = GymnasiumObstacle()
        simulated_env_info = GymnasiumObstacleSimulate()
    elif env_name == "obstacle_mid":
        env_info = GymnasiumObstacleMid()
        simulated_env_info = GymnasiumObstacleMidSimulate()
    elif env_name == "acc":
        env_info = Gymnasium_ACC()
        simulated_env_info = Gymnasium_ACCSimulate()
    elif env_name == "pendulum":
        env_info = GymnasiumPendulum()
        simulated_env_info = GymnasiumPendulumSimulate()
    elif env_name == "mountain_car":
        env_info = MountainCar()
    elif env_name == "road":
        env_info = GymnasiumRoad()
        simulated_env_info = GymnasiumRoadSimulate()
    elif env_name == "noisy_road":
        env_info = NoisyRoad()
    elif env_name == "noisy_road_2d":
        env_info = GymnasiumNoisyRoad2D()
        simulated_env_info = GymnasiumNoisyRoad2DSimulate()
    elif env_name == "car_racing":
        env_info = GymnasiumCarRacing()
        simulated_env_info = GymnasiumCarRacingSimulate()
    elif env_name == "road_2d":
        env_info = GymnasiumRoad2D()
        simulated_env_info = GymnasiumRoad2DSimulate()
    elif env_name == "cartpole_move":
        env_info = GymnasiumCartPoleMove()
        simulated_env_info = GymnasiumCartPoleMoveSimulate()
    elif env_name == "cartpole_swing":
        env_info = GymnasiumCartPoleSwing()
        simulated_env_info = GymnasiumCartPoleSwingSimulate()
    elif env_name == "tora":
        env_info = Gymnasium_Tora()
        simulated_env_info = Gymnasium_ToraSimulate()
    elif env_name == "lalo":
        env_info = Gymnasium_Lalo()
        simulated_env_info = Gymnasium_LaloSimulate()
    else:
        assert False

    return env_info, simulated_env_info
