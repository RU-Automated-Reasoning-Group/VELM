

from main_stable_baseline import compute_safe_agent
from utils.arguments import train_args
import utils.get_env
import pickle
from stable_baselines3 import SAC

def train(args):
    pi = "/mnt/c/Users/Yitong/Desktop/SRL_DSO/results/saved_agents/Marvelgymnasium_cartpole-v1/sac_warm_checkpoint/" + "best_model"
    # policy_model = pickle.load(open(pi, "rb"))
    policy_model = SAC.load(pi)

    # import pdb
    # pdb.set_trace()
    env_info, simulated_env_inf = utils.get_env.get_env(args.env)
    compute_safe_agent(args, env_info, learned_model=None, stds=None, neural_agent=policy_model, benchmark_config_dict=env_info.lagrange_config)

if __name__ == "__main__":
    args = train_args()
    train(args)