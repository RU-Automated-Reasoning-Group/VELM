import gym
import torch

# from environments.pendulum import PendulumSampler

# from environments.neural_lander import NeuLanderSampler


# gym environment (only pendulum for now)
class GymENV(gym.Wrapper):
    def __init__(self, args):
        self.env_name = args
        gym.Wrapper.__init__(self, gym.make(self.env_name))
        # init sampler
        # if "Pendulum" in self.env_name:
            # self.sampler = PendulumSampler()
        # elif "Lander" in self.env_name:
            # self.sampler = NeuLanderSampler()

    def reset(self):
        state = self.env.reset()
        # theta = np.arctan2(state[1], state[0])
        # theta_dot = state[2]

        # return [theta, theta_dot]
        return state

    def step(self, u):
        state, r, done, other = self.env.step(u)
        return state, r, done, other
        # theta = np.arctan2(state[1], state[0])
        # theta_dot = state[2]

        # return [theta, theta_dot], r, done, other

    def sample_state(self, sample_num):
        print("enter here")
        return self.sampler.sample_state(sample_num)

    # TODO
    def test_safe(self, state):
        return self.sampler.check_safe(state)

    def ground_truth_dynamic(self, state):
        return self.sampler.get_gt_dynamic(state)

    def ground_truth_approximate_step(self, state, u):
        # init
        x_next = self.sampler.get_gt_appro_step(state, u)

        return x_next


# custom environment (without gym)
class CustomENV:
    def __init__(self, args):
        self.env_name = args.env
        # init sampler
        if "Pendulum" in self.env_name:
            self.sampler = PendulumSampler()
        elif "lander" in self.env_name:
            self.sampler = NeuLanderSampler(dt=args.cbf_timestep)
            self.act_dim = self.sampler.n_control
            self.state_dim = self.sampler.n_dims

    def reset(self):
        self.state = self.sampler.sample_single_state()
        return self.state

    def step(self, u):
        next_state = self.sampler.get_gt_appro_step(
            torch.tensor(self.state).unsqueeze(0),
            torch.tensor(u).unsqueeze(0).unsqueeze(-1),
        )
        next_state = next_state.tolist()[0]
        self.state = next_state
        return next_state, 0, False, None

    def sample_state(self, sample_num):
        return self.sampler.sample_state(sample_num)

    # TODO
    def test_safe(self, state):
        return self.sampler.check_safe(state)

    def ground_truth_dynamic(self, state):
        return self.sampler.get_gt_dynamic(state)

    def ground_truth_approximate_step(self, state, u):
        # init
        x_next = self.sampler.get_gt_appro_step(state, u)

        return x_next
