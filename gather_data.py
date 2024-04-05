import gymnasium
import numpy as np

from pyoperon.sklearn import SymbolicRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sympy import parse_expr
from environments.gymnasium_cartpole import GymnasiumCartPole
a = GymnasiumCartPole()

env = gymnasium.make(a.env_name)

buffer = []

# env = gymnasium.make("CartPole-v1")
for i in range(0, 100):
    state, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, rwd, terminated, truncated, _ = env.step(action)
        buffer.append((state, action, next_state))
        done = terminated or truncated
        state = next_state

import pdb
pdb.set_trace()

x_list, a_list, y_list = zip(*buffer)

x_list = np.array(x_list)
a_list = np.array(a_list)
y_list = np.array(y_list)

# a_list = np.expand_dims(a_list, axis=1)

features = np.concatenate((x_list, a_list), axis=1)

pdb.set_trace()

reg = SymbolicRegressor(
        allowed_symbols='add,sub,mul,div,constant,variable,sin,cos',
        offspring_generator='basic',
        local_iterations=5,
        max_length=50,
        initialization_method='btc',
        n_threads=10,
        objectives = ['mse'],
        symbolic_mode=False,
        model_selection_criterion='mean_squared_error',
        random_state=4,
        generations=10000,
        population_size=5000
        )


for i in range(0, 4):
    X = features
    Y = y_list[:,i]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.04, test_size=0.5, shuffle=True)

    print(X_train.shape, y_train.shape)

    reg.fit(X_train, y_train)
    print(parse_expr(reg.get_model_string(reg.model_, 3)))
    # print(reg.get_model_string(reg.model_, names=['A', 'B', 'C', 'D' ], precision=2))
    print(reg.stats_)

    y_pred_train = reg.predict(X_train)
    print('r2 train (sklearn.r2_score): ', mean_squared_error(y_train, y_pred_train))

    y_pred_test = reg.predict(X_test)
    print('r2 test (sklearn.r2_score): ', mean_squared_error(y_test, y_pred_test))