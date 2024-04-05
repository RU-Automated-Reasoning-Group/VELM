import numpy as np
import pandas as pd
import scipy

from dso.task import HierarchicalTask
from dso.library import Library, Polynomial
from dso.functions import create_tokens
from dso.task.regression.dataset import BenchmarkDataset
from dso.task.regression.polyfit import PolyOptimizer, make_poly_data

# from dso.task.regression.dataset import BenchmarkDataset
# from dso.task.regression.sygusdataset import Dataset
# from metal.common.utils import CEHolder


class RegressionTask(HierarchicalTask):
    """
    Class for the symbolic regression task. Discrete objects are expressions,
    which are evaluated based on their fitness to a specified dataset.
    """

    task_type = "regression"

    def __init__(self, function_set, dataset, metric="inv_nrmse",
                 metric_params=(1.0,), extra_metric_test=None,
                 extra_metric_test_params=(), reward_noise=0.0,
                 reward_noise_type="r", threshold=1e-12,
                 normalize_variance=False, protected=False,
                 decision_tree_threshold_set=None,
                 poly_optimizer_params=None):
        """
        Parameters
        ----------
        function_set : list or None
            List of allowable functions. If None, uses function_set according to
            benchmark dataset.

        dataset : dict, str, or tuple
            If dict: .dataset.BenchmarkDataset kwargs.
            If str ending with .csv: filename of dataset.
            If other str: name of benchmark dataset.
            If tuple: (X, y) data

        metric : str
            Name of reward function metric to use.

        metric_params : list
            List of metric-specific parameters.

        extra_metric_test : str
            Name of extra function metric to use for testing.

        extra_metric_test_params : list
            List of metric-specific parameters for extra test metric.

        reward_noise : float
            Noise level to use when computing reward.

        reward_noise_type : "y_hat" or "r"
            "y_hat" : N(0, reward_noise * y_rms_train) is added to y_hat values.
            "r" : N(0, reward_noise) is added to r.

        threshold : float
            Threshold of NMSE on noiseless data used to determine success.

        normalize_variance : bool
            If True and reward_noise_type=="r", reward is multiplied by
            1 / sqrt(1 + 12*reward_noise**2) (We assume r is U[0,1]).

        protected : bool
            Whether to use protected functions.

        decision_tree_threshold_set : list
            A set of constants {tj} for constructing nodes (xi < tj) in decision trees.

        poly_optimizer_params : dict
            Parameters for PolyOptimizer if poly token is in the library.
        """

        super(HierarchicalTask).__init__()

        """
        Configure (X, y) train/test data. There are four supported use cases:
        (1) named benchmark, (2) benchmark config, (3) filename, and (4) direct
        (X, y) data.
        """
        self.X_test = self.y_test = self.y_test_noiseless = None

        # Case 1: Named benchmark dataset (shortcut for Case 2)
        if isinstance(dataset, str) and not dataset.endswith(".csv") and not dataset.endswith(".sl"):
            dataset = {"name" : dataset}

        # Case 2: Benchmark dataset config
        if isinstance(dataset, dict):
            benchmark = BenchmarkDataset(**dataset)
            self.X_train = benchmark.X_train
            self.y_train = benchmark.y_train
            self.X_test = benchmark.X_test
            self.y_test = benchmark.y_test
            self.y_test_noiseless = benchmark.y_test_noiseless
            self.name = benchmark.name

            # For benchmarks, always use the benchmark function_set.
            # Issue a warning if the user tried to supply a different one.
            if function_set is not None and function_set != benchmark.function_set:
                print("WARNING: function_set provided when running benchmark "
                      "problem. The provided function_set will be ignored; the "
                      "benchmark function_set will be used instead.\nProvided "
                      "function_set:\n  {}\nBenchmark function_set:\n  {}."
                      .format(function_set, benchmark.function_set))
            function_set = benchmark.function_set

        # Case 3: Dataset filename
        elif isinstance(dataset, str) and dataset.endswith("csv"):
            df = pd.read_csv(dataset, header=None) # Assuming data file does not have header rows
            self.X_train = df.values[:, :-1]
            self.y_train = df.values[:, -1]
            self.name = dataset.replace("/", "_")[:-4]

        # Case 4: sklearn-like (X, y) data
        elif isinstance(dataset, tuple):
            self.X_train = dataset[0]
            self.y_train = dataset[1]
            self.name = "regression"

        # Case 5: sygus data
        # ./benchmarks -file_list all -single_sample CrCy_10-sbox2-D5-sIn30.sl
        elif isinstance(dataset, str) and dataset.endswith(".sl"):
            # Load SYGUS
            dataset = Dataset(data_root="dso/task/regression/sygus_benchmarks/CrCi", single_sample=dataset)
            specsample_ls = dataset.sample_minibatch(1, replacement=True)
            print (f'spec: {specsample_ls[0].spectree.spec}')
            print (f'grammar: {specsample_ls[0].spectree.grammar}')
            print (f'vars: {specsample_ls[0].spectree.vars}')

            print (f'node_seq: {specsample_ls[0].spectree.node_seq}')
            print (f'node_type_seq: {specsample_ls[0].spectree.node_type_seq}')
            print (f'numOf_nodes: {specsample_ls[0].spectree.numOf_nodes}')
            print (f'nodename2ind: {specsample_ls[0].spectree.nodename2ind}')

            print (f'all_tests: {specsample_ls[0].spectree.all_tests}')

            # Context free grammar for synthesis
            cfg = specsample_ls[0].spectree.grammar
            root_symbol = cfg.start
            print (f'Grammar root: {root_symbol}')
            # IO examples holder.
            g = specsample_ls[0]
            holder = CEHolder(g)

            # Variables of the to-synthesize program.
            vars = specsample_ls[0].spectree.vars
            var_ids = {}
            for id, var in enumerate(vars):
                var_ids[var] = id


            max_depth = len(specsample_ls[0].spectree.grammar.productions)
            input_size = len(specsample_ls[0].spectree.vars)
            output_size = 1
            num_labels = 1
            input_type = output_type = "atom"


            examples = holder.all_ces
            # train_data must be sorted by var_ids
            train_data, train_labels = [], []
            for ex in examples:
                item = [None] * len(vars)
                for k in ex.config:
                    item[var_ids[k]] = 1. if ex.config[k] else 0.
                train_data.append(item)
                if ex.kind == 'T':
                    train_labels.append(1.)
                else:
                    train_labels.append(0.)
            train_data = np.array(train_data)
            train_labels = np.array(train_labels)
            # print (f"train_data = {train_data} and type ={type(train_data)}")
            # print (f"train_labels = {train_labels} and type = {type(train_labels)}")
            # assert False

            self.X_train = train_data
            self.y_train = train_labels
            self.name = "regression"

        # print (f"train_data = {self.X_train} and type = {type(self.X_train)}")
        # print (f"train_labels = {self.y_train} and type = {type(self.y_train)}")
        # assert False

        # If not specified, set test data equal to the training data
        if self.X_test is None:
            self.X_test = self.X_train
            self.y_test = self.y_train
            self.y_test_noiseless = self.y_test

        # Save time by only computing data variances once
        self.var_y_test = np.var(self.y_test)
        self.var_y_test_noiseless = np.var(self.y_test_noiseless)

        """
        Configure train/test reward metrics.
        """
        self.threshold = threshold
        self.metric, self.invalid_reward, self.max_reward = make_regression_metric(metric, self.y_train, *metric_params)
        self.extra_metric_test = extra_metric_test
        if extra_metric_test is not None:
            self.metric_test, _, _ = make_regression_metric(extra_metric_test, self.y_test, *extra_metric_test_params)
        else:
            self.metric_test = None

        """
        Configure reward noise.
        """
        self.reward_noise = reward_noise
        self.reward_noise_type = reward_noise_type
        self.normalize_variance = normalize_variance
        assert reward_noise >= 0.0, "Reward noise must be non-negative."
        if reward_noise > 0:
            assert reward_noise_type in ["y_hat", "r"], "Reward noise type not recognized."
            self.rng = np.random.RandomState(0)
            y_rms_train = np.sqrt(np.mean(self.y_train ** 2))
            if reward_noise_type == "y_hat":
                self.scale = reward_noise * y_rms_train
            elif reward_noise_type == "r":
                self.scale = reward_noise
        else:
            self.rng = None
            self.scale = None

        # Set the Library
        tokens = create_tokens(n_input_var=self.X_train.shape[1],
                               function_set=function_set,
                               protected=protected,
                               decision_tree_threshold_set=decision_tree_threshold_set)
        self.library = Library(tokens)

        # Set stochastic flag
        self.stochastic = reward_noise > 0.0

        # Set neg_nrmse as the metric for const optimization
        self.const_opt_metric, _, _ = make_regression_metric("neg_nrmse", self.y_train)

        # Function to optimize polynomial tokens
        if "poly" in self.library.names:
            if poly_optimizer_params is None:
                poly_optimizer_params = {
                        "degree": 3,
                        "coef_tol": 1e-6,
                        "regressor": "dso_least_squares",
                        "regressor_params": {}
                    }

            self.poly_optimizer = PolyOptimizer(**poly_optimizer_params)

    def reward_function(self, p, optimizing=False):
        # fit a polynomial if p contains a 'poly' token
        if p.poly_pos is not None:
            assert len(p.const_pos) == 0, "A program cannot contain 'poly' and 'const' tokens at the same time"
            poly_data_y = make_poly_data(p.traversal, self.X_train, self.y_train)
            if poly_data_y is None:  # invalid function evaluations (nan or inf) appeared in make_poly_data
                p.traversal[p.poly_pos] = Polynomial([(0,)*self.X_train.shape[1]], np.ones(1))
            else:
                p.traversal[p.poly_pos] = self.poly_optimizer.fit(self.X_train, poly_data_y)

        # Compute estimated values
        y_hat = p.execute(self.X_train)

        # For invalid expressions, return invalid_reward
        if p.invalid:
            return -1.0 if optimizing else self.invalid_reward

        # Observation noise
        # For reward_noise_type == "y_hat", success must always be checked to
        # ensure success cases aren't overlooked due to noise. If successful,
        # return max_reward.
        if self.reward_noise and self.reward_noise_type == "y_hat":
            if p.evaluate.get("success"):
                return self.max_reward
            y_hat += self.rng.normal(loc=0, scale=self.scale, size=y_hat.shape)

        # Compute and return neg_nrmse for constant optimization
        if optimizing:
            return self.const_opt_metric(self.y_train, y_hat)

        # Compute metric
        r = self.metric(self.y_train, y_hat)

        # Direct reward noise
        # For reward_noise_type == "r", success can for ~max_reward metrics be
        # confirmed before adding noise. If successful, must return np.inf to
        # avoid overlooking success cases.
        if self.reward_noise and self.reward_noise_type == "r":
            if r >= self.max_reward - 1e-5 and p.evaluate.get("success"):
                return np.inf
            r += self.rng.normal(loc=0, scale=self.scale)
            if self.normalize_variance:
                r /= np.sqrt(1 + 12 * self.scale ** 2)

        return r

    def evaluate(self, p):

        # Compute predictions on test data
        y_hat = p.execute(self.X_test)
        if p.invalid:
            nmse_test = None
            nmse_test_noiseless = None
            success = False

        else:
            # NMSE on test data (used to report final error)
            nmse_test = np.mean((self.y_test - y_hat) ** 2) / self.var_y_test

            # NMSE on noiseless test data (used to determine recovery)
            nmse_test_noiseless = np.mean((self.y_test_noiseless - y_hat) ** 2) / self.var_y_test_noiseless

            # Success is defined by NMSE on noiseless test data below a threshold
            success = nmse_test_noiseless < self.threshold

        info = {
            "nmse_test" : nmse_test,
            "nmse_test_noiseless" : nmse_test_noiseless,
            "success" : success
        }

        if self.metric_test is not None:
            if p.invalid:
                m_test = None
                m_test_noiseless = None
            else:
                m_test = self.metric_test(self.y_test, y_hat)
                m_test_noiseless = self.metric_test(self.y_test_noiseless, y_hat)

            info.update({
                self.extra_metric_test : m_test,
                self.extra_metric_test + '_noiseless' : m_test_noiseless
            })

        return info

def inv_nmse_noise_maker(var_y, *args):
    def inv_nmse_noise(y, y_hat):
        diff = y - y_hat
        _, std = scipy.stats.norm.fit(diff, floc=0.0)
        return 1/(1 + args[0]*np.mean((y - y_hat)**2)/var_y + args[1] * std)
    return inv_nmse_noise

def inv_mse_noise_maker(*args):
    def inv_mse_noise(y, y_hat):
        diff = y - y_hat
        _, std = scipy.stats.norm.fit(diff, floc=0.0)
        # print(f"fist part {np.mean((y - y_hat)**2)}, second_part {std}")
        return 1/(1 + args[0]*np.mean((y - y_hat)**2) + args[1] * std)
        # return 1/(1 + args[1] * std)
    return inv_mse_noise


def make_regression_metric(name, y_train, *args):
    """
    Factory function for a regression metric. This includes a closures for
    metric parameters and the variance of the training data.

    Parameters
    ----------

    name : str
        Name of metric. See all_metrics for supported metrics.

    args : args
        Metric-specific parameters

    Returns
    -------

    metric : function
        Regression metric mapping true and estimated values to a scalar.

    invalid_reward: float or None
        Reward value to use for invalid expression. If None, the training
        algorithm must handle it, e.g. by rejecting the sample.

    max_reward: float
        Maximum possible reward under this metric.
    """

    var_y = np.var(y_train)

    all_metrics = {

        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        "neg_mse" :     (lambda y, y_hat : -np.mean((y - y_hat)**2),
                        0),

        # Negative root mean squared error
        # Range: [-inf, 0]
        # Value = -sqrt(var(y)) when y_hat == mean(y)
        "neg_rmse" :     (lambda y, y_hat : -np.sqrt(np.mean((y - y_hat)**2)),
                        0),

        # Negative normalized mean squared error
        # Range: [-inf, 0]
        # Value = -1 when y_hat == mean(y)
        "neg_nmse" :    (lambda y, y_hat : -np.mean((y - y_hat)**2)/var_y,
                        0),

        # Negative normalized root mean squared error
        # Range: [-inf, 0]
        # Value = -1 when y_hat == mean(y)
        "neg_nrmse" :   (lambda y, y_hat : -np.sqrt(np.mean((y - y_hat)**2)/var_y),
                        0),

        # (Protected) negative log mean squared error
        # Range: [-inf, 0]
        # Value = -log(1 + var(y)) when y_hat == mean(y)
        "neglog_mse" : (lambda y, y_hat : -np.log(1 + np.mean((y - y_hat)**2)),
                        0),

        # (Protected) inverse mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]*var(y)) when y_hat == mean(y)
        "inv_mse" : (lambda y, y_hat : 1/(1 + args[0]*np.mean((y - y_hat)**2)),
                        1),

        # (Protected) inverse normalized mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]) when y_hat == mean(y)
        "inv_nmse" :    (lambda y, y_hat : 1/(1 + args[0]*np.mean((y - y_hat)**2)/var_y),
                        1),

        # (Protected) inverse mean squared error with noise std
        # Range: [0, 1]
        # Value = 1/(1 + args[0]*var(y)) when y_hat == mean(y)
        "inv_mse_noise" : (inv_mse_noise_maker(*args),
                        2),

        # (Protected) inverse normalized mean squared error with noise std
        # Range: [0, 1]
        # Value = 1/(1 + args[0]) when y_hat == mean(y)
        "inv_nmse_noise" :    (inv_nmse_noise_maker(var_y, *args),
                        2),

        # (Protected) inverse normalized root mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]) when y_hat == mean(y)
        "inv_nrmse" :    (lambda y, y_hat : 1/(1 + args[0]*np.sqrt(np.mean((y - y_hat)**2)/var_y)),
                        1),

        # Fraction of predicted points within p0*abs(y) + p1 band of the true value
        # Range: [0, 1]
        "fraction" :    (lambda y, y_hat : np.mean(abs(y - y_hat) < args[0]*abs(y) + args[1]),
                        2),

        # Pearson correlation coefficient
        # Range: [0, 1]
        "pearson" :     (lambda y, y_hat : scipy.stats.pearsonr(y, y_hat)[0],
                        0),

        # Spearman correlation coefficient
        # Range: [0, 1]
        "spearman" :    (lambda y, y_hat : scipy.stats.spearmanr(y, y_hat)[0],
                        0)
    }

    assert name in all_metrics, "Unrecognized reward function name."
    assert len(args) == all_metrics[name][1], "For {}, expected {} reward function parameters; received {}.".format(name,all_metrics[name][1], len(args))
    metric = all_metrics[name][0]

    # For negative MSE-based rewards, invalid reward is the value of the reward function when y_hat = mean(y)
    # For inverse MSE-based rewards, invalid reward is 0.0
    # For non-MSE-based rewards, invalid reward is the minimum value of the reward function's range
    all_invalid_rewards = {
        "neg_mse" : -var_y,
        "neg_rmse" : -np.sqrt(var_y),
        "neg_nmse" : -1.0,
        "neg_nrmse" : -1.0,
        "neglog_mse" : -np.log(1 + var_y),
        "inv_mse" : 0.0, #1/(1 + args[0]*var_y),
        "inv_nmse" : 0.0, #1/(1 + args[0]),
        "inv_nrmse" : 0.0, #1/(1 + args[0]),
        "fraction" : 0.0,
        "pearson" : 0.0,
        "spearman" : 0.0,
        "inv_mse_noise": 0.0,
        "inv_nmse_noise": 0.0,
    }
    invalid_reward = all_invalid_rewards[name]

    all_max_rewards = {
        "neg_mse" : 0.0,
        "neg_rmse" : 0.0,
        "neg_nmse" : 0.0,
        "neg_nrmse" : 0.0,
        "neglog_mse" : 0.0,
        "inv_mse" : 1.0,
        "inv_nmse" : 1.0,
        "inv_nrmse" : 1.0,
        "fraction" : 1.0,
        "pearson" : 1.0,
        "spearman" : 1.0,
        "inv_mse_noise": 1.0,
        "inv_nmse_noise": 1.0,
    }
    max_reward = all_max_rewards[name]

    return metric, invalid_reward, max_reward
