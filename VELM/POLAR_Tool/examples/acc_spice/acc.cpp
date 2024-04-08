#include "../../POLAR/NeuralNetwork.h"
#include "../../flowstar/flowstar-toolbox/Discrete.h"
#include <chrono>
#include <fstream>
#include <string>
// #include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

double my_relu(double v) { return max(v, 0.0); }

double calculate_safe_loss(const vector<vector<double>> &reached_set,
                           const vector<vector<double>> &unsafe_set)
{
  double loss = 0.0;
  bool initialized = false;
  for (int i = 0; i < reached_set.size(); ++i)
  {
    if (!initialized)
    {
      initialized = true;
      loss = my_relu(unsafe_set[i][1] - reached_set[i][0]);
      loss = min(loss, my_relu(reached_set[i][1] - unsafe_set[i][0]));
    }
    else
    {
      loss = min(loss, my_relu(unsafe_set[i][1] - reached_set[i][0]));
      loss = min(loss, my_relu(reached_set[i][1] - unsafe_set[i][0]));
    }
  }
  return loss;
}

int main(int argc, char *argv[])
{
  bool plot = false;
  bool print_safe_sets = false;
  if (argc >= 3)
  {
    if (string(argv[2]) == "--plot")
    {
      plot = true;
    }
    else if (string(argv[2]) == "--safe_sets")
    {
      print_safe_sets = true;
    }
  }
  string benchmark_name = "acc";
  // Declaration of the state variables.
  constexpr unsigned int state_vars = 2;
  constexpr unsigned int action_vars = 1;
  constexpr unsigned int random_vars = 2;

  constexpr unsigned int numVars = state_vars + action_vars + random_vars;

  intervalNumPrecision = 200;

  Variables vars;

  int x1_id = vars.declareVar("x1");
  int x2_id = vars.declareVar("x2");
  int x3_id = vars.declareVar("x3");
  int x4_id = vars.declareVar("x4");
  int x5_id = vars.declareVar("x5");

  int domainDim = numVars + 1;

  // Define the discrete dynamics.
  ifstream dynamics_file(
      "./results/learned_dynamics/Marvelgymnasium_acc-v1/model.txt");
  // cout << "trying to load dynamics" << endl;
  vector<string> dynamics_str;
  for (int i = 0; i < state_vars; ++i)
  {
    string equation;
    dynamics_file >> equation;
    equation.append("+x" + to_string(i + 1 + state_vars + action_vars));
    // cout << "successfully get one line" << endl;
    // cout << equation << endl;
    dynamics_str.push_back(equation);
  }

  ifstream stds_file(
      "./results/learned_dynamics/Marvelgymnasium_acc-v1/std.txt");
  vector<double> stds;
  for (int i = 0; i < state_vars; ++i)
  {
    string std;
    stds_file >> std;
    // cout << "successfully get one line" << endl;
    // cout << std << endl;
    stds.push_back(stod(std));
  }

  // vector<string> dynamics_str{"x1+0.02*x2+x4", "x2+0.02*x3+x5", "0"};
  DDE<Real> dynamics(dynamics_str, vars);

  // Specify the parameters for reachability computation.
  Computational_Setting setting(vars);
  // Computational_Setting setting;

  unsigned int order = 4;

  // stepsize and order for reachability analysis
  setting.setFixedStepsize(0.01, order); // the stepsize will be ignored

  // cutoff threshold
  setting.setCutoffThreshold(1e-8);

  // print out the steps
  setting.printOff();

  /*
   * Initial set can be a box which is represented by a vector of intervals.
   * The i-th component denotes the initial set of the i-th state variable.
   */
  int steps = 300;
  Interval init_x1(-1.1, -0.9), init_x2(-0.1, 0.1);
  Interval init_x3(0);
  Interval init_x4(-3 * stds[0], 3 * stds[0]);
  Interval init_x5(-3 * stds[1], 3 * stds[1]);

  std::vector<Interval> X0;
  X0.push_back(init_x1);
  X0.push_back(init_x2);
  X0.push_back(init_x3);
  X0.push_back(init_x4);
  X0.push_back(init_x5);

  // translate the initial set to a flowpipe
  Flowpipe initial_set(X0);

  Symbolic_Remainder symbolic_remainder(initial_set, 100);

  // no unsafe set
  vector<Constraint> safeSet;
  // vector<Constraint> unsafeSet;

  // result of the reachability computation
  Result_of_Reachability result;

  // the order in use
  // unsigned int order = 5;
  Interval cutoff_threshold(-1e-12, 1e-12);
  unsigned int bernstein_order = 2;
  // unsigned int partition_num = 4000;
  unsigned int partition_num = 200;

  double err_max = 0;

  string controller_base = string(argv[1]); //+net_name;
  // cout << "controller base is " << controller_base;
  int interval = 1;
  NeuralNetwork *nn = nullptr;
  double safe_loss = 0.0;
  for (int iter = 0; iter < steps; ++iter)
  {
    // cout << "Step " << iter << " starts.      " << endl;
    // vector<Interval> box;
    // initial_set.intEval(box, order, setting.tm_setting.cutoff_threshold);
    // define the neural network controller
    if (iter % interval == 0)
    {
      nn = new NeuralNetwork(controller_base + "_" + to_string(iter));
    }

    TaylorModelVec<Real> tmv_input;

    tmv_input.tms.push_back(initial_set.tmvPre.tms[0]);
    tmv_input.tms.push_back(initial_set.tmvPre.tms[1]);

    // taylor propagation
    PolarSetting polar_setting(order, bernstein_order, partition_num, "Mix",
                               "Concrete");
    polar_setting.set_num_threads(-1);
    TaylorModelVec<Real> tmv_output;

    auto inner_begin = std::chrono::high_resolution_clock::now();

    // using symbolic remainder
    nn->get_output_tmv_symbolic(tmv_output, tmv_input, initial_set.domain,
                                polar_setting, setting);

    // Matrix<Interval> rm1(1, 1);
    // tmv_output.Remainder(rm1);
    // cout << "Neural network taylor remainder: " << rm1 << endl;

    initial_set.tmvPre.tms[x3_id] = tmv_output.tms[0];

    // Always using symbolic remainder
    // cout << "before reach is called" << endl;
    dynamics.reach(result, setting, initial_set, 1, safeSet,
                   symbolic_remainder);

    if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE ||
        result.status == COMPLETED_UNKNOWN)
    {
      initial_set = result.fp_end_of_time;
      // for (int i = 0; i < 4; ++i) {
      // initial_set.tmvPre.tms[i].remainder =
      // initial_set.tmvPre.tms[i].remainder + Interval(-3.0 * stds[i], 3.0 *
      // stds[i]);
      // }
      vector<Interval> inter_box;
      result.fp_end_of_time.intEval(inter_box, order,
                                    setting.tm_setting.cutoff_threshold);
      vector<vector<double>> reached_set = {
          {inter_box[0].inf(), inter_box[0].sup()},
          {inter_box[1].inf(), inter_box[1].sup()}};

      safe_loss = safe_loss + my_relu(reached_set[0][1]);
      // safe_loss += calculate_safe_loss(reached_set, unsafe_set);
      if (plot || print_safe_sets)
      {
        cout << inter_box[0].inf() << " " << inter_box[0].sup() << " "
             << inter_box[1].inf() << " " << inter_box[1].sup() << "\n";
      }
      // cout << "Flowpipe taylor remainder: " <<
      // initial_set.tmv.tms[0].remainder << "     " <<
      // initial_set.tmv.tms[1].remainder << endl;
    }
    else
    {
      printf("Terminated due to too large overestimation.\n");
      return 1;
    }
  }

  // plot the flowpipes in the x-y plane
  if (plot)
  {
    result.transformToTaylorModels(setting);

    Plot_Setting plot_setting(vars);
    plot_setting.setOutputDims("x1", "x2");

    int mkres =
        mkdir("./outputs", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    if (mkres < 0 && errno != EEXIST)
    {
      printf("Can not create the directory for images.\n");
      exit(1);
    }

    // you need to create a subdir named outputs
    // the file name is example.m and it is put in the subdir outputs
    plot_setting.plot_2D_interval_GNUPLOT("./outputs/", benchmark_name,
                                          result.tmv_flowpipes, setting);
    // plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", benchmark_name + "_" +
    // to_string(if_symbo), result);
  }

  cout << safe_loss << endl;
  return 0;
}
