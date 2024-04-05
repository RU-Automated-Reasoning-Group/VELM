#include "../../POLAR/NeuralNetwork.h"
#include "../../flowstar/flowstar-toolbox/Discrete.h"
#include <chrono>
#include <fstream>
#include <string>
// #include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

double my_relu(double v)
{
	return max(v, 0.0);
}

double calculate_safe_loss(const vector<vector<double>> &reached_set, const vector<vector<double>> &unsafe_set)
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

double calculate_safe_loss_road_2d(const vector<vector<double>> &reached_set)
{
	double loss = 0;
	double x_min = reached_set[0][0];
	double x_max = reached_set[0][1];
	double y_min = reached_set[1][0];
	double y_max = reached_set[1][1];
	loss = max(loss, my_relu(x_min * x_min + y_min * y_min - 100));
	loss = max(loss, my_relu(x_min * x_min + y_max * y_max - 100));
	loss = max(loss, my_relu(x_max * x_max + y_min * y_min - 100));
	loss = max(loss, my_relu(x_max * x_max + y_max * y_max - 100));
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
	string benchmark_name = "obstacle_mid";
	// Declaration of the state variables.
	unsigned int numVars = 6;

	intervalNumPrecision = 200;

	Variables vars;

	int x1_id = vars.declareVar("x1");
	int x2_id = vars.declareVar("x2");
	int x3_id = vars.declareVar("x3");
	int x4_id = vars.declareVar("x4");
	int x5_id = vars.declareVar("x5");
	int x6_id = vars.declareVar("x6");

	int domainDim = numVars + 1;

	/* Old inteface
	// Define the discrete dynamics.
	// x0 is the position of the mountain car, x1 is the speed of the mountain car.
	Expression<Interval> deriv_x0("x0 + x1", vars); // Discrete: Next_x0 = x0 + x1
	Expression<Interval> deriv_x1("x1 + 0.0015 * u - 0.0025 * cos(3 * x0)", vars); // Discrete: Next_x1 = x1 + 0.0015 * u - 0.0025 * cos(3 * x0)
	Expression<Interval> deriv_u("u", vars);

	vector<Expression<Interval> > dde_rhs(numVars);
	dde_rhs[x0_id] = deriv_x0;
	dde_rhs[x1_id] = deriv_x1;
	dde_rhs[u_id] = deriv_u;


	Nonlinear_Discrete_Dynamics dynamics(dde_rhs);
	*/
	// Define the discrete dynamics.

	ifstream dynamics_file("./results/learned_dynamics/Marvelgymnasium_road_2d-v1/model.txt");
	// // cout << "trying to load dynamics" << endl;
	vector<string> dynamics_str;
	for (int i = 0; i < 4; ++i)
	{
		string equation;
		dynamics_file >> equation;
		// 	// cout << "successfully get one line" << endl;
		dynamics_str.push_back(equation);
	}

	for (int i = 0; i < 2; ++i)
	{
		dynamics_str.push_back("0");
	}
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

	/*	// DDE does not require a remainder estimation
		Interval I(-0.01, 0.01);
		vector<Interval> remainder_estimation(numVars, I);
		setting.setRemainderEstimation(remainder_estimation);
	*/
	// setting.printOn();

	// setting.prepare();

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
	int steps = 300;
	Interval init_x1(-0.1, 0.1), init_x2(-0.1, 0.1), init_x3(-0.1, 0.1), init_x4(-0.1, 0.1);
	// Interval init_x1(0.1), init_x2(0.1), init_x3(0.1), init_x4(0.1);
	Interval init_x5(0), init_x6(0);
	std::vector<Interval> X0;
	X0.push_back(init_x1);
	X0.push_back(init_x2);
	X0.push_back(init_x3);
	X0.push_back(init_x4);
	X0.push_back(init_x5);
	X0.push_back(init_x6);

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

	unsigned int if_symbo = 1;
	;

	double err_max = 0;
	time_t start_timer;
	time_t end_timer;
	double seconds;
	double nn_total_time = 0.0, flowstar_total_time = 0.0;
	time(&start_timer);

	vector<string> state_vars;
	state_vars.push_back("x0");
	state_vars.push_back("x1");

	if (if_symbo == 0)
	{
		// cout << "High order abstraction starts." << endl;
	}
	else
	{
		// cout << "High order abstraction with symbolic remainder starts." << endl;
	}

	auto begin = std::chrono::high_resolution_clock::now();
	string controller_base = string(argv[1]); //+net_name;
	// cout << "controller base is " << controller_base;
	int interval = 1;
	NeuralNetwork *nn = nullptr;
	double safe_loss = 0.0;
	vector<vector<double>> unsafe_set = {{1.0, 2.0}, {1.0, 2.0}};
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
		tmv_input.tms.push_back(initial_set.tmvPre.tms[2]);
		tmv_input.tms.push_back(initial_set.tmvPre.tms[3]);

		// taylor propagation
		PolarSetting polar_setting(order, bernstein_order, partition_num, "Mix", "Concrete");
		polar_setting.set_num_threads(-1);
		TaylorModelVec<Real> tmv_output;

		auto inner_begin = std::chrono::high_resolution_clock::now();

		if (if_symbo == 0)
		{
			// not using symbolic remainder
			nn->get_output_tmv(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
		}
		else
		{
			// using symbolic remainder
			nn->get_output_tmv_symbolic(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
		}
		auto nn_timing = std::chrono::high_resolution_clock::now();
		auto nn_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(nn_timing - inner_begin);
		seconds = nn_elapsed.count() * 1e-9;
		// printf("nn time is %.2f s.\n", seconds);
		nn_total_time += seconds;

		// Matrix<Interval> rm1(1, 1);
		// tmv_output.Remainder(rm1);
		// cout << "Neural network taylor remainder: " << rm1 << endl;

		initial_set.tmvPre.tms[x5_id] = tmv_output.tms[0];
		initial_set.tmvPre.tms[x6_id] = tmv_output.tms[1];

		// Always using symbolic remainder
		// cout << "before reach is called" << endl;
		dynamics.reach(result, setting, initial_set, 1, safeSet, symbolic_remainder);

		auto flowstar_timing = std::chrono::high_resolution_clock::now();
		auto flowstar_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(flowstar_timing - nn_timing);
		seconds = flowstar_elapsed.count() * 1e-9;
		// printf("flowstar time is %.2f s.\n", seconds);
		flowstar_total_time += seconds;

		// dynamics.reach_sr(result, setting, initial_set, 1, symbolic_remainder, unsafeSet);

		// not using a symbolic remainder
		// dynamics.reach(result, setting, initial_set, 1, unsafeSet);

		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
			vector<Interval> inter_box;
			result.fp_end_of_time.intEval(inter_box, order, setting.tm_setting.cutoff_threshold);
			vector<vector<double>> reached_set = {{inter_box[0].inf(), inter_box[0].sup()}, {inter_box[1].inf(), inter_box[1].sup()}};
			safe_loss += calculate_safe_loss_road_2d(reached_set);
			if (plot || print_safe_sets)
			{
				for (int i = 0; i < 3; ++i)
				{
					cout << inter_box[i].inf() << " " << inter_box[i].sup() << " ";
				}
				cout << inter_box[3].inf() << " " << inter_box[3].sup() << "\n";
			}
			// cout << "Flowpipe taylor remainder: " << initial_set.tmv.tms[0].remainder << "     " << initial_set.tmv.tms[1].remainder << endl;
		}
		else
		{
			printf("Terminated due to too large overestimation.\n");
			return 1;
		}
	}
	// printf("NN time: %.2fs, Flow* time: %.2fs.\n", nn_total_time, flowstar_total_time);

	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
	seconds = elapsed.count() * 1e-9;
	// printf("Time measured: %.3f seconds.\n", seconds);

	// time(&end_timer);
	// seconds = difftime(start_timer, end_timer);

	// plot the flowpipes in the x-y plane
	if (plot)
	{
		result.transformToTaylorModels(setting);

		Plot_Setting plot_setting(vars);
		plot_setting.setOutputDims("x1", "x2");

		int mkres = mkdir("./outputs", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
		if (mkres < 0 && errno != EEXIST)
		{
			printf("Can not create the directory for images.\n");
			exit(1);
		}

		std::string running_time = "Running Time: " + to_string(seconds) + " seconds";

		// you need to create a subdir named outputs
		// the file name is example.m and it is put in the subdir outputs
		plot_setting.plot_2D_interval_GNUPLOT("./outputs/", benchmark_name + "_" + to_string(if_symbo), result.tmv_flowpipes, setting);
		// plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", benchmark_name + "_" + to_string(if_symbo), result);
	}

	cout << safe_loss << endl;
	return 0;
}
