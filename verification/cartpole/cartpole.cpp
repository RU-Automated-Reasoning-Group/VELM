#include "../POLAR_Tool/POLAR/NeuralNetwork.h"
// #include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

double relu_my(double x)
{
	return max(x, 0.0);
}

string linear(const vector<string> &controller)
{
	return "x1 * (" + controller[0] + ") + x2 * (" + controller[1] + ") + x3 * (" + controller[2] + ") + x4 * (" + controller[3] + ") + (" + controller[4] + ")";
}

vector<double> vector_minus(const vector<double> &lhs, const vector<double> &rhs)
{
	assert(lhs.size() == rhs.size());
	vector<double> rst;
	for (int i = 0; i < lhs.size(); ++i)
	{
		rst.push_back(lhs[i] - rhs[i]);
	}
	return rst;
}

double vector_max(const vector<double> &input)
{
	assert(input.size() > 0);
	double m = input[0];
	for (int i = 1; i < input.size(); ++i)
	{
		m = max(m, input[i]);
	}
	return m;
}

double vector_min(const vector<double> &input)
{
	assert(input.size() > 0);
	double m = input[0];
	for (int i = 1; i < input.size(); ++i)
	{
		m = min(m, input[i]);
	}
	return m;
}

double reach_loss(const vector<double> &final_lbs, const vector<double> &final_ubs,
				  const vector<double> &target_set_lbs, const vector<double> &target_set_ubs)
{
	double lb_loss = relu_my(vector_max(vector_minus(target_set_lbs, final_lbs)));
	double ub_loss = relu_my(vector_max(vector_minus(final_ubs, target_set_ubs)));
	return max(lb_loss, ub_loss);
}

int main(int argc, char *argv[])
{
	// string net_name = "controller_single_pendulum_POLAR";
	string net_name = string(argv[1]);
	string benchmark_name = "cartpole";
	bool print = false;
	bool safe_sets = false;

	if (argc == 3)
	{
		print = (string(argv[2]) == "--print");
		safe_sets = (string(argv[2]) == "--safe_sets");
	}

	// Declaration of the state variables.
	unsigned int numVars = 6;

	//	intervalNumPrecision = 600;

	Variables vars;

	int x1_id = vars.declareVar("x1");
	int x2_id = vars.declareVar("x2");
	int x3_id = vars.declareVar("x3");
	int x4_id = vars.declareVar("x4");
	int t_id = vars.declareVar("t");
	int u_id = vars.declareVar("u");

	int domainDim = numVars + 1;

	// Define the continuous dynamics.
	ODE<Real> dynamics({"x2",
						" ((u + 0.05 * x4 * x4 * sin(x3)) / 1.1)  -  0.05 * ((9.8 * sin(x3) - cos(x3) *  ((u + 0.05 * x4 * x4 * sin(x3)) / 1.1)) / (0.5 * (4.0/3.0 - 0.1 * cos(x3) * cos(x3) / 1.1))) * cos(x3) / 1.1",
						"x4",
						"(9.8 * sin(x3) - cos(x3) *  ((u + 0.05 * x4 * x4 * sin(x3)) / 1.1)) / (0.5 * (4.0/3.0 - 0.1 * cos(x3) * cos(x3) / 1.1))",
						"1",
						"0"},
					   vars);

	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = 12;

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.002, order);

	// cutoff threshold
	setting.setCutoffThreshold(1e-12);

	// print out the steps
	setting.printOff();

	// remainder estimation
	Interval I(-1e-6, 1e-6);
	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);

	// setting.printOn();

	//	setting.prepare();

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */

	int steps = 200;
	Interval init_x1(0.0, 0.0), init_x2(0.0, 0.0), init_x3(0.0, 0.0), init_x4(0.0, 0.0), init_t(0), init_u(0);
	std::vector<Interval> X0;
	X0.push_back(init_x1);
	X0.push_back(init_x2);
	X0.push_back(init_x3);
	X0.push_back(init_x4);
	X0.push_back(init_t);
	X0.push_back(init_u);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

	Symbolic_Remainder symbolic_remainder(initial_set, 10000);

	// no unsafe set
	vector<Constraint> safeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// define the neural network controller
	// string nn_name = net_name;
	NeuralNetwork *nn;

	// the order in use
	// unsigned int order = 5;

	unsigned int bernstein_order = 2;
	unsigned int partition_num = 100;

	unsigned int if_symbo = 1;

	double err_max = 0;

	if (print)
	{
		if (if_symbo == 0)
		{
			cout << "High order abstraction starts." << endl;
		}
		else
		{
			cout << "High order abstraction with symbolic remainder starts." << endl;
		}
	}

	clock_t begin, end;
	begin = clock();

	double total_safe_loss = 0.0;
	double total_reach_loss = 0.0;
	int iters_run = 100;
	for (int iter = 0; iter < steps; ++iter)
	{
		if (iter % iters_run == 0)
		{
			// use the next new controller
			// cout << "reading file " << (net_name + "_" + to_string(iter / iters_run)) << endl;
			nn = new NeuralNetwork(net_name + "_" + to_string(iter / iters_run));
		}

		if (print)
		{
			cout << "Step " << iter << " starts.      " << endl;
		}
		// vector<Interval> box;
		// initial_set.intEval(box, order, setting.tm_setting.cutoff_threshold);
		TaylorModelVec<Real> tmv_input;

		tmv_input.tms.push_back(initial_set.tmvPre.tms[0]);
		tmv_input.tms.push_back(initial_set.tmvPre.tms[1]);
		tmv_input.tms.push_back(initial_set.tmvPre.tms[2]);
		tmv_input.tms.push_back(initial_set.tmvPre.tms[3]);

		// TaylorModelVec<Real> tmv_temp;
		// initial_set.compose(tmv_temp, order, cutoff_threshold);
		// tmv_input.tms.push_back(tmv_temp.tms[0]);
		// tmv_input.tms.push_back(tmv_temp.tms[1]);

		// taylor propagation
		PolarSetting polar_setting(order, bernstein_order, partition_num, "Mix", "Concrete");
		TaylorModelVec<Real> tmv_output;

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

		//		Matrix<Interval> rm1(1, 1);
		//		tmv_output.Remainder(rm1);
		//		cout << "Neural network taylor remainder: " << rm1 << endl;

		initial_set.tmvPre.tms[u_id] = tmv_output.tms[0];

		// if(if_symbo == 0){
		// 	dynamics.reach(result, setting, initial_set, unsafeSet);
		// }
		// else{
		// 	dynamics.reach_sr(result, setting, initial_set, unsafeSet, symbolic_remainder);
		// }

		// Always using symbolic remainder
		dynamics.reach(result, initial_set, 0.02, setting, safeSet, symbolic_remainder);

		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
			vector<Interval> final_box;
			result.fp_end_of_time.intEval(final_box, order, setting.tm_setting.cutoff_threshold);
			if (print || safe_sets)
			{
				for (int i = 0; i < 4; ++i)
				{
					cout << final_box[i].inf() << " ";
					cout << final_box[i].sup() << " ";
				}
				cout << "\n";
			}

			// safe loss for each timestep
			double safe_loss = final_box[0].sup();
			total_safe_loss += relu_my(safe_loss);
			if (print)
			{
				cout << "safe loss at iter " << iter << " is " << safe_loss << "\n";
			}
			//			cout << "Flowpipe taylor remainder: " << initial_set.tmv.tms[0].remainder << "     " << initial_set.tmv.tms[1].remainder << endl;
		}
		else
		{
			printf("Terminated due to too large overestimation.\n");
			return 1;
		}
	}
	cout << total_safe_loss << endl;
	// cout << total_reach_loss << endl;
	// cout << total_reach_loss + total_safe_loss << endl;

	// vector<Constraint> unsafeSet = {Constraint("-t + 0.5", vars), Constraint("-x0 + 1", vars)};

	// result.unsafetyChecking(unsafeSet, setting.tm_setting, setting.g_setting);

	// if(result.isUnsafe())
	// {
	// 	printf("The system is unsafe.\n");
	// }
	// else if(result.isSafe())
	// {
	// 	printf("The system is safe.\n");
	// }
	// else
	// {
	// 	printf("The safety is unknown.\n");
	// }

	end = clock();
	// printf("time cost: %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);

	// plot the flowpipes in the x-y plane
	if (print)
	{
		result.transformToTaylorModels(setting);

		Plot_Setting plot_setting(vars);
		plot_setting.setOutputDims("t", "x1");

		int mkres = mkdir("./outputs", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
		if (mkres < 0 && errno != EEXIST)
		{
			printf("Can not create the directory for images.\n");
			exit(1);
		}

		ofstream result_output("./outputs/" + benchmark_name + "_" + to_string(if_symbo) + ".txt");

		// you need to create a subdir named outputs
		// the file name is example.m and it is put in the subdir outputs
		plot_setting.plot_2D_interval_GNUPLOT("./outputs/", benchmark_name + "_x1", result.tmv_flowpipes, setting);
		plot_setting.setOutputDims("t", "x2");
		plot_setting.plot_2D_interval_GNUPLOT("./outputs/", benchmark_name + "_x2", result.tmv_flowpipes, setting);
		plot_setting.setOutputDims("t", "x3");
		plot_setting.plot_2D_interval_GNUPLOT("./outputs/", benchmark_name + "_x3", result.tmv_flowpipes, setting);
		plot_setting.setOutputDims("t", "x4");
		plot_setting.plot_2D_interval_GNUPLOT("./outputs/", benchmark_name + "_x4", result.tmv_flowpipes, setting);
	}

	return 0;
}
