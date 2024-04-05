#include "../../POLAR/NeuralNetwork.h"
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

double relu_my(double x) {
	return max(x, 0.0);
}

string linear(const vector<string> &controller) {
	return "x1 * (" + controller[0] + ") + x2 * (" + controller[1] + ") + x3 * (" + controller[2] + ") + x4 * (" + controller[3] + ") + (" + controller[4] + ")";
}

vector<double> vector_minus(const vector<double> &lhs, const vector<double> &rhs) {
    assert(lhs.size() == rhs.size());
    vector<double> rst;
    for (int i = 0; i < lhs.size(); ++i) {
        rst.push_back(lhs[i] - rhs[i]);
    }
    return rst;
}

double vector_max(const vector<double> &input) {
    assert(input.size() > 0);
    double m = input[0];
    for (int i = 1; i < input.size(); ++i) {
        m = max(m, input[i]);
    }
    return m;
}

double vector_min(const vector<double> &input) {
    assert(input.size() > 0);
    double m = input[0];
    for (int i = 1; i < input.size(); ++i) {
        m = min(m, input[i]);
    }
    return m;
}

double reach_loss(const vector<double> &final_lbs, const vector<double> &final_ubs,
                  const vector<double> &reach_set_lbs, const vector<double> &reach_set_ubs) {
    double lb_loss = relu_my(vector_max(vector_minus(final_lbs, reach_set_lbs)));
    double ub_loss = relu_my(vector_max(vector_minus(reach_set_ubs, final_ubs)));
    return max(lb_loss, ub_loss);
}

void print(const vector<double> &v) {
    for (double x : v) {
        cout << x << " ";
    }
    cout << "\n";
}

int main(int argc, char *argv[])
{
	int steps = atoi(argv[6]);
	int starting_step = atoi(argv[7]);
	string net_name = string(argv[1]);
	string benchmark_name = "Single_Pendulum";
	// Declaration of the state variables.
	unsigned int numVars = 4;

//	intervalNumPrecision = 600;

	Variables vars;

	int x0_id = vars.declareVar("x0");
	int x1_id = vars.declareVar("x1");
    int t_id = vars.declareVar("t");
	int u_id = vars.declareVar("u0");

	int domainDim = numVars + 1;

	// Define the continuous dynamics.
    ODE<Real> dynamics({"x1","2*sin(x0)+8*u0","1","0"}, vars);

	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = 4;

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.025, order);


	// cutoff threshold
	setting.setCutoffThreshold(1e-6);

	// print out the steps
	setting.printOff();

	// remainder estimation
	Interval I(-0.01, 0.01);
	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);

	//setting.printOn();

//	setting.prepare();

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */

	// int steps = 20;
	Interval init_x0(atof(argv[2]), atof(argv[3])), init_x1(atof(argv[4]), atof(argv[5])), init_t(0), init_u(0);
	std::vector<Interval> X0;
	X0.push_back(init_x0);
	X0.push_back(init_x1);
	X0.push_back(init_t);
	X0.push_back(init_u);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

	Symbolic_Remainder symbolic_remainder(initial_set, 100);

	// no unsafe set
	vector<Constraint> safeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// define the neural network controller
	string nn_name = net_name;
	NeuralNetwork nn(nn_name);

	// the order in use
	// unsigned int order = 5;

	unsigned int bernstein_order = 2;
	unsigned int partition_num = 100;

	unsigned int if_symbo = 1;

	double err_max = 0;


	// if (if_symbo == 0)
	// {
	// 	cout << "High order abstraction starts." << endl;
	// }
	// else
	// {
	// 	cout << "High order abstraction with symbolic remainder starts." << endl;
	// }

	clock_t begin, end;
	begin = clock();

	double loss = 0;
	for (int iter = 0; iter < steps; ++iter)
	{
		// cout << "Step " << iter << " starts.      " << endl;
		//vector<Interval> box;
		//initial_set.intEval(box, order, setting.tm_setting.cutoff_threshold);
		TaylorModelVec<Real> tmv_input;

		tmv_input.tms.push_back(initial_set.tmvPre.tms[0]);
		tmv_input.tms.push_back(initial_set.tmvPre.tms[1]);

		// TaylorModelVec<Real> tmv_temp;
		// initial_set.compose(tmv_temp, order, cutoff_threshold);
		// tmv_input.tms.push_back(tmv_temp.tms[0]);
		// tmv_input.tms.push_back(tmv_temp.tms[1]);


		// taylor propagation
        PolarSetting polar_setting(order, bernstein_order, partition_num, "Mix", "Concrete");
		TaylorModelVec<Real> tmv_output;

		if(if_symbo == 0){
			// not using symbolic remainder
			nn.get_output_tmv(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
		}
		else{
			// using symbolic remainder
			nn.get_output_tmv_symbolic(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
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
		dynamics.reach(result, initial_set, 0.05, setting, safeSet, symbolic_remainder);


		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
			// vector<Interval> inter_box;
			// result.fp_end_of_time.intEval(inter_box, order, setting.tm_setting.cutoff_threshold);
			// if (starting_step + iter >= 9) {
			// 	loss += (max(my_relu(0.0 - inter_box[0].inf()), my_relu(inter_box[0].sup() - 1.0)));
			// }

//			cout << "Flowpipe taylor remainder: " << initial_set.tmv.tms[0].remainder << "     " << initial_set.tmv.tms[1].remainder << endl;
		}
		else
		{
			printf("Terminated due to too large overestimation.\n");
			return 1;
		}
	}
	vector<Interval> final_box;
	result.fp_end_of_time.intEval(final_box, order, setting.tm_setting.cutoff_threshold);
	cout << final_box[0].inf() << "\n";
	cout << final_box[0].sup() << "\n";
	cout << final_box[1].inf() << "\n";
	cout << final_box[1].sup() << "\n";
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
	if (starting_step == 98) {
		// this is the last controller
		vector<double> final_lbs = {final_box[0].inf(), final_box[1].inf()};
		vector<double> final_ubs = {final_box[0].sup(), final_box[1].sup()};
		vector<double> reach_set_lbs = {-0.1, -0.1};
		vector<double> reach_set_ubs = {0.1, 0.1};
		loss = reach_loss(final_lbs, final_ubs, reach_set_lbs, reach_set_ubs);
	}
	cout << loss << endl;

	if (argc == 9 && string(argv[8]) == "print") {

		// plot the flowpipes in the x-y plane
		result.transformToTaylorModels(setting);

		Plot_Setting plot_setting(vars);
		plot_setting.setOutputDims("t", "x0");

		int mkres = mkdir("./outputs", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
		if (mkres < 0 && errno != EEXIST)
		{
			printf("Can not create the directory for images.\n");
			exit(1);
		}


		ofstream result_output("./outputs/" + benchmark_name + "_" + to_string(if_symbo) + ".txt");

		// you need to create a subdir named outputs
		// the file name is example.m and it is put in the subdir outputs
		plot_setting.plot_2D_interval_GNUPLOT("./outputs/", benchmark_name + "_x0_linear" + to_string(if_symbo), result.tmv_flowpipes, setting);
		plot_setting.setOutputDims("t", "x1");
		plot_setting.plot_2D_interval_GNUPLOT("./outputs/", benchmark_name + "_x1_linear" + to_string(if_symbo), result.tmv_flowpipes, setting);
	}
	return 0;
}
