#include "../../POLAR/NNCS.h"
//#include "../../flowstar/flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{
	// Check 'ARCH-COMP20_Category_Report_Artificial_Intelligence_and_Neural_Network_Control_Systems_-AINNCS-_for_Continuous_and_Hybrid_Systems_Plants.pdf'
	// Declaration of the state variables.

	Variables vars;
	
	/*
	 * The variables should be declared in the following way:
	 * 1st group: Ordered observable state variables which will be used as input of controllers.
	 * 2nd group: Control variables which are only updated by the controller.
	 * 3rd group: State variables which are not observable.
	 */
	
	// input of the controller
	int x0_id = vars.declareVar("x0"); // v_set
	int x1_id = vars.declareVar("x1");	// T_gap
	int x2_id = vars.declareVar("x2");	// x_lead
	int x3_id = vars.declareVar("x3");	// x_ego
	int x4_id = vars.declareVar("x4"); // v_lead
	int x5_id = vars.declareVar("x5");	// v_ego
	int x6_id = vars.declareVar("x6");	// gamma_lead
	int x7_id = vars.declareVar("x7");	// gamma_ego

	// output of the controller
	int u0_id = vars.declareVar("u0");	// a_ego

	// the time variable t is not used by the controller
	int t_id = vars.declareVar("t");    // time t	
	
	// Define the continuous dynamics.
	vector<string> derivatives = {"0",
								"0",
								"x4",
								"x5",
								"x6",
								"x7",
								"-4 - 2 * x6 - 0.0001 * x4^2",
								"2 * u0 - 2 * x7 - 0.0001 * x5^2",
								"0","1"};

	// create the neural network object for the controller
	//string net_name = argv[6];
	//string benchmark_name = "nncs_acc_" + net_name;
	string nn_name = "acc_tanh20x20x20_";
 	NeuralNetwork nn_controller(nn_name);

	// create the NNCS object
	NNCS<Real> system(vars, 0.1, derivatives, nn_controller);

	// Flow* reachability setting
	Computational_Setting setting(vars);

	unsigned int order = stoi(argv[4]);

	// stepsize and order for reachability analysis
	//setting.setFixedStepsize(0.005, order);
	setting.setFixedStepsize(0.1, order);
	setting.setCutoffThreshold(1e-5);
	setting.printOff();

	// remainder estimation
	Interval I(-0.01, 0.01);
	vector<Interval> remainder_estimation(vars.size(), I);
	setting.setRemainderEstimation(remainder_estimation);

	
	// POLAR setting
	unsigned int taylor_order = order;	// same as the flowpipe order, but it is not necessary
	unsigned int bernstein_order = stoi(argv[3]);
	//unsigned int partition_num = 4000;
	unsigned int partition_num = 10;
	unsigned int if_symbo = stoi(argv[5]);
	
	PolarSetting polar_setting(order, bernstein_order, partition_num, "Mix", "Symbolic");
	polar_setting.set_num_threads(-1);
	if(if_symbo == 0){
			// not using symbolic remainder
			polar_setting.set_remainder_type("Concrete");
			polar_setting.symb_rem = if_symbo;
		}

	// initial set
	double w = stoi(argv[1]);

	// define the initial set which is a box
	Interval init_x0(30), init_x1(1.4), init_x2(90, 91), init_x3(10, 11), init_x4(32, 32.05), init_x5(30, 30.05), init_x6(0), init_x7(0);

	// the initial values of the rest of the state variables are 0
	vector<Interval> box(vars.size());
	box[x0_id] = init_x0;
	box[x1_id] = init_x1;
	box[x2_id] = init_x2;
	box[x3_id] = init_x3;
	box[x4_id] = init_x4;
	box[x5_id] = init_x5;
	box[x6_id] = init_x6;
	box[x7_id] = init_x7;

	// translate the initial set to a flowpipe
	Flowpipe initialSet(box);

	//unsafe set
	vector<Constraint> safeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// run the reachability computation
	time_t start_timer;
	time_t end_timer;
	double seconds;
	time(&start_timer);

	int n = stoi(argv[2]); // time horizon

	//Symbolic_Remainder sr(initialSet, 500);
	Symbolic_Remainder sr(initialSet, 100);

	system.reach(result, initialSet, n, setting, polar_setting, safeSet, sr);

	// end box or target set
	vector<Interval> end_box;
	string reach_result;
	reach_result = "Verification result: Unknown(35)";
	result.fp_end_of_time.intEval(end_box, order, setting.tm_setting.cutoff_threshold);
	result.transformToTaylorModels(setting);

	// time cost
	time(&end_timer);
	seconds = difftime(start_timer, end_timer);
	printf("time cost: %lf\n", -seconds);
	std::string running_time ="Running Time: %lf\n" + to_string(-seconds) + " seconds";
	
	// create a subdir named outputs to save result
	int mkres = mkdir("./outputs", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
	if (mkres < 0 && errno != EEXIST)
	{
		printf("Can not create the directory for images.\n");
		exit(1);
	}

	ofstream result_output("./outputs/nncs_acc_" + to_string(if_symbo) + ".txt");
	if (result_output.is_open())
	{
		result_output << reach_result << endl;
		result_output << running_time << endl;
	}

	// plot the flowpipes
	Plot_Setting plot_setting(vars);
	plot_setting.printOn();
	plot_setting.setOutputDims("x4", "x5");
	plot_setting.plot_2D_interval_GNUPLOT("./outputs/", "acc_"  + to_string(n) + "_"  + to_string(polar_setting.symb_rem), result.tmv_flowpipes, setting);
	
	//plot_setting.setOutputDims("t", "x1");
	//plot_setting.plot_2D_interval_GNUPLOT("./outputs/", "acc_" + to_string(n) + "_"  + to_string(if_symbo), result.tmv_flowpipes, setting);
	
	return 0;
}