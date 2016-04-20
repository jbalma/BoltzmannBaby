#include <iostream>
#include <vector>

using namespace std;

struct RBM
{
	unsigned num_v;
	unsigned num_h;

	unsigned avg_v;
	unsigned avg_h;

	vector< vector<float> > W;
	vector< vector<float> > dW;

	vector< float > V;	//Output Values of visible layer 	(v_data)
	vector< float > H;	//Output Values of hidden layer  	(h_data)

        vector< float > Vp;	//Output probabilities of visible neurons in layer	(p_model)
        vector< float > Hp;	//Output probabilities of hidden neurons in layer	(q_model)

        vector< float > Vs;	//Sample-based reconstruction of visible layer		(v_model)
        vector< float > Hs;	//Sample-based reconstruction of hidden layer		(h_model)

	vector< float> biasV;
	vector< float> biasH;	

	
};


class DeepNet
{

public:

	typedef vector< vector<int> > Matrix;

	DeepNet();

	vector<RBM> Chain;

	void InitRBM(RBM &myRBM, unsigned num_v_neurons, unsigned num_h_neurons);		//inits 2-layer neural net of given topology

	void BuildChain(RBM myRBM);		 						//builds chain of RBMs with selected number of neurons in each layer
	
	void FeedForwardVisible(RBM &myRBM, vector<float> inputVals, vector<float> dataVp);	//set output values of hidden layer given visible
	
	void FeedForwardHidden(RBM &myRBM, vector<float> inputHiddens, vector<float> dataHp);	//set output values of visible layer given hidden

	void GradientDecent(RBM &myRBM, vector<float> targetVals);				//Calculate Gradient based on targetVals, update weights

	void UpdateBias(RBM &myRBM, RBM dataRBM);

	void WeightGradient(RBM &modelRBM, RBM dataRBM);

	void getOutputs(RBM &myRMB);			 					//Update output and prob value arrays, set Vs and Hs sample arrays for next layer

	void UpdateWeights(RBM &sampleRBM, RBM myRBM, float &norm, float K_MAX);	



	void SetupInputs(string f, unsigned num_f_bins, unsigned num_t_bins, Matrix &visible);

	void SampleToString(Matrix Map, int num_f_bins, int num_t_bins);

	void FillStringVector(string Filename, vector<string> &Book, int num_t_bins);


	

	












};
