#include "DeepNet.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <random>
using namespace std;


DeepNet::DeepNet()
{

}


void DeepNet::InitRBM(RBM &myRBM, unsigned num_v_neurons, unsigned num_h_neurons)
{

	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(-1,1);

        //sparsity targets
        float prob_vis = 1.0/32;//1.0/32.0;

	float half_h = num_h_neurons/2.0;
        float prob_hid = half_h/((float)num_h_neurons);

	F=0;
	S=0;	

	myRBM.num_v = num_v_neurons;
	myRBM.num_h = num_h_neurons;

	myRBM.Vp.resize(num_v_neurons); //hold prob of visible neurons activating
	myRBM.Hp.resize(num_h_neurons); //hold prob of hidden neurons activating

        myRBM.V.resize(num_v_neurons); //hold output value of visible neurons
        myRBM.H.resize(num_h_neurons); //hold output value of hidden neurons

        myRBM.Vs.resize(num_v_neurons); //hold output sample of visible neurons
        myRBM.Hs.resize(num_h_neurons); //hold output sample of hidden neurons


        myRBM.biasV.resize(num_v_neurons,prob_vis);//log(prob_vis/(1-prob_vis))); 
        myRBM.biasH.resize(num_h_neurons,prob_hid);//log(prob_hid/(1-prob_hid))); 

	myRBM.res.resize(num_v_neurons,0);


	//myRBM.W.resize(num_h_neurons,vector<float>(num_v_neurons, 0) );
	myRBM.W.resize(num_h_neurons);
	myRBM.dW.resize(num_h_neurons);
	myRBM.dW_last.resize(num_h_neurons);

	//ACCUM_GRAD.resize(num_h_neurons);

	float norm=0;

	///Fill weight matrix with initial values
	///Initialize Hidden and Visible layers to 0
	for(unsigned i=0; i<num_h_neurons; ++i)
	{
		//float start_weight = (float)rand()/(float)RAND_MAX;
		//norm += start_weight;

		myRBM.W.at(i).resize(num_v_neurons);
		myRBM.dW.at(i).resize(num_v_neurons);
		myRBM.dW_last.at(i).resize(num_v_neurons);
		//ACCUM_GRAD.at(i).resize(num_v_neurons);

		for(unsigned j=0; j<num_v_neurons; ++j)
		{
			//float start_weight = (float)rand()/((float)RAND_MAX) ;
			float start_weight = distribution(generator);
			
	                norm += start_weight;
			myRBM.W.at(i).at(j) = start_weight;
			myRBM.dW.at(i).at(j) = 0;
			myRBM.dW_last.at(i).at(j) = 0;
			//ACCUM_GRAD.at(i).at(j) = 0;

			if(i==num_h_neurons)
			{
				//myRBM.W.at(i).at(j)=-1.0/((float)num_v_neurons);
			}

			if(j==num_v_neurons)
			{
				//myRBM.W.at(i).at(j)=-1.0/((float)num_h_neurons);
			}
		}

	}
	

	//myRBM.V.back() = 1;//1.0/((float)num_v_neurons);
	//myRBM.Vs.back() = 1;//1.0/((float)num_v_neurons);
	//myRBM.Vp.back() = log(prob_vis/(1-prob_vis));//32.0/((float)num_v_neurons);

	//myRBM.H.back() = 1;//1.0/((float)num_h_neurons);
	//myRBM.Hs.back() = 1;//1.0/((float)num_h_neurons);
	//myRBM.Hp.back() = log(prob_hid/(1-prob_hid));//1.0/((float)num_h_neurons);

	///Normalize Weight Matrix
	float sum = 0;
        for(unsigned i=0; i<num_h_neurons; ++i)
        {
		
                for(unsigned j=0; j<num_v_neurons; ++j)
                {
                        myRBM.W.at(i).at(j) = myRBM.W.at(i).at(j);///norm;
			sum += myRBM.W.at(i).at(j);
                }
        }


	cout << "Normalize weights to " << sum << " using a normalization of " << norm << endl;


}

void DeepNet::BuildChain(RBM myRBM)
{
	Chain.push_back(myRBM);
	cout << "W[i][j] size: " << myRBM.W.size() << "x" << myRBM.W.back().size() << endl;
}


void DeepNet::SetupInputs(string f, unsigned num_f_bins, unsigned num_t_bins, Matrix &visible)
{

// constructs a visible matrix (num_f_bins x num_t_bins)
// for character data, there are 27 (a-z) character bins (num_f_bins), and a fixed size of t bins
// where t represents the length of a given sample set of text data


//first build character mapping
//to bins, i.e. a->1, b->2 etc

vector<char> mapChar(num_f_bins);

Matrix temp(num_f_bins,vector<int>(num_t_bins));

    for(int i = 0; i < (num_f_bins); ++i)
    {
         mapChar.at(i)=(32+i);
    }


for(int i=0; i<(num_f_bins); ++i)
{

//int j=0;

//    for(string::iterator str_itr = msg.begin(); str_itr != msg.end(); ++str_itr)
//    {
    for(int j=0; j<num_t_bins; ++j)
    {
        if( (mapChar.at(i) == (char)f.at(j)) )
	{
            //cout << *str_itr << "," << i << " = " << mapChar.at(i);
	    temp.at(i).at(j) = 1;
	}    
        else
	{
            //cout << " ";
	    temp.at(i).at(j) = 0;
	}
//	++j;
    }
    //cout << endl;
}

visible=temp;

}


void DeepNet::SampleToString(Matrix Map, int num_f_bins, int num_t_bins)
{

string f;
vector<char> mapChar(num_f_bins);

    for(int i = 0; i < (num_f_bins); ++i)
    {
         mapChar.at(i)=(32+i);
    }

	for(int i=0; i<num_t_bins; ++i)
	{
		for(int j=0; j<(num_f_bins); ++j)
		{
			if(Map.at(j).at(i)==1)
			{
				f.push_back((char)(32+j));
			}else 
			{
				
			}
		}
	}

//cout << "SampleToString :: Reconstructed f: " << endl;
cout << "	|------------------------------------------------------------------------| " << endl;
cout << "	|>>>>" <<  f << "<<<<|" << endl;
cout << "	|------------------------------------------------------------------------| " << endl;
cout << endl;
}


void DeepNet::FillStringVector(string Filename, vector<string> &Book, int num_t_bins)
{
	int array_size = num_t_bins*2; // define the size of character array
	char * array = new char[array_size]; // allocating an array of 1kb
  
	ifstream fin(Filename); //opening an input stream for file test.txt
	/*checking whether file could be opened or not. If file does not exist or don't have read permissions, file
  stream could not be opened.*/
  if(fin.is_open())
	{
    //file opened successfully so we are here
    cout << "File Opened successfully. Reading data from file into array..." << endl;
    //this loop run until end of file (eof) does not occur
		while(!fin.eof())
		{
			int position = 0; //this will be used incremently to fill characters in the array
			while(position < array_size)
			{
				fin.get(array[position]); //reading one character from file to array
				position++;
			}
			array[position-1] = '\0'; //placing character array terminating character
			Book.push_back(array);

			
		}

    
	}
	else //file could not be opened
	{
		cout << "File could not be opened." << endl;
	}

}


float transportFunction(float x)
{
	//return log(1+exp(x));
        return 1.0/(1.0+exp(-x));
}


void DeepNet::GetError(vector<float> outputVals, vector<float> targetVals, float &cost_layer)
{
	int size=targetVals.size();

	if(size!=outputVals.size()){cout << "Arrays in error calc don't match size!" << endl;}

	float error=0;
	#pragma omp parallel for reduction (+:error)
	for(unsigned i=0; i<size; ++i)
	{
		error += (targetVals[i] - outputVals[i])*(targetVals[i] - outputVals[i]);
	}
	
	cost_layer=(error)/2.0;
}



void DeepNet::StochasticGradientDecent(RBM &myRBM, vector<float> targetVals, bool FORWARD)
{


int k=0;
float K_MAX=1;

float norm=0;
float error = 0;
float error_last=0;
float dwavg=1;
float dwnorm=1;


RBM modelRBM = myRBM;
RBM dataRBM = myRBM;

//NormalizeLayer(targetVals);
//dataRBM.V = targetVals;
modelRBM.Vs = targetVals;

FeedForwardVisible(modelRBM,targetVals,modelRBM.biasH);		//Compute Q(h0=1|v0_data), get random binary sample of hiddens

FeedForwardHidden(dataRBM, targetVals, dataRBM.biasH);
FeedForwardHidden(dataRBM, dataRBM.Hp, dataRBM.biasV);
//NormalizeLayer(modelRBM.Vs);
dataRBM.V = targetVals;
dataRBM.H = modelRBM.Hp;					//Assign h0_p to dataRBM.H for use in weight update

while(k<K_MAX)
{
	FeedForwardHidden(modelRBM,modelRBM.Hs,modelRBM.biasH);		//Compute P(v1=1|h0), get random binary sample of reconstruction distribution
	WeightGradient(modelRBM,dataRBM,dwavg);
	dwnorm++;

	UpdateWeights(modelRBM,dataRBM,dwavg,dwnorm,FORWARD);
        //UpdateBias(modelRBM,dataRBM);
        ResetGradient(modelRBM);

	//NormalizeLayer(modelRBM.Vs);
	
	FeedForwardVisible(modelRBM,modelRBM.Vp,modelRBM.biasV); 	//Compute Q(h1=1|v1)
	WeightGradient(modelRBM,dataRBM,dwavg);
	dwnorm++;


	UpdateWeights(modelRBM,dataRBM,dwavg,dwnorm,FORWARD);
	//UpdateBias(modelRBM,dataRBM);
	ResetGradient(modelRBM);

	//NormalizeGradient(modelRBM);

	error_last=error;
	GetError(modelRBM.Vs, targetVals, error);

	if(error==0)
	{
		k=K_MAX;
	}

++k;
}


if(ITERATION==NUM_SAMPLES)
{
//UpdateWeights(modelRBM,dataRBM,dwavg,dwnorm,FORWARD);
//UpdateBias(modelRBM,dataRBM);
//NormalizeWeights(modelRBM);
//NormalizeLayer(modelRBM);
//NormalizeGradient(modelRBM);
//UpdateWeights(modelRBM,dataRBM,dwavg,dwnorm,FORWARD);
//UpdateBias(modelRBM,dataRBM);
//if(BATCH==4)
//{
//NormalizeGradient(modelRBM);
//UpdateWeights(modelRBM,dataRBM,dwavg,dwnorm,FORWARD);
//if(DEPTH==4)
//{
//ResetGradient(modelRBM);
//}
//NormalizeWeights(modelRBM);
//}
//cin.get();
}

//myRBM = modelRBM;
myRBM.W  = modelRBM.W;
myRBM.Vs = modelRBM.Vs;
myRBM.Vp = modelRBM.Vp;
myRBM.Hs = modelRBM.Hs;
myRBM.Hp = modelRBM.Hp;
myRBM.biasH = modelRBM.biasH;
myRBM.biasV = modelRBM.biasV;


	
	myRBM.error = error;			
	ERROR = ERROR + (error)/(ITERATION*DEPTH*TIME);	//average layer cost over the mini-batch



}

void DeepNet::Backprop(RBM &modelRBM, vector<float> inputVals, vector<float> targetVals)
{
	
        int size_h = modelRBM.num_h;
        int size_v = modelRBM.num_v;


		//backprop error to use previous layer's sample
                RBM inputRBM = modelRBM;
                RBM targetRBM = modelRBM;

                FeedForwardVisible(targetRBM,targetVals,targetRBM.biasH);   	// Calc the model.hp[i] dist that gives correct outputVals (target hp) to hit the target
                FeedForwardHidden(targetRBM,targetRBM.Hp,targetRBM.biasV);    	// Calc the model.vp[j] dist implied by that

                FeedForwardVisible(inputRBM,inputVals,inputRBM.biasH);    	// Calc the model.hp[i] dist for the inputVals (target hp if SGD) 
                FeedForwardHidden(inputRBM,inputRBM.Hs,inputRBM.biasV);   	// Calc the vp[j] dist implied by inputVals hp[i] (target vp if SGD)


                //Update weights due to error in hidden layers
		#pragma omp parallel for
                for(unsigned i=0; i<size_h; i++)
                {
			#pragma omp simd
                        for(unsigned j=0; j<size_v; ++j)
                        {
				modelRBM.dW[i][j] = (targetVals[j]-inputRBM.Vs[j])*(inputRBM.Vs[j])*(1-inputRBM.Vs[j])*inputRBM.Hp[i];
				//modelRBM.W[i][j] = modelRBM.W[i][j] + (targetVals[j]-inputRBM.Vs[j])*(inputRBM.Vs[j])*(1-inputRBM.Vs[j])*inputRBM.Hp[i];
				//ACCUM_GRAD[i][j] = ACCUM_GRAD[i][j] + modelRBM.dW[i][j];
                        }
                }

		float norm;

		UpdateWeights(modelRBM, modelRBM, norm, 0, 0);
		//UpdateBias(modelRBM,modelRBM);
		ResetGradient(modelRBM);

		FeedForwardVisible(modelRBM,inputVals,modelRBM.biasH);   //Calc the hp[i] dist for the inputVals
                FeedForwardHidden(modelRBM,modelRBM.Hs,modelRBM.biasV);   //Calc the vp[j] dist implied by sampled hp[i]


		float error=0;
              	GetError(inputRBM.Vp, modelRBM.Vp, error);
		
		TERROR += error/(DEPTH+ITERATION);



}


void DeepNet::FeedForwardVisible(RBM &myRBM, vector<float> inputVals, vector<float> biasV)
{
	/// Given a visible layer, assign 
	/// probabilities and sampled outputs
	/// to hidden layer.

	/// Assign raw input values to visible layer ///
	//myRBM.V = inputVals;

	int size_h = myRBM.num_h;
	int size_v = myRBM.num_v;

	/// Calculate probabilities of 
	/// hidden neuron activations
	/// using the input values and
	/// weight matrix W

	float avg_prob = 0;

	#pragma omp parallel for
	for(unsigned i=0; i<size_h; ++i)
	{
		float sum=0;

		#pragma omp simd reduction(+:sum)
		for(unsigned j=0; j<size_v; ++j)
		{
			sum += (biasV[j]+inputVals[j])*myRBM.W[i][j];
			//sum += myRBM.Vp[j]*myRBM.W[i][j];
		}

		/// Sample the resulting probability
		/// and assign 1 or 0 to H depending
		/// on the result. Use the probability
		/// to assign to Hp

		float prob = transportFunction(sum);

		myRBM.H[i] = (prob);
		myRBM.Hp[i] = prob;

		float samp = (float)rand()/(float)RAND_MAX;

		if(prob>samp)	{ myRBM.Hs[i]=1; }
		else 		{ myRBM.Hs[i]=0; }
	}

}

void DeepNet::FeedForwardHidden(RBM &myRBM, vector<float> inputHiddens, vector<float> biasH)
{
        /// Given a hidden layer, assign
        /// probabilities and sampled outputs
        /// to the sample visible layer.


        int size_h = myRBM.num_h;
        int size_v = myRBM.num_v;

        /// Calculate probabilities of
        /// visible neuron activations
        /// using freshly sampled hidden
	/// values and weight matrix W

	#pragma omp parallel for
        for(unsigned j=0; j<size_v; ++j)
        {
                float sum=0;
		float avg_hid_val=0;

		#pragma omp simd reduction(+:sum)
                for(unsigned i=0; i<size_h; ++i)
                {
                        sum += (biasH[i]+inputHiddens[i])*myRBM.W[i][j];
                }

	
                /// Sample the resulting probability
                /// and assign 1 or 0 to H depending
                /// on the result. Use the probability
                /// to assign to Hp

                float prob = transportFunction(sum);

		myRBM.Vp[j] = prob;

                float samp = (float)rand()/(float)RAND_MAX;


                if(prob>samp)	{ myRBM.Vs[j]=1; }
                else 		{ myRBM.Vs[j]=0; }
        }

}

void DeepNet::GetVisibleSample(RBM &myRBM)
{
	// Used to generate samples for the middle 
	// and last RBM in the stack; sample from 
	// the previous RBM's visible probability
	// distribution

	int size_v = myRBM.num_v;

        #pragma omp parallel for
        for(unsigned j=0; j<size_v; ++j)
        {
                float samp = (float)rand()/(float)RAND_MAX;


                if(myRBM.Vp[j]>samp)   	{ myRBM.Vs[j]=1; }
                else          		{ myRBM.Vs[j]=0; }

	}
}

void DeepNet::GetHiddenSample(RBM &myRBM)
{
        // Used to generate samples for the middle
        // and last RBM in the stack; sample from
        // the previous RBM's visible probability
        // distribution

        int size_h = myRBM.num_h;

        #pragma omp parallel for
        for(unsigned i=0; i<size_h; ++i)
        {
                float samp = (float)rand()/(float)RAND_MAX;


                if(myRBM.Hp[i]>samp)    { myRBM.Hs[i]=1; }
                else                    { myRBM.Hs[i]=0; }

        }
}


void DeepNet::UpdateBias(RBM &myRBM, RBM dataRBM)
{
        int size_h = myRBM.num_h;
        int size_v = myRBM.num_v;
	
	
	float avg_hid_val = 0;
	#pragma omp parallel for reduction(+:avg_hid_val)
	for(unsigned i=0; i<size_h; ++i)
	{
		avg_hid_val += (dataRBM.Hs[i])/((float)size_h);
	}

	float avg_vis_val = 0;	
	#pragma omp parallel for reduction(+:avg_vis_val)
        for(unsigned j=0; j<size_v; ++j)
        {
                avg_vis_val += dataRBM.Vs[j]/((float)size_v);
        }


        //update visible bias
	float num_vis=((float)size_v);
        float q = 32.0/num_vis;
	float avg_vbias_val=0;

	#pragma omp parallel for
	for(unsigned j=0; j<size_v; ++j)
	{
		float sum=0;
		#pragma omp simd reduction(+:sum)
        	for(unsigned i=0; i<size_h; ++i)
        	{
                	sum += (myRBM.Hp[i]-dataRBM.Hp[i])*myRBM.W[i][j];
        	}
		
		//myRBM.biasV[j] = (myRBM.Vp[j]-dataRBM.Vp[j]);//log(avg_vis_val/(1-avg_vis_val))-q*sum;
		myRBM.biasV[j] += log(q/(1-q))-avg_hid_val*sum;
	}


        //update hidden bias
	float num_hid = ((float)size_h);
	q=1.0/num_hid;
	float avg_hbias_val=0;

	#pragma omp parallel for
	for(unsigned i=0; i<size_h; ++i)
	{
		float sum=0;
		#pragma omp simd reduction(+:sum)
        	for(unsigned j=0; j<size_v; ++j)
        	{
                	sum += (myRBM.Vp[j]-dataRBM.Vp[j])*myRBM.W[i][j];
        	}
		//myRBM.biasH[i] = (myRBM.Hs[i]-dataRBM.Hp[i]);//log(q/(1-q))-avg_vis_val*sum;
		myRBM.biasH[i] += log(q/(1-q))-avg_vis_val*sum;
	}

}


void DeepNet::WeightGradient(RBM &modelRBM,RBM dataRBM,float &dwavg)
{
        unsigned num_h_neurons = modelRBM.num_h;
        unsigned num_v_neurons = modelRBM.num_v;

	float sum = 0;

        #pragma omp parallel for
        for(unsigned i=0; i<num_h_neurons; ++i)
        {
                for(unsigned j=0; j<num_v_neurons; ++j)
                {
			
                        //modelRBM.dW.at(i).at(j) =  modelRBM.dW.at(i).at(j) + (modelRBM.Hp[i]*modelRBM.Vp[j] - dataRBM.Hp[i]*dataRBM.Vp[j]);
			modelRBM.dW.at(i).at(j) =  modelRBM.dW.at(i).at(j) + (dataRBM.H[i]*dataRBM.V[j]-modelRBM.Hp[i]*modelRBM.Vp[j])/((1E-8+RATE));
			//sum += modelRBM.dW.at(i).at(j);
			//ACCUM_GRAD.at(i).at(j) = ACCUM_GRAD.at(i).at(j) + modelRBM.dW.at(i).at(j);///(1 + RATE);
			//RATE += ACCUM_GRAD.at(i).at(j)*ACCUM_GRAD.at(i).at(j);
                }
        }
	
}




void DeepNet::UpdateWeights(RBM &modelRBM, RBM dataRBM, float &L, float dnorm, bool FORWARD)
{

	unsigned num_h_neurons = modelRBM.num_h;
	unsigned num_v_neurons = modelRBM.num_v;

	vector<float> norm(num_h_neurons);

	//cout << "Norm of Updated sample matrix: ";

	float myrate=0;

        #pragma omp parallel for reduction (+:myrate)
        for(unsigned i=0; i<num_h_neurons; ++i)
        {
                for(unsigned j=0; j<num_v_neurons; ++j)
                {
			myrate += modelRBM.dW.at(i).at(j)*modelRBM.dW.at(i).at(j);
                }
        }

	RATE = sqrt(myrate);

	#pragma omp parallel for
        for(unsigned i=0; i<num_h_neurons; ++i)
        {
                for(unsigned j=0; j<num_v_neurons; ++j)
                {
 	                //modelRBM.W.at(i).at(j) = modelRBM.W.at(i).at(j) + ACCUM_GRAD.at(i).at(j) - (1E-6)*RATE;//(dnorm*NUM_SAMPLES);
			modelRBM.W.at(i).at(j) = modelRBM.W.at(i).at(j) + modelRBM.dW.at(i).at(j) - (1E-6)*RATE;
                }
        }

}

void DeepNet::Stats(RBM &m)
{

        unsigned num_h_neurons = m.num_h;
        unsigned num_v_neurons = m.num_v;

	float s = 0; //Entropy of Hiddens for this RBM
	float avg_h = 0;
	float avg_v = 0;
	float avg_bv = 0;
	float avg_bh = 0;
	float eta = 0;

	
        #pragma omp parallel for reduction(+:avg_v,avg_bv)
        for(unsigned j=0; j<num_v_neurons; ++j)
        {
                avg_v += m.Vs[j];
		avg_bv += m.biasV[j];

        }


        #pragma omp parallel for reduction(+:s,avg_h)
        for(unsigned i=0; i<num_h_neurons; ++i)
        {
		avg_h += m.Hs[i];
		avg_bh += m.biasH[i];
		
		s += -(m.Hp[i])*log(m.Hp[i]) + (1-(m.Hp[i]))*(log(1-(m.Hp[i])));
        }

	cout << "avg hidden : " << avg_h/m.num_h << endl;
	cout << "average:	visible:	" << avg_v/m.num_v << endl;
	cout << "		hidden:		" << avg_h/m.num_h << endl;
	cout << "		bias V:		" << avg_bv/m.num_v << endl;
	cout << "		bias H:		" << avg_bh/m.num_h << endl;


        float f = 0;  //Gibbs Free energy of this RBM
	
        #pragma omp parallel for reduction(+:f)
        for(unsigned i=0; i<num_h_neurons; ++i)
        {
                for(unsigned j=0; j<num_v_neurons; ++j)
                {
                      f += - m.Vp[j]*m.W[i][j]*m.Hp[i] - m.Vp[j]*m.biasV[j] - m.Hp[i]*m.biasH[i] - (m.Hp[i])*log(m.Hp[i]) + (1-(m.Hp[i]))*(log(1-(m.Hp[i])));				
                }
        }

	float temp = 0;

	temp = f/((float)(num_v_neurons + num_h_neurons));

	if(f<m.f && f>0)
	{
		//ResetGradient(m);
	}

	if(f==m.f)
	{
		//NormalizeWeights(m);
	}

	F += f; //Total energy of RBM Chain
	S += s; //Total entropy of RBM Chain

	m.f = f;
	m.s = s;
	
	cout << "===============================================" << endl;
	cout << "RBM STATS:"					  << endl;
	cout << "-----------------------------------------------" << endl;
	cout << "	Temperature:		" << temp << endl;
	cout << "	Gibbs Free Energy:	" << m.f << endl;
	cout << "	Entropy:		" << m.s << endl;
        cout << "	Average Error:		" << m.error << endl;
	cout << "===============================================" << endl;
	cout << "NETWORK STATS:"				  << endl;
	cout << "-----------------------------------------------" << endl;
        cout << "	Total Network Error:	" << ERROR << endl;
        cout << "	Learning Rate:		" << RATE << endl;
	cout << "	Training Error:		" << TERROR << endl;
	cout << "-----------------------------------------------" << endl;
	cout << "Total F: " << F << ", Total S: " << S << endl;
	cout << "|||||||||||||||||||||||||||||||||||||||||||||||" << endl;	
	

}

void DeepNet::ResetGradient(RBM &modelRBM)
{

        unsigned num_h_neurons = modelRBM.num_h;
        unsigned num_v_neurons = modelRBM.num_v;

        #pragma omp parallel for
        for(unsigned i=0; i<num_h_neurons; ++i)
        {
                for(unsigned j=0; j<num_v_neurons; ++j)
                {
			modelRBM.dW.at(i).at(j)=0;
                      //ACCUM_GRAD.at(i).at(j)=0;
                }
        }

}


void DeepNet::NormalizeGradient(RBM &modelRBM)
{

        unsigned num_h_neurons = modelRBM.num_h;
        unsigned num_v_neurons = modelRBM.num_v;

        float norm =0;
        #pragma omp parallel for reduction(+:norm)
        for(unsigned i=0; i<num_h_neurons; ++i)
        {
                for(unsigned j=0; j<num_v_neurons; ++j)
                {
                     // norm += ACCUM_GRAD.at(i).at(j);
                }
        }

        #pragma omp parallel for
        for(unsigned i=0; i<num_h_neurons; ++i)
        {
                for(unsigned j=0; j<num_v_neurons; ++j)
                {
                      //ACCUM_GRAD.at(i).at(j)=ACCUM_GRAD.at(i).at(j)/BATCH;
                }
        }

}



void DeepNet::NormalizeWeights(RBM &modelRBM)
{

        unsigned num_h_neurons = modelRBM.num_h;
        unsigned num_v_neurons = modelRBM.num_v;

        float norm =0;

        #pragma omp parallel for reduction(+:norm)
        for(unsigned i=0; i<num_h_neurons; ++i)
        {
                for(unsigned j=0; j<num_v_neurons; ++j)
                {
                      norm += modelRBM.W.at(i).at(j);
                }
        }

        #pragma omp parallel for
        for(unsigned i=0; i<num_h_neurons; ++i)
        {
                for(unsigned j=0; j<num_v_neurons; ++j)
                {
                      modelRBM.W.at(i).at(j)=modelRBM.W.at(i).at(j)/norm;
                }
        }
}


void DeepNet::NormalizeLayer(vector<float> &layer)
{

        unsigned num_neurons = layer.size();

        float mean = 0;
	float variance = 0;
	float stddev = 0;

        #pragma omp parallel for reduction(+:mean)
        for(unsigned i=0; i<num_neurons; ++i)
        {
		mean += layer.at(i)/((float)num_neurons);
        }


        #pragma omp parallel for reduction(+:variance)
        for(unsigned i=0; i<(num_neurons); ++i)
        {
                variance += layer.at(i)*(layer.at(i)-mean)*(layer.at(i)-mean);
        }

	stddev=sqrt(variance);

        #pragma omp parallel for
        for(unsigned i=0; i<num_neurons; ++i)
        {
                layer.at(i) = (layer.at(i) - mean)/stddev;
        }


}


