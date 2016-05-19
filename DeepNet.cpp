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
        float prob_vis = 1.0/96.0;
        float prob_hid = 1.0/((float)num_h_neurons);

	

	myRBM.num_v = num_v_neurons;
	myRBM.num_h = num_h_neurons;

	myRBM.Vp.resize(num_v_neurons); //hold prob of visible neurons activating
	myRBM.Hp.resize(num_h_neurons); //hold prob of hidden neurons activating

        myRBM.V.resize(num_v_neurons); //hold output value of visible neurons
        myRBM.H.resize(num_h_neurons); //hold output value of hidden neurons

        myRBM.Vs.resize(num_v_neurons); //hold output sample of visible neurons
        myRBM.Hs.resize(num_h_neurons); //hold output sample of hidden neurons


        myRBM.biasV.resize(num_v_neurons,log(prob_vis/(1-prob_vis))); 
        myRBM.biasH.resize(num_h_neurons,log(prob_hid/(1-prob_hid))); 


	//myRBM.W.resize(num_h_neurons,vector<float>(num_v_neurons, 0) );
	myRBM.W.resize(num_h_neurons);
	myRBM.dW.resize(num_h_neurons);
	myRBM.dW_last.resize(num_h_neurons);

	ACCUM_GRAD.resize(num_h_neurons);

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
		ACCUM_GRAD.at(i).resize(num_v_neurons);

		for(unsigned j=0; j<num_v_neurons; ++j)
		{
			//float start_weight = (float)rand()/((float)RAND_MAX) ;
			float start_weight = distribution(generator);
			
	                norm += start_weight;
			myRBM.W.at(i).at(j) = start_weight;
			myRBM.dW.at(i).at(j) = 0;
			myRBM.dW_last.at(i).at(j) = 0;
			ACCUM_GRAD.at(i).at(j) = 0;

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
                        myRBM.W.at(i).at(j) = myRBM.W.at(i).at(j)/norm;
			sum += myRBM.W.at(i).at(j);
                }
        }


	cout << "Normalize weights to " << sum << " using a normalization of " << norm << endl;


}

void DeepNet::BuildChain(RBM myRBM)
{
	Chain.push_back(myRBM);
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




void DeepNet::StochasticGradientDecent(RBM &myRBM, vector<float> targetVals, bool OUTPUT_LAYER)
{

vector<float> zeroVals(myRBM.num_v,0.1);



myRBM.V =  targetVals;

RBM modelRBM = myRBM;
RBM dataRBM = myRBM;

FeedForwardVisible(dataRBM,targetVals,dataRBM.biasH);	//Get random sample of H_data
FeedForwardHidden(dataRBM,dataRBM.H,dataRBM.biasV);		//Get random sample of V_data


//We are done with sampling
//of <H[i]V[j}>_data at this
//point, which is contained
//in dataRBM


//Now, sample <V[i]H[j]>_model
//to get unbiased sample
//for the second term...

//Start from any random visible
//state...Can use dataRBM to
//initialized... 

FeedForwardVisible(modelRBM,targetVals,modelRBM.biasH);        //Samples H_model from q_model=q_data
FeedForwardHidden(modelRBM,modelRBM.H,modelRBM.biasV);       //Determines p_model = f(bias_v + W[i][j]*h_model[i])


dataRBM.Vp=targetVals;
modelRBM.Vs=dataRBM.Vs;
modelRBM.Vp=dataRBM.Vp;
//modelRBM.Hs=dataRBM.Hs;
//modelRBM.Hp=dataRBM.Hp;

int k=0;
float K_MAX=10;

	float norm=0;
	float error = 0;
	float myerror=0;
	//float CUTOFF=1;
	float dwavg=1;
	float dwnorm=1;
	//float dwavg_last=0;
	modelRBM.dW_last = ACCUM_GRAD;//modelRBM.dW;
	//modelRBM.dW = ACCUM_GRAD;
	
	//Start of SGD sub mini-batch

if(OUTPUT_LAYER)
{
        while(k<K_MAX)
        {
                ///HiddenLayer Prob given inputs
                FeedForwardVisible( modelRBM, dataRBM.Vp, modelRBM.biasH);             //update sample of h_model from q_model

                //UpdateBias(modelRBM,dataRBM);
                WeightGradient(modelRBM,dataRBM, dwavg);
                dwnorm++;
		//UpdateWeights(modelRBM,dataRBM,dwavg,dwnorm,OUTPUT_LAYER);

                ///VisibleLayer Prob given sample hiddens
                FeedForwardHidden( modelRBM, dataRBM.Hp, modelRBM.biasV);              //update p_model = f(bias_v + W h_model)

                //UpdateBias(modelRBM,dataRBM);
                WeightGradient(modelRBM,dataRBM, dwavg);
                dwnorm++;
                //UpdateWeights(modelRBM,dataRBM,dwavg,dwnorm,OUTPUT_LAYER);


                #pragma omp parallel for reduction (+:error)
                for(unsigned i=0; i<myRBM.num_v; ++i)
                {
                        error += targetVals[i] - modelRBM.Vp[i];
                }

                k++;
        }

        //UpdateBias(modelRBM,dataRBM);
        UpdateWeights(modelRBM,dataRBM,dwavg,dwnorm,OUTPUT_LAYER);


}else
{
        while(k<K_MAX)
        {
                ///HiddenLayer Prob given inputs
                FeedForwardVisible( modelRBM, modelRBM.Vs, modelRBM.biasH);             //update sample of h_model from q_model

                //UpdateBias(modelRBM,dataRBM);
                //WeightGradient(modelRBM,dataRBM, dwavg);
                //dwnorm++;
                //UpdateWeights(modelRBM,dataRBM,dwavg,dwnorm,OUTPUT_LAYER);

                ///VisibleLayer Prob given sample hiddens
                FeedForwardHidden( modelRBM, modelRBM.Hs, modelRBM.biasV);              //update p_model = f(bias_v + W h_model)

                //UpdateBias(modelRBM,dataRBM);
                WeightGradient(modelRBM,dataRBM, dwavg);
                dwnorm++;
                //UpdateWeights(modelRBM,dataRBM,dwavg,dwnorm,OUTPUT_LAYER);


                #pragma omp parallel for reduction (+:error)
                for(unsigned i=0; i<myRBM.num_v; ++i)
                {
                        error += targetVals[i] - modelRBM.Vs[i];
                }

                k++;
        }

        UpdateBias(modelRBM,dataRBM);
	UpdateWeights(modelRBM,dataRBM,dwavg,dwnorm,OUTPUT_LAYER);

}

	cout << "Current RBM dW norm (# of gradients in batch): " << dwnorm << endl;
	cout << "Current RBM dW average: " << dwavg/dwnorm << endl;
	
	
	myRBM.W = modelRBM.W;
	//myRBM.dW = modelRBM.dW;
	myRBM.Vs = modelRBM.Vs;
	myRBM.Vp = modelRBM.Vp;
	//myRBM.Hs = modelRBM.Hs;
	//myRBM.Hp = modelRBM.Hp;
	//myRBM.H = modelRBM.H;

	error = error/((float)k);	//average error over the number of times we calcualted it mini-batch k loop

	ERROR += (error)/(TIME+1);	//average error over each RBM

	cout << "Current RBM Average Error: " << error << endl;
	cout << "Current Total Network Error: " << ERROR << endl;
	cout << "Current Learning Rate: " << RATE << endl;


}


void DeepNet::FeedForwardVisible(RBM &myRBM, vector<float> inputVals, vector<float> biasH)
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
			sum += (inputVals[j])*myRBM.W[i][j];
			//sum += myRBM.Vp[j]*myRBM.W[i][j];
		}

		/// Sample the resulting probability
		/// and assign 1 or 0 to H depending
		/// on the result. Use the probability
		/// to assign to Hp

		float prob = transportFunction(biasH[i]+sum);

		myRBM.H[i] = (prob);
		myRBM.Hp[i] = prob;

		float samp = (float)rand()/(float)RAND_MAX; 

		if(prob>samp)	{ myRBM.Hs[i]=1; }
		else 		{ myRBM.Hs[i]=0; }
	}

}

void DeepNet::FeedForwardHidden(RBM &myRBM, vector<float> inputHiddens, vector<float> biasV)
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
                        sum += (inputHiddens[i])*myRBM.W[i][j];
                }

	
                /// Sample the resulting probability
                /// and assign 1 or 0 to H depending
                /// on the result. Use the probability
                /// to assign to Hp

                float prob = transportFunction(biasV[j]+sum);

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

void DeepNet::UpdateBias(RBM &myRBM, RBM dataRBM)
{
        int size_h = myRBM.num_h;
        int size_v = myRBM.num_v;
	
	
	float avg_hid_val = 0;
	#pragma omp parallel for reduction(+:avg_hid_val)
	for(unsigned i=0; i<size_h; ++i)
	{
		avg_hid_val += round(myRBM.H[i])/((float)size_h);
	}

	float avg_vis_val = 0;	
	#pragma omp parallel for reduction(+:avg_vis_val)
        for(unsigned j=0; j<size_v; ++j)
        {
                avg_vis_val += dataRBM.V[j]/((float)size_v);
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
                	sum += (myRBM.Hp[i]-dataRBM.H[i])*myRBM.W[i][j];
        	}
		
		myRBM.biasV[j] = log(avg_vis_val/(1-avg_vis_val))-q*sum;
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
		myRBM.biasH[i] = log(q/(1-q))-avg_vis_val*sum;
	}

}

void DeepNet::getOutputs(RBM &myRBM)
{
	//myRBM.V = myRBM.Vs;
	//myRBM.H = myRBM.Hs;
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
			
                        modelRBM.dW.at(i).at(j) =  modelRBM.dW.at(i).at(j) + (modelRBM.Hs[i]*modelRBM.Vp[j] - dataRBM.Hs[i]*dataRBM.Vp[j]);
			//modelRBM.dW.at(i).at(j) =  modelRBM.dW.at(i).at(j) + (dataRBM.Hp[i]*dataRBM.V[j]-modelRBM.Hp[i]*modelRBM.Vp[j]);
			//sum += modelRBM.dW.at(i).at(j);
			ACCUM_GRAD.at(i).at(j) = ACCUM_GRAD.at(i).at(j) + modelRBM.dW.at(i).at(j)/(1E-6 + RATE);
			//RATE += ACCUM_GRAD.at(i).at(j)*ACCUM_GRAD.at(i).at(j);
                }
        }
	
	//RATE=sqrt(RATE);
	
	//dwavg = sum;
}

void DeepNet::Backprop(RBM &modelRBM, vector<float> inputVals, vector<float> targetVals)
{
        int size_h = modelRBM.num_h;
        int size_v = modelRBM.num_v;

	vector<float> H_temp(size_h);
	vector<float> V_temp(size_v);

        #pragma omp parallel for
        for(unsigned i=0; i<size_h; ++i)
        {
                float sum_want=0;

                #pragma omp simd reduction(+:sum_want)
                for(unsigned j=0; j<size_v; ++j)
                {
			// inputs-targets is the actual gradient
			// for the SUPERVISED learning of the
			// the last RBM in the chain
                        sum_want += (targetVals[j])*modelRBM.W[i][j];
                }

                float prob_want = transportFunction(sum_want);

		// This is the required activation
		// of the hidden layer to produce
		// the desired output values
		// which match the targets given a set of
		// inputs in the previous layer
                H_temp[i] = prob_want;

		
                float sum_got=0;

                #pragma omp simd reduction(+:sum_got)
                for(unsigned j=0; j<size_v; ++j)
                {
			// Vp_model is the prob of the 
			// visible layer given the hidden 
			// we actually got in sampling
			// from the previous layer
                        sum_got += (inputVals[j])*modelRBM.W[i][j]; 
                }

                float prob_got = transportFunction(sum_got);

		modelRBM.Hp[i] = prob_got;

                
        }

        #pragma omp parallel for
        for(unsigned j=0; j<size_v; ++j)
        {
                float sum_want=0;

                #pragma omp simd reduction(+:sum_want)
                for(unsigned i=0; i<size_h; ++i)
                {		
                        // (Hp_model - H_temp) is the actual gradient
                        // for the SUPERVISED learning of the
                        // the last RBM in the chain. We
			// use the error between the required hidden
			// activation and what we actually got
			// given the previous RBMs inputs
                        sum_want += (H_temp[i])*modelRBM.W[i][j];
                }

                float prob_want = (transportFunction(sum_want));

                // This is the required activation
                // of the visible layer we need to produce
                // the desired hidden activations,
                // which would be needed to reproduce the
                // targets in the previous layer
		V_temp[j]=prob_want;

		float sum_got =0;

                for(unsigned i=0; i<size_h; ++i)
                {
                        // Hp_model is the prob of the
                        // visible layer given the hidden
                        // we actually got in sampling
                        // from the previous layer
                        sum_got += (modelRBM.H[i])*modelRBM.W[i][j];
                }

                float prob_got = transportFunction(sum_got);


		modelRBM.Vp[j] = prob_got;

		float samp = (float)rand()/(float)RAND_MAX;

                if(prob_got>samp)  { modelRBM.Vs[j]=1; }
                else               { modelRBM.Vs[j]=0; }

		//ERROR += (modelRBM.Vs[j]-targetVals[j])/(TIME+1);
        }


        unsigned num_h_neurons = modelRBM.num_h;
        unsigned num_v_neurons = modelRBM.num_v;


        #pragma omp parallel for
        for(unsigned i=0; i<num_h_neurons; ++i)
        {
                for(unsigned j=0; j<num_v_neurons; ++j)
                {
                        modelRBM.W.at(i).at(j) = modelRBM.W.at(i).at(j) - (modelRBM.H[i]*inputVals[j] - H_temp[j]*targetVals[i] );
							//+ modelRBM.dW.at(i).at(j)*(modelRBM.dW[i][j]-modelRBM.dW_last[i][j]);
                }
        }
}

void DeepNet::UpdateWeights(RBM &modelRBM, RBM dataRBM, float &L, float K_MAX, bool OUTPUT_LAYER)
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
			myrate += ACCUM_GRAD.at(i).at(j)*ACCUM_GRAD.at(i).at(j);///((DEPTH+K_MAX+RATE)*(DEPTH+K_MAX+RATE));
                }
        }

	RATE = sqrt(myrate);


if(!OUTPUT_LAYER)
{
	#pragma omp parallel for
        for(unsigned i=0; i<num_h_neurons; ++i)
        {
                for(unsigned j=0; j<num_v_neurons; ++j)
                {
 	                modelRBM.W.at(i).at(j) = modelRBM.W.at(i).at(j) - ACCUM_GRAD.at(i).at(j);

						//+ ACCUM_GRAD.at(i).at(j)*(ACCUM_GRAD.at(i).at(j)-modelRBM.dW_last.at(i).at(j))/sqrt(diagQ);

						//modelRBM.W.at(i).at(j) = modelRBM.W.at(i).at(j) + RATE*modelRBM.dW.at(i).at(j);
						//+ modelRBM.dW.at(i).at(j)*(modelRBM.dW.at(i).at(j)-modelRBM.dW_last.at(i).at(j))/sqrt(diagQ);
                }

        }


}else
{

        #pragma omp parallel for
        for(unsigned i=0; i<num_h_neurons; ++i)
        {
                for(unsigned j=0; j<num_v_neurons; ++j)
                {
                        modelRBM.W.at(i).at(j) = modelRBM.W.at(i).at(j) - ACCUM_GRAD.at(i).at(j);
			
			//if(ITERATION==31)
			//{
				ACCUM_GRAD.at(i).at(j)=0;
			//}
                }
        }
	
}

}
