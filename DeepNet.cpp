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
	std::uniform_real_distribution<float> distribution(-0.5,0.5);

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

	float norm=0;

	///Fill weight matrix with initial values
	///Initialize Hidden and Visible layers to 0
	for(unsigned i=0; i<num_h_neurons; ++i)
	{
		//float start_weight = (float)rand()/(float)RAND_MAX;
		//norm += start_weight;

		myRBM.W.at(i).resize(num_v_neurons);
		myRBM.dW.at(i).resize(num_v_neurons);

		for(unsigned j=0; j<num_v_neurons; ++j)
		{
			//float start_weight = (float)rand()/((float)RAND_MAX) ;
			float start_weight = distribution(generator);
			
	                norm += start_weight;
			myRBM.W.at(i).at(j) = start_weight;
			myRBM.dW.at(i).at(j) = 0;

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
                        myRBM.W.at(i).at(j) = myRBM.W.at(i).at(j);
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
cout << "	|---------------------------------------------------------------| " << endl;
cout << "	|>>>>	" <<  f << "	<<<<" << endl;
cout << "	|---------------------------------------------------------------| " << endl;
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
        return 1.0/(1.0+exp(-x));
}




void DeepNet::GradientDecent(RBM &myRBM, vector<float> targetVals)
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

modelRBM.Vs=targetVals;

int k=0;
float K_MAX=3;

	float norm=0;
	float error = 0;
	float myerror=0;
	float CUTOFF=1;

	while(k<K_MAX)
	{
		///HiddenLayer Prob given inputs
		FeedForwardVisible( modelRBM, modelRBM.Vp, modelRBM.biasH); 		//update sample of h_model from q_model
		
		//UpdateBias(modelRBM,dataRBM);
		WeightGradient(modelRBM,dataRBM);
		//UpdateWeights(modelRBM,modelRBM,norm);

		///VisibleLayer Prob given sample hiddens
		FeedForwardHidden( modelRBM, modelRBM.Hs, modelRBM.biasV); 		//update p_model = f(bias_v + W h_model)

		//UpdateBias(modelRBM,dataRBM);
		WeightGradient(modelRBM,dataRBM);
		//UpdateWeights(modelRBM,modelRBM,norm);
		myerror=0;
		for(unsigned i=0; i<myRBM.num_v; ++i)
		{
			//float val=0;
			error += dataRBM.Vs[i] - modelRBM.V[i];
			//val = (dataRBM.Vs[i] - modelRBM.V[i]);
			//myerror += (dataRBM.Vs[i] - modelRBM.V[i]);
		}

		//cout << "error: " << error <<", ";

	        //FeedForwardVisible(modelRBM,modelRBM.Vp,myRBM.Vp);		//update q_model = f(bias_hid + W p_model)
	        //FeedForwardHidden(modelRBM,modelRBM.Hp,myRBM.Hp);		
		//if(abs(myerror)<CUTOFF){CUTOFF=abs(myerror);}
		//else{	//k=k+1;	}
		++k;
	}

FeedForwardVisible( modelRBM, modelRBM.Vp, modelRBM.biasH); 
WeightGradient(modelRBM,dataRBM);
FeedForwardHidden( modelRBM, modelRBM.Hp, modelRBM.biasV);
WeightGradient(modelRBM,dataRBM);

	UpdateBias(modelRBM,dataRBM);
	UpdateWeights(modelRBM,dataRBM,norm,(K_MAX+2));
	cout << "Norm: " << norm << endl;
	
	//myRBM=modelRBM;
	myRBM.W=modelRBM.W;
	myRBM.Vs=modelRBM.Vs;

	error = error/K_MAX;
	cout << "Average Error: " << error << endl;


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

		//#pragma omp simd reduction(+:sum)
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

		//cout << "FFVisible: Average samp, prob: " << samp << ", " << prob << endl;
		avg_prob += prob/(float)size_h;

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

	float avg_prob =0;
	#pragma omp parallel for
        for(unsigned j=0; j<size_v; ++j)
        {
                float sum=0;
		float avg_hid_val=0;

		//#pragma omp simd reduction(+:sum)
                for(unsigned i=0; i<size_h; ++i)
                {
                        sum += (inputHiddens[i])*myRBM.W[i][j];
			//sum += (myRBM.Hp[i])*myRBM.W[i][j];
                }

	
                /// Sample the resulting probability
                /// and assign 1 or 0 to H depending
                /// on the result. Use the probability
                /// to assign to Hp

                float prob = transportFunction(biasV[j]+sum);

		myRBM.Vp[j] = prob;

                float samp = (float)rand()/(float)RAND_MAX;

		//cout << "FFHidden: samp, prob: " << samp << ", " << prob << endl;
		avg_prob += prob/(float)size_v;
		

                if(prob>samp)	{ myRBM.Vs[j]=1; }
                else 		{ myRBM.Vs[j]=0; }

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
		//#pragma omp parallel for reduction(+:sum)
		//#pragma omp simd reduction(+:sum)
        	for(unsigned i=0; i<size_h; ++i)
        	{
                	//sum += (myRBM.Hp[j]-dataHp[j])*myRBM.W[j].back();
                	sum += (myRBM.Hp[i]-dataRBM.H[i])*myRBM.W[i][j];
			//myRBM.avgH[i] = avg_hid_val;
        	}
		
		//myRBM.biasV[j] += (myRBM.Vp[j]-dataRBM.Vp[j]);
		myRBM.biasV[j] = log(avg_vis_val/(1-avg_vis_val))-q*sum;
		//avg_vbias_val += myRBM.biasV[j]/((float)size_v);
	}

	//myRBM.V.back() = log(avg_vis_val/(1-avg_vis_val))-q*sum;
        //myRBM.Vs.back() = log(avg_vis_val/(1-avg_vis_val))-q*sum;
	//myRBM.Vp.back() = log(avg_vis_val/(1-avg_vis_val))-q*sum;
	//myRBM.Vp.back() = log(q/(1-q));



        //cout << "FFHidden: Average prob of visible neuron: " << avg_prob << endl;
        //cout << "Update Bias: Avg Visible Bias Prob: " << avg_vbias_val << ", Avg Visible Val: " << avg_vis_val << endl;
	

        //update hidden bias
	float num_hid = ((float)size_h);
	//q=1.0/num_hid;
	float avg_hbias_val=0;

	#pragma omp parallel for
	for(unsigned i=0; i<size_h; ++i)
	{
		float sum=0;
		//#pragma omp simd reduction(+:sum)
        	for(unsigned j=0; j<size_v; ++j)
        	{
                	//sum += (myRBM.Vp[j])*myRBM.W.back()[j];
                	sum += (myRBM.Vp[j]-dataRBM.Vp[j])*myRBM.W[i][j];
			//myRBM.avgV[j] = avg_vis_val;
        	}
		//myRBM.biasH[i] += (myRBM.Hp[i]-dataRBM.H[i]);
		myRBM.biasH[i] = log(q/(1-q))-avg_vis_val*sum;
		//avg_hbias_val += myRBM.biasH[i]/((float)size_h);

	}

	//cout << "Update Bias: Avg Hidden Bias Prob: " << avg_hbias_val << ", Avg Hidden Val: " << avg_hid_val << endl;

	

}

void DeepNet::getOutputs(RBM &myRBM)
{
	//myRBM.V = myRBM.Vs;
	//myRBM.H = myRBM.Hs;
}


void DeepNet::WeightGradient(RBM &modelRBM,RBM dataRBM)
{
        unsigned num_h_neurons = modelRBM.num_h;
        unsigned num_v_neurons = modelRBM.num_v;

        #pragma omp parallel for
        for(unsigned i=0; i<num_h_neurons; ++i)
        {
                for(unsigned j=0; j<num_v_neurons; ++j)
                {
                        modelRBM.dW.at(i).at(j) =  modelRBM.dW.at(i).at(j) + (dataRBM.H[i]*dataRBM.V[j]-modelRBM.Hs[i]*modelRBM.Vp[j]);
                }
        }
}

void DeepNet::UpdateWeights(RBM &modelRBM, RBM dataRBM, float &L2, float K_MAX)
{

	unsigned num_h_neurons = modelRBM.num_h;
	unsigned num_v_neurons = modelRBM.num_v;

	vector<float> norm(num_h_neurons);

	float normal=0;

	/*
	//L2=0;
        //cout << "Norm of Updated sample matrix: ";
        for(unsigned i=0; i<num_h_neurons; ++i)
        {
                norm[i]=0;
                for(unsigned j=0; j<num_v_neurons; ++j)
                {
                        L2 += modelRBM.W.at(i).at(j) * modelRBM.W.at(i).at(j)/2.0;//
                        //epsilon(<V[i]H[j]>_data - <V[i]H[j]>_model)
                }
        }
	*/

	//cout << "Norm of Updated sample matrix: ";
	#pragma omp parallel for
        for(unsigned i=0; i<num_h_neurons; ++i)
        {
		//norm[i]=0;
		
                for(unsigned j=0; j<num_v_neurons; ++j)
                {
 	                modelRBM.W.at(i).at(j) = modelRBM.W.at(i).at(j) + RATE*modelRBM.dW.at(i).at(j)/(K_MAX);// - 0.000001*L2;//
			//epsilon(<V[i]H[j]>_data - <V[i]H[j]>_model)

                        //normal += modelRBM.W.at(i).at(j);
			//norm[i] += modelRBM.W.at(i).at(j);
                }

		//cout << norm[i] << ", ";
        }
/*
	for(unsigned j=0; j<num_v_neurons; ++j)
        {
		modelRBM.biasV[j] += modelRBM.Vp[j]-dataRBM.V[j];	
        }

        for(unsigned i=0; i<num_h_neurons; ++i)
        {
                modelRBM.biasH[i] += modelRBM.Hp[i]-dataRBM.H[i];
        }
*/

	
}
