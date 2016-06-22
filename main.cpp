//#include "Genetics.h"
//#include "Net.h"
#include <time.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <iomanip>
//#include <cstdlib>
//#include <math.h>
#include <fstream>
#include "gnuplot_i.hpp"
#include "DeepNet.h"


using namespace std;

typedef vector< vector<int> > Matrix;

float randomInt(unsigned start, unsigned end)
{
//return random unsigned from uniform distribution of unsignedegers between range start and end

random_device rd;
mt19937 gen(rd());
uniform_int_distribution<unsigned> dis(start, end);

return dis(gen);

}

float f(float ptime)
{
	//float f=2*sin(M_PI*ptime/2.0)+0.1*cos(M_PI*ptime*2) + 0.2*sin(M_PI*ptime*4);
	float noise=(float)rand()/(float)RAND_MAX;
	float s=sin(ptime)*sin(ptime);//+noise*noise/10.0;
	//if(f>=0.5){ return 0;}else{return 1;}

return s;
}


void Bin2(vector<float> f, unsigned num_f_bins, unsigned num_t_bins, Matrix &visible)
{

// constructs a visible matrix (num_f_bins x num_t_bins)
// by binning up values within f(t) +/- df(t) and t +/- dt

	for(unsigned i=0; i<num_f_bins; ++i)
	{ 
		for(unsigned j=0; j<num_t_bins; ++j)
		{
			for(unsigned t=0; t<f.size(); ++t)
			{
				float chunk_f = 1.0/(float)num_f_bins;
				float chunk_t = (float)f.size()/(float)num_t_bins;

				if( (f.at(t) >= (chunk_f*i)) && (f.at(t)<(i+1)*chunk_f) )
				{ 
					if( t>=(j*chunk_t) && t<((j+1)*chunk_t) )
					{ visible.at(i).at(j) = 1; }
				}
			}
		}
	}
}

void ScaleVector(vector<float> &Data)
{

        auto max = max_element(begin(Data), end(Data));
        auto min = min_element(begin(Data), end(Data));

//        cout << "Found max to be: " << *max << endl;
//        cout << "Found min to be: " << *min << endl;

	
        for(unsigned i=0; i<Data.size(); ++i)
        {
                Data[i] = ( Data[i] - *min )/( *max - *min );
        }

}


void NewMin(float &current_min, float current_pounsigned)
{
	if(current_pounsigned < current_min)
	{
		current_min=current_pounsigned;
	}
}

void NewMax(float &current_max, float current_pounsigned)
{
	if(current_pounsigned > current_max)
	{
		current_max=current_pounsigned;
	}
}



void ScalePounsigned(float &min, float &max, vector<float> &Data)
{
        for(unsigned i=0; i<Data.size(); ++i)
        {
                NewMin(min, Data.at(i));
                NewMax(max, Data.at(i));
        }

        for(unsigned i=0; i<Data.size(); ++i)
        {
                Data[i] = (Data[i]-min)/(max-min);
        }
}

int main()
{

	srand (time(NULL));

	Gnuplot g0;
	Gnuplot g1;

	std::vector<int> f_inputVals, f_targetVals;

	DeepNet DRBM;

        unsigned f_bin_size = 96;
        unsigned t_bin_size = 64;
	unsigned num_samples = 128;
	unsigned num_iterations = 1600;
	unsigned num_epochs = 320;
	unsigned num_batches = num_iterations/num_epochs;

	unsigned num_RBM_in_chain = 4;
	unsigned num_visible = f_bin_size*t_bin_size;
	int num_hidden = 128;

	DRBM.NUM_SAMPLES=num_samples;
	DRBM.NUM_EPOCHS=num_epochs;



	/// Create topology for network
	/// Each element of nhidden
	/// or nvisible describes the number of 
	/// hidden and visible
	//  neurons for each RBM[i]
	
	vector<int> nhidden(num_RBM_in_chain);
	vector<int> nvisible(num_RBM_in_chain);

	nhidden[0]=128;
	nhidden[1]=126;
	nhidden[2]=124;
	nhidden[3]=120;

	for(int i=0; i<num_RBM_in_chain; ++i)
	{
		//nhidden[n-1]=num_hidden/(2*n);//128/(n+1);

		cout << "num_hidden[" << i << "] :" << nhidden[i] << endl;
		nvisible[i]=num_visible;
	}

	

	/// Build RBM markov chain from 

	for(unsigned i=0; i<num_RBM_in_chain; ++i)
	{	
		RBM myRBM;		
		DRBM.InitRBM(myRBM, num_visible, nhidden[i]);
		DRBM.BuildChain(myRBM);
	}

	DeepNet myTools;
	Matrix charGrid(f_bin_size,vector<int>(t_bin_size));
	Matrix tarGrid(f_bin_size,vector<int>(t_bin_size));

	vector<string> myBook;
	myTools.FillStringVector("test.txt", myBook,t_bin_size);

	for(unsigned page=0; page<myBook.size(); ++page)
	{
		myTools.SetupInputs(myBook.at(page),f_bin_size,t_bin_size,charGrid); 	//map 128-char sample of text to charGrid
		//myTools.SampleToString(charGrid,f_bin_size,t_bin_size);			//Reconstruct a string from the charGrid
	}

	if(myBook.size()<(num_batches*num_epochs))
	{
		cout << "Error: Not enough pages in book:" << myBook.size() << endl;
		cout << "Required: " << num_samples*num_epochs << endl;
	}
	
		cout << "Constructed RBM Chain Status:  " << endl;
		cout << "  Number of Visible:		" << num_visible << endl;
		cout << "  Number of Hidden:		" ; for(unsigned i=0; i<nhidden.size(); ++i){cout << nhidden[i] << ", ";} cout << endl;
		cout << "  Book Size	:		" << myBook.size() << " strings" << endl;
		cout << "  Page Length	:		" << t_bin_size << " characters" << endl;
		cout << "  Map (position x chars):	" << f_bin_size << " x " << t_bin_size << endl;
		cout << "  Number of Samples per Epoch:	" << num_samples << endl;
		cout << "  Number of Epochs:		" << num_epochs << endl;
	

	cin.get();

	float min_in=1e32, min_tar=1e32, min_res=1e32;
	float max_in=-1e32, max_tar=-1e32, max_res=-1e32;

///----------------------------------------------------------------------------------------///

///------------------------------------Gnuplot Setup---------------------------------------///
        g0.cmd("reset");
        g0.cmd("set term gif animate delay 1 size 640,480 optimize");
        g0.cmd("set output \"deep_activation.gif\"");
        g0.cmd("set xlabel \"Neuron\"");
        g0.cmd("set ylabel \"RBM Hidden Layer\"");
	g0.set_xrange(0,t_bin_size);
	g0.set_yrange(0,f_bin_size*num_RBM_in_chain);
        g0.set_grid();
        //g0.cmd("set hidden3d");

	g1.cmd("reset");
        g1.cmd("set term gif animate delay 1 size 640,480 optimize");
        g1.cmd("set output \"outputdist.gif\"");
        g1.cmd("set xlabel \"t\"");
        g1.cmd("set ylabel \"f(t)\"");
        g1.cmd("set title \"Output Distribution for Binned f(t)\"");
	g1.set_xrange(0,t_bin_size);
	g1.set_yrange(0,f_bin_size);

        g1.set_grid();

///----------------------------------------------------------------------------------------///
        system("rm /tmp/activation.dat /tmp/function.dat");



float time=0;

int INTERVAL=1;
bool SUPERVISED=false;
bool UNSUPERVISED=true;

//for(unsigned s=0; s<num_epochs; ++s)
//int s=0;
//while(s<(num_epochs))
//{


DRBM.ERROR=0;
DRBM.TERROR=0;

int s=0;


for(unsigned epoch=0; epoch<num_epochs; ++epoch)
{

for(unsigned i=0; i<DRBM.Chain.size(); ++i) ///layer of RBM chain
{

int b=0;
s = num_batches*epoch;

while(b<(num_batches))
{
    for(unsigned t=(s); t<(s+num_samples); ++t)///training loop over mini-batch samples 
    {

	cout << "Epoch		" << epoch << "/"<< (num_epochs-1) << endl;
	cout << "Batch		" << b << "/" << (num_batches-1) << endl;
	cout << "Mini-Batch	" << (t-s) << "/" << (num_samples-1) << endl;
	cout << "Sample		" << s << "/" << (num_iterations) << endl;


	int t_r = t;//t + rand()%(num_samples);
        std::vector<float> inputVals(num_visible), targetVals(num_visible);

	Matrix visible(f_bin_size,vector<int>(t_bin_size));
	//Matrix target(f_bin_size, vector<float>(t_bin_size));

	// now bin up this function
	// to create binary matrix 
	// to use as a visible layer


	//Bin2(f_inputVals, f_bin_size, t_bin_size, visible);
	//Bin2(f_targetVals, f_bin_size, t_bin_size, target);
	string sample_string,target_string;
	sample_string.resize(4*t_bin_size);
	target_string.resize(4*t_bin_size);
	sample_string = myBook.at(s) + myBook.at(s+1) + myBook.at(s+2) + myBook.at(s+3);
	target_string = myBook.at(s) + myBook.at(s+1) + myBook.at(s+2) + myBook.at(s+3);
	string sample, tarsample;
	sample.resize(t_bin_size);
	tarsample.resize(t_bin_size);
	int shift = t_r;
	//cout << "sample_string: " << sample_string << endl;
	int iter=0;
	for(unsigned sh = shift; sh<shift+t_bin_size; ++sh)
	{	
		sample[iter] = sample_string[sh];
		tarsample[iter] = target_string[sh+t_bin_size];
		iter = iter+1;
	}

	//cout << "shifted sample: " << sample << endl;

	myTools.SetupInputs(sample,f_bin_size,t_bin_size,charGrid);  //map 128-char sample of text to charGrid for input
	myTools.SetupInputs(tarsample,f_bin_size,t_bin_size,tarGrid);  //map 128-char sample of text to charGrid for target
	//cout << "Raw Input for Layer: " << i << endl;
        //myTools.SampleToString(charGrid,f_bin_size,t_bin_size);             		   //Reconstruct a string from the charGrid


	unsigned track=0;

	for(unsigned j=0; j<f_bin_size; ++j)
	{
		for(unsigned k=0; k<t_bin_size; ++k)
		{
			inputVals.at(track)=(charGrid.at(j).at(k));
			targetVals.at(track)=(tarGrid.at(j).at(k));
			track++;
		}
	}	
	

	DRBM.TIME++;			//the current sample
	DRBM.DEPTH = (i+1);		//the current layer of the RBM chain
	DRBM.ITERATION = (t-s)+1;	//a sample number starting at 0
	DRBM.BATCH=b+1;
	//if(s<INTERVAL){SUPERVISED=false; UNSUPERVISED=true;}
	//else if(s>=INTERVAL){SUPERVISED=true; UNSUPERVISED=false;}
	//if(TRAIN_TYPE==0){UNSUPERVISED=true;SUPERVISED=false;}
	//if(TRAIN_TYPE==1){SUPERVISED=true;UNSUPERVISED=false;}

	SUPERVISED=false;
	UNSUPERVISED=true;

	if(UNSUPERVISED)
	{

		if(i>0)
		{
			//DRBM.NormalizeLayer(DRBM.Chain.at(i-1));		
			DRBM.FeedForwardVisible(DRBM.Chain.at(0), inputVals, DRBM.Chain.at(0).biasH);
			//DRBM.StochasticGradientDecent(DRBM.Chain.at(0),inputVals,0);

			for(unsigned k=1; k<i; ++k)
			{
				DRBM.GetHiddenSample(DRBM.Chain.at(k-1));
				DRBM.FeedForwardHidden(DRBM.Chain.at(k), DRBM.Chain.at(k-1).Hs, DRBM.Chain.at(k).biasH);
				DRBM.GetVisibleSample(DRBM.Chain.at(k-1));
				DRBM.FeedForwardVisible(DRBM.Chain.at(k), DRBM.Chain.at(k-1).Vs, DRBM.Chain.at(k).biasV);
				//DRBM.StochasticGradientDecent(DRBM.Chain.at(k),DRBM.Chain.at(k-1).Vs,0);
			}
			//DRBM.NormalizeLayer(DRBM.Chain.at(i-1).Vs);
			DRBM.StochasticGradientDecent(DRBM.Chain.at(i),DRBM.Chain.at(i-1).Vs,0);

		}else
		{
			 DRBM.StochasticGradientDecent(DRBM.Chain.at(0),inputVals,0);
		}


                        cout << "SGD: Input for Layer[" << i << "] on sample " << (t-s) << "/" << (num_samples-1) << endl;
                        myTools.SampleToString(charGrid,f_bin_size,t_bin_size);                            //Reconstruct a string from the charGrid


	}

	if(SUPERVISED)
	{
		//Place holder for backprop
		if(i>0)
		{
			DRBM.FeedForwardVisible(DRBM.Chain.at(0), inputVals, DRBM.Chain.at(0).biasH);

                        for(unsigned k=1; k<i; ++k)
                        {
                                DRBM.GetVisibleSample(DRBM.Chain.at(k-1));
				DRBM.FeedForwardVisible(DRBM.Chain.at(k), DRBM.Chain.at(k-1).Vs, DRBM.Chain.at(k).biasH);
                                //DRBM.StochasticGradientDecent(DRBM.Chain.at(k),DRBM.Chain.at(k-1).Vs,0);
                        }

			//DRBM.NormalizeLayer(DRBM.Chain.at(i-1).Vs);
			DRBM.Backprop(DRBM.Chain.at(i), DRBM.Chain.at(i-1).Vs, targetVals);

		}else
		{
			DRBM.Backprop(DRBM.Chain.at(0), inputVals, targetVals);
		}

			
                        //cout << "BackProp: Input for Layer[" << i << "] on sample " << (t-s) << "/" << (num_samples-1) << endl;
                        //myTools.SampleToString(charGrid,f_bin_size,t_bin_size);
                        cout << "BackProp: Target for Layer[" << i << "] on sample " << (t-s) << "/" << (num_samples-1) << endl;
                        myTools.SampleToString(tarGrid,f_bin_size,t_bin_size);
	}


	

                unsigned tracker = 0;

		//DRBM.GetVisibleSample(DRBM.Chain.at(i));

                for(unsigned j=0; j<f_bin_size; ++j)
                {
                        for(unsigned k=0; k<t_bin_size; ++k)
                        {
                                charGrid.at(j).at(k) = DRBM.Chain.at(i).Vs.at(tracker);
                                tracker++;
                        }
                }


               	cout << "Reconstruction:	" << endl;
               	myTools.SampleToString(charGrid,f_bin_size,t_bin_size);

		
		DRBM.Stats(DRBM.Chain.at(i));

		if((DRBM.ITERATION)==(num_samples))
		{
			if(b==(num_batches-1))
			{
				//DRBM.ResetGradient(DRBM.Chain.at(i));
				//DRBM.NormalizeWeights(DRBM.Chain.at(i));
				//DRBM.NormalizeGradient(DRBM.Chain.at(i));
			}
			//DRBM.NormalizeLayer(DRBM.Chain.at(i));
			//DRBM.NormalizeWeights(DRBM.Chain.at(i)); 
		}


	}//end of sample loop t

        //Last iteration of the sample, but before reset of gradient, lets test the current network on an unknown sample

	std::vector<float> inputVals(num_visible), targetVals(num_visible);

        string sample_string,target_string;
        sample_string.resize(4*t_bin_size);
        target_string.resize(4*t_bin_size);
        sample_string = myBook.at(s) + myBook.at(s+1) + myBook.at(s+2) + myBook.at(s+3);
        target_string = myBook.at(s) + myBook.at(s+1) + myBook.at(s+2) + myBook.at(s+3);
        string sample, tarsample;
        sample.resize(t_bin_size);
        tarsample.resize(t_bin_size);
        //cout << "sample_string: " << sample_string << endl;
        int iter=0;
        for(unsigned sh = 0; sh<t_bin_size; ++sh)
        {

                sample[iter] = sample_string[sh]; //test inputval
                tarsample[iter] = target_string[sh+t_bin_size];	//inputval
                iter = iter+1;
        }

        //cout << "shifted sample: " << sample << endl;

        myTools.SetupInputs(sample,f_bin_size,t_bin_size,charGrid);  //map 128-char sample of text to charGrid for input
        myTools.SetupInputs(tarsample,f_bin_size,t_bin_size,tarGrid);  //map 128-char sample of text to charGrid for target

	
        unsigned track=0;

        for(unsigned j=0; j<f_bin_size; ++j)
        {
                for(unsigned k=0; k<t_bin_size; ++k)
                {
                        inputVals.at(track)=(charGrid.at(j).at(k));
                        targetVals.at(track)=(tarGrid.at(j).at(k));
                        track++;
                }
        }

	
/*
	cout << "/////////////////////////////////////////////////" << endl;
        cout << "///	Training Sample Input   : " << endl;
	cout << "///	";
	myTools.SampleToString(charGrid,f_bin_size,t_bin_size);

	
                unsigned tracker = 0;
                for(unsigned j=0; j<f_bin_size; ++j)
                {
                        for(unsigned k=0; k<t_bin_size; ++k)
                        {
                                //DRBM.getOutputs(DRBM.Chain.at(i));
				
			        //FeedForwardVisible(RBM &myRBM, vector<float> inputVals, vector<float> biasH)
        			DRBM.FeedForwardVisible(DRBM.Chain.at(0), inputVals, DRBM.Chain.at(0).biasH);

        			for(unsigned k=0; k<(DRBM.Chain.size()); ++k)
        			{
                			DRBM.GetVisibleSample(DRBM.Chain.at(k));
        			}


                                charGrid.at(j).at(k) = DRBM.Chain.back().Vs.at(tracker);
                                tracker++;
                        }
                }



	cout << "///	Training Sample Output	: " << endl;
	cout << "///	";
	myTools.SampleToString(charGrid,f_bin_size,t_bin_size);
	cout << "/////////////////////////////////////////////////" << endl;
	cout << endl;
	cout << endl;
*/	

//	DRBM.ResetGradient(DRBM.Chain.back());


///---------------------------------Plot Generation-------------------------------------------///





///---------------------------------Plot Generation-------------------------------------------///

			
		ofstream activation_file("/tmp/activation.dat");
		for(unsigned r=i; r<DRBM.Chain.size(); ++r)
		{
			unsigned tracker = 0;


			for(unsigned j=0; j<f_bin_size; ++j)
			{
			for(unsigned k=0; k<t_bin_size; ++k)
			{
                        		activation_file << DRBM.Chain.at(r).Vs.at(tracker) << ", ";
				
				charGrid.at(j).at(k) = DRBM.Chain.at(r).Vs.at(tracker);
				tracker++;
			}
			activation_file << endl;
			}
                	//activation_file << endl;
			//cout << "Finished Sample of Layer: " << r << " " << endl;

			//myTools.SampleToString(charGrid,f_bin_size,t_bin_size);
		}
		activation_file.close();
		activation_file.flush();
		


		ofstream sample_file("/tmp/sample.dat");
		unsigned tracker2=0;
		for(unsigned j=0; j<f_bin_size; ++j)
		{
		        for(unsigned k=0; k<t_bin_size; ++k)
        		{
		                sample_file << DRBM.Chain.back().Vp.at(tracker2) << ", ";
				tracker2++;
        		}
			sample_file << endl;
		}
		sample_file.close();
		sample_file.flush();
		


        string settime;
        string time_string;
        time_string = to_string(s);
        settime = "set title \"Layer " + to_string(i) + ", Time = " + time_string + " \" ";

        string splotcmd1, splotcmd2, field;
        string splotcmd;
	string mplotcmd;
	string mfield;
        splotcmd1 = "plot '";
        field = "/tmp/activation.dat";
        splotcmd2 = "' matrix with image";
        splotcmd = splotcmd1 + field + splotcmd2;

	mfield="/tmp/sample.dat";
	mplotcmd = splotcmd1 + mfield + splotcmd2;

	//g0.cmd("set palette defined ( 0 \"black\", 1 \"white\")");
	g0.cmd("set cbrange[0:1]");
        g0.cmd(settime);
        g0.cmd(splotcmd);
        g0.cmd("reread");

	//g1.cmd("set palette defined ( 0 \"black\", 1 \"white\")");
	g1.cmd("set cbrange[0:1]");
        g1.cmd(settime);
        g1.cmd(mplotcmd);
        g1.cmd("reread");

///---------------------------------End Plot Generation----------------------------------------///

//s++;
b++;
s++;
}//end of sample s loop


}//end of RBM layer loop

//s++;

}//end of epoch (full update of all weights in network)


	
return 0;
}
