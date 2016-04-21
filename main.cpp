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
	float training_chunk_size=10.0;
	float dt = 2*M_PI/10.0;

	Gnuplot g0;
	Gnuplot g1;

	std::vector<int> f_inputVals, f_targetVals;

	DeepNet DRBM;

        unsigned f_bin_size = 96;
        unsigned t_bin_size = 64;
	unsigned num_samples = 4;
	unsigned num_epochs = 1500;

	unsigned num_RBM_in_chain = 1;
	unsigned num_visible = f_bin_size*t_bin_size;
	unsigned num_hidden = num_visible;
	vector<int> nhidden(num_RBM_in_chain);
	nhidden[0]=500;
	//nhidden[1]=1000; //1536
	//nhidden[2]=192;

	for(unsigned i=0; i<num_RBM_in_chain; ++i)
	{	
		RBM myRBM;		
		DRBM.InitRBM(myRBM, num_visible, nhidden[i]);
		DRBM.BuildChain(myRBM);
	}

	DeepNet myTools;
	string myphrase = "hello my name is";
	Matrix charGrid(f_bin_size,vector<int>(t_bin_size));
	vector<string> myBook;
	myTools.FillStringVector("test.txt", myBook,t_bin_size);

	for(unsigned page=0; page<myBook.size(); ++page)
	{
		myTools.SetupInputs(myBook.at(page),f_bin_size,t_bin_size,charGrid); 	//map 128-char sample of text to charGrid
		myTools.SampleToString(charGrid,f_bin_size,t_bin_size);			//Reconstruct a string from the charGrid
	}

	if(myBook.size()<num_samples*num_epochs)
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

//for(unsigned s=0; s<num_epochs; ++s)
int s=0;
while(s<(num_epochs-num_samples))
{

for(unsigned i=0; i<DRBM.Chain.size(); ++i) ///layer of RBM chain
{

cout << "Training RBM: " << i << endl;

    for(unsigned t=s; t<(s+num_samples); ++t)///training loop of training_chunk_size number of samples for this "class"
    {
	int t_r = t;//t + rand()%num_samples;
        std::vector<float> inputVals(num_visible), targetVals(num_visible);

	Matrix visible(f_bin_size,vector<int>(t_bin_size));
	//Matrix target(f_bin_size, vector<float>(t_bin_size));

	// now bin up this function
	// to create binary matrix 
	// to use as a visible layer


	//Bin2(f_inputVals, f_bin_size, t_bin_size, visible);
	//Bin2(f_targetVals, f_bin_size, t_bin_size, target);
	string sample_string;
	sample_string.resize(2*t_bin_size);
	sample_string = myBook.at(t_r) + myBook.at(t_r+1);
	string sample;
	sample.resize(t_bin_size);
	int shift = rand()%(t_bin_size);
	//cout << "sample_string: " << sample_string << endl;
	int iter=0;
	for(unsigned sh = shift; sh<shift+t_bin_size; ++sh)
	{
		
		sample[iter] = sample_string[sh];
		iter = iter+1;
	}

	//cout << "shifted sample: " << sample << endl;

	myTools.SetupInputs(sample,f_bin_size,t_bin_size,charGrid);  //map 128-char sample of text to charGrid

	//cout << "Raw Input for Layer: " << i << endl;
        //myTools.SampleToString(charGrid,f_bin_size,t_bin_size);             		   //Reconstruct a string from the charGrid


	unsigned track=0;

	for(unsigned j=0; j<f_bin_size; ++j)
	{
		for(unsigned k=0; k<t_bin_size; ++k)
		{
			
			//cout << setiosflags(std::ios::fixed)
          		//<< setprecision(0)
          		//<< setw(1)
          		//<< left
          		//<< visible.at(j).at(k) << " ";
			
			//inputVals.at(track)=(visible.at(j).at(k));

			inputVals.at(track)=(charGrid.at(j).at(k));

			//float targetProb = 1.0/(1.0+exp(-inputVals.at(track)));
			//targetVals.at(track)=(targetProb);
			track++;
			
		}
		
		//cout << endl;
	}	
	

		if(i==0)
		{
		//Calculate Gradient based on targetVals, update weights
		DRBM.GradientDecent(DRBM.Chain.at(i),inputVals);
		
		}else if(i<DRBM.Chain.size() && i>0)
		{
                //Calculate Gradient based on targetVals, update weights
                DRBM.GradientDecent(DRBM.Chain.at(i),DRBM.Chain.at(i-1).Vp);
		}else
		{
                //Calculate Gradient based on targetVals, update weights
                //DRBM.GradientDecent(DRBM.Chain.at(i),DRBM.Chain.at(i-1).Vp);
		}

	        cout << "Input for Layer[" << i << "] on sample " << (t-s) << "/" << (num_samples-1) << endl;
        	myTools.SampleToString(charGrid,f_bin_size,t_bin_size);                            //Reconstruct a string from the charGrid

                unsigned tracker = 0;


                for(unsigned j=0; j<f_bin_size; ++j)
                {
                        for(unsigned k=0; k<t_bin_size; ++k)
                        {
                                //DRBM.getOutputs(DRBM.Chain.at(i));

                                charGrid.at(j).at(k) = DRBM.Chain.at(i).Vs.at(tracker);
                                tracker++;
                        }
                }

                cout << "Reconstruction:	" << endl;
                myTools.SampleToString(charGrid,f_bin_size,t_bin_size);




	}//end of sample loop t

//s=s+num_samples;
//s++;

	
	//#pragma omp flush(DRBM)
		
		ofstream activation_file("/tmp/activation.dat");
		for(unsigned r=i; r<i+1; ++r)
		{
			unsigned tracker = 0;


			for(unsigned j=0; j<f_bin_size; ++j)
			{
			for(unsigned k=0; k<t_bin_size; ++k)
			{
				DRBM.getOutputs(DRBM.Chain.at(r));				
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


}//end of RBM layer loop

cout << "Epoch " <<  s << "/"<< (num_epochs-1) << endl;
s++;

}//end of s loop

/*
///RBM sample loop
std::vector<float> p = resultVals;
std::default_random_engine gen;
std::discrete_distribution<unsigned> dis {p.at(0),p.at(1),p.at(2),p.at(3),p.at(4),p.at(5),p.at(6),p.at(7),p.at(8),p.at(9)};


ofstream function_file("/tmp/function.dat");
for(unsigned j=0; j<10; ++j)
{
	function_file << j*dt << ", " <<  f_targetVals.at(j) << ", " << ", " << f_inputVals.at(j) << endl;
}
function_file.close();
function_file.flush();
*/


/*
ofstream sample_file("sample.dat");
for(unsigned j=0; j<sample_resultVals.size(); ++j)
{
	for(unsigned k=0; k<t_bin_size; ++k)
	{
        	sample_file << resultVals.at(j) << ", " << j << endl;
	}
}
sample_file.close();
sample_file.flush();
*/


//g1.set_grid();
//g1.cmd("set palette");
//g1.cmd("set pm3d map");
//g1.cmd("set view map");
//g1.cmd("splot '/tmp/function.dat' using 1:2:1 with lines lc rgb 'blue' title 'f_target',\
		'/tmp/function.dat' using 1:3:1 with lines lc rgb 'green' title 'raw output rbm',\
		'/tmp/function.dat' using 1:4:1 with lines lc rgb 'red' title 'f_input',\
		'/tmp/sample.dat' using 1:2:3 with pounsigneds lc palette title 'sample inv_RBM output'");

//g1.cmd("plot '/tmp/sample.dat' matrix with image");

	
return 0;
}
