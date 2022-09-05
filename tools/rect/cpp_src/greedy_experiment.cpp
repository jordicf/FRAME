// (c) Víctor Franco Sanchez 2022
// For the FRAME Framework project.
// This code is licensed under MIT license (see LICENSE.txt on our git for details)

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include "boxfinder_cubic.cpp"
#include "boxfinder_slicing.cpp"
#include "boxfinder_complete.cpp"

#include <chrono>
std::chrono::steady_clock::time_point mark;
double time_sw;

inline void start_stopwatch(){
	time_sw = 0.0;
	mark = std::chrono::steady_clock::now();
}

inline void stop_stopwatch(){
	time_sw += ((float) ( std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - mark).count() )) / 1000000.0;
}

#define MIN_AREA 0
#define MIN_ERROR 1

#define print_box(b) b.x1 << " " << b.y1 << " " << b.x2 << " " << b.y2 << std::endl

unsigned char METHOD = MIN_ERROR;

int nboxes;
double w, h, proportion;
std::vector<cubic::box>    cubic_inputProblem;
std::vector<cubic::box>    cubic_allBoxes;
std::vector<slicing::box>  slicing_inputProblem;
std::vector<slicing::box>  slicing_allBoxes;
std::vector<complete::box> complete_inputProblem;
std::vector<complete::box> complete_allBoxes;


void readFile(const char * filename){
	std::ifstream indata;
	indata.open(filename);
	if(!indata) {
		std::cerr << "Error: File " << filename << " could not be opened!" << std::endl;
		exit(-2);
	}
	
	indata >> w >> h >> nboxes >> proportion;
	if(proportion < 1) METHOD == MIN_AREA;
	cubic_inputProblem    = std::vector<   cubic::box>( nboxes, (   cubic::box) { 0.0, 0.0, 0.0, 0.0, 0.0 } );
	slicing_inputProblem  = std::vector< slicing::box>( nboxes, ( slicing::box) { 0.0, 0.0, 0.0, 0.0, 0.0 } );
	complete_inputProblem = std::vector<complete::box>( nboxes, (complete::box) { 0.0, 0.0, 0.0, 0.0, 0.0 } );
	for(int i = 0; i < nboxes; ++i){
		indata >> cubic_inputProblem[i].x1 >> cubic_inputProblem[i].y1 >> cubic_inputProblem[i].x2 >> cubic_inputProblem[i].y2 >> cubic_inputProblem[i].p;
		slicing_inputProblem[i].x1 = cubic_inputProblem[i].x1;
		slicing_inputProblem[i].x2 = cubic_inputProblem[i].x2;
		slicing_inputProblem[i].y1 = cubic_inputProblem[i].y1;
		slicing_inputProblem[i].y2 = cubic_inputProblem[i].y2;
		slicing_inputProblem[i].p  = cubic_inputProblem[i].p;
		complete_inputProblem[i].x1 = cubic_inputProblem[i].x1;
		complete_inputProblem[i].x2 = cubic_inputProblem[i].x2;
		complete_inputProblem[i].y1 = cubic_inputProblem[i].y1;
		complete_inputProblem[i].y2 = cubic_inputProblem[i].y2;
		complete_inputProblem[i].p  = cubic_inputProblem[i].p;
	}
	
	indata.close();
}


void usage(const char * appname){
	std::cerr << "Usage: " << appname << " [inputfile] [preout]?\n";
	exit(-1);
}


double elapsed(struct timeval & t0, struct timeval & t1){
	long seconds = t1.tv_sec - t0.tv_sec;
    long microseconds = t1.tv_usec - t0.tv_usec;
    double elapsed = seconds + microseconds * 1e-6;
}

int main(int argc, char * argv[]){
	if(argc != 2 and argc != 3){
		usage(argv[0]);
	}
	readFile(argv[1]);
	if(argc == 3){
		std::cout << argv[2] << " ";
	}
	
	struct timeval cub0, cub1, sli0, sli1, com0, com1;
	
	cubic_allBoxes = std::vector<cubic::box>(0);
	start_stopwatch();
	cubic::all_rectangles(cubic_inputProblem, cubic_allBoxes);
	stop_stopwatch();
	std::cout << time_sw << " ";
	
	slicing_allBoxes = std::vector<slicing::box>(0);
	start_stopwatch();
	slicing::all_rectangles(slicing_inputProblem, slicing_allBoxes);
	stop_stopwatch();
	std::cout << time_sw << " ";
	
	complete_allBoxes = std::vector<complete::box>(0);
	start_stopwatch();
	complete::all_rectangles(complete_inputProblem, complete_allBoxes);
	stop_stopwatch();
	std::cout << time_sw << '\n';
}
