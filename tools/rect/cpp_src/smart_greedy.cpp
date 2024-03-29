// (c) Víctor Franco Sanchez 2022
// For the FRAME Project.
// Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

#include <iostream>
#include <fstream>
#include <vector>
#include "boxfinder_complete.cpp"
#define boxfinder complete

#define MIN_AREA 0
#define MIN_ERROR 1

#define print_box(b) b.x1 << " " << b.y1 << " " << b.x2 << " " << b.y2 << " " << b.p << std::endl

unsigned char METHOD = MIN_ERROR;

int nboxes;
double w, h, proportion;
std::vector<boxfinder::box> inputProblem;
std::vector<boxfinder::box> allBoxes;


void readFile(const char * filename){
	std::ifstream indata;
	indata.open(filename);
	if(!indata) {
		std::cerr << "Error: File " << filename << " could not be opened!" << std::endl;
		exit(-2);
	}
	
	indata >> w >> h >> nboxes >> proportion;
	if(proportion < 1) METHOD == MIN_AREA;
	inputProblem = std::vector<boxfinder::box>( nboxes, (boxfinder::box) { 0.0, 0.0, 0.0, 0.0, 0.0 } );
	for(int i = 0; i < nboxes; ++i){
		indata >> inputProblem[i].x1 >> inputProblem[i].y1 >> inputProblem[i].x2 >> inputProblem[i].y2 >> inputProblem[i].p;
	}
	
	indata.close();
}


void usage(const char * appname){
	std::cerr << "Usage: " << appname << " [inputfile] [outputfile]?\n";
	exit(-1);
}


int main(int argc, char * argv[]){
	if(argc != 3 && argc != 2){
		usage(argv[0]);
	}
	readFile(argv[1]);
	allBoxes = std::vector<boxfinder::box>(0);
	boxfinder::all_rectangles(inputProblem, allBoxes);
	
	double totalArea;
	if( METHOD == MIN_AREA ){
		totalArea = 0.0;
		for(int i = 0; i < nboxes; ++i){
			totalArea += area(inputProblem[i]) * inputProblem[i].p;
		}
		totalArea *= proportion;
	}
	
	double maxP = - 1.0;
	int maxi = -1;
	for(int i = 0; i < allBoxes.size(); ++i){
		if( METHOD == MIN_AREA ){
			if( area(allBoxes[i]) < totalArea ) continue;
			if( allBoxes[i].p > maxP) {
				maxP = allBoxes[i].p;
				maxi = i;
			}
		}
		
		if( METHOD == MIN_ERROR ){
			double newP = (proportion * allBoxes[i].p - 1.0) * area(allBoxes[i]);
			if(newP > maxP){
				maxP = newP;
				maxi = i;
			}
		}
	}
	
	if(argc == 3){
		std::ofstream MyFile(argv[2]);
		MyFile << print_box(allBoxes[maxi]);
		MyFile.close();
	} else {
		std::cout << print_box(allBoxes[maxi]);
	}
}
