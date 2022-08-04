#include <iostream>
#include <fstream>
#include <vector>

#define MIN_AREA 0
#define MIN_ERROR 1


unsigned char METHOD = MIN_ERROR;

struct box {
	double x1, y1, x2, y2;
	double p;
};

double w, h;
int nboxes;
double proportion;
struct box * inputProblem;
std::vector<struct box> allBoxes;


void readFile(const char * filename){
	std::ifstream indata;
	indata.open(filename);
	if(!indata) {
		std::cerr << "Error: File " << filename << " could not be opened!" << std::endl;
		exit(-2);
	}
	
	indata >> w >> h >> nboxes >> proportion;
	if(proportion < 1) METHOD = MIN_AREA;
	inputProblem = (struct box *) malloc( nboxes * sizeof(struct box) );
	for(int i = 0; i < nboxes; ++i){
		indata >> inputProblem[i].x1 >> inputProblem[i].y1 >> inputProblem[i].x2 >> inputProblem[i].y2 >> inputProblem[i].p;
	}
	
	indata.close();
}

bool isOutside(const struct box & b1, const struct box & b2){
	// Returns true if box2 is outside box1
	return b2.x1 >= b1.x2 or b2.x2 <= b1.x1 or b2.y1 >= b1.y2 or b2.y2 <= b1.y1;
}

bool isInside(const struct box & b1, const struct box & b2){
	// Returns true if box2 is completely enclosed inside of box1
	return b2.x2 <= b1.x2 and b2.x1 >= b1.x1 and b2.y2 <= b1.y2 and b2.y1 >= b1.y1;
}

double area(const struct box & b){
	return (double) ( (b.x2 - b.x1) * (b.y2 - b.y1) );
}

bool isBox(const int i1, const int i2){
	if(inputProblem[i2].x1 < inputProblem[i1].x1) return false;
	if(inputProblem[i2].x2 < inputProblem[i1].x2) return false;
	if(inputProblem[i2].y1 < inputProblem[i1].y1) return false;
	if(inputProblem[i2].y2 < inputProblem[i1].y2) return false;
	
	struct box tmp;
	tmp.x1 = inputProblem[i1].x1;
	tmp.y1 = inputProblem[i1].y1;
	tmp.x2 = inputProblem[i2].x2;
	tmp.y2 = inputProblem[i2].y2;
	tmp.p = 0;
	
	double tmpa = area(tmp);
	
	for(int i = 0; i < nboxes; ++i){
		if(isOutside(tmp, inputProblem[i])) continue;
		if(not isInside(tmp, inputProblem[i])) return false;
		tmp.p += inputProblem[i].p * area(inputProblem[i]) / tmpa;
	}
	allBoxes.push_back(tmp);
	return true;
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
	
	allBoxes = std::vector<struct box>(0);
	
	for(int i = 0; i < nboxes; ++i){
		for(int j = 0; j < nboxes; ++j) isBox(i, j);
	}
	
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
		MyFile << allBoxes[maxi].x1 << " " << allBoxes[maxi].y1 << " " << allBoxes[maxi].x2 << " " << allBoxes[maxi].y2 << " " << allBoxes[maxi].p << std::endl;
		MyFile.close();
	} else {
		std::cout << allBoxes[maxi].x1 << " " << allBoxes[maxi].y1 << " " << allBoxes[maxi].x2 << " " << allBoxes[maxi].y2 << " " << allBoxes[maxi].p << std::endl;
	}
}
