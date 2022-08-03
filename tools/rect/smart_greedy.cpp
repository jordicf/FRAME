#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>

#define MIN_AREA 0
#define MIN_ERROR 1


unsigned char METHOD = MIN_ERROR;

struct triplet {
	double a, b, c;
};

struct box {
	double x1, y1, x2, y2, p;
};

typedef std::map<struct triplet, struct box> memo;

int nboxes;
double w, h, proportion;
std::vector<struct box> inputProblem;
std::vector<struct box> allBoxes;


operator< (const triplet & t1, const triplet & t2){
	if(t1.a < t2.a) return true;
	if(t1.a > t2.a) return false;
	if(t1.b < t2.b) return true;
	if(t1.b > t2.b) return false;
	return t1.c < t2.c;
}

double area(const struct box & b){
	return (b.x2 - b.x1) * (b.y2 - b.y1);
}

void readFile(const char * filename){
	std::ifstream indata;
	indata.open(filename);
	if(!indata) {
		std::cerr << "Error: File " << filename << " could not be opened!" << std::endl;
		exit(-2);
	}
	
	indata >> w >> h >> nboxes >> proportion;
	if(proportion < 1) METHOD == MIN_AREA;
	inputProblem = std::vector<struct box>( nboxes, (struct box) { 0.0, 0.0, 0.0, 0.0, 0.0 } );
	for(int i = 0; i < nboxes; ++i){
		indata >> inputProblem[i].x1 >> inputProblem[i].y1 >> inputProblem[i].x2 >> inputProblem[i].y2 >> inputProblem[i].p;
	}
	
	indata.close();
}

bool compare(box b1, box b2){
	if(b1.x2 < b2.x2) return true;
	if(b1.x2 > b2.x2) return false;
	return b1.y2 < b2.y2;
}

box join(box b1, box b2){
	double a1 = area(b1);
	double a2 = area(b2);
	b1.p = (b1.p * a1) / (a1 + a2) + (b2.p * a2) / (a1 + a2);
	if(b2.x1 < b1.x1) b1.x1 = b2.x1;
	if(b2.y1 < b1.y1) b1.y1 = b2.y1;
	if(b2.x2 > b1.x2) b1.x2 = b2.x2;
	if(b2.y2 > b1.y2) b1.y2 = b2.y2;
	return b1;
}

void insert_rect(box b1, bool h, memo & H, memo & V, std::vector<box> & boxes){
	boxes.push_back(b1);
	
	double x1 = b1.x1, y1 = b1.y1, x2 = b1.x2, y2 = b1.y2;
	memo::iterator iH = H.find( (triplet) {x2, y1, y2} );
	if(iH == H.end() or x1 > iH->second.x1){
		H[(triplet) {x2, y1, y2}] = b1;
	}
	memo::iterator iV = V.find( (triplet) {y2, x1, x2} );
	if(iV == V.end() or y1 > iV->second.y1){
		V[(triplet) {y2, x1, x2}] = b1;
	}
	
	bool wall = false;
	if(h){
		memo::iterator iH = H.find( (triplet) {x1, y1, y2} );
		if(iH != H.end()){
			insert_rect(join(b1, iH->second), true, H, V, boxes);
		} else {
			wall = true;
		}
	}
	{
		memo::iterator iV = V.find( (triplet) {y1, x1, x2} );
		if(iV != V.end()){
			insert_rect(join(b1, iV->second), wall, H, V, boxes);
		}
	}
}

void all_rectangles(std::vector<box> & B, std::vector<box> & output){
	memo H, V;
	std::sort(B.begin(), B.end(), compare);
	for(auto b : B){
		insert_rect(b, true, H, V, output);
	}
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
	all_rectangles(inputProblem, allBoxes);
	
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


