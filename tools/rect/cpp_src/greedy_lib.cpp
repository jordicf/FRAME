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

#if defined(_MSC_VER)
    //  Microsoft 
    #define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
    //  GCC
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#else
    //  do nothing and hope for the best?
    #define EXPORT
    #define IMPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif


unsigned char METHOD = MIN_ERROR;

long nboxes;
double w, h, proportion;
std::vector<boxfinder::box> inputProblem;
std::vector<boxfinder::box> allBoxes;

/**
 * This fixes error LNK1120: unresolved external symbol PyInit_rect_greedy
 */
void PyInit_rect_greedy(void){ }

extern "C" {
	EXPORT boxfinder::box find_best_box(boxfinder::box * inboxes, double ww, double hh, long nb, double pr){
		w = ww;
		h = hh;
		nboxes = nb;
		proportion = pr;

		if(proportion < 1) METHOD = MIN_AREA;
		
		inputProblem = std::vector<boxfinder::box>( nboxes, boxfinder::box { 0.0, 0.0, 0.0, 0.0, 0.0 } );
		for(int i = 0; i < nboxes; ++i){
			inputProblem[i].x1 = inboxes[i].x1;
			inputProblem[i].y1 = inboxes[i].y1;
			inputProblem[i].x2 = inboxes[i].x2;
			inputProblem[i].y2 = inboxes[i].y2;
			inputProblem[i].p = inboxes[i].p;
		}
		
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
		
		return allBoxes[maxi];
	}	
}


