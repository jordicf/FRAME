#ifndef CUBIC_CPP
#define CUBIC_CPP

#include <iostream>
#include <fstream>
#include <vector>

namespace cubic {
	struct box {
		double x1, y1, x2, y2;
		double p;
	};


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
	
	
	bool isBox(const int i1, const int i2, const std::vector<struct box> & inputProblem, std::vector<struct box> & allBoxes){
		// Determines whether the two boxes i1 and i2 form a box
		// And if they do, the box, together with its proportion is added
		// to allBoxes
		// Time complexity: O(n)
		
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
		
		for(int i = 0; i < inputProblem.size(); ++i){
			if(isOutside(tmp, inputProblem[i])) continue;
			if(not isInside(tmp, inputProblem[i])) return false;
			tmp.p += inputProblem[i].p * area(inputProblem[i]) / tmpa;
		}
		allBoxes.push_back(tmp);
		return true;
	}
	
	
	void all_rectangles(std::vector<box> & B, std::vector<box> & output){
		// Computes every rectangle and returns it in the output vector
		// Time complexity: O(n³)
		
		for(int i = 0; i < B.size(); ++i){
			for(int j = 0; j < B.size(); ++j) isBox(i, j, B, output);
		}
	}
}

#endif /* CUBIC_CPP */
