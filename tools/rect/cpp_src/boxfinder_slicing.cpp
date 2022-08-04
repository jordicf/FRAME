#ifndef SLICING_CPP
#define SLICING_CPP

#include <algorithm>
#include <vector>
#include <map>

namespace slicing {
	struct triplet {
		double a, b, c;
	};

	struct box {
		double x1, y1, x2, y2, p;
	};

	typedef std::map<struct triplet, struct box> memo;

	
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
	
	
	bool compare(const box & b1, const box & b2){
		if(b1.x2 < b2.x2) return true;
		if(b1.x2 > b2.x2) return false;
		return b1.y2 < b2.y2;
	}
	
	
	box join(box b1, box & b2){
		// Joins two boxes into one
		// Time complexity: O(1)
		
		double a1 = area(b1);
		double a2 = area(b2);
		b1.p = (b1.p * a1) / (a1 + a2) + (b2.p * a2) / (a1 + a2);
		if(b2.x1 < b1.x1) b1.x1 = b2.x1;
		if(b2.y1 < b1.y1) b1.y1 = b2.y1;
		if(b2.x2 > b1.x2) b1.x2 = b2.x2;
		if(b2.y2 > b1.y2) b1.y2 = b2.y2;
		return b1;
	}
	
	
	void insert_rect(const box b1, const bool h, memo & H, memo & V, std::vector<box> & boxes){
		// Inserts every block that has b1 in the bottom right corner
		// Assumes boxes is a slicing floorplan
		// Time complexity: Either O(n) or O(n log n)
		
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
		// Computes every rectangle and returns it in the output vector
		// Assumes B is a slicing floorplan
		// Time complexity: Either O(n²) or O(n² log n)
		
		memo H, V;
		std::sort(B.begin(), B.end(), compare);
		for(auto b : B){
			insert_rect(b, true, H, V, output);
		}
	}
}

#endif /* SLICING_CPP */
