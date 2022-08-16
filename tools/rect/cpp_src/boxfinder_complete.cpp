#ifndef COMPLETE_CPP
#define COMPLETE_CPP

// For better compatibility
#define and &&
#define or ||

#include <algorithm>
#include <vector>
#include <set> 
#include <map>
#include <iostream>

namespace complete {
	struct quad { 
		double x1, y1, x2, y2;
	};
	
	typedef struct quad line_segment;
	typedef struct quad pless_box;
	
	struct box {
		double x1, y1, x2, y2, p;
	};
	
	struct vpoint {
		double x, y;
	};
	
	struct hpoint {
		double x, y;
	};
	
	typedef struct vpoint point;
	
	typedef std::set<line_segment> segments;
	typedef std::set< struct vpoint > vset;
	typedef std::set< struct hpoint > hset;
	typedef std::set< point > pset;
	typedef std::map< point, double > integral;
	
	
	bool operator< (const quad & t1, const quad & t2){
		if(t1.x1 < t2.x1) return true;
		if(t1.x1 > t2.x1) return false;
		if(t1.y1 < t2.y1) return true;
		if(t1.y1 > t2.y1) return false;
		if(t1.x2 < t2.x2) return true;
		if(t1.x2 > t2.x2) return false;
		return t1.y2 < t2.y2;
	}
	
	
	bool operator< (const vpoint & t1, const vpoint & t2){
		if(t1.x < t2.x) return true;
		if(t1.x > t2.x) return false;
		return t1.y < t2.y;
	}
	
	
	bool operator< (const hpoint & t1, const hpoint & t2){
		if(t1.y < t2.y) return true;
		if(t1.y > t2.y) return false;
		return t1.x < t2.x;
	}
	
	
	double area(const struct box & b){
		return (b.x2 - b.x1) * (b.y2 - b.y1);
	}
	
	
	void report_line_segment(line_segment ls, segments & S){
		S.insert( ls );
		//std::cerr << "REPORT: (" << ls.x1 << ", " << ls.y1 << ") - (" << ls.x2 << ", " << ls.y2 << ")\n";
	}
	
	
	void horizontal_iteration(std::vector<hpoint> & Blocks, const std::vector<hpoint> & T, const std::vector<hpoint> & B, segments & S){
		// Computes every horizontal line segment at a particular y coordinate
		// Time complexity: Either O(n²) or O(n² log n)
		// (Amortized: Either O(n) or O(n log n) per vertex)
		
		std::vector<hpoint> B2(0);
		{
			int ib = 0, ir = 0;
			while(ib < Blocks.size()){
				if( 2 * ir < B.size() and B[2 * ir].x == Blocks[ib].x){
					++ib;
				} else if(2 * ir < B.size() and B[2 * ir].x < Blocks[ib].x){
					++ir;
				} else {
					B2.push_back( Blocks[ib] );
					++ib;
				}
			}
		}
		
		
		std::vector<hpoint> TB(0);
		{
			int it = 0, ib = 0;
			while(it < T.size() or ib < B.size()){
				if(it < T.size() and (ib >= B.size() or T[it].x < B[ib].x)){
					TB.push_back( T[it] );
					++it;
				} else {
					TB.push_back( B[ib] );
					++ib;
				}
			}
		}
		
		{
			int ib = 0, itb1 = 0;
			while(itb1 < TB.size()){
				while(itb1 < TB.size() and ib < B2.size() and B2[ib].x <= TB[itb1].x){
					if(B2[ib].y <= TB[itb1].x) ++ib;
					else while(itb1 < TB.size() and B2[ib].y > TB[itb1].x) ++itb1;
				}
				double xcap = 1e200, lastx1 = 1e200, lastx2 = 1e200;
				if(ib < B2.size()){
					xcap = B2[ib].x;
				}
				
				while(itb1 < TB.size() and TB[itb1].x <= xcap){
					if(TB[itb1].x == lastx1) {
						++itb1;
						continue;
					}
					lastx1 = TB[itb1].x;
					lastx2 = 1e200;
					for(int itb2 = itb1 + 1; itb2 < TB.size() and TB[itb2].x <= xcap; ++itb2){
						if(TB[itb1].x == TB[itb2].x or TB[itb2].x == lastx2) continue;
						lastx2 = TB[itb2].x;
						report_line_segment( complete::line_segment { TB[itb1].x, TB[itb1].y, TB[itb2].x, TB[itb2].y }, S );
					}
					++itb1;
				}
			}
		}
		
		std::vector<hpoint> B3(0);
		{
			int ib = 0, it = 0;
			while(ib < B2.size() or 2 * it < T.size()){
				if(ib < B2.size() and (2 * it >= T.size() or B2[ib].x < T[2*it].x)){
					B3.push_back( B2[ib] );
					++ib;
				} else {
					B3.push_back( struct complete::hpoint { T[2*it].x, T[2*it+1].x } );
					++it;
				}
			}
		}
		
		while(Blocks.size() > 0) Blocks.pop_back();
		for(auto b : B3) Blocks.push_back(b);
	}
	
	void vertical_iteration(std::vector<vpoint> & Blocks, const std::vector<vpoint> & L, const std::vector<vpoint> & R, segments & S){
		// Computes every vertical line segment at a particular x coordinate
		// Time complexity: Either O(n²) or O(n² log n)
		// (Amortized: Either O(n) or O(n log n) per vertex)
		
		std::vector<vpoint> B2(0);
		{
			int ib = 0, ir = 0;
			while(ib < Blocks.size()){
				if(2 * ir < R.size() and R[2 * ir].y == Blocks[ib].x){
					++ib;
				} else if(2 * ir < R.size() and R[2 * ir].y < Blocks[ib].x){
					++ir;
				} else {
					B2.push_back( Blocks[ib] );
					++ib;
				}
			}
		}
		
		std::vector<vpoint> LR(0);
		{
			int il = 0, ir = 0;
			while(il < L.size() or ir < R.size()){
				if(il < L.size() and (ir >= R.size() or L[il].y < R[ir].y)){
					LR.push_back( L[il] );
					++il;
				} else {
					LR.push_back( R[ir] );
					++ir;
				}
			}
		}
		
		{
			int ib = 0, ilr1 = 0;
			while(ilr1 < LR.size()){
				while(ilr1 < LR.size() and ib < B2.size() and B2[ib].x <= LR[ilr1].y){
					if(B2[ib].y <= LR[ilr1].y) ++ib;
					else while(ilr1 < LR.size() and B2[ib].y > LR[ilr1].y) ++ilr1;
				}
				double ycap = 1e200, lasty1 = 1e200, lasty2 = 1e200;
				if(ib < B2.size()){
					ycap = B2[ib].x;
				}
				
				while(ilr1 < LR.size() and LR[ilr1].y <= ycap){
					if(LR[ilr1].y == lasty1) {
						++ilr1;
						continue;
					}
					lasty1 = LR[ilr1].y;
					lasty2 = 1e200;
					for(int ilr2 = ilr1 + 1; ilr2 < LR.size() and LR[ilr2].y <= ycap; ++ilr2){
						if(LR[ilr1].y == LR[ilr2].y or LR[ilr2].y == lasty2) continue;
						lasty2 = LR[ilr2].y;
						report_line_segment( complete::line_segment { LR[ilr1].x, LR[ilr1].y, LR[ilr2].x, LR[ilr2].y }, S );
					}
					++ilr1;
				}
			}
		}
		
		std::vector<vpoint> B3(0);
		{
			int ib = 0, il = 0;
			while(ib < B2.size() or 2 * il < L.size()){
				if(ib < B2.size() and (2 * il >= L.size() or B2[ib].x < L[2*il].y)){
					B3.push_back( B2[ib] );
					++ib;
				} else {
					B3.push_back( struct complete::vpoint { L[2*il].y, L[2*il+1].y } );
					++il;
				}
			}
		}
		
		while(Blocks.size() > 0) Blocks.pop_back();
		for(auto b : B3) Blocks.push_back(b);
	}
	
	
	void every_vertical_line_segment(const std::vector<struct box> & B, segments & S){
		// Computes every vertical line segment that begins and ends at a vertex
		// Time complexity: Either O(n²) or O(n² log n) (Amortized)
		
		std::vector<struct vpoint> LSet, RSet;
		for(auto b : B){
			LSet.push_back( struct complete::vpoint { b.x1, b.y1 } );
			LSet.push_back( struct complete::vpoint { b.x1, b.y2 } );
			RSet.push_back( struct complete::vpoint { b.x2, b.y1 } );
			RSet.push_back( struct complete::vpoint { b.x2, b.y2 } );
		}
		std::sort(LSet.begin(), LSet.end());
		std::sort(RSet.begin(), RSet.end());
		
		std::vector<struct vpoint> Blocks(0);
		int Lit = 0, Rit = 0;
		while(Lit < LSet.size() or Rit < RSet.size()){
			std::vector<struct vpoint> L(0), R(0);
			double min = 1e200;
			if(Lit < LSet.size()) min = LSet[Lit].x;
			if(Rit < RSet.size() and RSet[Rit].x < min) min = RSet[Rit].x;
			while(Lit < LSet.size() and LSet[Lit].x == min){
				L.push_back(LSet[Lit]);
				++Lit;
			}
			while(Rit < RSet.size() and RSet[Rit].x == min){
				R.push_back(RSet[Rit]);
				++Rit;
			}
			vertical_iteration(Blocks, L, R, S);
		}
	}
	
	
	void every_horizontal_line_segment(const std::vector<struct box> & B, segments & S){
		// Computes every horizontal line segment that begins and ends at a vertex
		// Time complexity: Either O(n²) or O(n² log n) (Amortized)
		
		std::vector<struct hpoint> TSet, BSet;
		for(auto b : B){
			TSet.push_back( struct complete::hpoint { b.x1, b.y1 } );
			TSet.push_back( struct complete::hpoint { b.x2, b.y1 } );
			BSet.push_back( struct complete::hpoint { b.x1, b.y2 } );
			BSet.push_back( struct complete::hpoint { b.x2, b.y2 } );
		}
		std::sort(TSet.begin(), TSet.end());
		std::sort(BSet.begin(), BSet.end());
		
		std::vector<struct hpoint> Blocks(0);
		int Tit = 0, Bit = 0;
		while(Tit < TSet.size() or Bit < BSet.size()){
			std::vector<struct hpoint> T(0), B(0);
			double min = 1e200;
			if(Tit < TSet.size()) min = TSet[Tit].y;
			if(Bit < BSet.size() and BSet[Bit].y < min) min = BSet[Bit].y;
			while(Tit < TSet.size() and TSet[Tit].y == min){
				T.push_back(TSet[Tit]);
				++Tit;
			}
			while(Bit < BSet.size() and BSet[Bit].y == min){
				B.push_back(BSet[Bit]);
				++Bit;
			}
			horizontal_iteration(Blocks, T, B, S);
		}
	}
	
	
	void compute_integral_blocks(std::vector<box> & B, integral & intmap){
		// Computes the "juice integral"
		// "Juice" is just proportion times area
		// And the "integral"  part means that a point contains as much
		// "juce" as every block in the upper left of it
		// Time complexity: Either O(n²) or O(n² log n)
		
		pset points;
		
		// Compute the total set of points
		// O(n log n), since we're indirectly sorting the points
		for(auto b : B){
			points.insert( struct complete::point { b.x1, b.y1 } );
			points.insert( struct complete::point { b.x1, b.y2 } );
			points.insert( struct complete::point { b.x2, b.y1 } );
			points.insert( struct complete::point { b.x2, b.y2 } );
			intmap[ struct complete::point { b.x1, b.y1 } ] = 0.0;
			intmap[ struct complete::point { b.x1, b.y2 } ] = 0.0;
			intmap[ struct complete::point { b.x2, b.y1 } ] = 0.0;
			intmap[ struct complete::point { b.x2, b.y2 } ] = 0.0;
		}
		
		// Compute the juice integral
		// Either O(n²) or O(n² log n)
		for(auto b : B){
			double juice = b.p * area(b);
			pset::iterator i = points.begin();
			while(i != points.end()){
				if( i->x > b.x1 and i->y > b.y1 ){
					intmap[ *i ] += juice;
				}
				++i;
			}
		}
	}
	
	
	bool isBox(const int i1, 
	           const int i2, 
	           const std::vector<struct box> & inputProblem, 
	           std::vector<struct box> & allBoxes,
	           const segments & segset,
	           integral & intmap){
		
		// Determines whether the two boxes i1 and i2 form a box
		// And if they do, the box, together with its proportion is added
		// to allBoxes
		// Time complexity: Either O(1) or O(log n)
		
		if(inputProblem[i2].x1 < inputProblem[i1].x1) return false;
		if(inputProblem[i2].x2 < inputProblem[i1].x2) return false;
		if(inputProblem[i2].y1 < inputProblem[i1].y1) return false;
		if(inputProblem[i2].y2 < inputProblem[i1].y2) return false;
		
		struct box tmp;
		tmp.x1 = inputProblem[i1].x1;
		tmp.y1 = inputProblem[i1].y1;
		tmp.x2 = inputProblem[i2].x2;
		tmp.y2 = inputProblem[i2].y2;
		
		line_segment top = complete::line_segment { tmp.x1, tmp.y1, tmp.x2, tmp.y1 };
		line_segment bot = complete::line_segment { tmp.x1, tmp.y2, tmp.x2, tmp.y2 };
		line_segment lft = complete::line_segment { tmp.x1, tmp.y1, tmp.x1, tmp.y2 };
		line_segment rgt = complete::line_segment { tmp.x2, tmp.y1, tmp.x2, tmp.y2 };
		
		if( segset.find(top) == segset.end() ) return false;
		if( segset.find(bot) == segset.end() ) return false;
		if( segset.find(lft) == segset.end() ) return false;
		if( segset.find(rgt) == segset.end() ) return false;
		
		double juice = intmap[ point {tmp.x2, tmp.y2} ] - intmap[ point {tmp.x1, tmp.y2} ] - intmap[ point {tmp.x2, tmp.y1} ] + intmap[ point {tmp.x1, tmp.y1} ];
		tmp.p = juice / area(tmp);
		
		allBoxes.push_back(tmp);
		return true;
	}
	
	
	void all_rectangles(std::vector<box> & B, std::vector<box> & output){
		// Computes every rectangle and returns it in the output vector
		// Time complexity: Either O(n²) or O(n² log n)
		
		segments segset;
		integral intmap;
		every_vertical_line_segment(B, segset);
		every_horizontal_line_segment(B, segset);
		compute_integral_blocks(B, intmap);
		for(int i = 0; i < B.size(); ++i){
			for(int j = 0; j < B.size(); ++j) isBox(i, j, B, output, segset, intmap);
		}
	}
}

#endif /* COMPLETE_CPP */
