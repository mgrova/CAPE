/*
 * Copyright 2018 Pedro Proenza <p.proenca@surrey.ac.uk> (University of Surrey)
 *
 */

#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

#include <Eigen/Dense>
#include <vector>
#include <iostream>

#define DMAX std::numeric_limits<float>::max()
#define DMIN std::numeric_limits<float>::min()

class Histogram{
public:
	Histogram(int nr_bins_per_coord);
	~Histogram(void);

	void initHistogram(Eigen::MatrixXd & Points, std::vector<bool> & Flags);
	
	std::vector<int> getPointsFromMostFrequentBin();
	void removePoint(int point_id);
private:
	std::vector<int> H;
	std::vector<int> B;
	int nr_bins_per_coord;
	int nr_bins;
	int nr_points;
};

#endif
