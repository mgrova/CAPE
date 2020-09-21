/*
 * Copyright 2018 Pedro Proenza <p.proenca@surrey.ac.uk> (University of Surrey)
 *
 */

#ifndef PLANE_SEGMENTATION_CAPE_H_
#define PLANE_SEGMENTATION_CAPE_H_

#include <iostream>
#include "Params.h"
#include <Eigen/Dense>
#include <ctime>

class PlaneSeg{
public:
	PlaneSeg(Eigen::MatrixXf & cloud_array, int cell_id, int nr_pts_per_cell, int cell_width);
	~PlaneSeg(void);
	
	void fitPlane();
	void expandSegment(PlaneSeg * plane_seg);
	void clearPoints();
	
public:
	int nr_pts, min_nr_pts;
	double x_acc, y_acc, z_acc,
		xx_acc, yy_acc, zz_acc,
		xy_acc, xz_acc, yz_acc;
	float score;
	float MSE;
	bool planar;

	// Plane params
	double mean[3];
	double normal[3];
	double d;
};

#endif