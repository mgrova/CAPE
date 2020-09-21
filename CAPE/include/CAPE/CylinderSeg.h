/*
 * Copyright 2018 Pedro Proenza <p.proenca@surrey.ac.uk> (University of Surrey)
 *
 */

#ifndef CYLINDER_SEGMENTATION_CAPE_H_
#define CYLINDER_SEGMENTATION_CAPE_H_

#include "PlaneSeg.h"
#include <vector>
#include <iostream>
#include <Eigen/Dense>

typedef Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>  MatrixXb;

class CylinderSeg{	
public:
    CylinderSeg(){}
	CylinderSeg(std::vector<PlaneSeg*> & Grid, bool * activated_mask, int nr_cells_activated);
	~CylinderSeg(void);
	
	float distance(Eigen::Vector3f & point, int segment_id);
public:
	int nr_segments;
	std::vector<float> radii;
	double axis[3];
	std::vector<Eigen::MatrixXd> centers;
	std::vector<MatrixXb> inliers;
	std::vector<double> MSEs;
	int * local2global_map;
	std::vector<bool> cylindrical_mask;
	std::vector<Eigen::Vector3f> P1;
	std::vector<Eigen::Vector3f> P2;
	std::vector<float> P1P2_norm;
};

#endif

