/*
 * sssp.cuh
 *
 *  Created on: Nov 29, 2015
 *      Author: reneoctavio
 */

#ifndef SSSP_CUH_
#define SSSP_CUH_

#include "definitions.cuh"


void near_far_pile_sssp(DCsrMatrix *d_graph, DVector *d_distance, const int k_delta);


#endif /* SSSP_CUH_ */
