/*
 * main.cu
 *
 *  Created on: Nov 13, 2015
 *      Author: reneoctavio
 */

#include "definitions.cuh"
#include "sssp.cuh"

int main(int argc, char* argv[]) {
	if (argc != 4)
		return 1;

	// Read properties
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	const int k_warp_size = prop.warpSize;
	const int k_avg_degree = std::atoi(argv[1]);
	const int k_avg_edge_length = std::atoi(argv[2]);
	const int k_delta = k_warp_size * k_avg_edge_length / k_avg_degree;

	// Print properties
	std::cout << "Device: " << prop.name << std::endl;
	std::cout << "Warp Size: " << prop.warpSize << std::endl;
	std::cout << "Average degree: " << k_avg_degree << std::endl;
	std::cout << "Average edge length: " << k_avg_edge_length << std::endl;
	std::cout << "Calculated delta: " << k_delta << std::endl;

	// Read graph
	DCsrMatrix d_graph;
	cusp::io::read_dimacs_file(d_graph, argv[3]);
	d_graph.row_offsets.shrink_to_fit();
	d_graph.column_indices.shrink_to_fit();
	d_graph.values.shrink_to_fit();

	// Create distance vector
	DVector d_distance(d_graph.num_rows);
	d_distance.shrink_to_fit();

	// Calculate time
    struct timeval tim;
    gettimeofday(&tim, NULL);
    double t1=tim.tv_sec+(tim.tv_usec/1000000.0);

    // Run SSSP
    near_far_pile_sssp(&d_graph, &d_distance, k_delta);

    // Get elapsed time
    gettimeofday(&tim, NULL);
    double t2=tim.tv_sec+(tim.tv_usec/1000000.0);

    // Print distances
	std::cout << "Distances: " << std::endl;
	thrust::copy(d_distance.begin(), d_distance.end(),
			std::ostream_iterator<int>(std::cout, "\n"));
	std::cout << std::endl;

	// Print elapsed time
	printf("%.6lf seconds elapsed\n", t2-t1);

	return 0;
}
