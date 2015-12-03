/*
 * main.cu
 *
 *  Created on: Nov 13, 2015
 *      Author: reneoctavio
 */

#include "sssp.cuh"

int main(int argc, char* argv[]) {
	if (argc != 3)
		return 1;

	// Read graph
	////////////////////////////////////////////////////////////////////////////
    struct timeval time;
    gettimeofday(&time, NULL);
    double t1_read = time.tv_sec + (time.tv_usec / 1000000.0);

	DCsrMatrix d_graph;

	if 		(std::strcmp(argv[1], "d") == 0) { cusp::io::read_dimacs_file(d_graph, argv[2]); }
	else if (std::strcmp(argv[1], "m") == 0) { cusp::io::read_matrix_market_file(d_graph, argv[2]); }
	else 					 				 { printf("Invalid format.\n"); return 1; }
	d_graph.row_offsets.shrink_to_fit();
	d_graph.column_indices.shrink_to_fit();
	d_graph.values.shrink_to_fit();

	gettimeofday(&time, NULL);
    double t2_read = time.tv_sec + (time.tv_usec / 1000000.0);
    printf("\nRead data time: %.6lf seconds\n\n", t2_read - t1_read);
    ////////////////////////////////////////////////////////////////////////////

    // Calculate degree and average edge length
    ////////////////////////////////////////////////////////////////////////////
	DVector d_edges_count(d_graph.row_offsets.size());
	thrust::adjacent_difference(d_graph.row_offsets.begin(),
								d_graph.row_offsets.end(),
								d_edges_count.begin());
	d_edges_count.erase(d_edges_count.begin());

	const int k_avg_degree = thrust::reduce(d_edges_count.begin(), d_edges_count.end()) / d_edges_count.size();
	const int k_avg_edge_length = thrust::reduce(d_graph.values.begin(), d_graph.values.end()) / d_graph.values.size();

	d_edges_count.clear();
	d_edges_count.shrink_to_fit();
	////////////////////////////////////////////////////////////////////////////

	// Update Properties
	////////////////////////////////////////////////////////////////////////////
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int delta = prop.warpSize * k_avg_edge_length / k_avg_degree;
	const int k_delta = (delta > 0) ? delta : 1;
	delta = k_delta;
	////////////////////////////////////////////////////////////////////////////

	// Print properties
	std::cout << "Device: " << prop.name << std::endl;
	std::cout << "Warp Size: " << prop.warpSize << std::endl;
	std::cout << "Average degree: " << k_avg_degree << std::endl;
	std::cout << "Average edge length: " << k_avg_edge_length << std::endl;
	std::cout << "Calculated delta: " << k_delta << std::endl;

	// Create distance vector
	DVector d_distance(d_graph.num_rows);
	d_distance.shrink_to_fit();

	// STATS
	int sple = 1;
	int n_sple = 1;
#if STATS
	int num_samples = 10;
	n_sple = num_samples;

	int num_deltas = 20;

	cusp::array1d<int, cusp::host_memory> deltas(num_deltas);
/*
	deltas[0] = 1000;    	deltas[1] = 20688;
	deltas[2] = 100000;			deltas[3] = 1000000;
	deltas[4] = 10000000; 			deltas[5] = 10000;
	deltas[6] = 500000; 		deltas[7] = 1000000;
	deltas[8] = 5000000; 		deltas[9] = INT_MAX;
	deltas[10] = k_delta;    	deltas[11] = 1000;
	deltas[12] = 5000;			deltas[13] = 10000;
	deltas[14] = 50000; 		deltas[15] = 100000;
	deltas[16] = 500000; 		deltas[17] = 1000000;
	deltas[18] = 5000000; 		deltas[19] = INT_MAX;
	deltas[20] = k_delta;    	deltas[21] = 1000;
	deltas[22] = 5000;			deltas[23] = 10000;
	deltas[24] = 50000; 		deltas[25] = 100000;
	deltas[26] = 500000; 		deltas[27] = 1000000;
	deltas[28] = 5000000; 		deltas[29] = INT_MAX;*/
	for (int i = 0; i < num_deltas; i++) deltas[i] = i + 1;
	thrust::sort(thrust::host, deltas.begin(), deltas.end());

	cusp::array2d<double, cusp::host_memory> samples_sssp(num_deltas, num_samples);

	for (int delta_idx = 0; delta_idx < num_deltas; delta_idx++) {
		delta = deltas[delta_idx];
		for (int sample = 0; sample < num_samples; sample++) {
			sple = sample + 1;
#endif
			// Run SSSP
			////////////////////////////////////////////////////////////////////////////
			gettimeofday(&time, NULL); double t1 = time.tv_sec + (time.tv_usec / 1000000.0);

			near_far_pile_sssp(&d_graph, &d_distance, delta, 0);

			gettimeofday(&time, NULL); double t2 = time.tv_sec + (time.tv_usec / 1000000.0);
			////////////////////////////////////////////////////////////////////////////
			printf("Delta: %d, Sample #: %d of %d, Computation Time: %.6lf seconds\n", delta, sple, n_sple, t2 - t1);
#if STATS
			samples_sssp(delta_idx, sample) = t2 - t1;
		}
	}
	// Write statistics
	cusp::io::write_matrix_market_file(deltas, "deltas.mtx");
	cusp::io::write_matrix_market_file(samples_sssp, "time_samples_sssp.mtx");
#endif

    // Write distances
    cusp::io::write_matrix_market_file(d_distance, "distance.mtx");

	return 0;
}
