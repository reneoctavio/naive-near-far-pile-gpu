/*
 * main.cu
 *
 *  Created on: Nov 13, 2015
 *      Author: reneoctavio
 */

#include "definitions.cuh"

int main(int argc, char* argv[]) {
	if (argc != 4)
		return 1;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	const int k_warp_size = prop.warpSize;

	const int k_avg_degree = std::atoi(argv[1]);
	const int k_avg_edge_length = std::atoi(argv[2]);

	const int k_delta = k_warp_size * k_avg_edge_length / k_avg_degree;

	std::cout << "Device: " << prop.name << std::endl;
	std::cout << "Warp Size: " << prop.warpSize << std::endl;
	std::cout << "Average degree: " << k_avg_degree << std::endl;
	std::cout << "Average edge length: " << k_avg_edge_length << std::endl;
	std::cout << "Calculated delta: " << k_delta << std::endl;

	DCsrMatrix d_graph;
	cusp::io::read_dimacs_file(d_graph, argv[3]);

	const int k_num_of_vertices = d_graph.num_rows;

	DVector d_distance(k_num_of_vertices);
	thrust::fill(thrust::device, d_distance.begin(), d_distance.end(), INT_MAX);

	DVector d_near_set;
	DVector d_far_pile;

	DVector d_out_edges_count(k_num_of_vertices + 1);
	thrust::adjacent_difference(thrust::device, d_graph.row_offsets.begin(),
			d_graph.row_offsets.end(), d_out_edges_count.begin());
	d_out_edges_count.erase(d_out_edges_count.begin());

	DVector d_current_out_edges_count(k_num_of_vertices + 1, 0);
	DVector d_current_out_edges_index(k_num_of_vertices + 1, 0);

	d_distance[0] = 0;
	d_near_set.push_back(0);

	int i = 0;
	int lower_distance = i * k_delta;
	int upper_distance = (i + 1) * k_delta;
	while (!d_far_pile.empty() || !d_near_set.empty()) {
		std::cout << "Iteration #: " << i << "\nLower: " << lower_distance
				<< " Upper: " << upper_distance << std::endl;

		while (!d_near_set.empty()) {
			// Copy current vertices set count of outgoing edges
			thrust::fill(thrust::device, d_current_out_edges_count.begin(),
					d_current_out_edges_count.end(), 0);
			thrust::copy(thrust::device,
					thrust::make_permutation_iterator(d_out_edges_count.begin(),
							d_near_set.begin()),
					thrust::make_permutation_iterator(d_out_edges_count.begin(),
							d_near_set.end()),
					thrust::make_permutation_iterator(
							d_current_out_edges_count.begin(),
							d_near_set.begin()));

			// Get number of outgoing edges
			int num_out_edges = thrust::reduce(thrust::device,
					d_current_out_edges_count.begin(),
					d_current_out_edges_count.end());

			// Scan count to get map
			thrust::exclusive_scan(thrust::device,
					d_current_out_edges_count.begin(),
					d_current_out_edges_count.end(),
					d_current_out_edges_index.begin());
#if DEBUG_MSG
			std::cout << "Current outgoing vertices count: " << std::endl;
			thrust::copy(d_current_out_edges_count.begin(),
					d_current_out_edges_count.end(),
					std::ostream_iterator<int>(std::cout, " "));
			std::cout << std::endl;

			std::cout << "Current outgoing vertices index: " << std::endl;
			thrust::copy(d_current_out_edges_index.begin(),
					d_current_out_edges_index.end(),
					std::ostream_iterator<int>(std::cout, " "));
			std::cout << std::endl;
#endif

			// Expanded source vertices
			DVector d_exp_source_vertices(num_out_edges);
			cusp::offsets_to_indices(d_current_out_edges_index,
					d_exp_source_vertices);
#if DEBUG_MSG
			std::cout << "Source Vertices: " << std::endl;
			thrust::copy(d_exp_source_vertices.begin(),
					d_exp_source_vertices.end(),
					std::ostream_iterator<int>(std::cout, " "));
			std::cout << std::endl;
#endif

			// Scatter to put row offset in position
			DVector d_frontier_position(num_out_edges, 1);
			thrust::scatter_if(thrust::device, d_graph.row_offsets.begin(),
					d_graph.row_offsets.end() - 1,
					d_current_out_edges_index.begin(),
					d_current_out_edges_count.begin(),
					d_frontier_position.begin());

			// Inclusive scan by key to sum get final frontier positions and values
			thrust::inclusive_scan_by_key(thrust::device,
					d_exp_source_vertices.begin(), d_exp_source_vertices.end(),
					d_frontier_position.begin(), d_frontier_position.begin());
#if DEBUG_MSG
			std::cout << "Frontier Index: " << std::endl;
			thrust::copy(d_frontier_position.begin(), d_frontier_position.end(),
					std::ostream_iterator<int>(std::cout, " "));
			std::cout << std::endl;
#endif

			// Get frontier vertices
			DVector d_frontier_vertices(num_out_edges);
			thrust::copy(thrust::device,
					thrust::make_permutation_iterator(
							d_graph.column_indices.begin(),
							d_frontier_position.begin()),
					thrust::make_permutation_iterator(
							d_graph.column_indices.begin(),
							d_frontier_position.end()),
					d_frontier_vertices.begin());
#if DEBUG_MSG
			std::cout << "Frontier Vertices: " << std::endl;
			thrust::copy(d_frontier_vertices.begin(), d_frontier_vertices.end(),
					std::ostream_iterator<int>(std::cout, " "));
			std::cout << std::endl;
#endif

			// Tentative distance
			DVector d_tent_distance(num_out_edges);
			thrust::transform(thrust::device,
					thrust::make_permutation_iterator(d_graph.values.begin(),
							d_frontier_position.begin()),
					thrust::make_permutation_iterator(d_graph.values.begin(),
							d_frontier_position.end()),
					thrust::make_permutation_iterator(d_distance.begin(),
							d_exp_source_vertices.begin()),
					d_tent_distance.begin(), thrust::plus<int>());
#if DEBUG_MSG
			std::cout << "Tentative Distance: " << std::endl;
			thrust::copy(d_tent_distance.begin(), d_tent_distance.end(),
					std::ostream_iterator<int>(std::cout, " "));
			std::cout << std::endl;
#endif

			// Remove if tent_dist[frontier vertex] > dist[frontier vertex]
			ZipTuple2TentDistIterator zip_removed_end;
			zip_removed_end = thrust::remove_if(thrust::device,
					thrust::make_zip_iterator(
							thrust::make_tuple(d_frontier_vertices.begin(),
									d_tent_distance.begin())),
					thrust::make_zip_iterator(
							thrust::make_tuple(d_frontier_vertices.end(),
									d_tent_distance.end())),
					is_tent_dist_greater(d_distance.data().get()));

			Tuple2TentDist tent_dist_iter =
					zip_removed_end.get_iterator_tuple();
			d_frontier_vertices.erase(thrust::get<0>(tent_dist_iter),
					d_frontier_vertices.end());
			d_tent_distance.erase(thrust::get<1>(tent_dist_iter),
					d_tent_distance.end());

			// Get key together maintaining increasing distance within pack of keys
			thrust::sort_by_key(thrust::device, d_frontier_vertices.begin(),
					d_frontier_vertices.end(), d_tent_distance.begin());

			// For each key, get the minimum value
			thrust::pair<DVectorIterator, DVectorIterator> new_end;
			new_end = thrust::reduce_by_key(thrust::device,
					d_frontier_vertices.begin(), d_frontier_vertices.end(),
					d_tent_distance.begin(), d_frontier_vertices.begin(),
					d_tent_distance.begin(), thrust::equal_to<int>(),
					thrust::minimum<int>());

			d_frontier_vertices.erase(new_end.first, d_frontier_vertices.end());
			d_tent_distance.erase(new_end.second, d_tent_distance.end());
#if DEBUG_MSG
			std::cout << "Update Frontier Vertices: " << std::endl;
			thrust::copy(d_frontier_vertices.begin(), d_frontier_vertices.end(),
					std::ostream_iterator<int>(std::cout, " "));
			std::cout << std::endl;

			std::cout << "Update Frontier Tent Dist: " << std::endl;
			thrust::copy(d_tent_distance.begin(), d_tent_distance.end(),
					std::ostream_iterator<int>(std::cout, " "));
			std::cout << std::endl;
#endif

			// Update distances
			thrust::copy(thrust::device, d_tent_distance.begin(),
					d_tent_distance.end(),
					make_permutation_iterator(d_distance.begin(),
							d_frontier_vertices.begin()));

			// Split frontier vertices between near and far pile
			int count_current_near = thrust::count_if(thrust::device,
					d_tent_distance.begin(), d_tent_distance.end(),
					is_within_range(upper_distance));
			int count_current_far = d_tent_distance.size() - count_current_near;

			// Resize near and far pile
			d_near_set.clear();
			d_near_set.resize(count_current_near);
			d_far_pile.resize(d_far_pile.size() + count_current_far);

			// Separate indices for near and far pile
			zip_removed_end = thrust::partition(thrust::device,
					make_zip_iterator(
							make_tuple(d_frontier_vertices.begin(),
									d_tent_distance.begin())),
					make_zip_iterator(
							make_tuple(d_frontier_vertices.end(),
									d_tent_distance.end())),
					is_within_range_tuple(upper_distance));

			tent_dist_iter = zip_removed_end.get_iterator_tuple();

			// Copy vertices to near set
			thrust::copy(thrust::device, d_frontier_vertices.begin(),
					thrust::get<0>(tent_dist_iter), d_near_set.begin());

			// Copy vertices to far pile
			thrust::copy(thrust::device, thrust::get<0>(tent_dist_iter),
					d_frontier_vertices.end(),
					(d_far_pile.end() - count_current_far));
		}

#if DEBUG_MSG
		std::cout << "Near Pile Vertices: " << std::endl;
		thrust::copy(d_near_set.begin(), d_near_set.end(),
				std::ostream_iterator<int>(std::cout, " "));
		std::cout << std::endl;

		std::cout << "Far Pile Vertices: " << std::endl;
		thrust::copy(d_far_pile.begin(), d_far_pile.end(),
				std::ostream_iterator<int>(std::cout, " "));
		std::cout << std::endl;
#endif

		i++;
		lower_distance = i * k_delta;
		upper_distance = (i + 1) * k_delta;

		// Compact (Remove duplicates)
		DVectorIterator new_compacted_end;
		thrust::sort(thrust::device, d_far_pile.begin(), d_far_pile.end());
		new_compacted_end = thrust::unique(thrust::device, d_far_pile.begin(),
				d_far_pile.end());
		d_far_pile.erase(new_compacted_end, d_far_pile.end());

		// Compact (Remove vertices with distance < i*delta)
		/*
		new_compacted_end = thrust::remove_if(thrust::device,
				d_far_pile.begin(), d_far_pile.end(),
				is_less_than_lower(lower_distance));
		d_far_pile.erase(new_compacted_end, d_far_pile.end());*/

		// Separate indices for near and far pile
		new_compacted_end = thrust::partition(thrust::device,
				d_far_pile.begin(), d_far_pile.end(), is_dist_set_in_near_pile(d_distance.data().get(), upper_distance));

		// Copy vertices to near set
		d_near_set.resize(
				thrust::distance(d_far_pile.begin(), new_compacted_end));
		thrust::copy(thrust::device, d_far_pile.begin(), new_compacted_end,
				d_near_set.begin());

		// Remove near vertices from far pile
		d_far_pile.erase(d_far_pile.begin(), new_compacted_end);
	}

	std::cout << "Distances: " << std::endl;
	thrust::copy(d_distance.begin(), d_distance.end(),
			std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;

	return 0;
}
