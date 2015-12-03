/*
 * sssp.cu
 *
 *  Created on: Nov 29, 2015
 *      Author: reneoctavio
 */

#include "sssp.cuh"

void near_far_pile_sssp(DCsrMatrix *d_graph, DVector *d_distance, const int k_delta, int ini_vertex) {
	// Set all distances to infinity
	thrust::fill(thrust::device, d_distance->begin(), d_distance->end(), INT_MAX);

	// Number of vertices
	const int k_num_of_vertices = d_graph->num_rows;

	// Create near set
	DVector d_near_set;

	DVector d_near_stencil(k_num_of_vertices);
	DVector d_processed_vx(k_num_of_vertices, 0);

	// Allocate Buffers
	DVector d_frontier_vertices;
	DVector d_tent_distance;

	// Get the number of outgoing edges for each vertex
	DVector d_out_edges_count(k_num_of_vertices + 1);
	thrust::adjacent_difference(thrust::device, d_graph->row_offsets.begin(),
			d_graph->row_offsets.end(), d_out_edges_count.begin());
	d_out_edges_count.erase(d_out_edges_count.begin());
	d_out_edges_count.shrink_to_fit();

	// Distance from vertex 0
	(*d_distance)[ini_vertex] = 0;

#if STATS_WORK
	// STATS
	DVector iteration_vertices;
	DVector edges_touched;
	DVector queued_vertices;
#endif

	// Iterator
	int i = 0;
	while (1) {
		int lower_distance =  i      * k_delta;
		int upper_distance = (i + 1) * k_delta;

#if DEBUG_MSG_BKT
		std::cout << "Iteration #: " << i << ", Lower: " << lower_distance << ", Upper: " << upper_distance << std::endl;
#endif
		// Create near stencil
		thrust::fill(d_near_stencil.begin(), d_near_stencil.end(), 0);
		thrust::transform(
				thrust::make_zip_iterator(thrust::make_tuple(
						d_distance->begin(),
						d_processed_vx.begin())),
				thrust::make_zip_iterator(thrust::make_tuple(
						d_distance->end(),
						d_processed_vx.end())),
				d_near_stencil.begin(),
				is_within_range(lower_distance, upper_distance));
		int near_set_sz = thrust::reduce(d_near_stencil.begin(), d_near_stencil.end());

		// Copy vertices to near set
		d_near_set.resize(near_set_sz);
		thrust::copy_if(thrust::make_counting_iterator(0),
				thrust::make_counting_iterator(k_num_of_vertices),
				d_near_stencil.begin(), d_near_set.begin(), thrust::identity<int>());

#if STATS_WORK
		// STATS
		iteration_vertices.push_back(near_set_sz);
		edges_touched.push_back(0);
		int queued_num = thrust::count_if(d_distance->begin(), d_distance->end(), is_valid_distance()) - near_set_sz;
		queued_vertices.push_back(queued_num);
#endif

		while (!d_near_set.empty()) {
			// Put origin vertices to processed
			thrust::fill(
					thrust::make_permutation_iterator(
							d_processed_vx.begin(),
							d_near_set.begin()),
					thrust::make_permutation_iterator(
							d_processed_vx.begin(),
							d_near_set.end()),
					1);

			// Update frontier and distances
			expand_edges(d_graph, d_distance,
					&d_out_edges_count, &d_near_set,
					d_near_set.size(), &d_frontier_vertices, &d_tent_distance);

#if STATS_WORK
			// STATS
			edges_touched[edges_touched.size() - 1] += d_frontier_vertices.size();
#endif

			// Remove tentative > current distance
			remove_invalid_distances(d_distance, &d_frontier_vertices, &d_tent_distance);

			// Update distance
			relax(d_distance, &d_frontier_vertices, &d_tent_distance);

			// Set frontier vertices ready
			thrust::fill(
					thrust::make_permutation_iterator(
							d_processed_vx.begin(),
							d_frontier_vertices.begin()),
					thrust::make_permutation_iterator(
							d_processed_vx.begin(),
							d_frontier_vertices.end()),
					0);

			// Create near stencil for next iteration
			thrust::fill(d_near_stencil.begin(), d_near_stencil.end(), 0);
			thrust::transform(
					thrust::make_zip_iterator(thrust::make_tuple(
							d_distance->begin(),
							d_processed_vx.begin())),
					thrust::make_zip_iterator(thrust::make_tuple(
							d_distance->end(),
							d_processed_vx.end())),
					d_near_stencil.begin(),
					is_within_range(lower_distance, upper_distance));
			near_set_sz = thrust::reduce(d_near_stencil.begin(), d_near_stencil.end());

			// Copy vertices to near set
			d_near_set.resize(near_set_sz);
			thrust::copy_if(thrust::make_counting_iterator(0),
					thrust::make_counting_iterator(k_num_of_vertices),
					d_near_stencil.begin(), d_near_set.begin(), thrust::identity<int>());

#if STATS_WORK
			// STATS
			iteration_vertices[bucket_vertices.size() - 1] += d_bucket.size();
#endif

		}
		// Find next position
		int next = thrust::transform_reduce(d_distance->begin(), d_distance->end(),
				invalidate_range(upper_distance), INT_MAX, thrust::minimum<int>());
		if (next == INT_MAX) break;
		i = next / k_delta;
	}
#if STATS_WORK
	// Write statistics
	cusp::io::write_matrix_market_file(iteration_vertices, "iteration_vertices.mtx");
	cusp::io::write_matrix_market_file(edges_touched, "edges_touched.mtx");
	cusp::io::write_matrix_market_file(queued_vertices, "queued_vertices.mtx");
#endif
}
/**
 * Given an origin vertex, expand it to get their edges and frontier vertices
 * @param d_graph a graph to be traversed (in)
 * @param d_distance a vector holding the current distance from initial vertex to others (in)
 * @param d_origin a vector with origin vertices (in)
 * @param num_origin_vertices number of origin vertices (in)
 * @param d_frontier_vertices a vector with frontier vertices (out)
 * @param d_tentative_distance a vector with tentative distance of these frontier edges (out)
 */
void expand_edges(DCsrMatrix *d_graph, DVector *d_distance, DVector *d_count, DVector *d_origin,
		int num_origin_vertices, DVector *d_frontier_vertices, DVector *d_tentative_distance) {

	// Calculate the offsets, where the frontier vertices and tentative distances will be placed
	DVector edges_offset(num_origin_vertices + 1, 0);
	thrust::copy(thrust::device,
			thrust::make_permutation_iterator(
					d_count->begin(),
					d_origin->begin()),
			thrust::make_permutation_iterator(
					d_count->begin(),
					d_origin->begin() + num_origin_vertices),
			edges_offset.begin());
	thrust::exclusive_scan(thrust::device, edges_offset.begin(), edges_offset.end(), edges_offset.begin());

	int num_out_edges = edges_offset.back();
	d_frontier_vertices->resize(num_out_edges);
	d_tentative_distance->resize(num_out_edges);

	// Copy frontier vertices and tentative distances from graph and others input
	thrust::for_each(thrust::device,
			make_zip_iterator(thrust::make_tuple(
					thrust::make_permutation_iterator(
						d_graph->row_offsets.begin(),
						d_origin->begin()),
					thrust::make_permutation_iterator(
						d_distance->begin(),
						d_origin->begin()),
					edges_offset.begin(),
					edges_offset.begin() + 1)),
			make_zip_iterator(thrust::make_tuple(
					thrust::make_permutation_iterator(
						d_graph->row_offsets.begin(),
						d_origin->begin() + num_origin_vertices),
					thrust::make_permutation_iterator(
						d_distance->begin(),
						d_origin->begin() + num_origin_vertices),
					edges_offset.begin() + num_origin_vertices,
					edges_offset.begin() + 1 + num_origin_vertices)),
			update_distance_and_vertices(d_tentative_distance->data().get(),
										 d_frontier_vertices->data().get(),
										 d_graph->column_indices.data().get(),
										 d_graph->values.data().get()));
}

/**
 * Remove tentative distances if they are greater or equal of current distance of a frontier vertex
 * Also remove associated frontier vertex
 * @param d_distance a vector with current distances (in)
 * @param d_frontier_vertices a vector with current frontier vertices (in/out)
 * @param d_tent_distance a vector with tentative distances (in/out)
 */
void remove_invalid_distances(DVector *d_distance, DVector *d_frontier_vertices, DVector *d_tent_distance) {
	Zip2DVecIterator zip_removed_end;
	Tuple2DVec tent_dist_iter;
	zip_removed_end = thrust::remove_if(thrust::device,
			thrust::make_zip_iterator(
					thrust::make_tuple(
							d_frontier_vertices->begin(),
							d_tent_distance->begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(
							d_frontier_vertices->end(),
							d_tent_distance->end())),
			is_invalid_tent_distance(d_distance->data().get()));

	tent_dist_iter = zip_removed_end.get_iterator_tuple();
	d_frontier_vertices->erase(thrust::get<0>(tent_dist_iter), d_frontier_vertices->end());
	d_tent_distance->erase(thrust::get<1>(tent_dist_iter), d_tent_distance->end());
}

/**
 * Relaxation phase
 * Update distance with the minimum tentative distance
 * @param d_distance a vector with current distances (in/out)
 * @param d_frontier_vertices a vector with current frontier vertices (in)
 * @param d_tent_distance a vector with tentative distances (in)
 */
void relax(DVector *d_distance, DVector *d_frontier_vertices, DVector *d_tent_distance) {
	thrust::for_each(thrust::device,
			thrust::make_zip_iterator(thrust::make_tuple(
					d_tent_distance->begin(),
					thrust::make_permutation_iterator(
							d_distance->begin(),
							d_frontier_vertices->begin())
					)),
			thrust::make_zip_iterator(thrust::make_tuple(
					d_tent_distance->end(),
					thrust::make_permutation_iterator(
							d_distance->begin(),
							d_frontier_vertices->end())
					)),
			update_distance());
}
