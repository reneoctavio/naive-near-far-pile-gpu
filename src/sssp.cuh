/*
 * sssp.cuh
 *
 *  Created on: Nov 29, 2015
 *      Author: reneoctavio
 */

#ifndef SSSP_CUH_
#define SSSP_CUH_

#ifndef DEBUG_MSG
#define DEBUG_MSG 0
#endif

#ifndef DEBUG_MSG_BKT
#define DEBUG_MSG_BKT 0
#endif

#ifndef STATS
#define STATS 1
#endif

#ifndef STATS_WORK
#define STATS_WORK 0
#endif

#include <sys/time.h>

#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/io/dimacs.h>
#include <cusp/io/matrix_market.h>

#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>

// Typedefs
typedef typename cusp::csr_matrix<int, int, cusp::device_memory> DCsrMatrix;
typedef typename cusp::coo_matrix<int, int, cusp::device_memory> DCooMatrix;

typedef typename cusp::array1d<int, cusp::device_memory> DVector;
typedef typename DVector::iterator DVectorIterator;

typedef typename thrust::tuple<DVectorIterator, DVectorIterator> Tuple2DVec;
typedef typename thrust::zip_iterator<Tuple2DVec> Zip2DVecIterator;

typedef typename thrust::tuple<DVectorIterator, DVectorIterator, DVectorIterator> Tuple3DVec;
typedef typename thrust::zip_iterator<Tuple3DVec> Zip3DVecIterator;

// Functions
void near_far_pile_sssp(DCsrMatrix *d_graph, DVector *d_distance, const int k_delta, int ini_vertex);

void remove_invalid_distances(DVector *d_distance, DVector *d_frontier_vertices, DVector *d_tent_distance);

void expand_edges(DCsrMatrix *d_graph, DVector *d_distance, DVector *d_count, DVector *d_origin,
		int num_origin_vertices, DVector *d_frontier_vertices, DVector *d_tentative_distance);

void relax(DVector *d_distance, DVector *d_frontier_vertices, DVector *d_tent_distance);

// Structs
struct update_distance_and_vertices
{
	int* distance;
	int* frontier_vertices;
	int* graph_target_vertices;
	int* graph_values;
	update_distance_and_vertices(int *_dist, int *_front, int *_tgt, int* _val) :
		distance(_dist), frontier_vertices(_front), graph_target_vertices(_tgt), graph_values(_val) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple tuple)
	{
		int graph_begin_offset = thrust::get<0>(tuple);
		int orgin_vertex_distance = thrust::get<1>(tuple);
		int target_begin_offset = thrust::get<2>(tuple);
		int target_end_offset = thrust::get<3>(tuple);
		for (int i = target_begin_offset; i < target_end_offset; i++) {
			int graph_offset = graph_begin_offset + (i - target_begin_offset);
			distance[i] = orgin_vertex_distance + graph_values[graph_offset];
			frontier_vertices[i] = graph_target_vertices[graph_offset];
		}
	}
};

struct update_distance
{
	template <typename Tuple>
	__device__
	void operator()(Tuple tuple)
	{
		int tent_distance = thrust::get<0>(tuple);

		int &tuple1 = thrust::get<1>(tuple);
		int *distance = const_cast<int*>(&tuple1);

		atomicMin(distance, tent_distance);
	}
};

struct is_invalid_tent_distance {
	int* distance_begin;
	is_invalid_tent_distance(int* _distance_begin) : distance_begin(_distance_begin) {}

	template<typename Tuple>
	__host__ __device__
	bool operator()(Tuple tuple) {
		int cur_dist = distance_begin[thrust::get<0>(tuple)];
		return thrust::get<1>(tuple) >= cur_dist;
	}
};

struct is_within_range {
	const int k_lower_bound;
	const int k_upper_bound;
	is_within_range(int _lower, int _upper) :
		k_lower_bound(_lower), k_upper_bound(_upper) {}

	template <typename Tuple>
	__host__ __device__
	bool operator()(Tuple tuple)
	{
		int distance = thrust::get<0>(tuple);
		int processed = thrust::get<1>(tuple);
		return (!processed) && (distance >= k_lower_bound) && (distance < k_upper_bound);
	}
};

struct invalidate_range
{
	const int k_upper_bound;
	invalidate_range(int _upper) : k_upper_bound(_upper) {}

	__host__ __device__
	int operator()(const int& x) const
	{
		return (x < k_upper_bound) ? INT_MAX : x;
	}
};

struct is_valid_distance
{
	__host__ __device__
	bool operator()(const int& x) const
	{
		return x != INT_MAX;
	}
};

#endif /* SSSP_CUH_ */
