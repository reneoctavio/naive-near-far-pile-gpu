/*
 * definitions.cuh
 *
 *  Created on: Nov 13, 2015
 *      Author: reneoctavio
 */

#ifndef DEFINITIONS_CUH_
#define DEFINITIONS_CUH_

#ifndef DEBUG_MSG
#define DEBUG_MSG 0
#endif

#include <cusp/csr_matrix.h>
#include <cusp/format_utils.h>
#include <cusp/print.h>

#include <cusp/io/dimacs.h>

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
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>

typedef typename cusp::csr_matrix<int, int, cusp::host_memory> HCsrMatrix;
typedef typename cusp::csr_matrix<int, int, cusp::device_memory> DCsrMatrix;

typedef typename cusp::array1d<int, cusp::device_memory> DVector;
typedef typename DVector::iterator DVectorIterator;

typedef typename thrust::permutation_iterator<DVectorIterator, DVectorIterator> PermIteratorIdx;

typedef typename thrust::tuple<DVectorIterator, DVectorIterator> Tuple2TentDist;
typedef typename thrust::zip_iterator<Tuple2TentDist> ZipTuple2TentDistIterator;

typedef typename thrust::tuple<DVectorIterator, DVectorIterator, DVectorIterator> Tuple3TentDist;
typedef typename thrust::zip_iterator<Tuple3TentDist> ZipTuple3TentDistIterator;

struct is_within_range {
	const int k_upper_distance;
	is_within_range(int _upper) :
			k_upper_distance(_upper) {
	}

	__host__ __device__
	bool operator()(const int& x) {
		return x <= k_upper_distance;
	}
};

struct is_less_than_lower {
	const int k_lower_distance;
	is_less_than_lower(int _lower) :
			k_lower_distance(_lower) { }

	__host__ __device__
	bool operator()(const int& x) {
		return x < k_lower_distance;
	}
};

struct is_heavy {
	const int k_upper_distance;
	is_heavy(int _upper) :
			k_upper_distance(_upper) {
	}

	__host__ __device__
	bool operator()(const int& x) {
		return x > k_upper_distance;
	}
};

struct is_within_range_tuple {
	const int k_upper;
	is_within_range_tuple(int _upper) :
			k_upper(_upper) {
	}
	is_within_range_tuple() :
			k_upper(0) {
	}

	template<typename Tuple>
	__host__ __device__ bool operator()(Tuple tuple) {
		int tent_dist = thrust::get<1>(tuple);
		return tent_dist <= k_upper;
	}
};

struct is_tent_dist_greater {
	int* distance_begin;
	is_tent_dist_greater(int* _distance_begin) :
			distance_begin(_distance_begin) {
	}

	template<typename Tuple>
	__host__ __device__ bool operator()(Tuple tuple) {
		int cur_dist = distance_begin[thrust::get<0>(tuple)];
		return thrust::get<1>(tuple) > cur_dist;
	}
};

struct is_tent_dist_greater_far {
	int* distance_begin;
	is_tent_dist_greater_far(int* _distance_begin) :
			distance_begin(_distance_begin) {
	}

	template<typename Tuple>
	__host__ __device__ bool operator()(Tuple tuple) {
		int cur_dist = distance_begin[thrust::get<0>(tuple)];
		return thrust::get<1>(tuple) > cur_dist;
	}
};

struct is_dist_set_in_near_pile {
	int* distance_begin;
	const int k_upper;
	is_dist_set_in_near_pile(int* _distance_begin, int _upper_distance) :
			distance_begin(_distance_begin), k_upper(_upper_distance) {
	}

	__host__ __device__ bool operator()(const int& x) {
		return distance_begin[x] <= k_upper;
	}
};

#endif /* DEFINITIONS_CUH_ */
