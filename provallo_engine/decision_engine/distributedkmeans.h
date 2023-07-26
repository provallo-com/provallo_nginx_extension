/*
 * Distributedkmeans.h
 *
 *  Created on: May 24, 2021
 *      Author: Kardon
 *
 *
 */

#ifndef DISTRIBUTEDKMEANS_H_
#define DISTRIBUTEDKMEANS_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>
#include <iostream>
#include <regex>
#include <assert.h>
#include <fstream>
#include "utils.h"
namespace provallo
{

//based on dkm MIT license :https://github.com/genbattle/dkm/blob/master/include/dkm.hpp
//adapters to provallo::attribute and provallo::matrix

  template<typename T>
    class clustering_parameters
    {
    public:
      explicit
      clustering_parameters (uint32_t k) :
	  _k (k), _has_max_iter (false), _max_iter (), _has_min_delta (false), _min_delta (), _has_rand_seed (
	      false), _rand_seed ()
      {
      }

      void
      set_max_iteration (uint64_t max_iter)
      {
        _max_iter = max_iter;
        _has_max_iter = true;
      }

      void
      set_min_delta (T min_delta)
      {
        _min_delta = min_delta;
        _has_min_delta = true;
      }

      void
      set_random_seed (uint64_t rand_seed)
      {
        _rand_seed = rand_seed;
        _has_rand_seed = true;
      }

      bool
      has_max_iteration () const
      {
	return _has_max_iter;
      }
      bool
      has_min_delta () const
      {
	return _has_min_delta;
      }
      bool
      has_random_seed () const
      {
	return _has_rand_seed;
      }

      uint32_t
      get_k () const
      {
      	return _k;
      }
 
      uint64_t
      get_max_iteration () const
      {
	return _max_iter;
      }
      T
      get_min_delta () const
      {
	return _min_delta;
      }
      uint64_t
      get_random_seed () const
      {
	return _rand_seed;
      }

    private:
      uint32_t _k;
      bool _has_max_iter;
      uint64_t _max_iter;
      bool _has_min_delta;
      T _min_delta;
      bool _has_rand_seed;
      uint64_t _rand_seed;
    };

  template<typename T, size_t N>
    std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>
    kmeans_lloyd (const std::vector<std::array<T, N>> &data,
		  const clustering_parameters<T> &parameters)
    {
      static_assert(std::is_arithmetic<T>::value && std::is_signed<T>::value,
	  "kmeans_lloyd requires the template parameter T to be a signed arithmetic type (e.g. float, double, int)");
      assert(parameters.get_k () > 0); // k must be greater than zero
      assert(data.size () >= parameters.get_k ()); // there must be at least k data points
      std::random_device rand_device;
      uint64_t seed =
	  parameters.has_random_seed () ?
	      parameters.get_random_seed () : rand_device ();
      std::vector<std::array<T, N>> means = random_plusplus (
	  data, parameters.get_k (), seed);

      std::vector<std::array<T, N>> old_means;
      std::vector<std::array<T, N>> old_old_means;
      std::vector<uint32_t> clusters;
      // Calculate new means until convergence is reached or we hit the maximum iteration count
      uint64_t count = 0;
      do
	{
	  clusters = calculate_clusters (data, means);
	  old_old_means = old_means;
	  old_means = means;
	  means = calculate_means (data, clusters, old_means,
				   parameters.get_k ());
	  ++count;
	}
      while (means != old_means && means != old_old_means
	  && !(parameters.has_max_iteration ()
	      && count == parameters.get_max_iteration ())
	  && !(parameters.has_min_delta ()
	      && deltas_below_limit (deltas (old_means, means),
				     parameters.get_min_delta ())));

      return std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> (
	  means, clusters);
    }

  template<typename T, size_t N>
    std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>
    kmeans_lloyd (const std::vector<std::array<T, N>> &data, uint32_t k,
		  uint64_t max_iter = 0, T min_delta = -1.0)
    {
      clustering_parameters<T> parameters (k);
      if (max_iter != 0)
	{
	  parameters.set_max_iteration (max_iter);
	}
      if (min_delta != 0)
	{
	  parameters.set_min_delta (min_delta);
	}
      return kmeans_lloyd (data, parameters);
    }
  template<typename T, size_t N>
    std::vector<T>
    closest_distance_parallel (const std::vector<std::array<T, N>> &means,
			       const std::vector<std::array<T, N>> &data)
    {
      std::vector<T> distances (data.size (), T ());
      for (size_t i = 0; i < data.size (); ++i)
	{
	  T closest = distance_squared (data[i], means[0]);
	  for (const auto &m : means)
	    {
	      T distance = distance_squared (data[i], m);
	      if (distance < closest)
		closest = distance;
	    }
	  distances[i] = closest;
	}
      return distances;
    }
  template<typename T, size_t N>
    std::vector<std::array<T, N>>
    random_plusplus_parallel (const std::vector<std::array<T, N>> &data,
			      uint32_t k, uint64_t seed)
    {
      assert(k > 0);
      assert(data.size () > 0);
      using input_size_t = typename std::array<T, N>::size_type;
      std::vector<std::array<T, N>> means;
      // Using a very simple PRBS generator, parameters selected according to
      // https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
      std::linear_congruential_engine<uint64_t, 6364136223846793005,
	  1442695040888963407, UINT64_MAX> rand_engine (seed);

      // Select first mean at random from the set
	{
	  std::uniform_int_distribution<input_size_t> uniform_generator (
	      0, data.size () - 1);
	  means.push_back (data[uniform_generator (rand_engine)]);
	}

      for (uint32_t count = 1; count < k; ++count)
	{
	  // Calculate the distance to the closest mean for each data point
	  auto distances = closest_distance_parallel (means, data);
	  // Pick a random point weighted by the distance from existing means
	  // TODO: This might convert floating point weights to ints, distorting the distribution for small weights
	  std::discrete_distribution<input_size_t> generator (
	      distances.begin (), distances.end ());
	  means.push_back (data[generator (rand_engine)]);
	}
      return means;
    }
  template<typename T, size_t N>
    std::vector<uint32_t>
    calculate_clusters_parallel (const std::vector<std::array<T, N>> &data,
				 const std::vector<std::array<T, N>> &means)
    {
      std::vector<uint32_t> clusters (data.size (), 0);
      for (size_t i = 0; i < data.size (); ++i)
	{
	  clusters[i] = closest_mean (data[i], means);
	}
      return clusters;
    }

  template<typename T, size_t N>
    std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>
    kmeans_lloyd_parallel (const std::vector<std::array<T, N>> &data,
			   const clustering_parameters<T> &parameters)
    {
      static_assert(std::is_arithmetic<T>::value && std::is_signed<T>::value,
	  "kmeans_lloyd requires the template parameter T to be a signed arithmetic type (e.g. float, double, int)");
      assert(parameters.get_k () > 0); // k must be greater than zero
      assert(data.size () >= parameters.get_k ()); // there must be at least k data points
      std::random_device rand_device;
      uint64_t seed =
	  parameters.has_random_seed () ?
	      parameters.get_random_seed () : rand_device ();
      std::vector<std::array<T, N>> means = random_plusplus_parallel (
	  data, parameters.get_k (), seed);

      std::vector<std::array<T, N>> old_means;
      std::vector<std::array<T, N>> old_old_means;
      std::vector<uint32_t> clusters;
      // Calculate new means until convergence is reached or we hit the maximum iteration count
      uint64_t count = 0;
      do
	{
	  clusters = calculate_clusters_parallel (data, means);
	  old_old_means = old_means;
	  old_means = means;
	  means = calculate_means (data, clusters, old_means,
				   parameters.get_k ());
	  ++count;
	}
      while ((means != old_means && means != old_old_means)
	  && !(parameters.has_max_iteration ()
	      && count == parameters.get_max_iteration ())
	  && !(parameters.has_min_delta ()
	      && deltas_below_limit (deltas (old_means, means),
				     parameters.get_min_delta ())));

      return std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> (
	  means, clusters);
    }
  template<typename T, size_t N>
    std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>
    kmeans_lloyd_parallel (const std::vector<std::array<T, N>> &data,
			   uint32_t k, uint64_t max_iter = 0,
			   T min_delta = -1.0)
    {
      clustering_parameters<T> parameters (k);
      if (max_iter != 0)
	{
	  parameters.set_max_iteration (max_iter);
	}
      if (min_delta != 0)
	{
	  parameters.set_min_delta (min_delta);
	}
      return kmeans_lloyd_parallel (data, parameters);
    }

  template<typename T, size_t N>
    std::vector<std::array<T, N>>
    get_cluster (const std::vector<std::array<T, N>> &points,
		 const std::vector<uint32_t> &labels, const uint32_t label)
    {
      assert(
	  points.size () == labels.size ()
	      && "Points and labels have different sizes");
      // construct the cluster
      std::vector<std::array<T, N>> cluster;
      for (size_t point_index = 0; point_index < points.size (); ++point_index)
	{
	  if (labels[point_index] == label)
	    {
	      cluster.push_back (points[point_index]);
	    }
	}
      return cluster;
    }

  template<typename T, size_t N>
    std::vector<T>
    dist_to_center (const std::vector<std::array<T, N>> &points,
		    const std::array<T, N> &center)
    {
      std::vector<T> result (points.size ());
      std::transform (points.begin (), points.end (), result.begin (), [&center]
      (const std::array<T, N> &p) 	{	  return distance(p, center);	});
      
      return result;
    }

  template<typename T, size_t N>
    T
    sum_dist (const std::vector<std::array<T, N>> &points,
	      const std::array<T, N> &center)
    {
      std::vector<T> distances = dist_to_center (points, center);
      return std::accumulate (distances.begin (), distances.end (), T ());
    }
  template<typename T, size_t N>
    T
    means_inertia (
	const std::vector<std::array<T, N>> &points,
	const std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> &means,
	uint32_t k)
    {
      std::vector<std::array<T, N>> centroids;
      std::vector<uint32_t> labels;
      std::tie (centroids, labels) = means;

      T inertia
	{ T () };
      for (uint32_t i = 0; i < k; ++i)
	{
	  auto cluster = get_cluster (points, labels, i);
	  inertia += sum_dist (cluster, centroids[i]);
	}
      return inertia;
    }
  template<typename T, size_t N>
    std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>
    get_best_means (const std::vector<std::array<T, N>> &points, uint32_t k,
		    uint32_t n_init = 10)
    {
      auto best_means = kmeans_lloyd (points, k);
      auto best_inertia = means_inertia (points, best_means, k);

      for (uint32_t i = 0; i < n_init - 1; ++i)
	{
	  auto curr_means = kmeans_lloyd (points, k);
	  auto curr_inertia = means_inertia (points, curr_means, k);
	  if (curr_inertia < best_inertia)
	    {
	      best_inertia = curr_inertia;
	      best_means = curr_means;
	    }
	}
      // copy and return
      return best_means;
    }
  template<typename T, size_t N>
    size_t
    predict (const std::vector<std::array<T, N>> &centroids,
	     const std::array<T, N> &query)
    {
      T min = distance (centroids[0], query);
      size_t index = 0;
      for (size_t i = 1; i < centroids.size (); i++)
	{
	  auto dist = distance (centroids[i], query);
	  if (dist < min)
	    {
	      min = dist;
	      index = i;
	    }
	}
      return index;
    }

  template<typename T, size_t N>
    class distributed_kmeans
    {
      std::string _datapath;
    public:
      distributed_kmeans (const std::string &path) :
	  _datapath (path)
      {
      }

      std::vector<std::string>
      split_commas (const std::string &line)
      {
	std::vector<std::string> split;
	std::regex reg (",");
	std::copy (
	    std::sregex_token_iterator (line.begin (), line.end (), reg, -1),
	    std::sregex_token_iterator (), std::back_inserter (split));
	return split;
      }

      std::vector<std::array<T, N>>
      load_csv (const std::string &path)
      {
	std::ifstream file (path);
	std::vector<std::array<T, N>> data;
	for (auto it = std::istream_iterator<std::string> (file);
	    it != std::istream_iterator<std::string> (); ++it)
	  {
	    auto split = split_commas (*it);
	    assert(split.size () == N); // number of values must match rows in file
	    std::array<T, N> row;
	    std::transform (split.begin (), split.end (), row.begin (), []
	    (const std::string &in) -> T 
	      {
          if(in.find('.')!=std::string::npos)
            return static_cast<T>(std::stof(in));
          else
		      return static_cast<T>(std::stod(in));
	      });
	    data.push_back (row);
	  }
	return data;
      }
      virtual
      ~distributed_kmeans () = default;
    };

} /* namespace provallo */

#endif /* DISTRIBUTEDKMEANS_H_ */
