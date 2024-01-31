/*
 * utils.h
 *
 *  Created on: Mar 22, 2023
 *      Author: kardon
 */

#ifndef DECISION_ENGINE_UTILS_H_
#define DECISION_ENGINE_UTILS_H_
#include <math.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <numeric>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <map>
#include <random>
#include <functional>
#include <vector>
#include <string>
#include <bits/unique_ptr.h>

#ifndef real_t
#define real_t double     /* supported: float, double */
#endif
#ifndef sparse_ix
#define sparse_ix int64_t  /* supported: int, int64_t, size_t */
#endif

#define square(x) ((x) * (x))

#define likely(x) __builtin_expect((bool)(x), true)

#define unlikely(x) __builtin_expect((bool)(x), false)
#define THRESHOLD_EXACT_S 87670 /* difference is <5e-4 */

#define pow2(n) ( ((size_t) 1) << (n) )
#define div2(n) ((n) >> 1)
#define mult2(n) ((n) << 1)
#define ix_parent(ix) (div2((ix) - (size_t)1))  /* integer division takes care of deciding left-right */
#define ix_child(ix)  (mult2(ix) + (size_t)1)
#define SD_MIN 1e-10
#define ix_comb_(i, j, n, ncomb) (  ((ncomb)  + ((j) - (i))) - (size_t)1 - div2(((n) - (i)) * ((n) - (i) - (size_t)1))  )
#define ix_comb(i, j, n, ncomb) (  ((i) < (j))? ix_comb_(i, j, n, ncomb) : ix_comb_(j, i, n, ncomb)  )
#define calc_ncomb(n) (((n) % 2) == 0)? (div2(n) * ((n)-(size_t)1)) : ((n) * div2((n)-(size_t)1))
#define THRESHOLD_LONG_DOUBLE (size_t)1e6

#define RNG_engine std::mt19937_64
#define UniformUnitInterval std::uniform_real_distribution<double>
#define hashed_set std::unordered_set
#define hashed_map std::unordered_map
#define is_na_or_inf(x) (std::isnan(x) || std::isinf(x))


//pendantic mode
#define UNDEF_REFERENCE(x)   ptrdiff_t var_unreference  = (ptrdiff_t) (&(x)) ; var_unreference++;
#define UNDEF_REFERENCE2(x)  var_unreference = (ptrdiff_t) (&(x)); var_unreference++;


#define set_return_position(x)  NULL
#define return_to_position(x,y)

namespace provallo
{
#if SIZE_MAX == UINT32_MAX /* 32-bit systems */

  constexpr static const uint32_t MultiplyDeBruijnBitPosition[32] =
    { 0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30, 8, 12, 20, 28,
	15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31 };

  size_t
  log2ceil (size_t v)
  {
    v--;
    v |= v >> 1; // first round down to one less than a power of 2
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;

    return MultiplyDeBruijnBitPosition[(uint32_t) (v * 0x07C4ACDDU) >> 27] + 1;
  }
#elif SIZE_MAX == UINT64_MAX /* 64-bit systems */
  constexpr static const uint64_t tab64[64] =
    { 63, 0, 58, 1, 59, 47, 53, 2, 60, 39, 48, 27, 54, 33, 42, 3, 61, 51, 37,
	40, 49, 18, 28, 20, 55, 30, 34, 11, 43, 14, 22, 4, 62, 57, 46, 52, 38,
	26, 32, 41, 50, 36, 17, 19, 29, 10, 13, 21, 56, 45, 25, 31, 35, 16, 9,
	12, 44, 24, 15, 8, 23, 7, 6, 5 };

  inline size_t
  log2ceil (size_t value)
  {
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;
    return tab64[((uint64_t) ((value - (value >> 1)) * 0x07EDD5E59A4E28C2))
	>> 58] + 1;
  }
#endif

#define EULERS_GAMMA 0.577215664901532860606512

template <typename T>
struct loss
{
	typedef T value_type;
	std::function<value_type(value_type,value_type)> loss_func;
	std::function<value_type(value_type,value_type)> loss_grad;
	std::function<value_type(value_type,value_type)> loss_hess;
	loss(std::function<T(T,T)> loss_func_, std::function<T(T,T)> loss_grad_, std::function<T(T,T)> loss_hess_):
		loss_func(loss_func_), loss_grad(loss_grad_), loss_hess(loss_hess_)
	{
	}

	T operator()(T x, T y) const
	{
		return loss_func(x,y);
	}
	 
	T grad(T x, T y) const
	{
		return loss_grad(x,y);
	}	
	T hess(T x, T y) const
	{
		return loss_hess(x,y);
	}
	//apply loss function on container
	template <typename T1> 
	T1 apply(T1 x, T1 y) const
	{
		T1 res(x);
		for(size_t i = 0; i < x.size1(); i++)
			for(size_t j = 0; j < x.size2(); j++)
				res(i,j) = loss_func(x(i,j),y(i,j));
		return res;
		

	}

};
//kullback-leibler divergence loss
template <class T>
struct kl_loss : public loss<T>
{
	kl_loss():
		loss<T>(
			kl_loss<T>::loss_func,
			kl_loss<T>::grad,
			kl_loss<T>::hess)
	{
	}
	static T loss_func(T x, T y)
	{
		return x*log(x/y) + (1-x)*log((1-x)/(1-y));
	}
	static T grad(T x, T y)
	{
		return (log((1-x)/(1-y)) - log(x/y));
	}
	static T hess(T x, T y)
	{
		x =x;
		return (1/(y*(1-y)) + 1/((1-y)*y));
	}
};
//cross-entropy loss
template <class T>
struct ce_loss : public loss<T>
{
	ce_loss():
		loss<T>(ce_loss<T>::loss_func,ce_loss<T>::grad, ce_loss<T>::hess) 
	{
	}
	static T loss_func(T x, T y)
	{
		return -x*log(y) - (1-x)*log(1-y);
	}
	static T grad(T x, T y)
	{
		return (y-x)/(y*(1-y));
	}
	static T hess(T x, T y)
	{
		return (y-x)/((y*y)*(1-y));
	}

};	
//hinge loss
template <class T>	
struct hinge_loss : public loss<T>
{
	hinge_loss():
		loss<T>(hinge_loss<T>::loss_func,hinge_loss<T>::grad, hinge_loss<T>::hess)
	{
	}
	static T loss_func(T x, T y)
	{
		return std::max(0,1-x*y);
	}
	static T grad(T x, T y)
	{
		return (x*y < 1)? -y : 0;
	}
	static T hess(T x, T y)
	{
		return 0 *((x+y)/(y+x));
	}

};	
//huber loss
template <class T>
struct huber_loss : public loss<T>
{
	huber_loss():
		loss<T>(huber_loss<T>::loss_func,huber_loss<T>::grad, huber_loss<T>::hess)
	{
	}
	static T loss_func(T x, T y)
	{
		return (x-y)*(x-y)/2;
	}
	static T grad(T x, T y)
	{
		return x-y;
	}
	static T hess(T x, T y)
	{
		return 1 *((x+y)/(y+x));
	}

};	
//logistic loss
template <class T>
struct logistic_loss :public  loss<T>
{
	logistic_loss():
		loss<T>(logistic_loss<T>::loss_func,logistic_loss<T>::grad, logistic_loss<T>::hess)
	{
	}
	static T loss_func(T x, T y)
	{
		return log(1+exp(-x*y));
	}
	static T grad(T x, T y)
	{
		return -y/(1+exp(x*y));
	}
	static T hess(T x, T y)
	{
		return y*y*exp(x*y)/((1+exp(x*y))*(1+exp(x*y)));
	}

};	
//modified huber loss
template <class T>
struct modified_huber_loss : public loss<T>
{
	modified_huber_loss():
		loss<T>(modified_huber_loss<T>::loss_func,modified_huber_loss<T>::grad, modified_huber_loss<T>::hess)
	{
	}
	static T loss_func(T x, T y)
	{
		return (x*y < 1)? (1-x*y)*(1-x*y) : 0;
	}
	static T grad(T x, T y)
	{
		return (x*y < 1)? -2*y*(1-x*y) : 0;
	}
	static T hess(T x, T y)
	{
		return (x*y < 1)? 2*y*y : 0;
	}

};	
//quantile loss
template <class T>
struct quantile_loss : public loss<T> 
{
	quantile_loss():
		loss<T>(quantile_loss<T>::loss_func,quantile_loss<T>::grad, quantile_loss<T>::hess)
	{
	}
	static T loss_func(T x, T y)
	{
		return (x-y)*(x-y)/2;
	}
	static T grad(T x, T y)
	{
		return (x-y);
	}
	static T hess(T x, T y)
	{
		return 1 *((x+y)/(y+x));
	}

};	
//squared loss
template <class T>
struct squared_loss : public loss<T>
{
	squared_loss():
		loss<T>(squared_loss<T>::loss_func,squared_loss<T>::grad, squared_loss<T>::hess)
	{
	}
	static T loss_func(T x, T y)
	{
		return (x-y)*(x-y)/2;
	}
	static T grad(T x, T y)
	{
		return (x-y);
	}
	static T hess(T x, T y)
	{
		//avoid warning
		return 1 *((x+y)/(y+x));
	}

};	
//smoothed hinge loss
template <class T>
struct smoothed_hinge_loss : public loss<T>
{
	smoothed_hinge_loss():
		loss<T>(smoothed_hinge_loss<T>::loss_func,smoothed_hinge_loss<T>::grad, smoothed_hinge_loss<T>::hess)
	{
	}
	static T loss_func(T x, T y)
	{
		return std::max(0,1-x*y);
	}
	static T grad(T x, T y)
	{
		return (x*y < 1)? -y : 0;
	}
	static T hess(T x, T y)
	{
		return 0.0*x*y;
	}

};		
//squared hinge loss
template <class T>
struct squared_hinge_loss : public loss<T>
{
	squared_hinge_loss():
		loss<T>(squared_hinge_loss<T>::loss_func,squared_hinge_loss<T>::grad, squared_hinge_loss<T>::hess)
	{
	}
	static T loss_func(T x, T y)
	{
		return std::max(0,1-x*y);
	}
	static T grad(T x, T y)
	{
		return (x*y < 1)? -y : 0;
	}
	static T hess(T x, T y)
	{
		//avoid warning
		return 0.0*x*y;
	}

};	
//welsch loss
template <class T>
struct welsch_loss : public loss<T>
{
	welsch_loss():
		loss<T>(welsch_loss<T>::loss_func,welsch_loss<T>::grad, welsch_loss<T>::hess)
	{
	}
	static T loss_func(T x, T y)
	{
		return 1-exp(-(x-y)*(x-y)/2);
	}
	static T grad(T x, T y)
	{
		return (x-y)*exp(-(x-y)*(x-y)/2);
	}
	static T hess(T x, T y)
	{
		return (1-(x-y)*(x-y))*exp(-(x-y)*(x-y)/2);
	}

};		

//loss functions for regression

//splitting criterion

  extern "C"
  {
    typedef enum NewCategAction
    {
      Weighted = 0, Smallest = 11, Random = 12
    } NewCategAction; /* Weighted means Impute in the extended model */
    typedef enum MissingAction
    {
      Divide = 21, Impute = 22, Fail = 0
    } MissingAction; /* Divide is only for non-extended model */
    typedef enum ColType
    {
      Numeric = 31, Categorical = 32, NotUsed = 0
    } ColType;
    typedef enum CategSplit
    {
      SubSet = 0, SingleCateg = 41
    } CategSplit;
    typedef enum CoefType
    {
      Uniform = 61, Normal = 0
    } CoefType; /* For extended model */
    typedef enum UseDepthImp
    {
      Lower = 71, Higher = 0, Same = 72
    } UseDepthImp; /* For NA imputation */
    typedef enum WeighImpRows
    {
      Inverse = 0, Prop = 81, Flat = 82
    } WeighImpRows; /* For NA imputation */
    typedef enum ScoringMetric
    {
      Depth = 0,
      Density = 92,
      BoxedDensity = 94,
      BoxedDensity2 = 96,
      BoxedRatio = 95,
      AdjDepth = 91,
      AdjDensity = 93
    } ScoringMetric;

    typedef enum ColCriterion
    {
      Uniformly = 0, ByRange = 1, ByVar = 2, ByKurt = 3
    } ColCriterion; /* For proportional choices */
    typedef enum GainCriterion
    {
      NoCrit = 0, Averaged = 1, Pooled = 2, FullGain = 3, DensityCrit = 4
    } Criterion; /* For guided splits */

  };//extern "C"
// for imputation
  typedef struct ImputeNode
  {
    std::vector<double> num_sum;
    std::vector<double> num_weight;
    std::vector<std::vector<double>> cat_sum;
    std::vector<double> cat_weight;
    size_t parent;
    ImputeNode () :
	parent (0)
    {
    } // default constructor
    explicit
    ImputeNode (const size_t &parent_) : num_sum(), num_weight(), cat_sum(),cat_weight(),
	parent (parent_)
    {
    }
    // copy constructor
	ImputeNode (const ImputeNode &other) :
	num_sum (other.num_sum),
	num_weight (other.num_weight),
	cat_sum (other.cat_sum),
	cat_weight (other.cat_weight),
		parent (other.parent)

	{
	}
	// move constructor
	ImputeNode (ImputeNode &&other) :
	num_sum (std::move (other.num_sum)),
	num_weight (std::move (other.num_weight)),
	cat_sum (std::move (other.cat_sum)),

	cat_weight (std::move (other.cat_weight)),
		parent (other.parent)
	{
	}
	// copy assignment
	ImputeNode &
	operator= (const ImputeNode &other)
	{
	  if (this != &other)
	    {
	      parent = other.parent;
	      num_sum = other.num_sum;
	      num_weight = other.num_weight;
	      cat_sum = other.cat_sum;
	      cat_weight = other.cat_weight;
	    }
	  return *this;
	}
	// move assignment
	ImputeNode &
	operator= (ImputeNode &&other)
	{
	  if (this != &other)
	    {
	      parent = other.parent;
	      num_sum = std::move (other.num_sum);
	      num_weight = std::move (other.num_weight);
	      cat_sum = std::move (other.cat_sum);
	      cat_weight = std::move (other.cat_weight);
	    }
	  return *this;
	}
	// destructor
	~ImputeNode ()
	{
	}
	// swap
	friend void
	swap (ImputeNode &lhs, ImputeNode &rhs)
	{
	  std::swap (lhs.parent, rhs.parent);
	  std::swap (lhs.num_sum, rhs.num_sum);
	  std::swap (lhs.num_weight, rhs.num_weight);
	  std::swap (lhs.cat_sum, rhs.cat_sum);
	  std::swap (lhs.cat_weight, rhs.cat_weight);
	}


  } ImputeNode; /* this is for each tree node */
//model parameters
  typedef struct
  {
    bool with_replacement;
    size_t sample_size;
    size_t ntrees;
    size_t ncols_per_tree;
    size_t max_depth;
    bool penalize_range;
    bool standardize_data;
    uint64_t random_seed;
    bool weigh_by_kurt;
    double prob_pick_by_gain_avg;
    double prob_pick_by_gain_pl;
    double prob_pick_by_full_gain;
    double prob_pick_by_dens;
    double prob_pick_col_by_range;
    double prob_pick_col_by_var;
    double prob_pick_col_by_kurt;
    double min_gain;
    CategSplit cat_split_type;
    NewCategAction new_cat_action;
    MissingAction missing_action;
    ScoringMetric scoring_metric;
    bool fast_bratio;
    bool all_perm;

    size_t ndim; /* only for extended model */
    size_t ntry; /* only for extended model */
    CoefType coef_type; /* only for extended model */
    bool coef_by_prop; /* only for extended model */

    bool calc_dist; /* checkbox for calculating distances on-the-fly */
    bool calc_depth; /* checkbox for calculating depths on-the-fly */
    bool impute_at_fit; /* checkbox for producing imputed missing values on-the-fly */

    UseDepthImp depth_imp; /* only when building NA imputer */
    WeighImpRows weigh_imp_rows; /* only when building NA imputer */
    size_t min_imp_obs; /* only when building NA imputer */
  } ModelParams;

  // for imputation
  void
  todense (size_t *ix_arr, size_t st, size_t end, size_t col_num, real_t *Xc,
  sparse_ix *Xc_ind,
	   sparse_ix *Xc_indptr, double *buffer_arr);

  // for imputation
  inline double
  harmonic_recursive (double a, double b)
  {
    if (b == a + 1)
      return 1. / a;
    double m = std::floor ((a + b) / 2.);
    return harmonic_recursive (a, m) + harmonic_recursive (m, b);
  }
  template<class ldouble_safe>
    inline double
    harmonic (size_t n)
    {
      ldouble_safe temp = (ldouble_safe) 1 / square((ldouble_safe )n);
      return -(ldouble_safe) 0.5 * temp
	  * ((ldouble_safe) 1 / (ldouble_safe) 6
	      - temp
		  * ((ldouble_safe) 1 / (ldouble_safe) 60
		      - ((ldouble_safe) 1 / (ldouble_safe) 126) * temp))
	  + (ldouble_safe) 0.5 * ((ldouble_safe) 1 / (ldouble_safe) n)
	  + std::log ((ldouble_safe) n) + (ldouble_safe) EULERS_GAMMA;
    }

  inline double
  digamma (double x)
  {
    double y, z, z2;
    /* check for positive integer up to 128 */
    if (unlikely((x <= 64) && (x == std::floor (x))))
      {
	return harmonic_recursive (1.0, (double) x) - EULERS_GAMMA;
      }

    if (likely(x < 1.0e17))
      {
	z = 1.0 / (x * x);
	z2 = square(z);
	y = z
	    * (8.33333333333333333333E-2 - 8.33333333333333333333E-3 * z
		+ 3.96825396825396825397E-3 * z2
		- 4.16666666666666666667E-3 * z2 * z
		+ 7.57575757575757575758E-3 * square(z2)
		- 2.10927960927960927961E-2 * square(z2) * z
		+ 8.33333333333333333333E-2 * square(z2) * z2);
      }
    else
      {
		y = 0.0;
      }

    y = ((-0.5 / x) - y) + std::log (x);
    return y;
  }

  template<class ldouble_safe>
    double
    expected_avg_depth (ldouble_safe approx_sample_size)
    {
      if (approx_sample_size <= 1)
	return 0;
      else if (approx_sample_size < (ldouble_safe) INT32_MAX)
	return 2. * (digamma (approx_sample_size + 1.) + EULERS_GAMMA - 1.);
      else
	{
	  ldouble_safe temp = (ldouble_safe) 1 / square(approx_sample_size);
	  return (ldouble_safe) 2 * std::log (approx_sample_size)
	      + (ldouble_safe) 2
		  * ((ldouble_safe) EULERS_GAMMA - (ldouble_safe) 1)
	      + ((ldouble_safe) 1 / approx_sample_size)
	      - temp
		  * ((ldouble_safe) 1 / (ldouble_safe) 6
		      - temp
			  * ((ldouble_safe) 1 / (ldouble_safe) 60
			      - ((ldouble_safe) 1 / (ldouble_safe) 126) * temp));
	}
    }

  inline double
  expected_separation_depth_hotstart (double curr, size_t n_curr,
				      size_t n_final)
  {
    if (n_final >= 1360)
      {
	if (n_final >= THRESHOLD_EXACT_S)
	  return 3;
	else if (n_final >= 40774)
	  return 2.999;
	else if (n_final >= 18844)
	  return 2.998;
	else if (n_final >= 11956)
	  return 2.997;
	else if (n_final >= 8643)
	  return 2.996;
	else if (n_final >= 6713)
	  return 2.995;
	else if (n_final >= 4229)
	  return 2.9925;
	else if (n_final >= 3040)
	  return 2.99;
	else if (n_final >= 2724)
	  return 2.989;
	else if (n_final >= 1902)
	  return 2.985;
	else if (n_final >= 1360)
	  return 2.98;
      }

    for (size_t i = n_curr + 1; i <= n_final; i++)
      curr += (-curr * (double) i + 3. * (double) i - 4.)
	  / ((double) i * ((double) (i - 1)));
    return curr;
  }

  inline double
  expected_separation_depth (size_t n)
  {
    const double val[] =
	  { 0., 0., 1., 1. + (1. / 3.), 1. + (1. / 3.) + (2. / 9.),
	      1.71666666667, 1.84, 1.93809524, 1.84, 1.93809524, 2.01836735,
	      2.08551587, 2.14268078 };
    return
	(n <= 10) ? val[n] :
	(n >= THRESHOLD_EXACT_S) ?
	    3. :
	    expected_separation_depth_hotstart ((double) 2.14268078,
						(size_t) 10, n);

  }

  inline void
  build_btree_sampler (std::vector<double> &btree_weights,
  real_t *sample_weights,
		       size_t nrows, size_t &log2_n, size_t &btree_offset)
  {
    log2_n = log2ceil (nrows);
    if (btree_weights.empty ())
      btree_weights.resize (pow2(log2_n + 1), 0);
    else
      btree_weights.assign (btree_weights.size (), 0);
    btree_offset = pow2(log2_n) - 1;

    for (size_t ix = 0; ix < nrows; ix++)
      btree_weights[ix + btree_offset] = std::fmax (0., sample_weights[ix]);
    for (size_t ix = btree_weights.size () - 1; ix > 0; ix--)
      btree_weights[ix_parent(ix)] += btree_weights[ix];
    if (std::isnan (btree_weights[0]) || btree_weights[0] <= 0)
      {
	fprintf (
	    stderr,
	    "Numeric precision error with sample weights, will not use them.\n");
	log2_n = 0;
	btree_weights.clear ();
	btree_weights.shrink_to_fit ();
      }

  }

  inline void
  unexpected_error ()
  {
    std::cerr << "error" << errno << std::endl;
  }

  //recursion for splitting criterion 
  //save the weights calculated by column_sampler 
  //and the indices of the rows that are not NA
  class RecursionState
  {
  public:
    size_t st;
    size_t st_NA;
    size_t end_NA;
    size_t split_ix;
    size_t end;
    size_t sampler_pos;
    size_t n_dropped;
    bool changed_weights;
    bool full_state;
    std::vector<size_t> ix_arr;
    std::vector<bool> cols_possible;
    std::vector<double> col_sampler_weights;
    std::unique_ptr<double[]> weights_arr;

    RecursionState () = default;
    template<class WorkerMemory>
      RecursionState (WorkerMemory &workspace, bool full_state);
    template<class WorkerMemory>
      void
      restore_state (WorkerMemory &workspace);
  };

  template<class WorkerMemory>
    void
    provallo::RecursionState::restore_state (WorkerMemory &workspace)
    {
      workspace.split_ix = this->split_ix;
      workspace.end = this->end;
      if (!workspace.col_sampler.has_weights ())
	workspace.col_sampler.curr_pos = this->sampler_pos;
      else
	{
	  workspace.col_sampler.tree_weights = std::move (
	      this->col_sampler_weights);
	  workspace.col_sampler.n_dropped = this->n_dropped;
	}

      if (this->full_state)
	{
	  workspace.st = this->st;
	  workspace.st_NA = this->st_NA;
	  workspace.end_NA = this->end_NA;

	  workspace.changed_weights = this->changed_weights;

	  if (workspace.comb_val.empty () && !this->ix_arr.empty ())
	    {
	      std::copy (this->ix_arr.begin (), this->ix_arr.end (),
			 workspace.ix_arr.begin () + this->st_NA);
	      if (this->changed_weights)
		{
		  size_t tot = workspace.end_NA - workspace.st_NA;
		  if (!workspace.weights_arr.empty ())
		    for (size_t ix = 0; ix < tot; ix++)
		      workspace.weights_arr[workspace.ix_arr[ix
			  + workspace.st_NA]] = this->weights_arr[ix];
		  else
		    for (size_t ix = 0; ix < tot; ix++)
		      workspace.weights_map[workspace.ix_arr[ix
			  + workspace.st_NA]] = this->weights_arr[ix];
		}
	    }
	}
    }

  template<class WorkerMemory>
    provallo::RecursionState::RecursionState (WorkerMemory &workspace,
					      bool full_state)
    {
      this->full_state = full_state;

      this->split_ix = workspace.split_ix;
      this->end = workspace.end;
      if (!workspace.col_sampler.has_weights ())
	this->sampler_pos = workspace.col_sampler.curr_pos;
      else
	{
	  this->col_sampler_weights = workspace.col_sampler.tree_weights;
	  this->n_dropped = workspace.col_sampler.n_dropped;
	}

      if (this->full_state)
	{
	  this->st = workspace.st;
	  this->st_NA = workspace.st_NA;
	  this->end_NA = workspace.end_NA;

	  this->changed_weights = workspace.changed_weights;

	  /* for the extended model, it's not necessary to copy everything */
	  if (workspace.comb_val.empty () && workspace.st_NA < workspace.end_NA)
	    {
	      this->ix_arr = std::vector<size_t> (
		  workspace.ix_arr.begin () + workspace.st_NA,
		  workspace.ix_arr.begin () + workspace.end_NA);
	      if (this->changed_weights)
		{
		  size_t tot = workspace.end_NA - workspace.st_NA;
		  this->weights_arr = std::unique_ptr<double[]> (
		      new double[tot]);
		  if (!workspace.weights_arr.empty ())
		    for (size_t ix = 0; ix < tot; ix++)
		      this->weights_arr[ix] =
			  workspace.weights_arr[workspace.ix_arr[ix
			      + workspace.st_NA]];
		  else
		    for (size_t ix = 0; ix < tot; ix++)
		      this->weights_arr[ix] =
			  workspace.weights_map[workspace.ix_arr[ix
			      + workspace.st_NA]];
		}
	    }
	}
    }

  inline bool
  is_boxed_metric (const ScoringMetric scoring_metric);
  template<class real_t_>
    void
    calc_mean_and_sd_t (size_t ix_arr[], size_t st, size_t end, real_t_ *x,
			MissingAction missing_action, double &x_sd,
			double &x_mean)
    {
      real_t_ m = 0;
      real_t_ s = 0;
      real_t_ m_prev = x[ix_arr[st]];
      real_t_ xval;

      if (missing_action == Fail)
	{
	  m_prev = x[ix_arr[st]];
	  for (size_t row = st; row <= end; row++)
	    {
	      xval = x[ix_arr[row]];
	      m += (xval - m) / (real_t) (row - st + 1);
	      s = std::fma (xval - m, xval - m_prev, s);
	      m_prev = m;
	    }

	  x_mean = m;
	  x_sd = std::sqrt (s / (real_t) (end - st + 1));
	}

      else
	{
	  size_t cnt = 0;
	  while (is_na_or_inf (m_prev) && st <= end)
	    {
	      m_prev = x[ix_arr[++st]];
	    }

	  for (size_t row = st; row <= end; row++)
	    {
	      xval = x[ix_arr[row]];
	      if (likely(!is_na_or_inf (xval)))
		{
		  cnt++;
		  m += (xval - m) / (real_t) cnt;
		  s = std::fma (xval - m, xval - m_prev, s);
		  m_prev = m;
		}
	    }

	  x_mean = m;
	  x_sd = std::sqrt (s / (real_t) cnt);
	}
    }

  template<class real_t_>
    double
    calc_mean_only (size_t ix_arr[], size_t st, size_t end, real_t_ *x)
    {
      size_t cnt = 0;
      double m = 0;
      real_t_ xval;
      for (size_t row = st; row <= end; row++)
	{
	  xval = x[ix_arr[row]];
	  if (likely(!is_na_or_inf (xval)))
	    {
	      cnt++;
	      m += (xval - m) / (double) cnt;
	    }
	}
		
      return m;
    }

  template<class real_t_, class mapping, typename ldouble_safe = long double>
    void
    calc_mean_and_sd_weighted (size_t ix_arr[], size_t st, size_t end,
			       real_t_ *x, mapping &w,
			       MissingAction missing_action, double &x_sd,
			       double &x_mean)
    {
      ldouble_safe cnt = 0;
      ldouble_safe w_this;
      ldouble_safe m = 0;
      ldouble_safe s = 0;
      ldouble_safe m_prev = x[ix_arr[st]];
      ldouble_safe xval;
      while (is_na_or_inf (m_prev) && st <= end)
	{
	  m_prev = x[ix_arr[++st]];
	}

      for (size_t row = st; row <= end; row++)
	{
	  xval = x[ix_arr[row]];
	  if (likely(!is_na_or_inf (xval)))
	    {
	      w_this = w[ix_arr[row]];
	      cnt += w_this;
	      m = std::fma (w_this, (xval - m) / cnt, m);
	      s = std::fma (w_this, (xval - m) * (xval - m_prev), s);
	      m_prev = m;
	    }
		//just reference missing_action.
		//
	

		} // end of for loop

		if (missing_action == Fail	)
		{
		  x_mean = m;
		  x_sd = std::sqrt (s / cnt);
		
		}
		else
		{
		  x_mean = m;
		  x_sd = std::sqrt (s / cnt);
		}		

	}

  template<class real_t_, class mapping>
    double
    calc_mean_only_weighted (size_t ix_arr[], size_t st, size_t end, real_t_ *x,
			     mapping &w)
    {
      double cnt = 0;
      double w_this;
      double m = 0;
      for (size_t row = st; row <= end; row++)
	{
	  if (likely(!is_na_or_inf (x[ix_arr[row]])))
	    {
	      w_this = w[ix_arr[row]];
	      cnt += w_this;
	      m = std::fma (w_this, (x[ix_arr[row]] - m) / cnt, m);
	    }
	}

      return m;
    }
  template<typename real_t_>
    void
    calc_mean_and_sd_ (size_t *ix_arr, size_t st, size_t end, size_t col_num,
		       real_t_ *Xc, sparse_ix *Xc_ind, sparse_ix *Xc_indptr,
		       double &x_sd, double &x_mean)
    {
      /* ix_arr must be already sorted beforehand */
      if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
	{
	  x_sd = 0;
	  x_mean = 0;
	  return;
	}
      size_t st_col = Xc_indptr[col_num];
      size_t end_col = Xc_indptr[col_num + 1] - 1;
      size_t curr_pos = st_col;
      size_t ind_end_col = (size_t) Xc_ind[end_col];
      size_t *ptr_st = std::lower_bound (ix_arr + st, ix_arr + end + 1,
					 (size_t) Xc_ind[st_col]);

      size_t cnt = end - st + 1;
      size_t added = 0;
      real_t m = 0;
      real_t s = 0;
      real_t m_prev = 0;

      for (size_t *row = ptr_st;
	  row != ix_arr + end + 1 && curr_pos != end_col + 1
	      && ind_end_col >= *row;)
	{
	  if (Xc_ind[curr_pos] == (sparse_ix) (*row))
	    {
	      if (unlikely(is_na_or_inf (Xc[curr_pos])))
		{
		  cnt--;
		}

	      else
		{
		  if (added == 0)
		    m_prev = Xc[curr_pos];
		  m += (Xc[curr_pos] - m) / (real_t) (++added);
		  s = std::fma (Xc[curr_pos] - m, Xc[curr_pos] - m_prev, s);
		  m_prev = m;
		}

	      if (row == ix_arr + end || curr_pos == end_col)
		break;
	      curr_pos = std::lower_bound (Xc_ind + curr_pos + 1,
					   Xc_ind + end_col + 1, *(++row))
		  - Xc_ind;
	    }

	  else
	    {
	      if (Xc_ind[curr_pos] > (sparse_ix) (*row))
		row = std::lower_bound (row + 1, ix_arr + end + 1,
					Xc_ind[curr_pos]);
	      else
		curr_pos = std::lower_bound (Xc_ind + curr_pos + 1,
					     Xc_ind + end_col + 1, *row)
		    - Xc_ind;
	    }
	}

      if (added == 0)
	{
	  x_mean = 0;
	  x_sd = 0;
	  return;
	}

      /* Note: up to this point:
       m = sum(x)/nnz
       s = sum(x^2) - (1/nnz)*(sum(x)^2)
       Here the standard deviation is given by:
       sigma = (1/n)*(sum(x^2) - (1/n)*(sum(x)^2))
       The difference can be put to a closed form. */
      if (cnt > added)
	{
	  s += square(m)
	      * ((real_t) added * ((real_t) 1 - (real_t) added / (real_t) cnt));
	  m *= (real_t) added / (real_t) cnt;
	}

      x_mean = m;
      x_sd = std::sqrt (s / (real_t) cnt);
    }

  template<class real_t_, class sparse_ix_, class ldouble_safe>
    void
    calc_mean_and_sd (size_t *ix_arr, size_t st, size_t end, size_t col_num,
		      real_t_ *Xc, sparse_ix_ *Xc_ind, sparse_ix_ *Xc_indptr,
		      double &x_sd, double &x_mean)
    {
      if (end - st + 1 < THRESHOLD_LONG_DOUBLE)
	calc_mean_and_sd_ (ix_arr, st, end, col_num, Xc, Xc_ind, Xc_indptr,
			   x_sd, x_mean);
      else
	calc_mean_and_sd_<ldouble_safe> (ix_arr, st, end, col_num,
					 (ldouble_safe*) Xc, Xc_ind, Xc_indptr,
					 x_sd, x_mean);
      x_sd = std::fmax (SD_MIN, x_sd);
    }

  template<class real_t_, class ldouble_safe>
    void
    calc_mean_and_sd (size_t ix_arr[], size_t st, size_t end, real_t_ *x,
		      MissingAction missing_action, double &x_sd,
		      double &x_mean)
    {
      if (end - st + 1 < THRESHOLD_LONG_DOUBLE)
	calc_mean_and_sd_t<real_t_> (ix_arr, st, end, x, missing_action, x_sd,
				     x_mean);
      else
	calc_mean_and_sd_t<real_t_> (ix_arr, st, end, x, missing_action, x_sd,
				     x_mean);
      x_sd = std::fmax (x_sd, SD_MIN);
    }

  template<class ldouble_safe>
    inline ldouble_safe
    calculate_sum_weights (std::vector<size_t> &ix_arr, size_t st, size_t end,
			   size_t curr_depth, std::vector<double> &weights_arr,
			   hashed_map<size_t, double> &weights_map)
    {
      if (curr_depth > 0 && !weights_arr.empty ())
	return std::accumulate (ix_arr.begin () + st, ix_arr.begin () + end + 1,
				(ldouble_safe) 0, [&weights_arr]
				(const ldouble_safe a, const size_t ix) 
				  { return a + weights_arr[ix];});
      else if (curr_depth > 0 && !weights_map.empty ())
	return std::accumulate (ix_arr.begin () + st, ix_arr.begin () + end + 1,
				(ldouble_safe) 0, [&weights_map]
				(const ldouble_safe a, const size_t ix) 
				  { return a + weights_map[ix];});
      else
	return -HUGE_VAL;
    }

  template<class InputData, class WorkerMemory, class ldouble_safe>
    inline void
    build_impute_node (ImputeNode &imputer, WorkerMemory &workspace,
		       InputData &input_data, ModelParams &model_params,
		       std::vector<ImputeNode> &imputer_tree, size_t curr_depth,
		       size_t min_imp_obs)
    {
		  
	  UNDEF_REFERENCE(model_params)
	  UNDEF_REFERENCE2(model_params)
	  UNDEF_REFERENCE2(min_imp_obs)
  	  UNDEF_REFERENCE2(imputer_tree)


      double wsum = 0.;
      bool has_weights = workspace.weights_arr.size ()
	  || workspace.weights_map.size ();
      if (!has_weights)
	wsum = (double) (workspace.end - workspace.st + 1);
      else
	wsum = calculate_sum_weights<ldouble_safe> (workspace.ix_arr,
						    workspace.st, workspace.end,
						    curr_depth,
						    workspace.weights_arr,
						    workspace.weights_map);

      
      

      imputer_tree.shrink_to_fit();
  
	 
      if(input_data.ncols_numeric)
	  imputer.num_sum.resize (input_data.ncols_numeric, wsum );
      if(input_data.ncols_numeric)
	  imputer.num_weight.resize (input_data.ncols_numeric, wsum);
      if(input_data.ncols_categ)
	  imputer.cat_sum.resize (input_data.ncols_categ);
      if(input_data.ncols_categ)
	  imputer.cat_weight.resize (input_data.ncols_categ, 0);
	   
	        //don't shrink yet.
	  //imputer.num_sum.shrink_to_fit ();
      //imputer.num_weight.shrink_to_fit ();
      //imputer.cat_sum.shrink_to_fit ();
      //imputer.cat_weight.shrink_to_fit ();

      /* Note: in theory, 'num_weight' could be initialized to 'wsum',
       and the entries could get subtracted the weight afterwards, but due to rounding
       error, this could produce some cases of no-present observations having positive
       weight, or cases of negative weight, so it's better to add it for each row after
       checking for possible NAs, even though it's less computationally efficient.
       For sparse matrices it's done the other way as otherwise it would be too slow. */

		for (size_t col = 0; col < input_data.ncols_categ; col++)
		{
			imputer.cat_sum[col].resize (input_data.ncat[col]);
		// imputer.cat_sum[col].shrink_to_fit ();
		}

      double xnum = 0.;
      int xcat = 0;
      double weight	= 0.0;
      size_t ix	= 0;

      if ((input_data.Xc_indptr == NULL && input_data.ncols_numeric)
	  || input_data.ncols_categ)
	{
	  if (!has_weights)
	    {
	      size_t cnt;
	      if (input_data.numeric_data != NULL)
		{
		  for (size_t col = 0; col < input_data.ncols_numeric; col++)
		    {
		      cnt = 0;
		      for (size_t row = workspace.st; row <= workspace.end;
			  row++)
			{
			  xnum = input_data.numeric_data[workspace.ix_arr[row]
			      + col * input_data.nrows];
			  if (!is_na_or_inf(xnum))
			    {
			      cnt++;
				  real_t sum = imputer.num_sum[col];

				  imputer.num_sum[col] +=  sum/ (double) cnt;
				  imputer.num_weight[col] += 1.0;
				  
			    }
			}//end of for loop
		      imputer.num_weight[col] = (double) cnt;
		    }
		}//	end of if (input_data.numeric_data != NULL)
	      if (input_data.ncols_categ)

		{
		  for (size_t col = 0; col < input_data.ncols_categ; col++)
		    {
		      cnt = 0;
		      for (size_t row = workspace.st; row <= workspace.end;
			  row++)
			{
			  xcat = input_data.categ_data[workspace.ix_arr[row]
			      + col * input_data.nrows];
			  if (xcat >= 0)
			    {
			      cnt++;
			      imputer.cat_sum[col][xcat]++; /* later gets divided */
			    }
			}
		      imputer.cat_weight[col] = (double) cnt;
		    }
		}//end of if (input_data.ncols_categ)
	    }//end of if (!has_weights)
	  else
	    {
		//has weights
	      ldouble_safe prod_sum, corr, val, diff;
	      if (input_data.numeric_data != NULL)
		{
		  for (size_t col = 0; col < input_data.ncols_numeric; col++)
		    {
		      prod_sum = 0;
		      corr = 0;
		      for (size_t row = workspace.st; row <= workspace.end;
			  row++)
			{
			  xnum = input_data.numeric_data[workspace.ix_arr[row]
			      + col * input_data.nrows];
			  if (!is_na_or_inf(xnum))
			    {
			      if (workspace.weights_arr.size ())
				weight =
				    workspace.weights_arr[workspace.ix_arr[row]];
			      else
				weight =
				    workspace.weights_map[workspace.ix_arr[row]];

			      imputer.num_weight[col] += weight; /* these are always <= 1 */
			      val = (xnum * weight) - corr;
			      diff = prod_sum + val;
			      corr = (diff - prod_sum) - val;
			      prod_sum = diff;
			    }
			}
		      imputer.num_sum[col] = prod_sum / imputer.num_weight[col];
		    }
		}
	      if (input_data.ncols_categ)
		{
		  for (size_t row = workspace.st; row <= workspace.end; row++)
		    {
		      ix = workspace.ix_arr[row];
		      if (workspace.weights_arr.size ())
			weight = workspace.weights_arr[ix];
		      else
			weight = workspace.weights_map[ix];

		      for (size_t col = 0; col < input_data.ncols_categ; col++)
			{
			  xcat = input_data.categ_data[ix
			      + col * input_data.nrows];
			  if (xcat >= 0)
			    {
			      imputer.cat_sum[col][xcat] += weight; /* later gets divided */
			      imputer.cat_weight[col] += weight;
			    }
			}
		    }
		}
	    }
	}
    }

  template<class InputData, class WorkerMemory, class ldouble_safe>
    void
    calc_kurt_all_cols (InputData &input_data, WorkerMemory &workspace,
			ModelParams &model_params, double *kurtosis,
			double *saved_xmin, double *saved_xmax);

  template<class ldouble_safe>
    double
    eval_guided_crit_weighted (double *x, size_t n, GainCriterion criterion,
			       double min_gain, bool as_relative_gain,
			       double *buffer_sd, double &split_point,
			       double &xmin, double &xmax, double *w,
			       size_t *buffer_indices, size_t *ix_arr_plus_st,
			       size_t *cols_use, size_t ncols_use,
			       bool force_cols_use, double *X_row_major,
			       size_t ncols, double *Xr, size_t *Xr_ind,
			       size_t *Xr_indptr);
  template<typename ldouble_safe/*=long double*/>
    void
    add_linear_comb (size_t *ix_arr, size_t st, size_t end, double *res,
		     int x[], int ncat, double *cat_coef,
		     double single_cat_coef, int chosen_cat, double &fill_val,
		     double &fill_new, size_t *buffer_cnt, size_t *buffer_pos,
		     NewCategAction new_cat_action,
		     MissingAction missing_action, CategSplit cat_split_type,
		     bool first_run);

  void
  increase_comb_counter (size_t ix_arr[], size_t st, size_t end, size_t n,
			 double counter[], hashed_map<size_t, double> &weights,
			 double exp_remainder);
  void
  increase_comb_counter (size_t ix_arr[], size_t st, size_t end, size_t n,
			 double counter[], double exp_remainder);
  void
  increase_comb_counter (size_t ix_arr[], size_t st, size_t end, size_t n,
			 double *counter, double *weights,
			 double exp_remainder);
  template<class InputData, class WorkerMemory>
    void
    add_separation_step (WorkerMemory &workspace, InputData &data,
			 double remainder)
    {
      if (!workspace.changed_weights)
	increase_comb_counter (workspace.ix_arr.data (), workspace.st,
			       workspace.end, data.nrows,
			       workspace.tmat_sep.data (), remainder);
      else if (!workspace.weights_arr.empty ())
	increase_comb_counter (workspace.ix_arr.data (), workspace.st,
			       workspace.end, data.nrows,
			       workspace.tmat_sep.data (),
			       workspace.weights_arr.data (), remainder);
      else
	increase_comb_counter (workspace.ix_arr.data (), workspace.st,
			       workspace.end, data.nrows,
			       workspace.tmat_sep.data (),
			       workspace.weights_map, remainder);
    }

  template<class InputData, class WorkerMemory, class ldouble_safe>
    inline void
    add_remainder_separation_steps (WorkerMemory &workspace,
				    InputData &input_data,
				    ldouble_safe sum_weight)
    {
      if ((workspace.end - workspace.st) > 0
	  && (!workspace.changed_weights || sum_weight > 0))
	{
	  double expected_dsep;
	  if (!workspace.changed_weights)
	    expected_dsep = expected_separation_depth (
		workspace.end - workspace.st + 1);
	  else
	    expected_dsep = expected_separation_depth (sum_weight);

	  add_separation_step (workspace, input_data, expected_dsep + 1);
	}
    }
  template<class InputData, class WorkerMemory>
    void
    calc_ranges_all_cols (InputData &input_data, WorkerMemory &workspace,
			  ModelParams &model_params, double *ranges,
			  double *saved_xmin, double *saved_xmax)
    {
	  
	  UNDEF_REFERENCE(model_params)
	  UNDEF_REFERENCE2(model_params)

      workspace.col_sampler.prepare_full_pass ();
      while (workspace.col_sampler.sample_col (workspace.col_chosen))
	{
	  get_split_range (workspace, input_data, model_params);

	  if (workspace.unsplittable)
	    {
	      workspace.col_sampler.drop_col (workspace.col_chosen);
	      ranges[workspace.col_chosen] = 0;
	      if (saved_xmin != NULL)
		{
		  saved_xmin[workspace.col_chosen] = 0;
		  saved_xmax[workspace.col_chosen] = 0;
		}
	    }
	  else
	    {
	      ranges[workspace.col_chosen] = workspace.xmax - workspace.xmin;
	      if (workspace.tree_kurtoses != NULL)
		{
		  ranges[workspace.col_chosen] *=
		      workspace.tree_kurtoses[workspace.col_chosen];
		  ranges[workspace.col_chosen] = std::fmax (
		      ranges[workspace.col_chosen], 1e-100);
		}
	      else if (input_data.col_weights != NULL)
		{
		  ranges[workspace.col_chosen] *=
		      input_data.col_weights[workspace.col_chosen];
		  ranges[workspace.col_chosen] = std::fmax (
		      ranges[workspace.col_chosen], 1e-100);
		}
	      if (saved_xmin != NULL)
		{
		  saved_xmin[workspace.col_chosen] = workspace.xmin;
		  saved_xmax[workspace.col_chosen] = workspace.xmax;
		}
	    }
	}
    }

  template<class real_t_, class sparse_ix, class ldouble_safe>
    double
    calc_mean_only (size_t *ix_arr, size_t st, size_t end, size_t col_num,
		    real_t_ *Xc, sparse_ix *Xc_ind, sparse_ix *Xc_indptr)
    {
      /* ix_arr must be already sorted beforehand */
      if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
	return 0;
      size_t st_col = Xc_indptr[col_num];
      size_t end_col = Xc_indptr[col_num + 1] - 1;
      size_t curr_pos = st_col;
      size_t ind_end_col = (size_t) Xc_ind[end_col];
      size_t *ptr_st = std::lower_bound (ix_arr + st, ix_arr + end + 1,
					 (size_t) Xc_ind[st_col]);

      size_t cnt = end - st + 1;
      size_t added = 0;
      double m = 0;

      for (size_t *row = ptr_st;
	  row != ix_arr + end + 1 && curr_pos != end_col + 1
	      && ind_end_col >= *row;)
	{
	  if (Xc_ind[curr_pos] == (sparse_ix) (*row))
	    {
	      if (unlikely(is_na_or_inf (Xc[curr_pos])))
		cnt--;
	      else
		m += (Xc[curr_pos] - m) / (double) (++added);

	      if (row == ix_arr + end || curr_pos == end_col)
		break;
	      curr_pos = std::lower_bound (Xc_ind + curr_pos + 1,
					   Xc_ind + end_col + 1, *(++row))
		  - Xc_ind;
	    }

	  else
	    {
	      if (Xc_ind[curr_pos] > (sparse_ix) (*row))
		row = std::lower_bound (row + 1, ix_arr + end + 1,
					Xc_ind[curr_pos]);
	      else
		curr_pos = std::lower_bound (Xc_ind + curr_pos + 1,
					     Xc_ind + end_col + 1, *row)
		    - Xc_ind;
	    }
	}

      if (added == 0)
	return 0;

      if (cnt > added)
	m *= ((ldouble_safe) added / (ldouble_safe) cnt);

      return m;
    }

  template<class real_t_, class sparse_ix, class mapping, class ldouble_safe>
    void
    calc_mean_and_sd_weighted (size_t *ix_arr, size_t st, size_t end,
			       size_t col_num, real_t_ *Xc, sparse_ix *Xc_ind,
			       sparse_ix *Xc_indptr,
			       double &x_sd, double &x_mean, mapping &w)
    {
      /* ix_arr must be already sorted beforehand */
      if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
	{
	  x_sd = 0;
	  x_mean = 0;
	  return;
	}
      size_t st_col = Xc_indptr[col_num];
      size_t end_col = Xc_indptr[col_num + 1] - 1;
      size_t curr_pos = st_col;
      size_t ind_end_col = (size_t) Xc_ind[end_col];
      size_t *ptr_st = std::lower_bound (ix_arr + st, ix_arr + end + 1,
					 (size_t) Xc_ind[st_col]);

      ldouble_safe cnt = 0.;
      for (size_t row = st; row <= end; row++)
	cnt += w[ix_arr[row]];
      ldouble_safe added = 0;
      ldouble_safe m = 0;
      ldouble_safe s = 0;
      ldouble_safe m_prev = 0;
      ldouble_safe w_this;

      for (size_t *row = ptr_st;
	  row != ix_arr + end + 1 && curr_pos != end_col + 1
	      && ind_end_col >= *row;)
	{
	  if (Xc_ind[curr_pos] == (sparse_ix) (*row))
	    {
	      if (unlikely(is_na_or_inf (Xc[curr_pos])))
		{
		  cnt -= w[*row];
		}

	      else
		{
		  w_this = w[*row];
		  if (added == 0)
		    m_prev = Xc[curr_pos];
		  added += w_this;
		  m = std::fma (w_this, (Xc[curr_pos] - m) / added, m);
		  s = std::fma (w_this,
				(Xc[curr_pos] - m) * (Xc[curr_pos] - m_prev),
				s);
		  m_prev = m;
		}

	      if (row == ix_arr + end || curr_pos == end_col)
		break;
	      curr_pos = std::lower_bound (Xc_ind + curr_pos + 1,
					   Xc_ind + end_col + 1, *(++row))
		  - Xc_ind;
	    }

	  else
	    {
	      if (Xc_ind[curr_pos] > (sparse_ix) (*row))
		row = std::lower_bound (row + 1, ix_arr + end + 1,
					Xc_ind[curr_pos]);
	      else
		curr_pos = std::lower_bound (Xc_ind + curr_pos + 1,
					     Xc_ind + end_col + 1, *row)
		    - Xc_ind;
	    }
	}

      if (added == 0)
	{
	  x_mean = 0;
	  x_sd = 0;
	  return;
	}

      /* Note: up to this point:
       m = sum(x)/nnz
       s = sum(x^2) - (1/nnz)*(sum(x)^2)
       Here the standard deviation is given by:
       sigma = (1/n)*(sum(x^2) - (1/n)*(sum(x)^2))
       The difference can be put to a closed form. */
      if (cnt > added)
	{
	  s += square(m)
	      * (added
		  * ((ldouble_safe) 1
		      - (ldouble_safe) added / (ldouble_safe) cnt));
	  m *= added / cnt;
	}

      x_mean = m;
      x_sd = std::sqrt (s / (ldouble_safe) cnt);
    }

  template<class real_t_, class sparse_ix, class mapping, class ldouble_safe>
    double
    calc_mean_only_weighted (size_t *ix_arr, size_t st, size_t end,
			     size_t col_num, real_t_ *Xc, sparse_ix *Xc_ind,
			     sparse_ix *Xc_indptr,
			     mapping &w)
    {
      /* ix_arr must be already sorted beforehand */
      if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
	return 0;
      size_t st_col = Xc_indptr[col_num];
      size_t end_col = Xc_indptr[col_num + 1] - 1;
      size_t curr_pos = st_col;
      size_t ind_end_col = (size_t) Xc_ind[end_col];
      size_t *ptr_st = std::lower_bound (ix_arr + st, ix_arr + end + 1,
					 (size_t) Xc_ind[st_col]);

      ldouble_safe cnt = 0.;
      for (size_t row = st; row <= end; row++)
	cnt += w[ix_arr[row]];
      ldouble_safe added = 0;
      ldouble_safe m = 0;
      ldouble_safe w_this;

      for (size_t *row = ptr_st;
	  row != ix_arr + end + 1 && curr_pos != end_col + 1
	      && ind_end_col >= *row;)
	{
	  if (Xc_ind[curr_pos] == (sparse_ix) (*row))
	    {
	      if (unlikely(is_na_or_inf (Xc[curr_pos])))
		{
		  cnt -= w[*row];
		}

	      else
		{
		  w_this = w[*row];
		  added += w_this;
		  m += w_this * (Xc[curr_pos] - m) / added;
		}

	      if (row == ix_arr + end || curr_pos == end_col)
		break;
	      curr_pos = std::lower_bound (Xc_ind + curr_pos + 1,
					   Xc_ind + end_col + 1, *(++row))
		  - Xc_ind;
	    }

	  else
	    {
	      if (Xc_ind[curr_pos] > (sparse_ix) (*row))
		row = std::lower_bound (row + 1, ix_arr + end + 1,
					Xc_ind[curr_pos]);
	      else
		curr_pos = std::lower_bound (Xc_ind + curr_pos + 1,
					     Xc_ind + end_col + 1, *row)
		    - Xc_ind;
	    }
	}

      if (added == 0)
	return 0;

      if (cnt > added)
	m *= (ldouble_safe) added / (ldouble_safe) cnt;

      return m;
    }

  /* Note about these functions: they write into an array that does not need to match to 'ix_arr',
   and instead, the index that is stored in ix_arr[n] will have the value in res[n] */

  /* for regular numerical */
  template<class real_t_>
    void
    add_linear_comb (size_t ix_arr[], size_t st, size_t end, double *res,
		     real_t_ *x, double &coef, double x_sd, double x_mean,
		     double &fill_val, MissingAction missing_action,
		     double *buffer_arr, size_t *buffer_NAs, bool first_run)
    {
      /* TODO: here don't need the buffer for NAs */

      if (first_run)
	coef /= x_sd;

      size_t cnt = 0;
      size_t cnt_NA = 0;
      double *res_write = res - st;

      if (missing_action == Fail)
	{
	  for (size_t row = st; row <= end; row++)
	    res_write[row] = std::fma (x[ix_arr[row]] - x_mean, coef,
				       res_write[row]);
	}

      else
	{
	  if (first_run)
	    {
	      for (size_t row = st; row <= end; row++)
		{
		  if (likely(!is_na_or_inf (x[ix_arr[row]])))
		    {
		      res_write[row] = std::fma (x[ix_arr[row]] - x_mean, coef,
						 res_write[row]);
		      buffer_arr[cnt++] = x[ix_arr[row]];
		    }

		  else
		    {
		      buffer_NAs[cnt_NA++] = row;
		    }

		}
	    }

	  else
	    {
	      for (size_t row = st; row <= end; row++)
		{
		  res_write[row] +=
		      (is_na_or_inf(x[ix_arr[row]])) ?
			  fill_val : ((x[ix_arr[row]] - x_mean) * coef);
		}
	      return;
	    }

	  size_t mid_ceil = cnt / 2;
	  std::partial_sort (buffer_arr, buffer_arr + mid_ceil + 1,
			     buffer_arr + cnt);

	  if ((cnt % 2) == 0)
	    fill_val = buffer_arr[mid_ceil - 1]
		+ (buffer_arr[mid_ceil] - buffer_arr[mid_ceil - 1]) / 2.0;
	  else
	    fill_val = buffer_arr[mid_ceil];

	  fill_val = (fill_val - x_mean) * coef;
	  if (cnt_NA && fill_val)
	    {
	      for (size_t row = 0; row < cnt_NA; row++)
		res_write[buffer_NAs[row]] += fill_val;
	    }

	}
    }

  /* for regular numerical */
  template<class real_t_, class mapping, class ldouble_safe>
    void
    add_linear_comb_weighted (size_t ix_arr[], size_t st, size_t end,
			      double *res, real_t_ *x, double &coef,
			      double x_sd, double x_mean, double &fill_val,
			      MissingAction missing_action, double *buffer_arr,
			      size_t *buffer_NAs, bool first_run, mapping &w)
    {
      /* TODO: here don't need the buffer for NAs */

      if (first_run)
	coef /= x_sd;

      size_t cnt = 0;
      size_t cnt_NA = 0;
      double *res_write = res - st;
      ldouble_safe cumw = 0;
      double w_this;
      /* TODO: these buffers should be allocated externally */
      std::vector<double> obs_weight;

      if (first_run && missing_action != Fail)
	{
	  obs_weight.resize (end - st + 1, 0.);
	}

      if (missing_action == Fail)
	{
	  for (size_t row = st; row <= end; row++)
	    res_write[row] = std::fma (x[ix_arr[row]] - x_mean, coef,
				       res_write[row]);
	}

      else
	{
	  if (first_run)
	    {
	      for (size_t row = st; row <= end; row++)
		{
		  if (likely(!is_na_or_inf (x[ix_arr[row]])))
		    {
		      w_this = w[ix_arr[row]];
		      res_write[row] = std::fma (x[ix_arr[row]] - x_mean, coef,
						 res_write[row]);
		      obs_weight[cnt] = w_this;
		      buffer_arr[cnt++] = x[ix_arr[row]];
		      cumw += w_this;
		    }

		  else
		    {
		      buffer_NAs[cnt_NA++] = row;
		    }

		}
	    }

	  else
	    {
	      for (size_t row = st; row <= end; row++)
		{
		  res_write[row] +=
		      (is_na_or_inf(x[ix_arr[row]])) ?
			  fill_val : ((x[ix_arr[row]] - x_mean) * coef);
		}
	      return;
	    }

	  ldouble_safe mid_point = cumw / (ldouble_safe) 2;
	  std::vector<size_t> sorted_ix (cnt);
	  std::iota (sorted_ix.begin (), sorted_ix.end (), (size_t) 0);
	  std::sort (sorted_ix.begin (), sorted_ix.end (), [&buffer_arr]
	  (const size_t a, const size_t b) 
	    { return buffer_arr[a] < buffer_arr[b];});
	  ldouble_safe currw = 0;
	  fill_val = buffer_arr[sorted_ix.back ()]; /* <- will overwrite later */
	  /* TODO: is this median calculation correct? should it do a weighted interpolation? */
	  for (size_t ix = 0; ix < cnt; ix++)
	    {
	      currw += obs_weight[sorted_ix[ix]];
	      if (currw >= mid_point)
		{
		  if (currw == mid_point && ix < cnt - 1)
		    fill_val = buffer_arr[sorted_ix[ix]]
			+ (buffer_arr[sorted_ix[ix + 1]]
			    - buffer_arr[sorted_ix[ix]]) / 2.0;
		  else
		    fill_val = buffer_arr[sorted_ix[ix]];
		  break;
		}
	    }

	  fill_val = (fill_val - x_mean) * coef;
	  if (cnt_NA && fill_val)
	    {
	      for (size_t row = 0; row < cnt_NA; row++)
		res_write[buffer_NAs[row]] += fill_val;
	    }

	}
    }

  /* for sparse numerical */
  template<class real_t_, class sparse_ix_ >
    void
    add_linear_comb (size_t *ix_arr, size_t st, size_t end, size_t col_num,
		     double *res, real_t_ *Xc, sparse_ix_ *Xc_ind,
		     sparse_ix_ *Xc_indptr, double &coef, double x_sd,
		     double x_mean, double &fill_val,
		     MissingAction missing_action, double *buffer_arr,
		     size_t *buffer_NAs, bool first_run)
    {
      /* ix_arr must be already sorted beforehand */

      /* if it's all zeros, no need to do anything, but this is not supposed
       to happen while fitting because the range is determined before calling this */
      if (Xc_indptr[col_num] == Xc_indptr[col_num + 1]
	  || Xc_ind[Xc_indptr[col_num]] > (sparse_ix) ix_arr[end]
	  || Xc_ind[Xc_indptr[col_num + 1] - 1] < (sparse_ix) ix_arr[st])
	{
	  if (first_run)
	    {
	      coef /= x_sd;
	      if (missing_action != Fail)
		fill_val = 0;
	    }

	  double *res_write = res - st;
	  double offset = x_mean * coef;
	  if (offset)
	    {
	      for (size_t row = st; row <= end; row++)
		res_write[row] -= offset;
	    }

	  return;
	}

      size_t st_col = Xc_indptr[col_num];
      size_t end_col = Xc_indptr[col_num + 1] - 1;
      size_t curr_pos = st_col;
      size_t *ptr_st = std::lower_bound (ix_arr + st, ix_arr + end + 1,
					 (size_t) Xc_ind[st_col]);

      size_t cnt_non_NA = 0; /* when NAs need to be imputed */
      size_t cnt_NA = 0; /* when NAs need to be imputed */
      size_t n_sample = end - st + 1;
      size_t *ix_arr_plus_st = ix_arr + st;

      if (first_run)
	coef /= x_sd;

      double *res_write = res - st;
      double offset = x_mean * coef;
      if (offset)
	{
	  for (size_t row = st; row <= end; row++)
	    res_write[row] -= offset;
	}

      size_t ind_end_col = Xc_ind[end_col];
      size_t nmatches = 0;

      if (missing_action != Fail)
	{
	  if (first_run)
	    {
	      for (size_t *row = ptr_st;
		  row != ix_arr + end + 1 && curr_pos != end_col + 1
		      && ind_end_col >= *row;)
		{
		  if (Xc_ind[curr_pos] == (sparse_ix) (*row))
		    {
		      if (unlikely(is_na_or_inf (Xc[curr_pos])))
			{
			  buffer_NAs[cnt_NA++] = row - ix_arr_plus_st;
			}

		      else
			{
			  buffer_arr[cnt_non_NA++] = Xc[curr_pos];
			  res[row - ix_arr_plus_st] = std::fma (
			      Xc[curr_pos], coef, res[row - ix_arr_plus_st]);
			}

		      nmatches++;
		      if (row == ix_arr + end || curr_pos == end_col)
			break;
		      curr_pos = std::lower_bound (Xc_ind + curr_pos + 1,
						   Xc_ind + end_col + 1,
						   *(++row)) - Xc_ind;
		    }

		  else
		    {
		      if (Xc_ind[curr_pos] > (sparse_ix) (*row))
			row = std::lower_bound (row + 1, ix_arr + end + 1,
						Xc_ind[curr_pos]);
		      else
			curr_pos = std::lower_bound (Xc_ind + curr_pos + 1,
						     Xc_ind + end_col + 1, *row)
			    - Xc_ind;
		    }
		}
	    }

	  else
	    {
	      /* when impute value for missing has already been determined */
	      for (size_t *row = ptr_st;
		  row != ix_arr + end + 1 && curr_pos != end_col + 1
		      && ind_end_col >= *row;)
		{
		  if (Xc_ind[curr_pos] == (sparse_ix) (*row))
		    {
		      res[row - ix_arr_plus_st] +=
		      is_na_or_inf (Xc[curr_pos]) ?
			  (fill_val + offset) : (Xc[curr_pos] * coef);
		      if (row == ix_arr + end)
			break;
		      curr_pos = std::lower_bound (Xc_ind + curr_pos + 1,
						   Xc_ind + end_col + 1,
						   *(++row)) - Xc_ind;
		    }

		  else
		    {
		      if (Xc_ind[curr_pos] > (sparse_ix) (*row))
			row = std::lower_bound (row + 1, ix_arr + end + 1,
						Xc_ind[curr_pos]);
		      else
			curr_pos = std::lower_bound (Xc_ind + curr_pos + 1,
						     Xc_ind + end_col + 1, *row)
			    - Xc_ind;
		    }
		}

	      return;
	    }

	  /* Determine imputation value */
	  std::sort (buffer_arr, buffer_arr + cnt_non_NA);
	  size_t mid_ceil = (n_sample - cnt_NA) / 2;
	  size_t nzeros = (end - st + 1) - nmatches;
	  if (nzeros > mid_ceil && buffer_arr[0] > 0)
	    {
	      fill_val = 0;
	      return;
	    }

	  else
	    {
	      size_t n_neg =
		  (buffer_arr[0] > 0) ?
		      0 :
		      ((buffer_arr[cnt_non_NA - 1] < 0) ?
			  cnt_non_NA :
			  std::lower_bound (buffer_arr, buffer_arr + cnt_non_NA,
					    (double) 0) - buffer_arr);

	      if (n_neg < (mid_ceil - 1) && n_neg + nzeros > mid_ceil)
		{
		  fill_val = 0;
		  return;
		}

	      else
		{
		  /* if the sample size is odd, take the middle, otherwise take a simple average */
		  if (((n_sample - cnt_NA) % 2) != 0)
		    {
		      if (mid_ceil < n_neg)
			fill_val = buffer_arr[mid_ceil];
		      else if (mid_ceil < n_neg + nzeros)
			fill_val = 0;
		      else
			fill_val = buffer_arr[mid_ceil - nzeros];
		    }

		  else
		    {
		      if (mid_ceil < n_neg)
			{
			  fill_val = (buffer_arr[mid_ceil - 1]
			      + buffer_arr[mid_ceil]) / 2;
			}

		      else if (mid_ceil < n_neg + nzeros)
			{
			  if (mid_ceil == n_neg)
			    fill_val = buffer_arr[mid_ceil - 1] / 2;
			  else
			    fill_val = 0;
			}

		      else
			{
			  if (mid_ceil == n_neg + nzeros && nzeros > 0)
			    fill_val = buffer_arr[n_neg] / 2;
			  else
			    fill_val = (buffer_arr[mid_ceil - nzeros - 1]
				+ buffer_arr[mid_ceil - nzeros]) / 2; /* WRONG!!!! */
			}
		    }

		  /* fill missing if any */
		  fill_val *= coef;
		  if (cnt_NA && fill_val)
		    for (size_t ix = 0; ix < cnt_NA; ix++)
		      res[buffer_NAs[ix]] += fill_val;

		  /* next time, it will need to have the offset added */
		  fill_val -= offset;
		}
	    }
	}

      else /* no NAs */
	{
	  for (size_t *row = ptr_st;
	      row != ix_arr + end + 1 && curr_pos != end_col + 1
		  && ind_end_col >= *row;)
	    {
	      if (Xc_ind[curr_pos] == (sparse_ix) (*row))
		{
		  res[row - ix_arr_plus_st] += Xc[curr_pos] * coef;
		  if (row == ix_arr + end || curr_pos == end_col)
		    break;
		  curr_pos = std::lower_bound (Xc_ind + curr_pos + 1,
					       Xc_ind + end_col + 1, *(++row))
		      - Xc_ind;
		}

	      else
		{
		  if (Xc_ind[curr_pos] > (sparse_ix) (*row))
		    row = std::lower_bound (row + 1, ix_arr + end + 1,
					    Xc_ind[curr_pos]);
		  else
		    curr_pos = std::lower_bound (Xc_ind + curr_pos + 1,
						 Xc_ind + end_col + 1, *row)
			- Xc_ind;
		}
	    }
	}
    }

  template<class real_t_, class sparse_ix, class mapping, class ldouble_safe>
    void
    add_linear_comb_weighted (size_t *ix_arr, size_t st, size_t end,
			      size_t col_num, double *res, real_t_ *Xc,
			      sparse_ix *Xc_ind,
			      sparse_ix *Xc_indptr, double &coef, double x_sd,
			      double x_mean, double &fill_val,
			      MissingAction missing_action, double *buffer_arr,
			      size_t *buffer_NAs, bool first_run, mapping &w)
    {
      /* TODO: there's likely a better way of doing this directly with sparse inputs.
       Think about some way of doing it efficiently. */
      if (first_run && missing_action != Fail)
	{
	  std::vector<double> denseX (end - st + 1, 0.);
	  todense (ix_arr, st, end, col_num, Xc, Xc_ind, Xc_indptr,
		   denseX.data ());
	  std::vector<double> obs_weight (end - st + 1);
	  for (size_t row = st; row <= end; row++)
	    obs_weight[row - st] = w[ix_arr[row]];

	  size_t end_new = end - st + 1;
	  for (size_t ix = 0; ix < end - st + 1; ix++)
	    {
	      if (unlikely(is_na_or_inf (denseX[ix])))
		{
		  std::swap (denseX[ix], denseX[--end_new]);
		  std::swap (obs_weight[ix], obs_weight[end_new]);
		}
	    }

	  ldouble_safe cumw = std::accumulate (obs_weight.begin (),
					       obs_weight.begin () + end_new,
					       (ldouble_safe) 0);
	  ldouble_safe mid_point = cumw / (ldouble_safe) 2;
	  std::vector<size_t> sorted_ix (end_new);
	  std::iota (sorted_ix.begin (), sorted_ix.end (), (size_t) 0);
	  std::sort (sorted_ix.begin (), sorted_ix.end (), [&denseX]
	  (const size_t a, const size_t b) 
	    { return denseX[a] < denseX[b];});
	  ldouble_safe currw = 0;
	  fill_val = denseX[sorted_ix.back ()]; /* <- will overwrite later */
	  /* TODO: is this median calculation correct? should it do a weighted interpolation? */
	  for (size_t ix = 0; ix < end_new; ix++)
	    {
	      currw += obs_weight[sorted_ix[ix]];
	      if (currw >= mid_point)
		{
		  if (currw == mid_point && ix < end_new - 1)
		    fill_val = denseX[sorted_ix[ix]]
			+ (denseX[sorted_ix[ix + 1]] - denseX[sorted_ix[ix]])
			    / 2.0;
		  else
		    fill_val = denseX[sorted_ix[ix]];
		  break;
		}
	    }

	  fill_val = (fill_val - x_mean) * (coef / x_sd);
	  denseX.clear ();
	  obs_weight.clear ();
	  sorted_ix.clear ();

	  add_linear_comb (ix_arr, st, end, col_num, res, Xc, Xc_ind, Xc_indptr,
			   coef, x_sd, x_mean, fill_val, missing_action,
			   buffer_arr, buffer_NAs, false);
	}

      else
	{
	  add_linear_comb (ix_arr, st, end, col_num, res, Xc, Xc_ind, Xc_indptr,
			   coef, x_sd, x_mean, fill_val, missing_action,
			   buffer_arr, buffer_NAs, first_run);
	}
    }

  /* for categoricals */
  template<typename ldouble_safe/*=long double*/>
    void
    add_linear_comb (size_t *ix_arr, size_t st, size_t end, double *res,
		     int x[], int ncat, double *cat_coef,
		     double single_cat_coef, int chosen_cat, double &fill_val,
		     double &fill_new, size_t *buffer_cnt, size_t *buffer_pos,
		     NewCategAction new_cat_action,
		     MissingAction missing_action, CategSplit cat_split_type,
		     bool first_run)
    {
      double *res_write = res - st;
      switch (cat_split_type)
	{
	case SingleCateg:
	  {
	    /* in this case there's no need to make-up an impute value for new categories, only for NAs */
	    switch (missing_action)
	      {
	      case Fail:
		{
		  for (size_t row = st; row <= end; row++)
		    res_write[row] +=
			(x[ix_arr[row]] == chosen_cat) ? single_cat_coef : 0;
		  return;
		}

	      case Impute:
		{
		  size_t cnt_NA = 0;
		  size_t cnt_this = 0;
		  size_t cnt = end - st + 1;
		  if (first_run)
		    {
		      for (size_t row = st; row <= end; row++)
			{
			  if (unlikely(x[ix_arr[row]] < 0))
			    {
			      cnt_NA++;
			    }

			  else if (x[ix_arr[row]] == chosen_cat)
			    {
			      cnt_this++;
			      res_write[row] += single_cat_coef;
			    }
			}
		    }

		  else
		    {
		      for (size_t row = st; row <= end; row++)
			res_write[row] +=
			    (x[ix_arr[row]] < 0) ?
				fill_val :
				((x[ix_arr[row]] == chosen_cat) ?
				    single_cat_coef : 0);
		      return;
		    }

		  fill_val =
		      (cnt_this > (cnt - cnt_NA - cnt_this)) ?
			  single_cat_coef : 0;
		  if (cnt_NA && fill_val)
		    {
		      for (size_t row = st; row <= end; row++)
			if (x[ix_arr[row]] < 0)
			  res_write[row] += fill_val;
		    }
		  return;
		}

	      default:
		{
		  unexpected_error ();
		  break;
		}
	      }
	  }
	  break;
	case SubSet:
	  {
	    /* in this case, since the splits are by more than 1 variable, it's not possible to
	     divide missing/new categoricals by assigning weights, so they have to be imputed
	     in both cases, unless using random weights for the new ones, in which case they won't
	     need to be imputed for new, but sill need it for NA */

	    if (new_cat_action == Random && missing_action == Fail)
	      {
		for (size_t row = st; row <= end; row++)
		  res_write[row] += cat_coef[x[ix_arr[row]]];
		return;
	      }

	    if (!first_run)
	      {
		if (missing_action == Fail)
		  {
		    for (size_t row = st; row <= end; row++)
		      res_write[row] +=
			  (x[ix_arr[row]] >= ncat) ?
			      fill_new : cat_coef[x[ix_arr[row]]];
		  }

		else
		  {
		    for (size_t row = st; row <= end; row++)
		      res_write[row] +=
			  (x[ix_arr[row]] < 0) ?
			      fill_val :
			      ((x[ix_arr[row]] >= ncat) ?
				  fill_new : cat_coef[x[ix_arr[row]]]);
		  }
		return;
	      }

	    std::fill (buffer_cnt, buffer_cnt + ncat + 1, 0);
	    switch (missing_action)
	      {
	      case Fail:
		{
		  for (size_t row = st; row <= end; row++)
		    {
		      buffer_cnt[x[ix_arr[row]]]++;
		      res_write[row] += cat_coef[x[ix_arr[row]]];
		    }
		  break;
		}

	      default:
		{
		  for (size_t row = st; row <= end; row++)
		    {
		      if (x[ix_arr[row]] >= 0)
			{
			  buffer_cnt[x[ix_arr[row]]]++;
			  res_write[row] += cat_coef[x[ix_arr[row]]];
			}

		      else
			{
			  buffer_cnt[ncat]++;
			}

		    }
		  break;
		}
	      }
	    // new cat action: 
	    switch (new_cat_action)
	      {
	      case Smallest:
		{
		  size_t smallest = SIZE_MAX;
		  int cat_smallest = 0;
		  for ( size_t cat= 0; cat < size_t(ncat); cat++)
		    {
		      if (buffer_cnt[cat] > 0 && buffer_cnt[cat] < smallest)
			{
			  smallest = buffer_cnt[cat];
			  cat_smallest = cat;
			}
		    }
		  fill_new = cat_coef[cat_smallest];
		  if (missing_action == Fail)
		    break;
		}
		[[fallthrough]];
		//fall through:
	      default:
		{
		  /* Determine imputation value as the category in sorted order that gives 50% + 1 */
		  ldouble_safe cnt_l = (ldouble_safe) ((end - st + 1)
		      - buffer_cnt[ncat]);
		  std::iota (buffer_pos, buffer_pos + ncat, (size_t) 0);
		  std::sort (buffer_pos, buffer_pos + ncat, [&cat_coef]
		  (const size_t a, const size_t b) 
		    { return cat_coef[a] < cat_coef[b];});

		  double cumprob = 0;
		  int cat;
		  for (cat = 0; cat < ncat; cat++)
		    {
		      cumprob += (ldouble_safe) buffer_cnt[buffer_pos[cat]]
			  / cnt_l;
		      if (cumprob >= .5)
			break;
		    }
		  // cat = std::min(cat, ncat); /* in case it picks the last one */
		  fill_val = cat_coef[buffer_pos[cat]];
		  if (new_cat_action != Smallest)
		    fill_new = fill_val;

		  if (buffer_cnt[ncat] > 0 && fill_val) /* NAs */
		    for (size_t row = st; row <= end; row++)
		      if (unlikely(x[ix_arr[row]] < 0))
			res_write[row] += fill_val;
		}
	      }

	    /* now fill unseen categories */
	    if (new_cat_action != Random)
	      for (size_t cat = 0; cat < size_t(ncat); cat++)
		if (!buffer_cnt[cat])
		  cat_coef[cat] = fill_new;

	  }
	}
    }

  template<class mapping, class ldouble_safe>
    void
    add_linear_comb_weighted (size_t *ix_arr, size_t st, size_t end,
			      double *res, int x[], int ncat, double *cat_coef,
			      double single_cat_coef, int chosen_cat,
			      double &fill_val, double &fill_new,
			      size_t *buffer_pos, NewCategAction new_cat_action,
			      MissingAction missing_action,
			      CategSplit cat_split_type, bool first_run,
			      mapping &w)
    {
      double *res_write = res - st;
      /* TODO: this buffer should be allocated externally */

      switch (cat_split_type)
	{
	case SingleCateg:
	  {
	    /* in this case there's no need to make-up an impute value for new categories, only for NAs */
	    switch (missing_action)
	      {
	      case Fail:
		{
		  for (size_t row = st; row <= end; row++)
		    res_write[row] +=
			(x[ix_arr[row]] == chosen_cat) ? single_cat_coef : 0;
		  return;
		}

	      case Impute:
		{
		  bool has_NA = false;
		  ldouble_safe cnt_this = 0;
		  ldouble_safe cnt_other = 0;
		  if (first_run)
		    {
		      for (size_t row = st; row <= end; row++)
			{
			  if (unlikely(x[ix_arr[row]] < 0))
			    {
			      has_NA = true;
			    }

			  else if (x[ix_arr[row]] == chosen_cat)
			    {
			      cnt_this += w[ix_arr[row]];
			      res_write[row] += single_cat_coef;
			    }

			  else
			    {
			      cnt_other += w[ix_arr[row]];
			    }
			}
		    }

		  else
		    {
		      for (size_t row = st; row <= end; row++)
			res_write[row] +=
			    (x[ix_arr[row]] < 0) ?
				fill_val :
				((x[ix_arr[row]] == chosen_cat) ?
				    single_cat_coef : 0);
		      return;
		    }

		  fill_val = (cnt_this > cnt_other) ? single_cat_coef : 0;
		  if (has_NA && fill_val)
		    {
		      for (size_t row = st; row <= end; row++)
			if (unlikely(x[ix_arr[row]] < 0))
			  res_write[row] += fill_val;
		    }
		  return;
		}

	      default:
		{
		  unexpected_error ();
		  break;
		}
	      }
	  }
	  break;
	case SubSet:
	  {
	    /* in this case, since the splits are by more than 1 variable, it's not possible to
	     divide missing/new categoricals by assigning weights, so they have to be imputed
	     in both cases, unless using random weights for the new ones, in which case they won't
	     need to be imputed for new, but sill need it for NA */

	    if (new_cat_action == Random && missing_action == Fail)
	      {
		for (size_t row = st; row <= end; row++)
		  res_write[row] += cat_coef[x[ix_arr[row]]];
		return;
	      }

	    if (!first_run)
	      {
		if (missing_action == Fail)
		  {
		    for (size_t row = st; row <= end; row++)
		      res_write[row] +=
			  (x[ix_arr[row]] >= ncat) ?
			      fill_new : cat_coef[x[ix_arr[row]]];
		  }

		else
		  {
		    for (size_t row = st; row <= end; row++)
		      res_write[row] +=
			  (x[ix_arr[row]] < 0) ?
			      fill_val :
			      ((x[ix_arr[row]] >= ncat) ?
				  fill_new : cat_coef[x[ix_arr[row]]]);
		  }
		return;
	      }

	    /* TODO: this buffer should be allocated externally */
	    std::vector<ldouble_safe> buffer_cnt (ncat + 1, 0.);
	    switch (missing_action)
	      {
	      case Fail:
		{
		  for (size_t row = st; row <= end; row++)
		    {
		      buffer_cnt[x[ix_arr[row]]] += w[ix_arr[row]];
		      res_write[row] += cat_coef[x[ix_arr[row]]];
		    }
		  break;
		}

	      default:
		{
		  for (size_t row = st; row <= end; row++)
		    {
		      if (likely(x[ix_arr[row]] >= 0))
			{
			  buffer_cnt[x[ix_arr[row]]] += w[ix_arr[row]];
			  res_write[row] += cat_coef[x[ix_arr[row]]];
			}

		      else
			{
			  buffer_cnt[ncat] += w[ix_arr[row]];
			}

		    }
		  break;
		}
	      }

	    switch (new_cat_action)
	      {
	      case Smallest:
		{
		  ldouble_safe smallest =
		      std::numeric_limits<ldouble_safe>::infinity ();
		  int cat_smallest = 0;
		  for (int cat = 0; cat < ncat; cat++)
		    {
		      if (buffer_cnt[cat] > 0 && buffer_cnt[cat] < smallest)
			{
			  smallest = buffer_cnt[cat];
			  cat_smallest = cat;
			}
		    }
		  fill_new = cat_coef[cat_smallest];
		  if (missing_action == Fail)
		    break;
		[[fallthrough]];

		}

	      default:
		{
		  /* Determine imputation value as the category in sorted order that gives 50% + 1 */
		  ldouble_safe cnt_l = std::accumulate (
		      buffer_cnt.begin (), buffer_cnt.begin () + ncat,
		      (ldouble_safe) 0);
		  std::iota (buffer_pos, buffer_pos + ncat, (size_t) 0);
		  std::sort (buffer_pos, buffer_pos + ncat, [&cat_coef]
		  (const size_t a, const size_t b) 
		    { return cat_coef[a] < cat_coef[b];});

		  double cumprob = 0;
		  int cat;
		  for (cat = 0; cat < ncat; cat++)
		    {
		      cumprob += buffer_cnt[buffer_pos[cat]] / cnt_l;
		      if (cumprob >= .5)
			break;
		    }
		  // cat = std::min(cat, ncat); /* in case it picks the last one */
		  fill_val = cat_coef[buffer_pos[cat]];
		  if (new_cat_action != Smallest)
		    fill_new = fill_val;

		  if (buffer_cnt[ncat] > 0 && fill_val) /* NAs */
		    for (size_t row = st; row <= end; row++)
		      if (unlikely(x[ix_arr[row]] < 0))
			res_write[row] += fill_val;
		}
	      }

	    /* now fill unseen categories */
	    if (new_cat_action != Random)
	      for (int cat = 0; cat < ncat; cat++)
		if (!buffer_cnt[cat])
		  cat_coef[cat] = fill_new;

	  }
	}

    }

  template<class int_t, class ldouble_safe>
    double
    expected_sd_cat (double p[], size_t n, int_t pos[])
    {
      if (n <= 1)
	return 0;

      ldouble_safe cum_var = -square(p[pos[0]]) / 3.0
	  - p[pos[0]] * p[pos[1]] / 2.0 + p[pos[0]] / 3.0
	  - square(p[pos[1]]) / 3.0 + p[pos[1]] / 3.0;
      for (size_t cat1 = 2; cat1 < n; cat1++)
	{
	  cum_var += p[pos[cat1]] / 3.0 - square(p[pos[cat1]]) / 3.0;
	  for (size_t cat2 = 0; cat2 < cat1; cat2++)
	    cum_var -= p[pos[cat1]] * p[pos[cat2]] / 2.0;
	}
      return std::sqrt (std::fmax (cum_var, (ldouble_safe) 0));
    }

  template<class number, class int_t, class ldouble_safe>
    double
    expected_sd_cat (number *counts, double *p, size_t n, int_t *pos)
    {
      if (n <= 1)
	return 0;

      number tot = std::accumulate (pos, pos + n, (number) 0, [&counts]
      (number tot, const size_t ix) 
	{ return tot + counts[ix];});
      ldouble_safe cnt_div = (ldouble_safe) tot;
      for (size_t cat = 0; cat < n; cat++)
	p[pos[cat]] = (ldouble_safe) counts[pos[cat]] / cnt_div;

      return expected_sd_cat<int_t, ldouble_safe> (p, n, pos);
    }

  template<class number, class int_t, class ldouble_safe>
    double
    expected_sd_cat_single (number *counts, double *p, size_t n, int_t *pos,
			    size_t cat_exclude, number cnt)
    {
      if (cat_exclude == 0)
	return expected_sd_cat<number, int_t, ldouble_safe> (counts, p, n - 1,
							     pos + 1);

      else if (cat_exclude == (n - 1))
	return expected_sd_cat<number, int_t, ldouble_safe> (counts, p, n - 1,
							     pos);

      size_t ix_exclude = pos[cat_exclude];

      ldouble_safe cnt_div = (ldouble_safe) (cnt - counts[ix_exclude]);
      for (size_t cat = 0; cat < n; cat++)
	p[pos[cat]] = (ldouble_safe) counts[pos[cat]] / cnt_div;

      ldouble_safe cum_var;
      if (cat_exclude != 1)
	cum_var = -square(p[pos[0]]) / 3.0 - p[pos[0]] * p[pos[1]] / 2.0
	    + p[pos[0]] / 3.0 - square(p[pos[1]]) / 3.0 + p[pos[1]] / 3.0;
      else
	cum_var = -square(p[pos[0]]) / 3.0 - p[pos[0]] * p[pos[2]] / 2.0
	    + p[pos[0]] / 3.0 - square(p[pos[2]]) / 3.0 + p[pos[2]] / 3.0;
      for (size_t cat1 = (cat_exclude == 1) ? 3 : 2; cat1 < n; cat1++)
	{
	  if (pos[cat1] == ix_exclude)
	    continue;
	  cum_var += p[pos[cat1]] / 3.0 - square(p[pos[cat1]]) / 3.0;
	  for (size_t cat2 = 0; cat2 < cat1; cat2++)
	    {
	      if (pos[cat2] == ix_exclude)
		continue;
	      cum_var -= p[pos[cat1]] * p[pos[cat2]] / 2.0;
	    }

	}
      return std::sqrt (std::fmax (cum_var, (ldouble_safe) 0));
    }

  template<class number, class int_t, class ldouble_safe>
    double
    expected_sd_cat_internal (int ncat, number *buffer_cnt, ldouble_safe cnt_l,
			      int_t *buffer_pos, double *buffer_prob)
    {
      /* move zero-valued to the beginning */
      std::iota (buffer_pos, buffer_pos + ncat, (int_t) 0);
      int_t st_pos = 0;
      int ncat_present = 0;
      int_t temp;
      for (int cat = 0; cat < ncat; cat++)
	{
	  if (buffer_cnt[cat])
	    {
	      ncat_present++;
	      buffer_prob[cat] = (ldouble_safe) buffer_cnt[cat] / cnt_l;
	    }

	  else
	    {
	      temp = buffer_pos[st_pos];
	      buffer_pos[st_pos] = buffer_pos[cat];
	      buffer_pos[cat] = temp;
	      st_pos++;
	    }
	}

      if (ncat_present <= 1)
	return 0;
      return expected_sd_cat<int_t, ldouble_safe> (buffer_prob, ncat_present,
						   buffer_pos + st_pos);
    }

  template<class mapping, class int_t, class ldouble_safe>
    double
    expected_sd_cat_weighted (size_t *ix_arr, size_t st, size_t end, int x[],
			      int ncat, MissingAction missing_action,
			      mapping &w, double *buffer_cnt, int_t *buffer_pos,
			      double *buffer_prob)
    {
      /* generate counts */
      std::fill (buffer_cnt, buffer_cnt + ncat + 1, 0.);
      ldouble_safe cnt = 0;

      if (missing_action != Fail)
	{
	  int xval;
	  double w_this;
	  for (size_t row = st; row <= end; row++)
	    {
	      xval = x[ix_arr[row]];
	      w_this = w[ix_arr[row]];

	      if (unlikely(xval < 0))
		{
		  buffer_cnt[ncat] += w_this;
		}
	      else
		{
		  buffer_cnt[xval] += w_this;
		  cnt += w_this;
		}
	    }
	  if (cnt == 0)
	    return 0;
	}

      else
	{
	  for (size_t row = st; row <= end; row++)
	    {
	      if (likely(x[ix_arr[row]] >= 0))
		{
		  buffer_cnt[x[ix_arr[row]]] += w[ix_arr[row]];
		}
	    }
	  for (int cat = 0; cat < ncat; cat++)
	    cnt += buffer_cnt[cat];
	  if (unlikely(cnt == 0))
	    return 0;
	}

      return expected_sd_cat_internal<double, int_t, ldouble_safe> (ncat,
								    buffer_cnt,
								    cnt,
								    buffer_pos,
								    buffer_prob);
    }

  /* Note: this isn't exactly comparable to the pooled gain from numeric variables,
   but among all the possible options, this is what happens to end up in the most
   similar scale when considering standardized gain. */
  template<class number, class ldouble_safe>
    double
    categ_gain (number cnt_left, number cnt_right, ldouble_safe s_left,
		ldouble_safe s_right, ldouble_safe base_info, ldouble_safe cnt)
    {
      return (base_info
	  - (((cnt_left <= 1) ?
	      0 : ((ldouble_safe) cnt_left * std::log ((ldouble_safe) cnt_left)))
	      - s_left)
	  - (((cnt_right <= 1) ?
	      0 :
	      ((ldouble_safe) cnt_right * std::log ((ldouble_safe) cnt_right)))
	      - s_right)) / cnt;
    }

  template<class int_t, class ldouble_safe>
    double
    expected_sd_cat (size_t *ix_arr, size_t st, size_t end, int x[], int ncat,
		     MissingAction missing_action, size_t *buffer_cnt,
		     int_t *buffer_pos, double buffer_prob[])
    {
      /* generate counts */
      std::fill (buffer_cnt, buffer_cnt + ncat + 1, (size_t) 0);
      size_t cnt = end - st + 1;

      if (missing_action != Fail)
	{
	  int xval;
	  for (size_t row = st; row <= end; row++)
	    {
	      xval = x[ix_arr[row]];
	      if (unlikely(xval < 0))
		buffer_cnt[ncat]++;
	      else
		buffer_cnt[xval]++;
	    }
	  cnt -= buffer_cnt[ncat];
	  if (cnt == 0)
	    return 0;
	}

      else
	{
	  for (size_t row = st; row <= end; row++)
	    {
	      if (likely(x[ix_arr[row]] >= 0))
		buffer_cnt[x[ix_arr[row]]]++;
	    }
	}

      return expected_sd_cat_internal<size_t, int_t, ldouble_safe> (ncat,
								    buffer_cnt,
								    cnt,
								    buffer_pos,
								    buffer_prob);
    }
  template<class InputData, class WorkerMemory, class ldouble_safe>
    void
    calc_var_all_cols (InputData &input_data, WorkerMemory &workspace,
		       ModelParams &model_params, double *variances,
		       double *saved_xmin, double *saved_xmax,
		       double *saved_means, double *saved_sds)
    {
      double xmean, xsd;
      if (saved_means != NULL)
	workspace.has_saved_stats = true;

      workspace.col_sampler.prepare_full_pass ();
      while (workspace.col_sampler.sample_col (workspace.col_chosen))
	{
	  if (workspace.col_chosen < input_data.ncols_numeric)
	    {
	      get_split_range (workspace, input_data, model_params);
	      if (workspace.unsplittable)
		{
		  workspace.col_sampler.drop_col (workspace.col_chosen);
		  variances[workspace.col_chosen] = 0;
		  if (saved_xmin != NULL)
		          saved_xmin[workspace.col_chosen] = 0;
		  if (saved_xmax != NULL)		    
			  saved_xmax[workspace.col_chosen] = 0;
		    
		  continue;
		}

	      if (saved_xmin != NULL)
		{
		  saved_xmin[workspace.col_chosen] = workspace.xmin;
		  if(saved_xmax != NULL)
		 	 saved_xmax[workspace.col_chosen] = workspace.xmax;
		}

	    if (input_data.Xc_indptr == NULL)
		{
		  if (workspace.weights_arr.empty ()
		      && workspace.weights_map.empty ())
		    {
		      calc_mean_and_sd<
			  typename std::remove_pointer<
			      decltype(input_data.numeric_data)>::type,
			  ldouble_safe> (
			  workspace.ix_arr.data (),
			  workspace.st,
			  workspace.end,
			  input_data.numeric_data
			      + workspace.col_chosen * input_data.nrows,
			  model_params.missing_action, xsd, xmean);
		    }

		  else if (!workspace.weights_arr.empty ())
		    {
		      calc_mean_and_sd_weighted<
			  typename std::remove_pointer<
			      decltype(input_data.numeric_data)>::type,
			  decltype(workspace.weights_arr), ldouble_safe> (
			  workspace.ix_arr.data (),
			  workspace.st,
			  workspace.end,
			  input_data.numeric_data
			      + workspace.col_chosen * input_data.nrows,
			  workspace.weights_arr, model_params.missing_action,
			  xsd, xmean);
		    }

		  else
		    {
		      calc_mean_and_sd_weighted<
			  typename std::remove_pointer<
			      decltype(input_data.numeric_data)>::type,
			  decltype(workspace.weights_map), ldouble_safe> (
			  workspace.ix_arr.data (),
			  workspace.st,
			  workspace.end,
			  input_data.numeric_data
			      + workspace.col_chosen * input_data.nrows,
			  workspace.weights_map, model_params.missing_action,
			  xsd, xmean);
		    }
		}

	      else
		{
		  if (workspace.weights_arr.empty ()
		      && workspace.weights_map.empty ())
		    {
		      calc_mean_and_sd<
			  typename std::remove_pointer<decltype(input_data.Xc)>::type,
			  typename std::remove_pointer<
			      decltype(input_data.Xc_indptr)>::type,
			  ldouble_safe> (workspace.ix_arr.data (), workspace.st,
					 workspace.end, workspace.col_chosen,
					 input_data.Xc, input_data.Xc_ind,
					 input_data.Xc_indptr, xsd, xmean);
		    }

		  else if (!workspace.weights_arr.empty ())
		    {
		      calc_mean_and_sd_weighted<
			  typename std::remove_pointer<decltype(input_data.Xc)>::type,
			  typename std::remove_pointer<
			      decltype(input_data.Xc_indptr)>::type,
			  decltype(workspace.weights_arr), ldouble_safe> (
			  workspace.ix_arr.data (), workspace.st, workspace.end,
			  workspace.col_chosen, input_data.Xc,
			  input_data.Xc_ind, input_data.Xc_indptr, xsd, xmean,
			  workspace.weights_arr);
		    }

		  else
		    {
		      calc_mean_and_sd_weighted<
			  typename std::remove_pointer<decltype(input_data.Xc)>::type,
			  typename std::remove_pointer<
			      decltype(input_data.Xc_indptr)>::type,
			  decltype(workspace.weights_map), ldouble_safe> (
			  workspace.ix_arr.data (), workspace.st, workspace.end,
			  workspace.col_chosen, input_data.Xc,
			  input_data.Xc_ind, input_data.Xc_indptr, xsd, xmean,
			  workspace.weights_map);
		    }
		}

	      if (saved_means != NULL)
		saved_means[workspace.col_chosen] = xmean;
	      if (saved_sds != NULL)
		saved_sds[workspace.col_chosen] = xsd;
	    }

	  else
	    {
	      size_t col = workspace.col_chosen - input_data.ncols_numeric;
	      if (workspace.weights_arr.empty ()
		  && workspace.weights_map.empty ())
		{
		  if (workspace.buffer_szt.size ()
		      < (size_t) 2 * (size_t) input_data.ncat[col] + 1)
		    workspace.buffer_szt.resize (
			(size_t) 2 * (size_t) input_data.ncat[col] + 1);
		  xsd = expected_sd_cat<size_t, ldouble_safe> (
		      workspace.ix_arr.data (), workspace.st, workspace.end,
		      input_data.categ_data + col * input_data.nrows,
		      input_data.ncat[col], model_params.missing_action,
		      workspace.buffer_szt.data (),
		      workspace.buffer_szt.data () + input_data.ncat[col] + 1,
		      workspace.buffer_dbl.data ());
		}

	      else if (!workspace.weights_arr.empty ())
		{
		  if (workspace.buffer_dbl.size ()
		      < (size_t) 2 * (size_t) input_data.ncat[col] + 1)
		    workspace.buffer_dbl.resize (
			(size_t) 2 * (size_t) input_data.ncat[col] + 1);
		  xsd = expected_sd_cat_weighted<
		      decltype(workspace.weights_arr), size_t, ldouble_safe> (
		      workspace.ix_arr.data (), workspace.st, workspace.end,
		      input_data.categ_data + col * input_data.nrows,
		      input_data.ncat[col], model_params.missing_action,
		      workspace.weights_arr, workspace.buffer_dbl.data (),
		      workspace.buffer_szt.data (),
		      workspace.buffer_dbl.data () + input_data.ncat[col] + 1);
		}

	      else
		{
		  if (workspace.buffer_dbl.size ()
		      < (size_t) 2 * (size_t) input_data.ncat[col] + 1)
		    workspace.buffer_dbl.resize (
			(size_t) 2 * (size_t) input_data.ncat[col] + 1);
		  xsd = expected_sd_cat_weighted<
		      decltype(workspace.weights_map), size_t, ldouble_safe> (
		      workspace.ix_arr.data (), workspace.st, workspace.end,
		      input_data.categ_data + col * input_data.nrows,
		      input_data.ncat[col], model_params.missing_action,
		      workspace.weights_map, workspace.buffer_dbl.data (),
		      workspace.buffer_szt.data (),
		      workspace.buffer_dbl.data () + input_data.ncat[col] + 1);
		}
	    }

	  if (xsd)
	    {
	      variances[workspace.col_chosen] = square(xsd);
	      if (workspace.tree_kurtoses != NULL)
		variances[workspace.col_chosen] *=
		    workspace.tree_kurtoses[workspace.col_chosen];
	      else if (input_data.col_weights != NULL)
		variances[workspace.col_chosen] *=
		    input_data.col_weights[workspace.col_chosen];
	      variances[workspace.col_chosen] = std::fmax (
		  variances[workspace.col_chosen], 1e-100);
	    }

	  else
	    {
	      workspace.col_sampler.drop_col (workspace.col_chosen);
	      variances[workspace.col_chosen] = 0;
	    }
	}
    }
  inline double
  sample_random_uniform (double xmin, double xmax, RNG_engine &rng) noexcept
  {
    double out;
    std::uniform_real_distribution<double> runif (xmin, xmax);
    for (int attempt = 0; attempt < 100; attempt++)
      {
	out = runif (rng);
	if (likely(out < xmax))
	  return out;
      }
    return xmin;
  }
  
  real_t
  calc_sd_right_to_left (real_t *x, size_t n, double *sd_arr);

  template<typename T, size_t N>
    T
    distance_squared (const std::array<T, N> &point_a,
		      const std::array<T, N> &point_b)
    {
      T d_squared = T ();
      for (typename std::array<T, N>::size_type i = 0; i < N; ++i)
	{
	  auto delta = point_a[i] - point_b[i];
	  d_squared += delta * delta;
	}
      return d_squared;
    }

  template<typename T, size_t N>
    T
    distance (const std::array<T, N> &point_a, const std::array<T, N> &point_b)
    {
      return std::sqrt (distance_squared (point_a, point_b));
    }

  template<typename T, size_t N>
    std::vector<T>
    closest_distance (const std::vector<std::array<T, N>> &means,
		      const std::vector<std::array<T, N>> &data)
    {
      std::vector<T> distances;
      distances.reserve (data.size ());
      for (auto &d : data)
	{
	  T closest = distance_squared (d, means[0]);
	  for (auto &m : means)
	    {
	      T distance = distance_squared (d, m);
	      if (distance < closest)
		closest = distance;
	    }
	  distances.push_back (closest);
	}
      return distances;
    }

  template<typename T, size_t N>
    std::vector<std::array<T, N>>
    random_plusplus (const std::vector<std::array<T, N>> &data, uint32_t k,
		     uint64_t seed)
    {
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
	  auto distances = closest_distance (means, data);
	  // Pick a random point weighted by the distance from existing means
	  // TODO: This might convert floating point weights to ints, distorting the distribution for small weights
#if !defined(_MSC_VER) || _MSC_VER >= 1900
	  std::discrete_distribution<input_size_t> generator (
	      distances.begin (), distances.end ());
#else  // MSVC++ older than 14.0
		input_size_t i = 0;
		std::discrete_distribution<input_size_t> generator(distances.size(), 0.0, 0.0, [&distances, &i](double) { return distances[i++]; });
#endif
	  means.push_back (data[generator (rand_engine)]);
	}
      return means;
    }

  template<typename T, size_t N>
    uint32_t
    closest_mean (const std::array<T, N> &point,
		  const std::vector<std::array<T, N>> &means)
    {

      T smallest_distance = distance_squared (point, means[0]);
      typename std::array<T, N>::size_type index = 0;
      T distance;
      for (size_t i = 1; i < means.size (); ++i)
	{
	  distance = distance_squared (point, means[i]);
	  if (distance < smallest_distance)
	    {
	      smallest_distance = distance;
	      index = i;
	    }
	}
      return index;
    }

  template<typename T, size_t N>
    std::vector<uint32_t>
    calculate_clusters (const std::vector<std::array<T, N>> &data,
			const std::vector<std::array<T, N>> &means)
    {
      std::vector<uint32_t> clusters;
      for (auto &point : data)
	{
	  clusters.push_back (closest_mean (point, means));
	}
      return clusters;
    }

  template<typename T, size_t N>
    std::vector<std::array<T, N>>
    calculate_means (const std::vector<std::array<T, N>> &data,
		     const std::vector<uint32_t> &clusters,
		     const std::vector<std::array<T, N>> &old_means, uint32_t k)
    {
      std::vector<std::array<T, N>> means (k);
      std::vector<T> count (k, T ());
      for (size_t i = 0; i < std::min (clusters.size (), data.size ()); ++i)
	{
	  auto &mean = means[clusters[i]];
	  count[clusters[i]] += 1;
	  for (size_t j = 0; j < std::min (data[i].size (), mean.size ()); ++j)
	    {
	      mean[j] += data[i][j];
	    }
	}
      for (size_t i = 0; i < k; ++i)
	{
	  if (count[i] == 0)
	    {
	      means[i] = old_means[i];
	    }
	  else
	    {
	      for (size_t j = 0; j < means[i].size (); ++j)
		{
		  means[i][j] /= count[i];
		}
	    }
	}
      return means;
    }
  template<typename T, size_t N>
    std::vector<T>
    deltas (const std::vector<std::array<T, N>> &old_means,
	    const std::vector<std::array<T, N>> &means)
    {
      std::vector<T> distances;
      distances.reserve (means.size ());
      assert (old_means.size () == means.size ());
      for (size_t i = 0; i < means.size (); ++i)
	{
	  distances.push_back (distance (means[i], old_means[i]));
	}
      return distances;
    }

  template<typename T>
    bool
    deltas_below_limit (const std::vector<T> &deltas, T min_delta)
    {
      for (T d : deltas)
	{
	  if (d > min_delta)
	    {
	      return false;
	    }
	}
      return true;
    }

  template<typename real_value>
    int
    solveQuadratic (real_value a, real_value b, real_value c, real_value *x0,
		    real_value *x1)
    {
      real_value disc = b * b - 4 * a * c;

      if (a == 0)
	{
	  if (b == 0)
	    {
	      return 0;
	    }
	  else
	    {
	      *x0 = -c / b;
	      return 1;
	    };
	}

      if (disc > 0)
	{
	  if (b == 0)
	    {
	      real_value r = fabs (0.5 * sqrt (disc) / a);
	      *x0 = -r;
	      *x1 = r;
	    }
	  else
	    {
	      real_value sgnb = (b > 0 ? 1 : -1);
	      real_value temp = -0.5 * (b + sgnb * sqrt (disc));
	      real_value r1 = temp / a;
	      real_value r2 = c / temp;

	      if (r1 < r2)
		{
		  *x0 = r1;
		  *x1 = r2;
		}
	      else
		{
		  *x0 = r2;
		  *x1 = r1;
		}
	    }
	  return 2;
	}
      else if (disc == 0)
	{
	  *x0 = -0.5 * b / a;
	  *x1 = -0.5 * b / a;
	  return 2;
	}
      else
	{
	  return 0;
	}
    }

  template<typename real_value>
    real_value
    interpQuad (real_value f0, real_value fp0, real_value f1, real_value zl,
		real_value zh)
    {
      real_value fl = f0 + zl * (fp0 + zl * (f1 - f0 - fp0));
      real_value fh = f0 + zh * (fp0 + zh * (f1 - f0 - fp0));
      real_value c = 2 * (f1 - f0 - fp0);

      real_value zmin = zl, fmin = fl;

      if (fh < fmin)
	{
	  zmin = zh;
	  fmin = fh;
	}

      if (c > 0)
	{
	  real_value z = -fp0 / c;
	  if (z > zl && z < zh)
	    {
	      real_value f = f0 + z * (fp0 + z * (f1 - f0 - fp0));
	      if (f < fmin)
		{
		  zmin = z;
		  fmin = f;
		};
	    }
	}

      return zmin;
    }

  template<typename real_value>
    real_value
    cubic (real_value c0, real_value c1, real_value c2, real_value c3,
	   real_value z)
    {
      return c0 + z * (c1 + z * (c2 + z * c3));
    }

  template<typename real_value>
    void
    checkExtremum (real_value c0, real_value c1, real_value c2, real_value c3,
		   real_value z, real_value *zmin, real_value *fmin)
    {
      real_value y = cubic (c0, c1, c2, c3, z);
      if (y < *fmin)
	{
	  *zmin = z;
	  *fmin = y;
	}
    }

  template<typename real_value>
    real_value
    interpCubic (real_value f0, real_value fp0, real_value f1, real_value fp1,
		 real_value zl, real_value zh)
    {
      real_value eta = 3 * (f1 - f0) - 2 * fp0 - fp1;
      real_value xi = fp0 + fp1 - 2 * (f1 - f0);
      real_value c0 = f0, c1 = fp0, c2 = eta, c3 = xi;
      real_value zmin, fmin;
      real_value z0, z1;

      zmin = zl;
      fmin = cubic (c0, c1, c2, c3, zl);
      checkExtremum (c0, c1, c2, c3, zh, &zmin, &fmin);

      int n = solveQuadratic (3 * c3, 2 * c2, c1, &z0, &z1);

      if (n == 2)
	{
	  if (z0 > zl && z0 < zh)
	    checkExtremum (c0, c1, c2, c3, z0, &zmin, &fmin);
	  if (z1 > zl && z1 < zh)
	    checkExtremum (c0, c1, c2, c3, z1, &zmin, &fmin);
	}
      else if (n == 1)
	{
	  if (z0 > zl && z0 < zh)
	    checkExtremum (c0, c1, c2, c3, z0, &zmin, &fmin);
	}

      return zmin;
    }

  template<typename real_value>
    real_value
    interpolate (real_value a, real_value fa, real_value fpa, real_value b,
		 real_value fb, real_value fpb, real_value xmin,
		 real_value xmax, int order)
    {
      real_value z, alpha, zmin, zmax;

      zmin = (xmin - a) / (b - a);
      zmax = (xmax - a) / (b - a);

      if (zmin > zmax)
	{
	  real_value tmp = zmin;
	  zmin = zmax;
	  zmax = tmp;
	};

      if (order > 2 && not isnan (fpb))
	z = interpCubic (fa, fpa * (b - a), fb, fpb * (b - a), zmin, zmax);
      else
	z = interpQuad (fa, fpa * (b - a), fb, zmin, zmax);

      alpha = a + z * (b - a);

      return alpha;
    }
	
  struct wolfe_linear_search
  {
    template<typename Function, typename Gradient, typename Array>
      static typename Array::value_type
      alpha (typename Array::value_type alpha1, const Array &best,
	     const Array &dir, const Function &function,
	     typename Array::value_type fx, const Gradient &gradient,
	     const Array &current_gradient)
      {
	typedef typename Array::value_type real_value;

	// Max number of iterations
	static const size_t bracket_iters = 100, section_iters = 100;

	// Recommended values from Fletcher are :
	static const real_value rho = 0.01;
	static const real_value sigma = 0.1;
	static const real_value tau1 = 9;
	static const real_value tau2 = 0.05;
	static const real_value tau3 = 0.5;

	real_value falpha, fpalpha, delta, alpha_next;
	real_value alpha = alpha1, alpha_prev = 0.0;

	// Initialize function values
	real_value f0 = fx;
	real_value fp0 = dot (current_gradient, dir);

	real_value a (0.0), b (alpha), fa (f0), fb (0.0), fpa (fp0), fpb (0.0);

	// Initialize previous values
	real_value falpha_prev = f0;
	real_value fpalpha_prev = fp0;

	// Temporary value
	Array temp;

	// Begin bracketing
	size_t i = 0;
	while (i++ < bracket_iters)
	  {
	    // Calculate function in alpha
	    temp = best + alpha * dir;
	    falpha = function (temp);

	    // Fletcher's rho test
	    if (falpha > f0 + alpha * rho * fp0 || falpha >= falpha_prev)
	      {
		a = alpha_prev;
		fa = falpha_prev;
		fpa = fpalpha_prev;
		b = alpha;
		fb = falpha;
		fpb = NAN;
		break;
	      }

	    fpalpha = dot (gradient (temp), dir);

	    // Fletcher's sigma test
	    if (fabs (fpalpha) <= -sigma * fp0)
	      return alpha;

	    if (fpalpha >= 0)
	      {
		a = alpha;
		fa = falpha;
		fpa = fpalpha;
		b = alpha_prev;
		fb = falpha_prev;
		fpb = fpalpha_prev;
		break; // goto sectioning
	      }

	    delta = alpha - alpha_prev;

	    real_value lower = alpha + delta;
	    real_value upper = alpha + tau1 * delta;

	    alpha_next = interpolate (alpha_prev, falpha_prev, fpalpha_prev,
				      alpha, falpha, fpalpha, lower, upper, 3);

	    alpha_prev = alpha;
	    falpha_prev = falpha;
	    fpalpha_prev = fpalpha;
	    alpha = alpha_next;
	  }

	while (i++ < section_iters)
	  {
	    delta = b - a;

	    real_value lower = a + tau2 * delta;
	    real_value upper = b - tau3 * delta;

	    alpha = interpolate (a, fa, fpa, b, fb, fpb, lower, upper, 3);
	    temp = best + alpha * dir;
	    falpha = function (temp);

	    if ((a - alpha) * fpa
		<= std::numeric_limits<real_value>::epsilon ())
	      {
		// Roundoff prevents progress
		return alpha;
	      };

	    if (falpha > f0 + rho * alpha * fp0 || falpha >= fa)
	      {
		//  a_next = a;
		b = alpha;
		fb = falpha;
		fpb = NAN;
	      }
	    else
	      {
		fpalpha = dot (gradient (temp), dir);

		if (fabs (fpalpha) <= -sigma * fp0)
		  return alpha;  // terminate

		if (((b - a) >= 0 && fpalpha >= 0)
		    || ((b - a) <= 0 && fpalpha <= 0))
		  {
		    b = a;
		    fb = fa;
		    fpb = fpa;
		    a = alpha;
		    fa = falpha;
		    fpa = fpalpha;
		  }
		else
		  {
		    a = alpha;
		    fa = falpha;
		    fpa = fpalpha;
		  }
	      }
	  }
	return alpha;
      }
  };

  template<int Base>
    inline double
    log (double x)
    {
      return ::log (x) / ::log (Base);
    }

  // x log (x) function
  template<int Base>
    inline double
    xlog (double x)
    {
      if (x == 0.0)
	return 0.0;
      return x * log<Base> (x);
    }
	 
  	
	inline uint64_t
	fnv1a (const std::string &text)
	{
	 constexpr const  real_t fnv_prime = 16777619;
  	 constexpr const real_t fnv_offset_basis = 2166136261;

	  uint64_t hash = fnv_offset_basis;
	  for (size_t i = 0; i < text.size (); i++)
	    {
	      hash ^= text[i];
	      hash *= fnv_prime;
	    }
	  return hash;
	}	
	constexpr std::uint64_t wang64(const std::uint64_t &x) {
  		std::uint64_t y(x);
		y = (~y) + (y << 21);  // y = (y << 21) - y - 1;
		y = y ^ (y >> 24);
		y = (y + (y << 3)) + (y << 8);  // y * 265
		y = y ^ (y >> 14);
		y = (y + (y << 2)) + (y << 4);  // y * 21
		y = y ^ (y >> 28);
		y = y + (y << 31);
		return y;
	}
	constexpr std::uint32_t wang32(const std::uint32_t &x) {
  		std::uint32_t y(x);
		y = (~y) + (y << 15);  // y = (y << 15) - y - 1;
		y = y ^ (y >> 12);
		y = (y + (y << 2)) + (y << 4);  // y * 21
		y = y ^ (y >> 9);
		y = (y + (y << 3)) + (y << 4);  // y * 28
		y = y ^ (y >> 23);
		y = y + (y << 1) + (y << 4);  // y * 21 * 5
		return y;
	}
	constexpr std::uint16_t wang16(const std::uint16_t &x) {
  		std::uint16_t y(x);
		y = (~y) + (y << 7);  // y = (y << 7) - y - 1;
		y = y ^ (y >> 4);
		y = (y + (y << 3)) + (y << 4);  // y * 21
		y = y ^ (y >> 10);
		y = y + (y << 1) + (y << 4);  // y * 21 * 5
		return y;
	}
	template <typename UIntType>
	constexpr bool is_even(const UIntType val) {
  		return val & 1;
	}
	template <typename UIntType>
	inline bool is_pow2(const UIntType val) {
	if (val < 0) {  // safety against signed ints.

		throw std::invalid_argument(std::string("error: is_pow2 argument ") + std::to_string(val) + " should be nonnegative.");	
	}
	return !(val == 0) && !(val & (val - 1));
	}
	template <typename UIntType>
	inline UIntType next_pow2(const UIntType val) {
	if (val < 0) {  // safety against signed ints.

		throw std::invalid_argument(std::string("error: next_pow2 argument ") + std::to_string(val) + " should be nonnegative.");
	}
	if (is_pow2(val)) {
		return val;
	}
	UIntType next_pow2 = 1;
	while (next_pow2 < val) {
		next_pow2 <<= 1;
	}
	return next_pow2;
	}
	template <typename UIntType>
	inline UIntType prev_pow2(const UIntType val) {
	if (val < 0) {  // safety against signed ints.
	 
		throw std::invalid_argument(std::string("error: prev_pow2 argument ") + std::to_string(val) + " should be nonnegative.");
	}
	if (is_pow2(val)) {
		return val;
	}
	UIntType prev_pow2 = 1;
	while (prev_pow2 < val) {
		prev_pow2 <<= 1;
	}
	return prev_pow2 >> 1;
	}
	template <typename UIntType>
	inline UIntType log2(const UIntType val) {
	if (val < 0) {  // safety against signed ints.

	 			throw std::invalid_argument(std::string("error: log2 argument ") + std::to_string(val) + " should be nonnegative.");
	}
	if (val == 0) {
		return 0;
	}
	UIntType log2 = 0;
	UIntType value = val;
	while (value >>= 1) {
		++log2;
	}
	return log2;
	}
	template <typename UIntType>
	inline UIntType log2_ceil(const UIntType val) {
	if (val < 0) {  // safety against signed ints.

	 
		throw std::invalid_argument( std::string("error: log2_ceil argument ") + std::to_string(val) + " should be nonnegative." );
	}
	if (val == 0) {
		return 0;
	}
	UIntType log2 = 0;
	UIntType value = val - 1;
	while (value >>= 1) {
		++log2;
	}
	return log2 + 1;
	}//log2_ceil


   std::string
  trim (const std::string &pString, const std::string &pWhitespace);
   std::string
  reduce (const std::string &pString, const std::string &pFill,
	  const std::string &pWhitespace = " ");
}//namespace
#endif /* DECISION_ENGINE_UTILS_H_ */

