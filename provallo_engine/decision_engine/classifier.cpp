/*
 * classifier.cpp
 *
 *  Created on: Jan 19, 2022
 *      Author: kardon
 */



#include "utils.h"

#include "Density.hpp"
#include "classifier.h"
#include <cmath>
#include <memory.h>
#include <signal.h>
#include "crit.h"
#include <iostream>
#include <fstream>
#include <sstream>

#define square(x) std::pow(x, 2.0) // x^2 


namespace provallo
{

#ifndef _OPENMP
#define omp_get_thread_num() (0)
#endif

	using namespace provallo;

	bool interrupt_switch = false;

	template <typename in>
	bool
	check_more_than_two_unique_values(size_t ix_arr[], size_t st, size_t end,
									  in x[], MissingAction missing_action);

	typedef struct WorkerForSimilarity
	{
		std::vector<size_t> ix_arr;
		size_t st;
		size_t end;
		std::vector<real_t> weights_arr;
		std::vector<real_t> comb_val;
		std::vector<real_t> tmat_sep;
		std::vector<real_t> rmat;
		size_t n_from;
		bool assume_full_distr; /* doesn't need to have one copy per worker */
	} WorkerForSimilarity;

	typedef struct WorkerForPredictCSC
	{
		std::vector<size_t> ix_arr;
		size_t st;
		size_t end;
		std::vector<real_t> comb_val;
		std::vector<real_t> weights_arr;
		std::vector<real_t> depths;
	} WorkerForPredictCSC;

	/* Structs that are only used internally */
	struct InputData
	{
		real_t *numeric_data;
		size_t ncols_numeric;
		int *categ_data;
		int *ncat;
		int max_categ;
		size_t ncols_categ;
		size_t nrows;
		size_t ncols_tot;
		real_t *sample_weights;
		bool weight_as_sample;
		real_t *col_weights;
		real_t *Xc;								/* only for sparse matrices */
		sparse_ix *Xc_ind;						/* only for sparse matrices */
		sparse_ix *Xc_indptr;					/* only for sparse matrices */
		size_t log2_n;							/* only when using weights for sampling */
		size_t btree_offset;					/* only when using weights for sampling */
		std::vector<real_t> btree_weights_init; /* only when using weights for sampling */
		std::vector<char> has_missing;			/* only used when producing missing imputations on-the-fly */
		size_t n_missing;						/* only used when producing missing imputations on-the-fly */
		void *preinitialized_col_sampler;		/* only when using column weights */
		real_t *range_low;						/* only when calculating variable ranges or boxed densities with no sub-sampling */
		real_t *range_high;						/* only when calculating variable ranges or boxed densities with no sub-sampling */
		int *ncat_;								/* only when calculating boxed densities with no sub-sampling */
		std::vector<real_t> all_kurtoses;		/* only when using 'prob_pick_col_by_kurtosis' or mixing 'weigh_by_kurt' with 'prob_pick_col*' with no sub-sampling */
		std::vector<real_t> X_row_major;		/* created by this library, only used when calculating full gain */
		std::vector<real_t> Xr;					/* created by this library, only used when calculating full gain */
		std::vector<size_t> Xr_ind;				/* created by this library, only used when calculating full gain */
		std::vector<size_t> Xr_indptr;			/* created by this library, only used when calculating full gain */
	};
	// Impute generate/replaces N/A values
	struct ImputedData
	{
		std::vector<real_t> num_sum;
		std::vector<real_t> num_weight;
		std::vector<std::vector<real_t>> cat_sum;
		std::vector<real_t> cat_weight;
		std::vector<real_t> sp_num_sum;
		std::vector<real_t> sp_num_weight;

		std::vector<size_t> missing_num;
		std::vector<size_t> missing_cat;
		std::vector<sparse_ix> missing_sp;
		size_t n_missing_num;
		size_t n_missing_cat;
		size_t n_missing_sp;

		ImputedData()
		{
		}
		template <class InputData>
		ImputedData(InputData &input_data, size_t row);
	};
	// FW Declarations.

	size_t
	move_NAs_to_front(size_t ix_arr[], size_t st, size_t end, real_t x[]);
	template <class xreal, class real_t_, class mapping>
	real_t
	find_split_rel_gain_weighted_t(xreal *x, xreal xmean, size_t *ix_arr,
								   size_t st, size_t end, real_t &split_point,
								   size_t &split_ix, mapping &w);
	template <class InputData, class WorkerMemory>
	void
	add_separation_step(WorkerMemory &workspace, InputData &data,
						real_t remainder);
	size_t
	divide_subset_split(size_t ix_arr[], real_t x[], size_t st, size_t end,
						real_t split_point) noexcept;

	template <class xreal /*=real_t*/, class sparse_ix_ /*= sparse_ix*/,
			  class lreal_t_safe /*= long real_t*/>
	real_t
	eval_guided_crit(size_t ix_arr[], size_t st, size_t end, size_t col_num,
					 xreal Xc[], sparse_ix_ Xc_ind[], sparse_ix_ Xc_indptr[],
					 real_t buffer_arr[], size_t buffer_pos[],
					 bool as_relative_gain, real_t *saved_xmedian,
					 real_t &split_point, real_t &xmin, real_t &xmax,
					 GainCriterion criterion, real_t min_gain,
					 MissingAction missing_action, size_t *cols_use,
					 size_t ncols_use, bool force_cols_use,
					 real_t *X_row_major, size_t ncols, real_t *Xr,
					 size_t *Xr_ind, size_t *Xr_indptr);

	template <class lreal_t_safe>
	real_t
	eval_guided_crit_weighted(real_t *x, size_t n, GainCriterion criterion,
							  real_t min_gain, bool as_relative_gain,
							  real_t *buffer_sd, real_t &split_point,
							  real_t &xmin, real_t &xmax, real_t *w,
							  size_t *buffer_indices, size_t *ix_arr_plus_st,
							  size_t *cols_use, size_t ncols_use,
							  bool force_cols_use, real_t *X_row_major,
							  size_t ncols, real_t *Xr, size_t *Xr_ind,
							  size_t *Xr_indptr);

	template <class mapping, class lreal_t_safe>
	real_t
	eval_guided_crit_weighted(size_t *ix_arr, size_t st, size_t end, int *x,
							  int ncat, int *saved_cat_mode,
							  size_t *buffer_pos, real_t *buffer_prob,
							  int &chosen_cat, signed char *split_categ,
							  signed char *buffer_split,
							  GainCriterion criterion, real_t min_gain,
							  bool all_perm, MissingAction missing_action,
							  CategSplit cat_split_type, mapping &w);

	template <class xreal, class yreal>
	real_t
	find_split_std_gain_t(xreal *x, yreal xmean, size_t ix_arr[], size_t st,
						  size_t end, real_t *sd_arr, GainCriterion criterion,
						  real_t min_gain, real_t &split_point,
						  size_t &split_ix);

	void
	todense(size_t *ix_arr, size_t st, size_t end, size_t col_num, real_t *Xc,
			sparse_ix *Xc_ind,
			sparse_ix *Xc_indptr, real_t *buffer_arr);
	void
	fill_NAs_with_median(size_t *ix_arr, size_t st_orig, size_t st, size_t end,
						 real_t *x,
						 real_t *buffer_imputed_x, real_t *xmedian);

	template <class InputData, class WorkerMemory, class lreal_t_safe>
	void
	calc_var_all_cols(InputData &input_data, WorkerMemory &workspace,
					  ModelParams &model_params, real_t *variances,
					  real_t *saved_xmin, real_t *saved_xmax,
					  real_t *saved_means, real_t *saved_sds);

	template <class lreal_t_safe>
	real_t
	eval_guided_crit(real_t *x, size_t n, GainCriterion criterion,
					 real_t min_gain, bool as_relative_gain, real_t *buffer_sd,
					 real_t &split_point, real_t &xmin, real_t &xmax,
					 size_t *ix_arr_plus_st, size_t *cols_use,
					 size_t ncols_use, bool force_cols_use,
					 real_t *X_row_major, size_t ncols, real_t *Xr,
					 size_t *Xr_ind, size_t *Xr_indptr);
	template <class lreal_t_safe>
	real_t
	eval_guided_crit_weighted(real_t *x, size_t n, GainCriterion criterion,
							  real_t min_gain, bool as_relative_gain,
							  real_t *buffer_sd, real_t &split_point,
							  real_t &xmin, real_t &xmax, real_t *w,
							  size_t *buffer_indices, size_t *ix_arr_plus_st,
							  size_t *cols_use, size_t ncols_use,
							  bool force_cols_use, real_t *X_row_major,
							  size_t ncols, real_t *Xr, size_t *Xr_ind,
							  size_t *Xr_indptr);
	template <class real_t_, class lreal_t_safe>
	real_t
	eval_guided_crit(size_t *ix_arr, size_t st, size_t end, real_t_ *x,
					 real_t *buffer_sd, bool as_relative_gain,
					 real_t *buffer_imputed_x, real_t *saved_xmedian,
					 size_t &split_ix, real_t &split_point, real_t &xmin,
					 real_t &xmax, GainCriterion criterion, real_t min_gain,
					 MissingAction missing_action, size_t *cols_use,
					 size_t ncols_use, bool force_cols_use,
					 real_t *X_row_major, size_t ncols, real_t *Xr,
					 size_t *Xr_ind, size_t *Xr_indptr);
	template <class real_t_, class mapping, class lreal_t_safe>
	real_t
	eval_guided_crit_weighted(size_t *ix_arr, size_t st, size_t end,
							  real_t_ *x, real_t *buffer_sd,
							  bool as_relative_gain, real_t *buffer_imputed_x,
							  real_t *saved_xmedian, size_t &split_ix,
							  real_t &split_point, real_t &xmin, real_t &xmax,
							  GainCriterion criterion, real_t min_gain,
							  MissingAction missing_action, size_t *cols_use,
							  size_t ncols_use, bool force_cols_use,
							  real_t *X_row_major, size_t ncols, real_t *Xr,
							  size_t *Xr_ind, size_t *Xr_indptr, mapping &w);

	void
	weighted_shuffle(size_t *outp, size_t n, real_t *weights, real_t *buffer_arr,
					 RNG_engine &rnd_generator);
	real_t
	sample_random_uniform(real_t xmin, real_t xmax, RNG_engine &rng) noexcept;

	size_t
	move_NAs_to_front(size_t ix_arr[], size_t st, size_t end, real_t x[])
	{
		size_t st_non_na = st;
		size_t temp;

		for (size_t row = st; row <= end; row++)
		{
			if (unlikely(is_na_or_inf(x[ix_arr[row]])))
			{
				temp = ix_arr[st_non_na];
				ix_arr[st_non_na] = ix_arr[row];
				ix_arr[row] = temp;
				st_non_na++;
			}
		}

		return st_non_na;
	}
	template <class xreal, class sparse_ix_>
	size_t
	move_NAs_to_front(size_t *ix_arr, size_t st, size_t end, size_t col_num,
					  xreal Xc[], sparse_ix_ *Xc_ind, sparse_ix_ *Xc_indptr)
	{
		size_t st_non_na = st;
		size_t temp;

		size_t st_col = Xc_indptr[col_num];
		size_t end_col = Xc_indptr[col_num + 1] - 1;
		size_t curr_pos = st_col;
		size_t ind_end_col = Xc_ind[end_col];
		std::sort(ix_arr + st, ix_arr + end + 1);
		size_t *ptr_st = std::lower_bound(ix_arr + st, ix_arr + end + 1,
										  Xc_ind[st_col]);

		for (size_t *row = ptr_st;
			 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;)
		{
			if (Xc_ind[curr_pos] == *row)
			{
				if (unlikely(is_na_or_inf(Xc[curr_pos])))
				{
					temp = ix_arr[st_non_na];
					ix_arr[st_non_na] = *row;
					*row = temp;
					st_non_na++;
				}

				if (row == ix_arr + end || curr_pos == end_col)
					break;
				curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
											Xc_ind + end_col + 1, *(++row)) -
						   Xc_ind;
			}

			else
			{
				if (Xc_ind[curr_pos] > *row)
					row = std::lower_bound(row + 1, ix_arr + end + 1,
										   Xc_ind[curr_pos]);
				else
					curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
												Xc_ind + end_col + 1, *row) -
							   Xc_ind;
			}
		}

		return st_non_na;
	}

	size_t
	move_NAs_to_front(size_t ix_arr[], size_t st, size_t end, int x[])
	{
		size_t st_non_na = st;
		size_t temp;

		for (size_t row = st; row <= end; row++)
		{
			if (unlikely(x[ix_arr[row]] < 0))
			{
				temp = ix_arr[st_non_na];
				ix_arr[st_non_na] = ix_arr[row];
				ix_arr[row] = temp;
				st_non_na++;
			}
		}

		return st_non_na;
	}

	size_t
	center_NAs(size_t ix_arr[], size_t st_left, size_t st, size_t curr_pos)
	{
		size_t temp;
		for (size_t row = st_left; row < st; row++)
		{
			temp = ix_arr[--curr_pos];
			ix_arr[curr_pos] = ix_arr[row];
			ix_arr[row] = temp;
		}

		return curr_pos;
	}

	template <class InputData, class WorkerMemory, class lreal_t_safe>
	void
	calc_kurt_all_cols(InputData &data, WorkerMemory &workspace,
					   ModelParams &model_params, real_t *kurtosis,
					   real_t *saved_xmin, real_t *saved_xmax);

	inline bool
	is_boxed_metric(const ScoringMetric scoring_metric)
	{
		return scoring_metric == BoxedDensity || scoring_metric == BoxedDensity2 || scoring_metric == BoxedRatio;
	}
	void
	get_categs(size_t *ix_arr, int x[], size_t st, size_t end, int ncat,
			   MissingAction missing_action, signed char categs[],
			   size_t &npresent, bool &unsplittable) noexcept
	{
		std::fill(categs, categs + ncat, -1);

		npresent = (missing_action == MissingAction::Fail) ? 0 : 1;

		for (size_t row = st; row <= end; row++)
			if (likely(x[ix_arr[row]] >= 0))
				categs[x[ix_arr[row]]] = 1;

		npresent = std::accumulate(categs, categs + ncat, (size_t)0, [](const size_t a, const signed char b)
								   { return a + (b > 0); });

		unsplittable = npresent < 2;
	}

	inline void
	count_categs(size_t *ix_arr, size_t st, size_t end, int x[], int ncat,
				 size_t *counts)
	{
		std::fill(counts, counts + ncat, (size_t)0);
		for (size_t row = st; row <= end; row++)
			if (likely(x[ix_arr[row]] >= 0))
				counts[x[ix_arr[row]]]++;
	}

	int
	count_ncateg_in_col(const int x[], const size_t n, const int ncat,
						unsigned char buffer[])
	{
		::memset(buffer, 0, ncat * sizeof(char));
		for (size_t ix = 0; ix < n; ix++)
		{
			if (likely(x[ix] >= 0))
				buffer[x[ix]] = true;
		}

		int ncat_present = 0;
		for (int cat = 0; cat < ncat; cat++)
			ncat_present += buffer[cat];
		return ncat_present;
	}
	inline bool
	is_col_taken(std::vector<bool> &col_is_taken,
				 hashed_set<size_t> &col_is_taken_s,
				 size_t col_num)
	{
		if (!col_is_taken.empty())
			return col_is_taken[col_num];
		else
			return col_is_taken_s.find(col_num) != col_is_taken_s.end();
	}

	template <class InputData>
	void
	set_col_as_taken(std::vector<bool> &col_is_taken,
					 hashed_set<size_t> &col_is_taken_s,
					 InputData &input_data, size_t col_num, ColType col_type)
	{
		col_num += ((col_type == Numeric) ? 0 : input_data.ncols_numeric);
		if (!col_is_taken.empty())
			col_is_taken[col_num] = true;
		else
			col_is_taken_s.insert(col_num);
	}

	template <class InputData>
	void
	set_col_as_taken(std::vector<bool> &col_is_taken,
					 hashed_set<size_t> &col_is_taken_s,
					 InputData &input_data, size_t col_num)
	{
		UNDEF_REFERENCE(input_data)

		UNDEF_REFERENCE2(input_data)

		if (!col_is_taken.empty())
			col_is_taken[col_num] = true;
		else
			col_is_taken_s.insert(col_num);
	}

	real_t
	midpoint(real_t x, real_t y)
	{
		real_t m = x + (y - x) / (real_t)2;
		if (likely((real_t)m < (real_t)y))
			return m;
		else
		{
			m = std::nextafter(m, y);
			if (m > x && m < y)
				return m;
			else
				return x;
		}
	}
	/* for regular numeric columns */
	template <class real_t_>
	void
	get_range(size_t ix_arr[], real_t_ *x, size_t st, size_t end,
			  MissingAction missing_action, real_t &xmin, real_t &xmax,
			  bool &unsplittable) noexcept
	{
		xmin = HUGE_VAL;
		xmax = -HUGE_VAL;
		real_t xval;
		
		
		if (missing_action == Fail)
		{
			for (size_t row = st; row <= end; row++)
			{
				xval = x[ix_arr[row]];
				xmin = (xval < xmin) ? xval : xmin;
				xmax = (xval > xmax) ? xval : xmax;
			}
		}

		else
		{
			for (size_t row = st; row <= end; row++)
			{
				xval = x[ix_arr[row]];
				xmin = std::fmin(xmin, xval);
				xmax = std::fmax(xmax, xval);
			}
		}

		unsplittable = (xmin == xmax) || (xmin == HUGE_VAL && xmax == -HUGE_VAL) || std::isnan(xmin) || std::isnan(xmax);
	}

	template <class real_t_>
	void
	get_range(real_t_ *x, size_t n, MissingAction missing_action, real_t &xmin,
			  real_t &xmax, bool &unsplittable) noexcept
	{
		xmin = HUGE_VAL;
		xmax = -HUGE_VAL;

		if (missing_action == Fail)
		{
			for (size_t row = 0; row < n; row++)
			{
				xmin = (x[row] < xmin) ? x[row] : xmin;
				xmax = (x[row] > xmax) ? x[row] : xmax;
			}
		}

		else
		{
			for (size_t row = 0; row < n; row++)
			{
				xmin = std::fmin(xmin, x[row]);
				xmax = std::fmax(xmax, x[row]);
			}
		}

		unsplittable = (xmin == xmax) || (xmin == HUGE_VAL && xmax == -HUGE_VAL) || std::isnan(xmin) || std::isnan(xmax);
	}

	template <class real_t_, class sparse_ix_>
	void
	get_range(size_t *ix_arr, size_t st, size_t end, size_t col_num,
			  real_t_ *Xc, sparse_ix_ *Xc_ind, sparse_ix_ *Xc_indptr,
			  MissingAction missing_action, real_t &xmin, real_t &xmax,
			  bool &unsplittable) noexcept
	{
		/* ix_arr must already be sorted beforehand */
		xmin = HUGE_VAL;
		xmax = -HUGE_VAL;

		size_t st_col = Xc_indptr[col_num];
		size_t end_col = Xc_indptr[col_num + 1];
		size_t nnz_col = end_col - st_col;
		end_col--;
		size_t curr_pos = st_col;

		if (!nnz_col || Xc_ind[st_col] > (sparse_ix)ix_arr[end] || (sparse_ix)ix_arr[st] > Xc_ind[end_col])
		{
			unsplittable = true;
			return;
		}

		if (nnz_col < end - st + 1 || Xc_ind[st_col] > (sparse_ix)ix_arr[st] || Xc_ind[end_col] < (sparse_ix)ix_arr[end])
		{
			xmin = 0;
			xmax = 0;
		}

		size_t ind_end_col = Xc_ind[end_col];
		size_t nmatches = 0;

		if (missing_action == Fail)
		{
			for (size_t *row = std::lower_bound(ix_arr + st, ix_arr + end + 1,
												Xc_ind[st_col]);
				 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;)
			{
				if (Xc_ind[curr_pos] == (sparse_ix)(*row))
				{
					nmatches++;
					xmin = (Xc[curr_pos] < xmin) ? Xc[curr_pos] : xmin;
					xmax = (Xc[curr_pos] > xmax) ? Xc[curr_pos] : xmax;
					if (row == ix_arr + end || curr_pos == end_col)
						break;
					curr_pos = std::lower_bound(Xc_ind + curr_pos,
												Xc_ind + end_col + 1, *(++row)) -
							   Xc_ind;
				}

				else
				{
					if (Xc_ind[curr_pos] > (sparse_ix)(*row))
						row = std::lower_bound(row + 1, ix_arr + end + 1,
											   Xc_ind[curr_pos]);
					else
						curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
													Xc_ind + end_col + 1, *row) -
								   Xc_ind;
				}
			}
		}

		else /* can have NAs */
		{
			for (size_t *row = std::lower_bound(ix_arr + st, ix_arr + end + 1,
												Xc_ind[st_col]);
				 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;)
			{
				if (Xc_ind[curr_pos] == (sparse_ix)(*row))
				{
					nmatches++;
					xmin = std::fmin(xmin, Xc[curr_pos]);
					xmax = std::fmax(xmax, Xc[curr_pos]);
					if (row == ix_arr + end || curr_pos == end_col)
						break;
					curr_pos = std::lower_bound(Xc_ind + curr_pos,
												Xc_ind + end_col + 1, *(++row)) -
							   Xc_ind;
				}

				else
				{
					if (Xc_ind[curr_pos] > (sparse_ix)(*row))
						row = std::lower_bound(row + 1, ix_arr + end + 1,
											   Xc_ind[curr_pos]);
					else
						curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
													Xc_ind + end_col + 1, *row) -
								   Xc_ind;
				}
			}
		}

		if (nmatches < (end - st + 1))
		{
			xmin = std::fmin(xmin, 0);
			xmax = std::fmax(xmax, 0);
		}

		unsplittable = (xmin == xmax) || (xmin == HUGE_VAL && xmax == -HUGE_VAL) || std::isnan(xmin) || std::isnan(xmax);
	}
	template <class real_t_, class sparse_ix_>
	void
	get_range(size_t col_num, size_t nrows, real_t_ *Xc, sparse_ix_ *Xc_ind,
			  sparse_ix_ *Xc_indptr, MissingAction missing_action,
			  real_t &xmin, real_t &xmax, bool &unsplittable) noexcept
	{
		UNDEF_REFERENCE(Xc_ind)
		UNDEF_REFERENCE2(Xc_indptr)
		UNDEF_REFERENCE2(missing_action)
	
		xmin = HUGE_VAL;
		xmax = -HUGE_VAL;

		if ((size_t)(Xc_indptr[col_num + 1] - Xc_indptr[col_num]) < nrows)
		{
			xmin = 0;
			xmax = 0;
		}

		if (missing_action == Fail)
		{
			for (auto ix = Xc_indptr[col_num]; ix < Xc_indptr[col_num + 1]; ix++)
			{
				xmin = (Xc[ix] < xmin) ? Xc[ix] : xmin;
				xmax = (Xc[ix] > xmax) ? Xc[ix] : xmax;
			}
		}

		else
		{
			for (auto ix = Xc_indptr[col_num]; ix < Xc_indptr[col_num + 1]; ix++)
			{
				if (unlikely(std::isinf(Xc[ix])))
					continue;
				xmin = std::fmin(xmin, Xc[ix]);
				xmax = std::fmax(xmax, Xc[ix]);
			}
		}

		unsplittable = (xmin == xmax) || (xmin == HUGE_VAL && xmax == -HUGE_VAL) || std::isnan(xmin) || std::isnan(xmax);
	}

	template <class InputData, class WorkerMemory>
	void
	get_split_range(WorkerMemory &workspace, InputData &data,
					ModelParams &model_params, IsoTree &tree)
	{
		if (tree.col_num < data.ncols_numeric)
		{
			tree.col_type = Numeric;

			if (data.Xc_indptr == NULL)
				get_range(workspace.ix_arr.data(),
						  data.numeric_data + data.nrows * tree.col_num,
						  workspace.st, workspace.end, model_params.missing_action,
						  workspace.xmin, workspace.xmax, workspace.unsplittable);
			else
				get_range(workspace.ix_arr.data(), workspace.st, workspace.end,
						  tree.col_num, data.Xc, data.Xc_ind, data.Xc_indptr,
						  model_params.missing_action, workspace.xmin,
						  workspace.xmax, workspace.unsplittable);
		}

		else
		{
			tree.col_num -= data.ncols_numeric;
			tree.col_type = Categorical;

			get_categs(workspace.ix_arr.data(),
					   data.categ_data + data.nrows * tree.col_num, workspace.st,
					   workspace.end, data.ncat[tree.col_num],
					   model_params.missing_action, workspace.categs.data(),
					   workspace.npresent, workspace.unsplittable);
		}
	}

	template <class InputData, class WorkerMemory>
	void
	get_split_range(WorkerMemory &workspace, InputData &data,
					ModelParams &model_params)
	{
		if (workspace.col_chosen < data.ncols_numeric)
		{
			workspace.col_type = Numeric;

			if (data.Xc_indptr == NULL)
				get_range(workspace.ix_arr.data(),
						  data.numeric_data + data.nrows * workspace.col_chosen,
						  workspace.st, workspace.end, model_params.missing_action,
						  workspace.xmin, workspace.xmax, workspace.unsplittable);
			else
				get_range(workspace.ix_arr.data(), workspace.st, workspace.end,
						  workspace.col_chosen, data.Xc, data.Xc_ind,
						  data.Xc_indptr, model_params.missing_action,
						  workspace.xmin, workspace.xmax, workspace.unsplittable);
		}

		else
		{
			workspace.col_type = Categorical;
			workspace.col_chosen -= data.ncols_numeric;

			get_categs(workspace.ix_arr.data(),
					   data.categ_data + data.nrows * workspace.col_chosen,
					   workspace.st, workspace.end,
					   data.ncat[workspace.col_chosen],
					   model_params.missing_action, workspace.categs.data(),
					   workspace.npresent, workspace.unsplittable);
		}
	}

	/* for use in regular model with ntry>1 */
	template <class InputData, class WorkerMemory>
	void
	get_split_range_v2(WorkerMemory &workspace, InputData &data,
					   ModelParams &model_params)
	{
		get_split_range(workspace, data, model_params);
		if (workspace.col_type == Categorical)
			workspace.col_chosen += data.ncols_numeric;
	}

	template <class InputData, class WorkerMemory>
	int
	choose_cat_from_present(WorkerMemory &workspace, InputData &data,
							size_t col_num)
	{
		int chosen_cat = std::uniform_int_distribution<int>(
			0, workspace.npresent - 1)(workspace.rnd_generator);
		workspace.ncat_tried = 0;
		for (int cat = 0; cat < data.ncat[col_num]; cat++)
		{
			if (workspace.categs[cat] > 0)
			{
				if (workspace.ncat_tried == chosen_cat)
					return cat;
				else
					workspace.ncat_tried++;
			}
		}

		return -1; /* this will never be reached, but CRAN complains otherwise */
	}

	// FW:
	template <class real_t_, class mapping, class lreal_t_safe>
	real_t
	eval_guided_crit_weighted(size_t *ix_arr, size_t st, size_t end,
							  real_t_ *x, real_t *buffer_sd,
							  bool as_relative_gain, real_t *buffer_imputed_x,
							  real_t *saved_xmedian, size_t &split_ix,
							  real_t &split_point, real_t &xmin, real_t &xmax,
							  GainCriterion criterion, real_t min_gain,
							  MissingAction missing_action, size_t *cols_use,
							  size_t ncols_use, bool force_cols_use,
							  real_t *X_row_major, size_t ncols, real_t *Xr,
							  size_t *Xr_ind, size_t *Xr_indptr, mapping &w);

	template <class InputData, class WorkerMemory, class lreal_t_safe>
	void
	fit_itree(std::vector<IsoTree> *tree_root,
			  std::vector<IsoHPlane> *hplane_root, WorkerMemory &workspace,
			  InputData &input_data, ModelParams &model_params,
			  std::vector<ImputeNode> *impute_nodes, size_t tree_num);

	void
	shrink_impute_node(ImputeNode &imputer)
	{
		imputer.num_sum.clear();
		imputer.num_weight.clear();
		imputer.cat_sum.clear();
		imputer.cat_weight.clear();

		imputer.num_sum.shrink_to_fit();
		imputer.num_weight.shrink_to_fit();
		imputer.cat_sum.shrink_to_fit();
		imputer.cat_weight.shrink_to_fit();
	}

	void
	drop_nonterminal_imp_node(std::vector<ImputeNode> &imputer_tree,
							  std::vector<IsoTree> *trees,
							  std::vector<IsoHPlane> *hplanes)
	{
		if (trees != NULL)
		{
			for (size_t tr = 0; tr < trees->size(); tr++)
			{
				if ((*trees)[tr].tree_left != 0)
				{
					shrink_impute_node(imputer_tree[tr]);
				}

				else
				{
					/* cat_weight is not needed for anything else */
					imputer_tree[tr].cat_weight.clear();
					imputer_tree[tr].cat_weight.shrink_to_fit();
				}
			}
		}

		else
		{
			for (size_t tr = 0; tr < hplanes->size(); tr++)
			{
				if ((*hplanes)[tr].hplane_left != 0)
				{
					shrink_impute_node(imputer_tree[tr]);
				}

				else
				{
					/* cat_weight is not needed for anything else */
					imputer_tree[tr].cat_weight.clear();
					imputer_tree[tr].cat_weight.shrink_to_fit();
				}
			}
		}

		imputer_tree.shrink_to_fit();
	}

	template <class InputData, class WorkerMemory, class lreal_t_safe>
	void
	split_itree_recursive(std::vector<IsoTree> &trees, WorkerMemory &workspace,
						  InputData &data, ModelParams &model_params,
						  std::vector<ImputeNode> *impute_nodes,
						  size_t curr_depth)
	{
		lreal_t_safe sum_weight = -HUGE_VAL;

		/* calculate imputation statistics if desired */
		if (impute_nodes != NULL)
		{
			if (data.Xc_indptr != NULL)
				std::sort(workspace.ix_arr.begin() + workspace.st,
						  workspace.ix_arr.begin() + workspace.end + 1);
			build_impute_node<decltype(data), decltype(workspace), lreal_t_safe>(
				impute_nodes->back(), workspace, data, model_params,
				*impute_nodes, curr_depth, model_params.min_imp_obs);
		}

		/* check for potential isolated leafs or unique splits */
		if (workspace.end == workspace.st || (workspace.end - workspace.st) == 1 || curr_depth >= model_params.max_depth)
			goto terminal_statistics;

		/* when using weights, the split should stop when the sum of weights is <= 1 */
		if (workspace.changed_weights)
		{
			sum_weight = calculate_sum_weights<lreal_t_safe>(
				workspace.ix_arr, workspace.st, workspace.end, curr_depth,
				workspace.weights_arr, workspace.weights_map);
			if (curr_depth > 0 && sum_weight <= 1)
				goto terminal_statistics;
		}

		/* if there's no columns left to split, can end here */
		if (!workspace.col_sampler.get_remaining_cols())
			goto terminal_statistics;

		/* for sparse matrices, need to sort the indices */
		if (data.Xc_indptr != NULL && impute_nodes == NULL)
			std::sort(workspace.ix_arr.begin() + workspace.st,
					  workspace.ix_arr.begin() + workspace.end + 1);

		/* pick column to split according to criteria */
		workspace.prob_split_type = workspace.rbin(workspace.rnd_generator);

		/* case1: guided, pick column and/or point with best gain */
		if (workspace.prob_split_type < (model_params.prob_pick_by_gain_avg + model_params.prob_pick_by_gain_pl + model_params.prob_pick_by_full_gain + model_params.prob_pick_by_dens))
		{
			/* case 1.1: column and/or threshold is/are decided by averaged gain */
			if (workspace.prob_split_type < model_params.prob_pick_by_gain_avg)
				workspace.criterion = Averaged;

			/* case 1.2: column and/or threshold is/are decided by pooled gain */
			else if (workspace.prob_split_type < model_params.prob_pick_by_gain_avg + model_params.prob_pick_by_gain_pl)
				workspace.criterion = Pooled;
			/* case 1.3: column and/or threshold is/are decided by full gain (pooled gain in all columns) */
			else if (workspace.prob_split_type < model_params.prob_pick_by_gain_avg + model_params.prob_pick_by_gain_pl + model_params.prob_pick_by_full_gain)
				workspace.criterion = FullGain;
			/* case 1.4: column and/or threshold is/are decided by density pooled gain */
			else
				workspace.criterion = DensityCrit;

			workspace.determine_split = model_params.ntry <= 1 || workspace.col_sampler.get_remaining_cols() == 1;

			if (workspace.criterion == FullGain)
			{
				workspace.col_sampler.get_array_remaining_cols(
					workspace.col_indices);
			}
		}

		/* case2: column and split point is decided at random */
		else
		{
			workspace.criterion = NoCrit;
			workspace.determine_split = true;
		}

		/* pick column selection method also according to criteria */
		if ((workspace.criterion != NoCrit && std::max(workspace.ntry, (size_t)1) >= workspace.col_sampler.get_remaining_cols()) || (workspace.col_sampler.get_remaining_cols() == 1))
		{
			workspace.prob_split_type = 0;
		}
		else
		{
			workspace.prob_split_type = workspace.rbin(workspace.rnd_generator);
		}

		if (workspace.prob_split_type < model_params.prob_pick_col_by_range)
		{
			workspace.col_criterion = ByRange;
			if (curr_depth == 0 && is_boxed_metric(model_params.scoring_metric))
			{
				workspace.has_saved_stats = false;
				for (size_t col = 0; col < data.ncols_numeric; col++)
					workspace.node_col_weights[col] =
						workspace.density_calculator.box_high[col] - workspace.density_calculator.box_low[col];

			add_col_weights_to_ranges:
				if (workspace.tree_kurtoses != NULL)
				{
					for (size_t col = 0; col < data.ncols_numeric; col++)
					{
						if (workspace.node_col_weights[col] <= 0)
							continue;
						workspace.node_col_weights[col] *=
							workspace.tree_kurtoses[col];
						workspace.node_col_weights[col] = std::fmax(
							workspace.node_col_weights[col], 1e-100);
					}
				}
				else if (data.col_weights != NULL)
				{
					for (size_t col = 0; col < data.ncols_numeric; col++)
					{
						if (workspace.node_col_weights[col] <= 0)
							continue;
						workspace.node_col_weights[col] *= data.col_weights[col];
						workspace.node_col_weights[col] = std::fmax(
							workspace.node_col_weights[col], 1e-100);
					}
				}
			}

			else if (curr_depth == 0 && model_params.sample_size == data.nrows && !model_params.with_replacement && data.range_low != NULL && model_params.ncols_per_tree == data.ncols_tot)
			{
				workspace.has_saved_stats = false;
				for (size_t col = 0; col < data.ncols_numeric; col++)
					workspace.node_col_weights[col] = data.range_high[col] - data.range_low[col];
				goto add_col_weights_to_ranges;
			}

			else
			{
				workspace.has_saved_stats = workspace.criterion == NoCrit;
				calc_ranges_all_cols(
					data,
					workspace,
					model_params,
					workspace.node_col_weights.data(),
					workspace.has_saved_stats ? workspace.saved_stat1.data() : NULL,
					workspace.has_saved_stats ? workspace.saved_stat2.data() : NULL);
			}
		}

		else if (workspace.prob_split_type < (model_params.prob_pick_col_by_range + model_params.prob_pick_col_by_var))
		{
			workspace.col_criterion = ByVar;
			workspace.has_saved_stats = workspace.criterion == NoCrit;
			calc_var_all_cols<InputData, WorkerMemory, lreal_t_safe>(
				data, workspace, model_params, workspace.node_col_weights.data(),
				workspace.has_saved_stats ? workspace.saved_stat1.data() : NULL,
				workspace.has_saved_stats ? workspace.saved_stat2.data() : NULL,
				NULL,
				NULL);
		}

		else if (workspace.prob_split_type < (model_params.prob_pick_col_by_range + model_params.prob_pick_col_by_var + model_params.prob_pick_col_by_kurt))
		{
			workspace.col_criterion = ByKurt;
			workspace.has_saved_stats = workspace.criterion == NoCrit;
			calc_kurt_all_cols<decltype(data), decltype(workspace), lreal_t_safe>(
				data, workspace, model_params, workspace.node_col_weights.data(),
				workspace.has_saved_stats ? workspace.saved_stat1.data() : NULL,
				workspace.has_saved_stats ? workspace.saved_stat2.data() : NULL);
		}

		else
		{
			workspace.col_criterion = Uniformly;
		}

		if (workspace.col_criterion != Uniformly)
		{
			if (!workspace.node_col_sampler.initialize(
					workspace.node_col_weights.data(),
					&workspace.col_sampler.col_indices,
					workspace.col_sampler.curr_pos,
					(workspace.criterion == NoCrit) ? 1 : model_params.ntry, false))
			{
				goto terminal_statistics;
			}
		}

		/* when column is chosen at random */
		if (workspace.determine_split)
		{
			if (workspace.col_criterion != Uniformly)
			{
				if (!workspace.node_col_sampler.sample_col(
						trees.back().col_num, workspace.rnd_generator))
				{
					goto terminal_statistics;
				}

				if (trees.back().col_num < data.ncols_numeric)
				{
					trees.back().col_type = Numeric;
					if (workspace.has_saved_stats)
					{
						workspace.xmin =
							workspace.saved_stat1[trees.back().col_num];
						workspace.xmax =
							workspace.saved_stat2[trees.back().col_num];
					}

					else
					{
						get_split_range(workspace, data, model_params,
										trees.back());
						if (workspace.unsplittable)
							unexpected_error();
					}
				}

				else
				{
					get_split_range(workspace, data, model_params,
									trees.back());
					if (workspace.unsplittable)
						unexpected_error();
				}

				goto produce_split;
			}

			if (!workspace.col_sampler.has_weights())
			{
				while (workspace.col_sampler.sample_col(trees.back().col_num,
														workspace.rnd_generator))
				{
					if (interrupt_switch)
						return;

					get_split_range(workspace, data, model_params,
									trees.back());
					if (workspace.unsplittable)
						workspace.col_sampler.drop_col(
							trees.back().col_num + ((trees.back().col_type == Numeric) ? (size_t)0 : data.ncols_numeric));
					else
						goto produce_split;
				}
				goto terminal_statistics;
			}

			else
			{
				if (workspace.try_all)
					workspace.col_sampler.shuffle_remainder(
						workspace.rnd_generator);
				workspace.ntried = 0;
				size_t threshold_shuffle =
					(workspace.col_sampler.get_remaining_cols() + 1) / 2;

				while (
					workspace.try_all ? workspace.col_sampler.sample_col(trees.back().col_num) : workspace.col_sampler.sample_col(trees.back().col_num, workspace.rnd_generator))
				{
					if (interrupt_switch)
						return;

					get_split_range(workspace, data, model_params,
									trees.back());
					if (workspace.unsplittable)
					{
						workspace.col_sampler.drop_col(
							trees.back().col_num + ((trees.back().col_type == Numeric) ? (size_t)0 : data.ncols_numeric));
						workspace.ntried++;
						if (!workspace.try_all && workspace.ntried >= threshold_shuffle)
						{
							workspace.try_all = true;
							workspace.col_sampler.shuffle_remainder(
								workspace.rnd_generator);
						}
					}

					else
					{
						goto produce_split;
					}
				}
				goto terminal_statistics;
			}
		}

		/* when choosing both column and threshold */
		else
		{
			if (model_params.ntry >= workspace.col_sampler.get_remaining_cols())
				workspace.col_sampler.prepare_full_pass();
			else if (workspace.try_all && workspace.col_criterion == Uniformly)
				workspace.col_sampler.shuffle_remainder(workspace.rnd_generator);

			std::vector<bool> col_is_taken;
			hashed_set<size_t> col_is_taken_s;
			if (model_params.ntry < workspace.col_sampler.get_remaining_cols() && workspace.col_criterion == Uniformly)
			{
				if (data.ncols_tot < 1e5 || ((lreal_t_safe)model_params.ntry / (lreal_t_safe)workspace.col_sampler.get_remaining_cols()) > .25)
				{
					col_is_taken.resize(data.ncols_tot, false);
				}
				else
				{
					col_is_taken_s.reserve(model_params.ntry);
				}
			}

			size_t threshold_shuffle =
				(workspace.col_sampler.get_remaining_cols() + 1) / 2;
			workspace.ntried = 0;			/* <- used to determine when to shuffle the remainder */
			workspace.ntaken = 0;			/* <- used to count how many columns have been evaluated */
			trees.back().score = -HUGE_VAL; /* this is used to track the best gain */

			while (
				(workspace.col_criterion != Uniformly) ? workspace.node_col_sampler.sample_col(
															 workspace.col_chosen, workspace.rnd_generator)
													   : (workspace.try_all ? workspace.col_sampler.sample_col(workspace.col_chosen) : workspace.col_sampler.sample_col(workspace.col_chosen, workspace.rnd_generator)))
			{
				if (interrupt_switch)
					return;

				if (workspace.col_criterion != Uniformly)
				{
					workspace.ntaken++;
					goto probe_this_col;
				}

				workspace.ntried++;
				if (!workspace.try_all && workspace.ntried >= threshold_shuffle)
				{
					workspace.try_all = true;
					workspace.col_sampler.shuffle_remainder(
						workspace.rnd_generator);
				}

				if ((col_is_taken.size() || col_is_taken_s.size()) && !workspace.try_all)
				{
					if (is_col_taken(col_is_taken, col_is_taken_s,
									 workspace.col_chosen))
						continue;
					set_col_as_taken(col_is_taken, col_is_taken_s, data,
									 workspace.col_chosen);
				}

				get_split_range_v2(workspace, data, model_params);
				if (workspace.unsplittable)
				{
					workspace.col_sampler.drop_col(workspace.col_chosen);
					continue;
				}

				else
				{
				probe_this_col:
					if (workspace.col_chosen < data.ncols_numeric)
					{
						if (data.Xc_indptr == NULL)
						{
							if (!workspace.changed_weights)
								workspace.this_gain = eval_guided_crit<
									typename std::remove_pointer<
										decltype(data.numeric_data)>::type,
									lreal_t_safe>(
									workspace.ix_arr.data(),
									workspace.st,
									workspace.end,
									data.numeric_data + workspace.col_chosen * data.nrows,
									workspace.buffer_dbl.data(), false,
									workspace.imputed_x_buffer.data(),
									&workspace.saved_xmedian, workspace.split_ix,
									workspace.this_split_point, workspace.xmin,
									workspace.xmax, workspace.criterion,
									model_params.min_gain,
									model_params.missing_action,
									workspace.col_indices.data(),
									workspace.col_sampler.get_remaining_cols(),
									model_params.ncols_per_tree < data.ncols_tot,
									data.X_row_major.data(), data.ncols_numeric,
									data.Xr.data(), data.Xr_ind.data(),
									data.Xr_indptr.data());
							else if (!workspace.weights_arr.empty())
								workspace.this_gain = eval_guided_crit_weighted<
									typename std::remove_pointer<
										decltype(data.numeric_data)>::type,
									decltype(workspace.weights_arr), lreal_t_safe>(
									workspace.ix_arr.data(),
									workspace.st,
									workspace.end,
									data.numeric_data + workspace.col_chosen * data.nrows,
									workspace.buffer_dbl.data(), false,
									workspace.imputed_x_buffer.data(),
									&workspace.saved_xmedian, workspace.split_ix,
									workspace.this_split_point, workspace.xmin,
									workspace.xmax, workspace.criterion,
									model_params.min_gain,
									model_params.missing_action,
									workspace.col_indices.data(),
									workspace.col_sampler.get_remaining_cols(),
									model_params.ncols_per_tree < data.ncols_tot,
									data.X_row_major.data(), data.ncols_numeric,
									data.Xr.data(), data.Xr_ind.data(),
									data.Xr_indptr.data(), workspace.weights_arr);
							else
								workspace.this_gain = eval_guided_crit_weighted<
									typename std::remove_pointer<
										decltype(data.numeric_data)>::type,
									decltype(workspace.weights_map), lreal_t_safe>(
									workspace.ix_arr.data(),
									workspace.st,
									workspace.end,
									data.numeric_data + workspace.col_chosen * data.nrows,
									workspace.buffer_dbl.data(), false,
									workspace.imputed_x_buffer.data(),
									&workspace.saved_xmedian, workspace.split_ix,
									workspace.this_split_point, workspace.xmin,
									workspace.xmax, workspace.criterion,
									model_params.min_gain,
									model_params.missing_action,
									workspace.col_indices.data(),
									workspace.col_sampler.get_remaining_cols(),
									model_params.ncols_per_tree < data.ncols_tot,
									data.X_row_major.data(), data.ncols_numeric,
									data.Xr.data(), data.Xr_ind.data(),
									data.Xr_indptr.data(), workspace.weights_map);
						}

						else
						{
							if (!workspace.changed_weights)
								workspace.this_gain =
									eval_guided_crit<
										typename std::remove_pointer<
											decltype(data.Xc)>::type,
										typename std::remove_pointer<
											decltype(data.Xc_indptr)>::type,
										lreal_t_safe>(
										workspace.ix_arr.data(),
										workspace.st,
										workspace.end,
										workspace.col_chosen,
										data.Xc,
										data.Xc_ind,
										data.Xc_indptr,
										workspace.buffer_dbl.data(),
										workspace.buffer_szt.data(),
										false,
										&workspace.saved_xmedian,
										workspace.this_split_point,
										workspace.xmin,
										workspace.xmax,
										workspace.criterion,
										model_params.min_gain,
										model_params.missing_action,
										workspace.col_indices.data(),
										workspace.col_sampler.get_remaining_cols(),
										model_params.ncols_per_tree < data.ncols_tot,
										data.X_row_major.data(),
										data.ncols_numeric, data.Xr.data(),
										data.Xr_ind.data(),
										data.Xr_indptr.data());
							else if (!workspace.weights_arr.empty())
								workspace.this_gain =
									eval_guided_crit_weighted<
										typename std::remove_pointer<
											decltype(data.Xc)>::type,
										typename std::remove_pointer<
											decltype(data.Xc_indptr)>::type,
										decltype(workspace.weights_arr),
										lreal_t_safe>(
										workspace.ix_arr.data(),
										workspace.st,
										workspace.end,
										workspace.col_chosen,
										data.Xc,
										data.Xc_ind,
										data.Xc_indptr,
										workspace.buffer_dbl.data(),
										workspace.buffer_szt.data(),
										false,
										&workspace.saved_xmedian,
										workspace.this_split_point,
										workspace.xmin,
										workspace.xmax,
										workspace.criterion,
										model_params.min_gain,
										model_params.missing_action,
										workspace.col_indices.data(),
										workspace.col_sampler.get_remaining_cols(),
										model_params.ncols_per_tree < data.ncols_tot,
										data.X_row_major.data(),
										data.ncols_numeric, data.Xr.data(),
										data.Xr_ind.data(), data.Xr_indptr.data(),
										workspace.weights_arr);
							else
								workspace.this_gain =
									eval_guided_crit_weighted<
										typename std::remove_pointer<
											decltype(data.Xc)>::type,
										typename std::remove_pointer<
											decltype(data.Xc_indptr)>::type,
										decltype(workspace.weights_map),
										lreal_t_safe>(
										workspace.ix_arr.data(),
										workspace.st,
										workspace.end,
										workspace.col_chosen,
										data.Xc,
										data.Xc_ind,
										data.Xc_indptr,
										workspace.buffer_dbl.data(),
										workspace.buffer_szt.data(),
										false,
										&workspace.saved_xmedian,
										workspace.this_split_point,
										workspace.xmin,
										workspace.xmax,
										workspace.criterion,
										model_params.min_gain,
										model_params.missing_action,
										workspace.col_indices.data(),
										workspace.col_sampler.get_remaining_cols(),
										model_params.ncols_per_tree < data.ncols_tot,
										data.X_row_major.data(),
										data.ncols_numeric, data.Xr.data(),
										data.Xr_ind.data(), data.Xr_indptr.data(),
										workspace.weights_map);
						}
					}

					else
					{
						if (!workspace.changed_weights)
							workspace.this_gain =
								eval_guided_crit<lreal_t_safe>(
									workspace.ix_arr.data(),
									workspace.st,
									workspace.end,
									data.categ_data + (workspace.col_chosen - data.ncols_numeric) * data.nrows,
									data.ncat[workspace.col_chosen - data.ncols_numeric],
									&workspace.saved_cat_mode,
									workspace.buffer_szt.data(),
									workspace.buffer_szt.data() + data.max_categ,
									workspace.buffer_dbl.data(),
									workspace.this_categ,
									workspace.this_split_categ.data(),
									workspace.buffer_chr.data(),
									workspace.criterion, model_params.min_gain,
									model_params.all_perm,
									model_params.missing_action,
									model_params.cat_split_type);
						else if (!workspace.weights_arr.empty())
							workspace.this_gain =
								eval_guided_crit_weighted<
									decltype(workspace.weights_arr), lreal_t_safe>(
									workspace.ix_arr.data(),
									workspace.st,
									workspace.end,
									data.categ_data + (workspace.col_chosen - data.ncols_numeric) * data.nrows,
									data.ncat[workspace.col_chosen - data.ncols_numeric],
									&workspace.saved_cat_mode,
									workspace.buffer_szt.data(),
									workspace.buffer_dbl.data(),
									workspace.this_categ,
									workspace.this_split_categ.data(),
									workspace.buffer_chr.data(),
									workspace.criterion, model_params.min_gain,
									model_params.all_perm,
									model_params.missing_action,
									model_params.cat_split_type,
									workspace.weights_arr);
						else
							workspace.this_gain =
								eval_guided_crit_weighted<
									decltype(workspace.weights_map), lreal_t_safe>(
									workspace.ix_arr.data(),
									workspace.st,
									workspace.end,
									data.categ_data + (workspace.col_chosen - data.ncols_numeric) * data.nrows,
									data.ncat[workspace.col_chosen - data.ncols_numeric],
									&workspace.saved_cat_mode,
									workspace.buffer_szt.data(),
									workspace.buffer_dbl.data(),
									workspace.this_categ,
									workspace.this_split_categ.data(),
									workspace.buffer_chr.data(),
									workspace.criterion, model_params.min_gain,
									model_params.all_perm,
									model_params.missing_action,
									model_params.cat_split_type,
									workspace.weights_map);
					}

					if (std::isnan(
							workspace.this_gain) ||
						workspace.this_gain <= -HUGE_VAL)
						continue;

					if (workspace.this_gain > trees.back().score)
					{
						if (workspace.col_chosen < data.ncols_numeric)
						{
							trees.back().score = workspace.this_gain;
							trees.back().col_num = workspace.col_chosen;
							trees.back().col_type = Numeric;
							trees.back().num_split = workspace.this_split_point;
							if (model_params.penalize_range)
							{
								trees.back().range_low = workspace.xmin - workspace.xmax + trees.back().num_split;
								trees.back().range_high = workspace.xmax - workspace.xmin + trees.back().num_split;
							}

							if (model_params.scoring_metric != Depth && !is_boxed_metric(model_params.scoring_metric))
							{
								workspace.density_calculator.save_range(
									workspace.xmin, workspace.xmax);
							}

							workspace.best_xmedian = workspace.saved_xmedian;
						}

						else
						{
							trees.back().score = workspace.this_gain;
							trees.back().col_num = workspace.col_chosen - data.ncols_numeric;
							trees.back().col_type = Categorical;
							switch (model_params.cat_split_type)
							{
							case SingleCateg:
							{
								trees.back().chosen_cat = workspace.this_categ;
								break;
							}

							case SubSet:
							{
								trees.back().cat_split.assign(
									workspace.this_split_categ.begin(),
									workspace.this_split_categ.begin() + data.ncat[trees.back().col_num]);
								break;
							}
							}

							workspace.best_cat_mode = workspace.saved_cat_mode;

							if (model_params.scoring_metric != Depth && !is_boxed_metric(model_params.scoring_metric))
							{
								if (model_params.scoring_metric == Density)
								{
									switch (model_params.cat_split_type)
									{
									case SingleCateg:
									{
										workspace.density_calculator.save_n_present(
											workspace.buffer_szt.data(),
											data.ncat[trees.back().col_num]);
										break;
									}

									case SubSet:
									{
										workspace.density_calculator.save_n_present_and_left(
											workspace.this_split_categ.data(),
											data.ncat[trees.back().col_num]);
										break;
									}
									}
								}

								else
								{
									workspace.density_calculator.save_counts(
										workspace.buffer_szt.data(),
										data.ncat[trees.back().col_num]);
								}
							}
						}
					}

					if (++workspace.ntaken >= model_params.ntry)
						break;
				}
			}

			if (!workspace.ntaken)
				goto terminal_statistics;

			if (trees.back().score <= 0.)
				goto terminal_statistics;
			else
				trees.back().score = 0.;
		}

		/* for numeric, choose a random point, or pick the best point as determined earlier */
	produce_split:
		if (trees.back().col_type == Numeric)
		{
			if (workspace.determine_split)
			{
				switch (workspace.criterion)
				{
				case NoCrit:
				{
					trees.back().num_split = sample_random_uniform(
						workspace.xmin, workspace.xmax,
						workspace.rnd_generator);
					break;
				}

				default:
				{
					if (data.Xc_indptr == NULL)
					{
						if (!workspace.changed_weights)
							workspace.this_gain = eval_guided_crit<
								typename std::remove_pointer<
									decltype(data.numeric_data)>::type,
								lreal_t_safe>(
								workspace.ix_arr.data(),
								workspace.st,
								workspace.end,
								data.numeric_data + trees.back().col_num * data.nrows,
								workspace.buffer_dbl.data(), true,
								workspace.imputed_x_buffer.data(),
								&workspace.best_xmedian, workspace.split_ix,
								trees.back().num_split, workspace.xmin,
								workspace.xmax, workspace.criterion,
								model_params.min_gain,
								model_params.missing_action,
								workspace.col_indices.data(),
								workspace.col_sampler.get_remaining_cols(),
								model_params.ncols_per_tree < data.ncols_tot,
								data.X_row_major.data(), data.ncols_numeric,
								data.Xr.data(), data.Xr_ind.data(),
								data.Xr_indptr.data());
						else if (!workspace.weights_arr.empty())
							workspace.this_gain = eval_guided_crit_weighted<
								typename std::remove_pointer<
									decltype(data.numeric_data)>::type,
								decltype(workspace.weights_arr), lreal_t_safe>(
								workspace.ix_arr.data(),
								workspace.st,
								workspace.end,
								data.numeric_data + trees.back().col_num * data.nrows,
								workspace.buffer_dbl.data(), true,
								workspace.imputed_x_buffer.data(),
								&workspace.best_xmedian, workspace.split_ix,
								trees.back().num_split, workspace.xmin,
								workspace.xmax, workspace.criterion,
								model_params.min_gain,
								model_params.missing_action,
								workspace.col_indices.data(),
								workspace.col_sampler.get_remaining_cols(),
								model_params.ncols_per_tree < data.ncols_tot,
								data.X_row_major.data(), data.ncols_numeric,
								data.Xr.data(), data.Xr_ind.data(),
								data.Xr_indptr.data(), workspace.weights_arr);
						else
							workspace.this_gain = eval_guided_crit_weighted<
								typename std::remove_pointer<
									decltype(data.numeric_data)>::type,
								decltype(workspace.weights_map), lreal_t_safe>(
								workspace.ix_arr.data(),
								workspace.st,
								workspace.end,
								data.numeric_data + trees.back().col_num * data.nrows,
								workspace.buffer_dbl.data(), true,
								workspace.imputed_x_buffer.data(),
								&workspace.best_xmedian, workspace.split_ix,
								trees.back().num_split, workspace.xmin,
								workspace.xmax, workspace.criterion,
								model_params.min_gain,
								model_params.missing_action,
								workspace.col_indices.data(),
								workspace.col_sampler.get_remaining_cols(),
								model_params.ncols_per_tree < data.ncols_tot,
								data.X_row_major.data(), data.ncols_numeric,
								data.Xr.data(), data.Xr_ind.data(),
								data.Xr_indptr.data(), workspace.weights_map);

						if (std::isnan(
								workspace.this_gain) ||
							workspace.this_gain <= -HUGE_VAL)
							goto terminal_statistics;

						if (model_params.missing_action == Fail || (model_params.missing_action == Impute && data.Xc_indptr == NULL)) /* data is already split in this case */
						{
							if (model_params.missing_action == Impute)
							{
								workspace.st_NA = workspace.split_ix + 1;
								workspace.end_NA = workspace.st_NA;
							}

							workspace.split_ix++;
							if (model_params.penalize_range)
							{
								trees.back().range_low = workspace.xmin - workspace.xmax + trees.back().num_split;
								trees.back().range_high = workspace.xmax - workspace.xmin + trees.back().num_split;
							}
							goto follow_branches;
						}
					}

					else
					{
						if (!workspace.changed_weights)
							workspace.this_gain =
								eval_guided_crit<
									typename std::remove_pointer<decltype(data.Xc)>::type,
									typename std::remove_pointer<
										decltype(data.Xc_indptr)>::type,
									lreal_t_safe>(
									workspace.ix_arr.data(), workspace.st,
									workspace.end, trees.back().col_num, data.Xc,
									data.Xc_ind, data.Xc_indptr,
									workspace.buffer_dbl.data(),
									workspace.buffer_szt.data(), true,
									&workspace.best_xmedian,
									trees.back().num_split, workspace.xmin,
									workspace.xmax, workspace.criterion,
									model_params.min_gain,
									model_params.missing_action,
									workspace.col_indices.data(),
									workspace.col_sampler.get_remaining_cols(),
									model_params.ncols_per_tree < data.ncols_tot,
									data.X_row_major.data(), data.ncols_numeric,
									data.Xr.data(), data.Xr_ind.data(),
									data.Xr_indptr.data());
						else if (!workspace.weights_arr.empty())
							workspace.this_gain =
								eval_guided_crit_weighted<
									typename std::remove_pointer<decltype(data.Xc)>::type,
									typename std::remove_pointer<
										decltype(data.Xc_indptr)>::type,
									decltype(workspace.weights_arr), lreal_t_safe>(
									workspace.ix_arr.data(), workspace.st,
									workspace.end, trees.back().col_num, data.Xc,
									data.Xc_ind, data.Xc_indptr,
									workspace.buffer_dbl.data(),
									workspace.buffer_szt.data(), true,
									&workspace.best_xmedian,
									trees.back().num_split, workspace.xmin,
									workspace.xmax, workspace.criterion,
									model_params.min_gain,
									model_params.missing_action,
									workspace.col_indices.data(),
									workspace.col_sampler.get_remaining_cols(),
									model_params.ncols_per_tree < data.ncols_tot,
									data.X_row_major.data(), data.ncols_numeric,
									data.Xr.data(), data.Xr_ind.data(),
									data.Xr_indptr.data(),
									workspace.weights_arr);
						else
							workspace.this_gain =
								eval_guided_crit_weighted<
									typename std::remove_pointer<decltype(data.Xc)>::type,
									typename std::remove_pointer<
										decltype(data.Xc_indptr)>::type,
									decltype(workspace.weights_map), lreal_t_safe>(
									workspace.ix_arr.data(), workspace.st,
									workspace.end, trees.back().col_num, data.Xc,
									data.Xc_ind, data.Xc_indptr,
									workspace.buffer_dbl.data(),
									workspace.buffer_szt.data(), true,
									&workspace.best_xmedian,
									trees.back().num_split, workspace.xmin,
									workspace.xmax, workspace.criterion,
									model_params.min_gain,
									model_params.missing_action,
									workspace.col_indices.data(),
									workspace.col_sampler.get_remaining_cols(),
									model_params.ncols_per_tree < data.ncols_tot,
									data.X_row_major.data(), data.ncols_numeric,
									data.Xr.data(), data.Xr_ind.data(),
									data.Xr_indptr.data(),
									workspace.weights_map);
					}

					if (std::isnan(
							workspace.this_gain) ||
						workspace.this_gain <= -HUGE_VAL)
						goto terminal_statistics;

					break;
				}
				}

				if (model_params.penalize_range)
				{
					trees.back().range_low = workspace.xmin - workspace.xmax + trees.back().num_split;
					trees.back().range_high = workspace.xmax - workspace.xmin + trees.back().num_split;
				}
			}

			if (model_params.missing_action == Fail && std::isnan(trees.back().num_split))
				throw std::runtime_error(
					"Data has missing values. Try using a different value for 'missing_action'.\n");

			/* TODO: make this work, can end up messing with the start and end indices somehow */
			/* It should also consider that 'split_ix' might not match when using missing_action == Impute */
			// if (input_data.Xc_indptr == NULL && model_params.missing_action == Fail && workspace.ntaken == 1)
			//     goto follow_branches;
			if (data.Xc_indptr == NULL)
				divide_subset_split(
					workspace.ix_arr.data(),
					data.numeric_data + data.nrows * trees.back().col_num,
					workspace.st, workspace.end, trees.back().num_split,
					model_params.missing_action, workspace.st_NA, workspace.end_NA,
					workspace.split_ix);
			else
				divide_subset_split(workspace.ix_arr.data(), workspace.st,
									workspace.end, trees.back().col_num, data.Xc,
									data.Xc_ind, data.Xc_indptr,
									trees.back().num_split,
									model_params.missing_action, workspace.st_NA,
									workspace.end_NA, workspace.split_ix);
		}

		/* for categorical, there are different ways of splitting */
		else
		{
			/* if the columns is binary, there's only one possible split */
			if (data.ncat[trees.back().col_num] <= 2)
			{
				trees.back().chosen_cat = 0;
				divide_subset_split(
					workspace.ix_arr.data(),
					data.categ_data + data.nrows * trees.back().col_num,
					workspace.st, workspace.end, (int)0,
					model_params.missing_action, workspace.st_NA,
					workspace.end_NA, workspace.split_ix);
				trees.back().cat_split.clear();
				trees.back().cat_split.shrink_to_fit();
			}

			/* otherwise, split according to desired type (single/subset) */
			/* TODO: refactor this */
			else
			{

				switch (model_params.cat_split_type)
				{

				case SingleCateg:
				{

					if (workspace.determine_split)
					{
						switch (workspace.criterion)
						{
						case NoCrit:
						{
							trees.back().chosen_cat =
								choose_cat_from_present(
									workspace, data, trees.back().col_num);
							break;
						}

						default:
						{
							if (!workspace.changed_weights)
								workspace.this_gain = eval_guided_crit<
									lreal_t_safe>(
									workspace.ix_arr.data(),
									workspace.st,
									workspace.end,
									data.categ_data + trees.back().col_num * data.nrows,
									data.ncat[trees.back().col_num],
									&workspace.best_cat_mode,
									workspace.buffer_szt.data(),
									workspace.buffer_szt.data() + data.max_categ,
									workspace.buffer_dbl.data(),
									trees.back().chosen_cat,
									workspace.this_split_categ.data(),
									workspace.buffer_chr.data(),
									workspace.criterion, model_params.min_gain,
									model_params.all_perm,
									model_params.missing_action,
									model_params.cat_split_type);
							else if (!workspace.weights_arr.empty())
								workspace.this_gain = eval_guided_crit_weighted<
									decltype(workspace.weights_arr),
									lreal_t_safe>(
									workspace.ix_arr.data(),
									workspace.st,
									workspace.end,
									data.categ_data + trees.back().col_num * data.nrows,
									data.ncat[trees.back().col_num],
									&workspace.best_cat_mode,
									workspace.buffer_szt.data(),
									workspace.buffer_dbl.data(),
									trees.back().chosen_cat,
									workspace.this_split_categ.data(),
									workspace.buffer_chr.data(),
									workspace.criterion, model_params.min_gain,
									model_params.all_perm,
									model_params.missing_action,
									model_params.cat_split_type,
									workspace.weights_arr);
							else
								workspace.this_gain = eval_guided_crit_weighted<
									decltype(workspace.weights_map),
									lreal_t_safe>(
									workspace.ix_arr.data(),
									workspace.st,
									workspace.end,
									data.categ_data + trees.back().col_num * data.nrows,
									data.ncat[trees.back().col_num],
									&workspace.best_cat_mode,
									workspace.buffer_szt.data(),
									workspace.buffer_dbl.data(),
									trees.back().chosen_cat,
									workspace.this_split_categ.data(),
									workspace.buffer_chr.data(),
									workspace.criterion, model_params.min_gain,
									model_params.all_perm,
									model_params.missing_action,
									model_params.cat_split_type,
									workspace.weights_map);

							if (std::isnan(
									workspace.this_gain) ||
								workspace.this_gain <= -HUGE_VAL)
								goto terminal_statistics;

							break;
						}
						}
					}

					divide_subset_split(
						workspace.ix_arr.data(),
						data.categ_data + data.nrows * trees.back().col_num,
						workspace.st, workspace.end, trees.back().chosen_cat,
						model_params.missing_action, workspace.st_NA,
						workspace.end_NA, workspace.split_ix);
					break;
				}

				case SubSet:
				{

					if (workspace.determine_split)
					{
						switch (workspace.criterion)
						{
						case NoCrit:
						{
							workspace.unsplittable = true;
							while (workspace.unsplittable)
							{
								workspace.npresent = 0;
								workspace.ncols_tried = 0;
								for (int cat = 0;
									 cat < data.ncat[trees.back().col_num];
									 cat++)
								{
									if (workspace.categs[cat] >= 0)
									{
										workspace.categs[cat] =
											workspace.rbin(
												workspace.rnd_generator) < 0.5;
										workspace.npresent +=
											workspace.categs[cat];
										workspace.ncols_tried +=
											!workspace.categs[cat];
									}
									workspace.unsplittable =
										!(workspace.npresent && workspace.ncols_tried);
								}
							}

							trees.back().cat_split.assign(
								workspace.categs.begin(),
								workspace.categs.begin() + data.ncat[trees.back().col_num]);
							break; /* NoCrit */
						}

						default:
						{
							trees.back().cat_split.resize(
								data.ncat[trees.back().col_num]);
							if (!workspace.changed_weights)
								workspace.this_gain =
									eval_guided_crit<lreal_t_safe>(
										workspace.ix_arr.data(),
										workspace.st,
										workspace.end,
										data.categ_data + trees.back().col_num * data.nrows,
										data.ncat[trees.back().col_num],
										&workspace.best_cat_mode,
										workspace.buffer_szt.data(),
										workspace.buffer_szt.data() + data.max_categ,
										workspace.buffer_dbl.data(),
										trees.back().chosen_cat,
										(signed char *)trees.back().cat_split.data(),
										workspace.buffer_chr.data(),
										workspace.criterion,
										model_params.min_gain,
										model_params.all_perm,
										model_params.missing_action,
										model_params.cat_split_type);
							else if (!workspace.weights_arr.empty())
								workspace.this_gain =
									eval_guided_crit_weighted<
										decltype(workspace.weights_arr),
										lreal_t_safe>(
										workspace.ix_arr.data(),
										workspace.st,
										workspace.end,
										data.categ_data + trees.back().col_num * data.nrows,
										data.ncat[trees.back().col_num],
										&workspace.best_cat_mode,
										workspace.buffer_szt.data(),
										workspace.buffer_dbl.data(),
										trees.back().chosen_cat,
										(signed char *)trees.back().cat_split.data(),
										workspace.buffer_chr.data(),
										workspace.criterion,
										model_params.min_gain,
										model_params.all_perm,
										model_params.missing_action,
										model_params.cat_split_type,
										workspace.weights_arr);
							else
								workspace.this_gain =
									eval_guided_crit_weighted<
										decltype(workspace.weights_map),
										lreal_t_safe>(
										workspace.ix_arr.data(),
										workspace.st,
										workspace.end,
										data.categ_data + trees.back().col_num * data.nrows,
										data.ncat[trees.back().col_num],
										&workspace.best_cat_mode,
										workspace.buffer_szt.data(),
										workspace.buffer_dbl.data(),
										trees.back().chosen_cat,
										(signed char *)trees.back().cat_split.data(),
										workspace.buffer_chr.data(),
										workspace.criterion,
										model_params.min_gain,
										model_params.all_perm,
										model_params.missing_action,
										model_params.cat_split_type,
										workspace.weights_map);

							if (std::isnan(
									workspace.this_gain) ||
								workspace.this_gain <= -HUGE_VAL)
								goto terminal_statistics;
							break;
						}
						}
					}

					if (model_params.new_cat_action == Random)
					{
						if (model_params.scoring_metric == Density)
						{
							workspace.density_calculator.save_n_present_and_left(
								(signed char *)trees.back().cat_split.data(),
								data.ncat[trees.back().col_num]);
						}

						for (int cat = 0;
							 cat < data.ncat[trees.back().col_num]; cat++)
							if (trees.back().cat_split[cat] < 0)
								trees.back().cat_split[cat] = workspace.rbin(
																  workspace.rnd_generator) < 0.5;
					}

					divide_subset_split(
						workspace.ix_arr.data(),
						data.categ_data + data.nrows * trees.back().col_num,
						workspace.st, workspace.end,
						(signed char *)trees.back().cat_split.data(),
						model_params.missing_action, workspace.st_NA,
						workspace.end_NA, workspace.split_ix);
				}
				}
			}
		}

		/* if it hasn't reached the limit, continue splitting from here */
	follow_branches:
	{
		/* add another round of separation depth for distance */
		if (model_params.calc_dist && curr_depth > 0)
			add_separation_step(workspace, data, (real_t)(-1));

		/* if it split by a categorical variable with only 2 values,
		 the column will no longer be splittable in either branch */
		if (trees.back().col_type == Categorical && ((model_params.cat_split_type == SubSet && trees.back().cat_split.empty()) || (model_params.cat_split_type == SingleCateg && data.ncat[trees.back().col_num] == 2)))
		{
			workspace.col_sampler.drop_col(
				trees.back().col_num + data.ncols_numeric,
				workspace.end - workspace.st + 1);
		}

		size_t tree_from = trees.size() - 1;
		std::unique_ptr<RecursionState> recursion_state(
			new RecursionState(workspace,
							   model_params.missing_action != Fail));
		trees.back().score = -1;

		/* compute statistics for NAs and remember recursion indices/weights */
		if (model_params.missing_action != Fail)
		{
			if (model_params.missing_action == Impute && workspace.criterion != NoCrit && workspace.st_NA < workspace.end_NA)
			{
				bool move_NAs_left;
				if (trees.back().col_type == Numeric)
				{
					move_NAs_left = workspace.best_xmedian <= trees.back().num_split;
				}

				else
				{
					if (trees.back().cat_split.empty())
						move_NAs_left = workspace.best_cat_mode == trees.back().chosen_cat;
					else
						move_NAs_left =
							trees.back().cat_split[workspace.best_cat_mode] == 1;
				}

				if (move_NAs_left)
					workspace.st_NA = workspace.end_NA;
				else
					workspace.end_NA = workspace.st_NA;
			}

			if (!workspace.changed_weights)
			{
				trees.back().pct_tree_left = (lreal_t_safe)(workspace.st_NA - workspace.st) / (lreal_t_safe)(workspace.end - workspace.st + 1 - (workspace.end_NA - workspace.st_NA));

				if (model_params.missing_action == Divide && workspace.st_NA < workspace.end_NA)
				{
					workspace.changed_weights = true;

					if (data.Xc_indptr != NULL && model_params.sample_size < data.nrows / 20)
					{
						workspace.weights_arr.clear();
						workspace.weights_map.reserve(
							workspace.end - workspace.st + 1);
						for (size_t row = workspace.st;
							 row < workspace.end_NA; row++)
							workspace.weights_map[workspace.ix_arr[row]] = 1;
					}

					else
					{
						workspace.weights_arr.resize(data.nrows);
						for (size_t row = workspace.st;
							 row < workspace.end_NA; row++)
							workspace.weights_arr[workspace.ix_arr[row]] = 1;
					}
				}
			}

			else
			{
				lreal_t_safe sum_weight_left = 0;
				lreal_t_safe sum_weight_right = 0;

				if (!workspace.weights_arr.empty())
				{
					for (size_t row = workspace.st; row < workspace.st_NA;
						 row++)
						sum_weight_left +=
							workspace.weights_arr[workspace.ix_arr[row]];
					for (size_t row = workspace.end_NA; row <= workspace.end;
						 row++)
						sum_weight_right +=
							workspace.weights_arr[workspace.ix_arr[row]];
				}

				else
				{
					for (size_t row = workspace.st; row < workspace.st_NA;
						 row++)
						sum_weight_left +=
							workspace.weights_map[workspace.ix_arr[row]];
					for (size_t row = workspace.end_NA; row <= workspace.end;
						 row++)
						sum_weight_right +=
							workspace.weights_map[workspace.ix_arr[row]];
				}

				trees.back().pct_tree_left = sum_weight_left / (sum_weight_left + sum_weight_right);
			}

			switch (model_params.missing_action)
			{
			case Impute:
			{
				if (trees.back().pct_tree_left >= .5)
					workspace.end = workspace.end_NA - 1;
				else
					workspace.end = workspace.st_NA - 1;
				break;
			}

			case Divide:
			{
				if (!workspace.weights_arr.empty())
					for (size_t row = workspace.st_NA; row < workspace.end_NA;
						 row++)
						workspace.weights_arr[workspace.ix_arr[row]] *=
							trees.back().pct_tree_left;
				else
					for (size_t row = workspace.st_NA; row < workspace.end_NA;
						 row++)
						workspace.weights_map[workspace.ix_arr[row]] *=
							trees.back().pct_tree_left;
				workspace.end = workspace.end_NA - 1;
				break;
			}

			default:
			{
				unexpected_error();
				break;
			}
			}
		}

		else
		{
			trees.back().pct_tree_left = (lreal_t_safe)(workspace.split_ix - workspace.st) / (lreal_t_safe)(workspace.end - workspace.st + 1);
			workspace.end = workspace.split_ix - 1;
		}

		/* Depending on the scoring metric, might need to calculate fractions of data and volume */
		if (model_params.scoring_metric != Depth && !is_boxed_metric(model_params.scoring_metric))
		{
			switch (trees.back().col_type)
			{
			case Numeric:
			{
				if (!workspace.determine_split)
					workspace.density_calculator.restore_range(
						workspace.xmin, workspace.xmax);

				if (model_params.scoring_metric == Density)
					workspace.density_calculator.push_density(
						workspace.xmin, workspace.xmax,
						trees.back().num_split);
				else
					workspace.density_calculator.push_adj(
						workspace.xmin, workspace.xmax,
						trees.back().num_split, trees.back().pct_tree_left,
						model_params.scoring_metric);
				break;
			}

			case Categorical:
			{
				switch (model_params.cat_split_type)
				{
				case SingleCateg:
				{
					if (model_params.scoring_metric == Density)
					{
						if (workspace.determine_split)
						{
							if (workspace.criterion == NoCrit)
								workspace.density_calculator.push_density(
									workspace.npresent);
							else
								workspace.density_calculator.push_density(
									workspace.buffer_szt.data(),
									data.ncat[trees.back().col_num]);
						}

						else
						{
							workspace.density_calculator.push_density(
								workspace.density_calculator.counts.data(),
								data.ncat[trees.back().col_num]);
						}
					}

					else
					{
						if (workspace.determine_split)
						{
							if (workspace.criterion == NoCrit)
							{
								count_categs(
									workspace.ix_arr.data(),
									workspace.st,
									workspace.end,
									data.categ_data + trees.back().col_num * data.nrows,
									data.ncat[trees.back().col_num],
									workspace.density_calculator.counts.data());
								workspace.density_calculator.push_adj(
									workspace.density_calculator.counts.data(),
									data.ncat[trees.back().col_num],
									trees.back().chosen_cat,
									model_params.scoring_metric);
							}

							else
							{
								workspace.density_calculator.push_adj(
									workspace.buffer_szt.data(),
									data.ncat[trees.back().col_num],
									trees.back().chosen_cat,
									model_params.scoring_metric);
							}
						}

						else
						{

							workspace.density_calculator.push_adj(
								workspace.density_calculator.counts.data(),
								data.ncat[trees.back().col_num],
								trees.back().chosen_cat,
								model_params.scoring_metric);
						}
					}
					break;
				}

				case SubSet:
				{
					if (model_params.scoring_metric == Density)
					{
						if (!trees.back().cat_split.size())
						{
							workspace.density_calculator.push_density();
						}

						else
						{
							workspace.density_calculator.push_density(
								workspace.density_calculator.n_left,
								workspace.density_calculator.n_present);
						}
					}

					else
					{
						if (!trees.back().cat_split.size())
						{
							workspace.density_calculator.push_adj(
								trees.back().pct_tree_left,
								model_params.scoring_metric);
						}

						else
						{
							if (workspace.determine_split)
							{
								if (workspace.criterion == NoCrit)
								{
									count_categs(
										workspace.ix_arr.data(),
										workspace.st,
										workspace.end,
										data.categ_data + trees.back().col_num * data.nrows,
										data.ncat[trees.back().col_num],
										workspace.density_calculator.counts.data());
									workspace.density_calculator.push_adj(
										(signed char *)trees.back().cat_split.data(),
										workspace.density_calculator.counts.data(),
										data.ncat[trees.back().col_num],
										model_params.scoring_metric);
								}

								else
								{
									workspace.density_calculator.push_adj(
										(signed char *)trees.back().cat_split.data(),
										workspace.buffer_szt.data(),
										data.ncat[trees.back().col_num],
										model_params.scoring_metric);
								}
							}

							else
							{
								workspace.density_calculator.push_adj(
									(signed char *)trees.back().cat_split.data(),
									workspace.density_calculator.counts.data(),
									data.ncat[trees.back().col_num],
									model_params.scoring_metric);
							}
						}
					}
					break;
				}
				}
				break;
			}

			default:
			{
				assert(0);
			}
			}
		}

		else if (is_boxed_metric(model_params.scoring_metric))
		{
			switch (trees.back().col_type)
			{
			case Numeric:
			{
				workspace.density_calculator.push_bdens(
					trees.back().num_split, trees.back().col_num);
				break;
			}

			case Categorical:
			{
				switch (model_params.cat_split_type)
				{
				case SingleCateg:
				{
					workspace.density_calculator.push_bdens(
						(int)1, trees.back().col_num);
					break;
				}

				case SubSet:
				{
					if (trees.back().cat_split.empty())
					{
						workspace.density_calculator.push_bdens(
							(int)1, trees.back().col_num);
					}

					else
					{
						workspace.density_calculator.push_bdens(
							trees.back().cat_split,
							trees.back().col_num);
					}
					break;
				}
				}
				break;
			}

			default:
			{
				assert(0);
			}
			}
		}

		/* Branch where to assign new categories can be pre-determined in this case */
		if (trees.back().col_type == Categorical && model_params.cat_split_type == SubSet && data.ncat[trees.back().col_num] > 2 && model_params.new_cat_action == Smallest)
		{
			bool new_to_left = trees.back().pct_tree_left < 0.5;
			for (int cat = 0; cat < data.ncat[trees.back().col_num]; cat++)
				if (trees.back().cat_split[cat] < 0)
					trees.back().cat_split[cat] = new_to_left;
		}

		/* If doing single-category splits, the branch that got only one category will not
		 be splittable anymore, so it can be dropped for the remainder of that branch */
		if (trees.back().col_type == Categorical && model_params.cat_split_type == SingleCateg && data.ncat[trees.back().col_num] > 2 /* <- in this case, would have been dropped earlier */
		)
		{
			workspace.col_sampler.drop_col(
				trees.back().col_num + data.ncols_numeric,
				workspace.end - workspace.st + 1);
		}

		/* left branch */
		trees.back().tree_left = trees.size();
		trees.emplace_back();
		if (impute_nodes != NULL)
			impute_nodes->emplace_back(tree_from);
		split_itree_recursive<InputData, WorkerMemory, lreal_t_safe>(
			trees, workspace, data, model_params, impute_nodes,
			curr_depth + 1);

		/* right branch */
		recursion_state->restore_state(workspace);
		if (is_boxed_metric(model_params.scoring_metric))
		{
			if (trees[tree_from].col_type == Numeric)
				workspace.density_calculator.pop_bdens(
					trees[tree_from].col_num);
			else
				workspace.density_calculator.pop_bdens_cat(
					trees[tree_from].col_num);
		}
		else if (model_params.scoring_metric != Depth)
		{
			workspace.density_calculator.pop();
		}
		if (model_params.missing_action != Fail)
		{
			switch (model_params.missing_action)
			{
			case Impute:
			{
				if (trees[tree_from].pct_tree_left >= .5)
					workspace.st = workspace.end_NA;
				else
					workspace.st = workspace.st_NA;
				break;
			}

			case Divide:
			{
				if (!workspace.changed_weights && workspace.st_NA < workspace.end_NA)
				{
					workspace.changed_weights = true;

					if (!workspace.weights_arr.empty())
					{
						for (size_t row = workspace.st_NA;
							 row <= workspace.end; row++)
							workspace.weights_arr[workspace.ix_arr[row]] = 1;
					}

					else
					{
						for (size_t row = workspace.st_NA;
							 row <= workspace.end; row++)
							workspace.weights_map[workspace.ix_arr[row]] = 1;
					}
				}

				if (!workspace.weights_arr.empty())
					for (size_t row = workspace.st_NA; row < workspace.end_NA;
						 row++)
						workspace.weights_arr[workspace.ix_arr[row]] *= (1. - trees[tree_from].pct_tree_left);
				else
					for (size_t row = workspace.st_NA; row < workspace.end_NA;
						 row++)
						workspace.weights_map[workspace.ix_arr[row]] *= (1. - trees[tree_from].pct_tree_left);
				workspace.st = workspace.st_NA;
				break;
			}

			default:
			{
				unexpected_error();
				break;
			}
			}
		}

		else
		{
			workspace.st = workspace.split_ix;
		}

		trees[tree_from].tree_right = trees.size();
		trees.emplace_back();
		if (impute_nodes != NULL)
			impute_nodes->emplace_back(tree_from);
		split_itree_recursive<InputData, WorkerMemory, lreal_t_safe>(
			trees, workspace, data, model_params, impute_nodes,
			curr_depth + 1);
		if (is_boxed_metric(model_params.scoring_metric))
		{
			if (trees[tree_from].col_type == Numeric)
				workspace.density_calculator.pop_bdens_right(
					trees[tree_from].col_num);
			else
				workspace.density_calculator.pop_bdens_cat_right(
					trees[tree_from].col_num);
		}
		else if (model_params.scoring_metric != Depth)
		{
			workspace.density_calculator.pop_right();
		}
	}
		return;

		/* if it reached the limit, calculate terminal statistics */
	terminal_statistics:
	{
		trees.back().tree_left = 0;

		if (workspace.changed_weights)
		{
			if (sum_weight <= -HUGE_VAL)
				sum_weight = calculate_sum_weights<lreal_t_safe>(
					workspace.ix_arr, workspace.st, workspace.end, curr_depth,
					workspace.weights_arr, workspace.weights_map);
		}

		switch (model_params.scoring_metric)
		{
		case Depth:
		{
			if (!workspace.changed_weights)
				trees.back().score = curr_depth + expected_avg_depth<lreal_t_safe>(
													  workspace.end - workspace.st + 1);
			else
				trees.back().score = curr_depth + expected_avg_depth<lreal_t_safe>(sum_weight);
			break;
		}

		case AdjDepth:
		{
			if (!workspace.changed_weights)
				trees.back().score =
					workspace.density_calculator.calc_adj_depth() + expected_avg_depth<lreal_t_safe>(
																		workspace.end - workspace.st + 1);
			else
				trees.back().score =
					workspace.density_calculator.calc_adj_depth() + expected_avg_depth<lreal_t_safe>(sum_weight);
			break;
		}

		case Density:
		{
			if (!workspace.changed_weights)
				trees.back().score =
					workspace.density_calculator.calc_density(
						workspace.end - workspace.st + 1,
						model_params.sample_size);
			else
				trees.back().score =
					workspace.density_calculator.calc_density(
						sum_weight, model_params.sample_size);
			break;
		}

		case AdjDensity:
		{
			trees.back().score =
				workspace.density_calculator.calc_adj_density();
			break;
		}

		case BoxedRatio:
		{
			trees.back().score =
				workspace.density_calculator.calc_bratio();
			break;
		}

		case BoxedDensity:
		{
			if (!workspace.changed_weights)
				trees.back().score =
					workspace.density_calculator.calc_bdens(
						workspace.end - workspace.st + 1,
						model_params.sample_size);
			else
				trees.back().score =
					workspace.density_calculator.calc_bdens(
						sum_weight, model_params.sample_size);
			break;
		}

		case BoxedDensity2:
		{
			if (!workspace.changed_weights)
				trees.back().score =
					workspace.density_calculator.calc_bdens2(
						workspace.end - workspace.st + 1,
						model_params.sample_size);
			else
				trees.back().score =
					workspace.density_calculator.calc_bdens2(
						sum_weight, model_params.sample_size);
			break;
		}
		}

		trees.back().cat_split.clear();
		trees.back().cat_split.shrink_to_fit();

		trees.back().remainder =
			workspace.changed_weights ? (real_t)sum_weight : (real_t)(workspace.end - workspace.st + 1);

		/* for distance, assume also the elements keep being split */
		if (model_params.calc_dist)
			add_remainder_separation_steps<InputData, WorkerMemory, lreal_t_safe>(
				workspace, data, sum_weight);

		/* add this depth right away if requested */
		if (workspace.row_depths.size())
		{
			if (!workspace.changed_weights)
			{
				for (size_t row = workspace.st; row <= workspace.end; row++)
					workspace.row_depths[workspace.ix_arr[row]] +=
						trees.back().score;
			}

			else if (!workspace.weights_arr.empty())
			{
				for (size_t row = workspace.st; row <= workspace.end; row++)
					workspace.row_depths[workspace.ix_arr[row]] +=
						workspace.weights_arr[workspace.ix_arr[row]] * trees.back().score;
			}

			else
			{
				for (size_t row = workspace.st; row <= workspace.end; row++)
					workspace.row_depths[workspace.ix_arr[row]] +=
						workspace.weights_map[workspace.ix_arr[row]] * trees.back().score;
			}
		}

		/* add imputations from node if requested */
		if (model_params.impute_at_fit)
			add_from_impute_node(impute_nodes->back(), workspace, data);
	}
	}

	template <class InputData, class WorkerMemory, class lreal_t_safe>
	void
	split_hplane_recursive(std::vector<IsoHPlane> &hplanes,
						   WorkerMemory &workspace, InputData &input_data,
						   ModelParams &model_params,
						   std::vector<ImputeNode> *impute_nodes,
						   size_t curr_depth);
	// hplane tools :

	template <typename in = real_t>
	bool
	check_more_than_two_unique_values(size_t ix_arr[], size_t st, size_t end,
									  in x[], MissingAction missing_action);

	bool
	check_more_than_two_unique_values(size_t ix_arr[], size_t st, size_t end,
									  int x[], MissingAction missing_action);

	template <class real_t_ = real_t, class sparse_ix_ = sparse_ix>
	bool
	check_more_than_two_unique_values(size_t *ix_arr, size_t st, size_t end,
									  size_t col, sparse_ix_ *Xc_indptr,
									  sparse_ix_ *Xc_ind, real_t_ *Xc,
									  MissingAction missing_action)
	{
		UNDEF_REFERENCE(missing_action)
		UNDEF_REFERENCE2(missing_action)
		if (end - st <= 1)
			return false;
		if (Xc_indptr[col + 1] == Xc_indptr[col])
			return false;
		bool has_zeros = (end - st + 1) > (size_t)(Xc_indptr[col + 1] - Xc_indptr[col]);
		if (has_zeros && !is_na_or_inf(Xc[Xc_indptr[col]]) && Xc[Xc_indptr[col]] != 0)
			return true;

		size_t st_col = Xc_indptr[col];
		size_t end_col = Xc_indptr[col + 1] - 1;
		size_t curr_pos = st_col;
		size_t ind_end_col = Xc_ind[end_col];

		/* 'ix_arr' should be sorted beforehand */
		/* TODO: refactor this */
		real_t x0 = 0;
		size_t *row;
		for (row = std::lower_bound(ix_arr + st, ix_arr + end + 1,
									Xc_ind[st_col]);
			 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;)
		{
			if (Xc_ind[curr_pos] == (sparse_ix)(*row))
			{
				if (is_na_or_inf(Xc[curr_pos]) || (has_zeros && Xc[curr_pos] == 0))
				{
					if (row == ix_arr + end || curr_pos == end_col)
						return false;
					curr_pos = std::lower_bound(Xc_ind + curr_pos,
												Xc_ind + end_col + 1, *(++row)) -
							   Xc_ind;
				}

				x0 = Xc[curr_pos];
				if (has_zeros)
					return true;
				else if (x0 == 0)
					has_zeros = true;
				if (row == ix_arr + end || curr_pos == end_col)
					return false;
				curr_pos = std::lower_bound(Xc_ind + curr_pos,
											Xc_ind + end_col + 1, *(++row)) -
						   Xc_ind;
				break;
			}

			else
			{
				if (Xc_ind[curr_pos] > (sparse_ix)(*row))
					row = std::lower_bound(row + 1, ix_arr + end + 1,
										   Xc_ind[curr_pos]);
				else
					curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
												Xc_ind + end_col + 1, *row) -
							   Xc_ind;
			}

			for (;
				 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;)
			{
				if (Xc_ind[curr_pos] == (sparse_ix)(*row))
				{
					if (is_na_or_inf(Xc[curr_pos]) || (has_zeros && Xc[curr_pos] == 0))
					{
						if (row == ix_arr + end || curr_pos == end_col)
							break;
						curr_pos = std::lower_bound(Xc_ind + curr_pos,
													Xc_ind + end_col + 1,
													*(++row)) -
								   Xc_ind;
					}

					else if (Xc[curr_pos] != x0)
					{
						return true;
					}

					if (row == ix_arr + end || curr_pos == end_col)
						break;
					curr_pos = std::lower_bound(Xc_ind + curr_pos,
												Xc_ind + end_col + 1, *(++row)) -
							   Xc_ind;
				}

				else
				{
					if (Xc_ind[curr_pos] > (sparse_ix)(*row))
						row = std::lower_bound(row + 1, ix_arr + end + 1,
											   Xc_ind[curr_pos]);
					else
						curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
													Xc_ind + end_col + 1, *row) -
								   Xc_ind;
				}
			}
		}
		return false;
	}
	template <class real_t_ = real_t, class sparse_ix_ = sparse_ix>
	bool
	check_more_than_two_unique_values(size_t nrows, size_t col,
									  sparse_ix_ *Xc_indptr,
									  sparse_ix_ *Xc_ind, real_t_ *Xc,
									  MissingAction missing_action)
	{
		UNDEF_REFERENCE (missing_action);
		UNDEF_REFERENCE2(Xc_ind)

		if (nrows <= 1)
			return false;
		if (Xc_indptr[col + 1] == Xc_indptr[col])
			return false;
		bool has_zeros = nrows > (size_t)(Xc_indptr[col + 1] - Xc_indptr[col]);
		if (has_zeros && !is_na_or_inf(Xc[Xc_indptr[col]]) && Xc[Xc_indptr[col]] != 0)
			return true;

		real_t x0 = 0;
		sparse_ix ix;
		for (ix = Xc_indptr[col]; ix < Xc_indptr[col + 1]; ix++)
		{
			if (!is_na_or_inf(Xc[ix]))
			{
				if (has_zeros && Xc[ix] == 0)
					continue;
				if (has_zeros)
					return true;
				else if (Xc[ix] == 0)
					has_zeros = true;
				x0 = Xc[ix];
				ix++;
				break;
			}
		}

		for (ix = Xc_indptr[col]; ix < Xc_indptr[col + 1]; ix++)
		{
			if (!is_na_or_inf(Xc[ix]))
			{
				if (has_zeros && Xc[ix] == 0)
					continue;
				if (Xc[ix] != x0)
					return true;
			}
		}

		return false;
	}

	template <class ImputedData, class InputData>
	void
	initialize_impute_calc(ImputedData &imp, InputData &input_data, size_t row)
	{
		imp.n_missing_num = 0;
		imp.n_missing_cat = 0;
		imp.n_missing_sp = 0;

		if (input_data.numeric_data != NULL)
		{
			imp.missing_num.resize(input_data.ncols_numeric);
			for (size_t col = 0; col < input_data.ncols_numeric; col++)
				if (is_na_or_inf(
						input_data.numeric_data[row + col * input_data.nrows]))
					imp.missing_num[imp.n_missing_num++] = col;
			imp.missing_num.resize(imp.n_missing_num);
			imp.num_sum.assign(imp.n_missing_num, 0);
			imp.num_weight.assign(imp.n_missing_num, 0);
		}

		else if (input_data.Xc_indptr != NULL)
		{
			imp.missing_sp.resize(input_data.ncols_numeric);
			decltype(input_data.Xc_indptr) res;
			for (size_t col = 0; col < input_data.ncols_numeric; col++)
			{
				res = std::lower_bound(
					input_data.Xc_ind + input_data.Xc_indptr[col],
					input_data.Xc_ind + input_data.Xc_indptr[col + 1], row);
				if (res != input_data.Xc_ind + input_data.Xc_indptr[col + 1] &&
					*res == static_cast<typename std::remove_pointer<decltype(res)>::type>(row) &&
					is_na_or_inf(input_data.Xc[res - input_data.Xc_ind]))
				{
					imp.missing_sp[imp.n_missing_sp++] = col;
				}
			}
			imp.sp_num_sum.assign(imp.n_missing_sp, 0);
			imp.sp_num_weight.assign(imp.n_missing_sp, 0);
		}
		if (input_data.categ_data != NULL)
		{
			imp.missing_cat.resize(input_data.ncols_categ);
			for (size_t col = 0; col < input_data.ncols_categ; col++)
				if (input_data.categ_data[row + col * input_data.nrows] < 0)
					imp.missing_cat[imp.n_missing_cat++] = col;
			imp.missing_cat.resize(imp.n_missing_cat);
			imp.cat_weight.assign(imp.n_missing_cat, 0);
			imp.cat_sum.resize(input_data.ncols_categ);
			for (size_t cat = 0; cat < imp.n_missing_cat; cat++)
				imp.cat_sum[imp.missing_cat[cat]].assign(
					input_data.ncat[imp.missing_cat[cat]], 0);
		}
	}

	template <class ImputedData, class PredictionData>
	void
	initialize_impute_calc(ImputedData &imp, PredictionData &data,
						   Imputer &imputer, size_t row)
	{
		imp.n_missing_num = 0;
		imp.n_missing_cat = 0;
		imp.n_missing_sp = 0;

		if (data.numeric_data != NULL)
		{
			if (!imp.missing_num.size())
				imp.missing_num.resize(imputer.ncols_numeric);

			if (data.is_col_major)
			{
				for (size_t col = 0; col < imputer.ncols_numeric; col++)
					if (is_na_or_inf(
							data.numeric_data[row + col * data.nrows]))
						imp.missing_num[imp.n_missing_num++] = col;
			}

			else
			{
				for (size_t col = 0; col < imputer.ncols_numeric; col++)
					if (is_na_or_inf(
							data.numeric_data[col + row * imputer.ncols_numeric]))
						imp.missing_num[imp.n_missing_num++] = col;
			}

			if (!imp.num_sum.size())
			{
				imp.num_sum.resize(imputer.ncols_numeric, 0);
				imp.num_weight.resize(imputer.ncols_numeric, 0);
			}
			else
			{
				std::fill(imp.num_sum.begin(),
						  imp.num_sum.begin() + imp.n_missing_num, 0);
				std::fill(imp.num_weight.begin(),
						  imp.num_weight.begin() + imp.n_missing_num, 0);
			}
		}
		else if (data.Xr != NULL)
		{
			if (!imp.missing_sp.size())
				imp.missing_sp.resize(imputer.ncols_numeric);
			for (auto ix = data.Xr_indptr[row];
				 ix < data.Xr_indptr[row + 1]; ix++)
				if (is_na_or_inf(data.Xr[ix]))
					imp.missing_sp[imp.n_missing_sp++] = data.Xr_ind[ix];

			if (!imp.sp_num_sum.size())
			{
				imp.sp_num_sum.resize(imputer.ncols_numeric, 0);
				imp.sp_num_weight.resize(imputer.ncols_numeric, 0);
			}

			else
			{
				std::fill(imp.sp_num_sum.begin(),
						  imp.sp_num_sum.begin() + imp.n_missing_sp, 0);
				std::fill(imp.sp_num_weight.begin(),
						  imp.sp_num_weight.begin() + imp.n_missing_sp, 0);
			}
		}
		if (data.categ_data != NULL)
		{
			if (!imp.missing_cat.size())
				imp.missing_cat.resize(imputer.ncols_categ);

			if (data.is_col_major)
			{
				for (size_t col = 0; col < imputer.ncols_categ; col++)
				{
					if (data.categ_data[row + col * data.nrows] < 0)
						imp.missing_cat[imp.n_missing_cat++] = col;
				}
			}

			else
			{
				for (size_t col = 0; col < imputer.ncols_categ; col++)
				{
					if (data.categ_data[col + row * imputer.ncols_categ] < 0)
						imp.missing_cat[imp.n_missing_cat++] = col;
				}
			}

			if (!imp.cat_weight.size())
			{
				imp.cat_weight.resize(imputer.ncols_categ, 0);
				imp.cat_sum.resize(imputer.ncols_categ);
				for (size_t col = 0; col < imputer.ncols_categ; col++)
					imp.cat_sum[col].resize(imputer.ncat[col], 0);
			}
			else
			{
				std::fill(imp.cat_weight.begin(),
						  imp.cat_weight.begin() + imp.n_missing_cat, 0);
				for (size_t col = 0; col < imp.n_missing_cat; col++)
					std::fill(imp.cat_sum[imp.missing_cat[col]].begin(),
							  imp.cat_sum[imp.missing_cat[col]].end(), 0);
			}
		}
	}

	template <class ImputedData, class InputData>
	void
	initialize_impute_calc(ImputedData &imp, InputData &input_data,
						   size_t row);
	template <class ImputedData, class PredictionData>
	void
	initialize_impute_calc(ImputedData &imp, PredictionData &prediction_data,
						   Imputer &imputer, size_t row);

	template <class InputData>
	ImputedData::ImputedData(InputData &input_data, size_t row)
	{
		initialize_impute_calc(*this, input_data, row);
	}

	real_t
	expected_separation_depth(size_t n);
	real_t
	expected_separation_depth_hotstart(real_t curr, size_t n_curr,
									   size_t n_final);
	template <class lreal_t_safe>
	real_t
	expected_separation_depth(lreal_t_safe n)
	{
		if (n >= THRESHOLD_EXACT_S)
			return 3;
		real_t s_l = expected_separation_depth((size_t)std::floor(n));
		lreal_t_safe u = std::ceil(n);
		real_t s_u = s_l + (-s_l * u + 3. * u - 4.) / (u * (u - 1.));
		real_t diff = n - std::floor(n);
		return s_l + diff * s_u;
	}
	void
	increase_comb_counter(size_t ix_arr[], size_t st, size_t end, size_t n,
						  real_t counter[], real_t exp_remainder);

	real_t
	expected_separation_depth(size_t n);
	real_t
	expected_separation_depth_hotstart(real_t curr, size_t n_curr,
									   size_t n_final);
	template <class lreal_t_safe>
	real_t
	expected_separation_depth(lreal_t_safe n);

	void
	increase_comb_counter(size_t ix_arr[], size_t st, size_t end, size_t n,
						  real_t counter[], real_t exp_remainder)
	{
		size_t i, j;
		size_t ncomb = calc_ncomb(n);
		if (exp_remainder <= 1)
			for (size_t el1 = st; el1 < end; el1++)
			{
				for (size_t el2 = el1 + 1; el2 <= end; el2++)
				{
					// counter[i * (n - (i+1)/2) + j - i - 1]++; /* beaware integer division */
					i = ix_arr[el1];
					j = ix_arr[el2];
					counter[ix_comb(i, j, n, ncomb)]++;
				}
			}
		else
			for (size_t el1 = st; el1 < end; el1++)
			{
				for (size_t el2 = el1 + 1; el2 <= end; el2++)
				{
					i = ix_arr[el1];
					j = ix_arr[el2];
					counter[ix_comb(i, j, n, ncomb)] += exp_remainder;
				}
			}
	}

	void
	increase_comb_counter(size_t ix_arr[], size_t st, size_t end, size_t n,
						  real_t *counter, real_t *weights, real_t exp_remainder)
	{
		size_t i, j;
		size_t ncomb = calc_ncomb(n);
		if (exp_remainder <= 1)
			for (size_t el1 = st; el1 < end; el1++)
			{
				for (size_t el2 = el1 + 1; el2 <= end; el2++)
				{
					i = ix_arr[el1];
					j = ix_arr[el2];
					counter[ix_comb(i, j, n, ncomb)] += weights[i] * weights[j];
				}
			}
		else
			for (size_t el1 = st; el1 < end; el1++)
			{
				for (size_t el2 = el1 + 1; el2 <= end; el2++)
				{
					i = ix_arr[el1];
					j = ix_arr[el2];
					counter[ix_comb(i, j, n, ncomb)] += weights[i] * weights[j] * exp_remainder;
				}
			}
	}
	void
	increase_comb_counter(size_t ix_arr[], size_t st, size_t end, size_t n,
						  real_t counter[], hashed_map<size_t, real_t> &weights,
						  real_t exp_remainder)
	{
		size_t i, j;
		size_t ncomb = calc_ncomb(n);
		if (exp_remainder <= 1)
			for (size_t el1 = st; el1 < end; el1++)
			{
				for (size_t el2 = el1 + 1; el2 <= end; el2++)
				{
					i = ix_arr[el1];
					j = ix_arr[el2];
					counter[ix_comb(i, j, n, ncomb)] += weights[i] * weights[j];
				}
			}
		else
			for (size_t el1 = st; el1 < end; el1++)
			{
				for (size_t el2 = el1 + 1; el2 <= end; el2++)
				{
					i = ix_arr[el1];
					j = ix_arr[el2];
					counter[ix_comb(i, j, n, ncomb)] += weights[i] * weights[j] * exp_remainder;
				}
			}
	}

	void
	increase_comb_counter_in_groups(size_t ix_arr[], size_t st, size_t end,
									size_t split_ix, size_t n, real_t counter[],
									real_t exp_remainder)
	{
		size_t *ptr_split_ix = std::lower_bound(ix_arr + st, ix_arr + end + 1,
												split_ix);
		size_t n_group = std::distance(ix_arr + st, ptr_split_ix);
		n = n - split_ix;

		if (exp_remainder <= 1)
			for (size_t ix1 = st; ix1 < st + n_group; ix1++)
				for (size_t ix2 = st + n_group; ix2 <= end; ix2++)
					counter[ix_arr[ix1] * n + ix_arr[ix2] - split_ix]++;
		else
			for (size_t ix1 = st; ix1 < st + n_group; ix1++)
				for (size_t ix2 = st + n_group; ix2 <= end; ix2++)
					counter[ix_arr[ix1] * n + ix_arr[ix2] - split_ix] += exp_remainder;
	}

	void
	increase_comb_counter_in_groups(size_t ix_arr[], size_t st, size_t end,
									size_t split_ix, size_t n, real_t *counter,
									real_t *weights, real_t exp_remainder)
	{
		size_t *ptr_split_ix = std::lower_bound(ix_arr + st, ix_arr + end + 1,
												split_ix);
		size_t n_group = std::distance(ix_arr + st, ptr_split_ix);
		n = n - split_ix;

		if (exp_remainder <= 1)
			for (size_t ix1 = st; ix1 < st + n_group; ix1++)
				for (size_t ix2 = st + n_group; ix2 <= end; ix2++)
					counter[ix_arr[ix1] * n + ix_arr[ix2] - split_ix] +=
						weights[ix_arr[ix1]] * weights[ix_arr[ix2]];
		else
			for (size_t ix1 = st; ix1 < st + n_group; ix1++)
				for (size_t ix2 = st + n_group; ix2 <= end; ix2++)
					counter[ix_arr[ix1] * n + ix_arr[ix2] - split_ix] +=
						weights[ix_arr[ix1]] * weights[ix_arr[ix2]] * exp_remainder;
	}

	void
	tmat_to_dense(real_t *tmat, real_t *dmat, size_t n, real_t fill_diag);

	void
	sample_random_rows(std::vector<size_t> &ix_arr, size_t nrows,
					   bool with_replacement,
					   RNG_engine &rnd_generator,
					   std::vector<size_t> &ix_all,
					   real_t *sample_weights,
					   std::vector<real_t> &btree_weights, size_t log2_n,
					   size_t btree_offset, std::vector<bool> &is_repeated);

	template <class real_t_>
	void
	colmajor_to_rowmajor(real_t_ *X, size_t nrows, size_t ncols,
						 std::vector<real_t> &X_row_major)
	{

		X_row_major.resize(nrows * ncols);
		for (size_t row = 0; row < nrows; row++)
			for (size_t col = 0; col < ncols; col++)
				X_row_major[row + col * nrows] = X[col + row * ncols];
	}

	template <class real_t_, class sparse_ix_>
	void
	colmajor_to_rowmajor(real_t_ *Xc, sparse_ix_ *Xc_ind,
						 sparse_ix_ *Xc_indptr, size_t nrows, size_t ncols,
						 std::vector<real_t> &Xr, std::vector<size_t> &Xr_ind,
						 std::vector<size_t> &Xr_indptr)
	{
		size_t nnz = Xc_indptr[ncols];
		std::vector<size_t> row_indices(nnz);
		for (size_t col = 0; col < ncols; col++)
		{
			for (sparse_ix ix = Xc_indptr[col]; ix < Xc_indptr[col + 1]; ix++)
			{
				row_indices[ix] = Xc_ind[ix];
			}
		}
		std::vector<size_t> argsorted_indices(nnz);
		std::iota(argsorted_indices.begin(), argsorted_indices.end(),
				  (size_t)0);
		std::stable_sort(argsorted_indices.begin(), argsorted_indices.end(),
						 [&row_indices](const size_t a, const size_t b)
						 { return row_indices[a] < row_indices[b]; });
		Xr.resize(nnz);
		Xr_ind.resize(nnz);
		for (size_t ix = 0; ix < nnz; ix++)
		{
			Xr[ix] = Xc[argsorted_indices[ix]];
			Xr_ind[ix] = Xc_ind[argsorted_indices[ix]];
		}
		Xr_indptr.resize(nrows + 1);
		size_t curr_row = 0;
		size_t curr_n = 0;
		for (size_t ix = 0; ix < nnz; ix++)
		{
			if (row_indices[argsorted_indices[ix]] != curr_row)
			{
				Xr_indptr[curr_row + 1] = curr_n;
				curr_n = 0;
				curr_row = row_indices[argsorted_indices[ix]];
			}

			else
			{
				curr_n++;
			}
		}
		for (size_t row = 1; row < nrows; row++)
			Xr_indptr[row + 1] += Xr_indptr[row];
	}

	template <class Model>
	void
	build_distance_mappings(TreesIndexer &indexer, const Model &model,
							int nthreads)
	{

		build_terminal_node_mappings(indexer, model);
		signal_switcher ss = signal_switcher();

		size_t ntrees = model.ntrees();
		std::vector<size_t> n_terminal(ntrees);
		for (size_t tree = 0; tree < ntrees; tree++)
			n_terminal[tree] = indexer.indices[tree].n_terminal;

		size_t max_n_terminal = *std::max_element(n_terminal.begin(),
												  n_terminal.end());
		check_interrupt_switch(ss);
		if (max_n_terminal <= 1)
			return;

#ifndef _OPENMP
		nthreads = 1;
#endif
		std::vector<std::vector<size_t>> thread_buffer_indices(nthreads);
		for (std::vector<size_t> &v : thread_buffer_indices)
			v.reserve(max_n_terminal);
		check_interrupt_switch(ss);

		bool threw_exception = false;
		std::exception_ptr ex = NULL;

#ifdef OPENMP_
#pragma omp parallel for schedule(dynamic) num_threads(nthreads) shared(indexer, model, n_terminal, threw_exception, ex)
		for (size_t_for tree = 0; tree < (decltype(tree))ntrees; tree++)
#else
		for (size_t tree = 0; tree < (decltype(tree))ntrees; tree++)

#endif
		{
			if (signal_switcher::interrupt_switch || threw_exception)
				continue;

			try
			{
				size_t n_terminal_this = n_terminal[tree];
				size_t ncomb = calc_ncomb(n_terminal_this);
				indexer.indices[tree].node_distances.assign(ncomb, 0.);
				indexer.indices[tree].node_distances.shrink_to_fit();
				build_dindex(thread_buffer_indices[omp_get_thread_num()],
							 indexer.indices[tree].terminal_node_mappings,
							 indexer.indices[tree].node_distances,
							 indexer.indices[tree].node_depths, n_terminal_this,
							 get_tree(model, tree));
			}

			catch (...)
			{
#ifdef OPENMP_
#pragma omp critical
#endif // OPENMP
				{
					if (!threw_exception)
					{
						threw_exception = true;
						ex = std::current_exception();
					}
				}
			}
		}

		if (signal_switcher::interrupt_switch || threw_exception)
		{
			indexer.indices.clear();
		}

		check_interrupt_switch(ss);
		if (threw_exception)
			std::rethrow_exception(ex);
	}
	template <class Model>
	void
	build_tree_indices(TreesIndexer &indexer, const Model &model, int nthreads,
					   const bool with_distances)
	{
		if (!indexer.indices.empty() && !indexer.indices.front().reference_points.empty())
		{
			for (auto &ind : indexer.indices)
			{
				ind.reference_points.clear();
				ind.reference_indptr.clear();
				ind.reference_mapping.clear();
			}
		}

		try
		{
			if (with_distances)
			{
				build_distance_mappings(indexer, model, nthreads);
			}

			else
			{
				if (!indexer.indices.empty() && !indexer.indices.front().node_distances.empty())
				{
					for (auto &ind : indexer.indices)
					{
						ind.node_distances.clear();
						ind.node_depths.clear();
					}
				}

				build_terminal_node_mappings(indexer, model);
			}
		}

		catch (...)
		{
			indexer.indices.clear();
			throw;
		}
	}

	void
	build_tree_indices(TreesIndexer &indexer, const IsoForest &model,
					   int nthreads, const bool with_distances);

	void
	build_tree_indices(TreesIndexer &indexer, const ExtIsoForest &model,
					   int nthreads, const bool with_distances);

	void
	set_reference_points(TreesIndexer &indexer, ExtIsoForest &model,
						 const bool with_distances,
						 real_t *numeric_data,
						 int *categ_data, bool is_col_major, size_t ld_numeric,
						 size_t ld_categ,
						 real_t *Xc,
						 sparse_ix *Xc_ind, sparse_ix *Xc_indptr,
						 real_t *Xr,
						 sparse_ix *Xr_ind, sparse_ix *Xr_indptr, size_t nrows,
						 int nthreads);

	void
	set_reference_points(IsoForest *model_outputs,
						 ExtIsoForest *model_outputs_ext, TreesIndexer *indexer,
						 const bool with_distances,
						 real_t *numeric_data,
						 int *categ_data, bool is_col_major, size_t ld_numeric,
						 size_t ld_categ,
						 real_t *Xc,
						 sparse_ix *Xc_ind, sparse_ix *Xc_indptr,
						 real_t *Xr,
						 sparse_ix *Xr_ind, sparse_ix *Xr_indptr, size_t nrows,
						 int nthreads);

	template <class sparse_ix_ = sparse_ix>
	void
	get_num_nodes(IsoForest &model_outputs, sparse_ix_ *n_nodes,
				  sparse_ix_ *n_terminal, int nthreads) noexcept;

	template <class sparse_ix_ = sparse_ix>
	void
	get_num_nodes(ExtIsoForest &model_outputs, sparse_ix_ *n_nodes,
				  sparse_ix_ *n_terminal, int nthreads) noexcept;

	void
	calc_similarity(real_t numeric_data[], int categ_data[],
					real_t Xc[],
					sparse_ix Xc_ind[], sparse_ix Xc_indptr[], size_t nrows,
					bool use_long_real_t, int nthreads, bool assume_full_distr,
					bool standardize_dist, bool as_kernel,
					IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
					real_t tmat[], real_t rmat[], size_t n_from,
					bool use_indexed_references, TreesIndexer *indexer,
					bool is_col_major, size_t ld_numeric, size_t ld_categ);

	void
	calc_similarity(real_t numeric_data[], int categ_data[],
					real_t Xc[],
					int Xc_ind[], int Xc_indptr[], size_t nrows,
					bool use_long_real_t, int nthreads, bool assume_full_distr,
					bool standardize_dist, bool as_kernel,
					IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
					real_t tmat[], real_t rmat[], size_t n_from,
					bool use_indexed_references, TreesIndexer *indexer,
					bool is_col_major, size_t ld_numeric, size_t ld_categ);

	void
	impute_missing_values(real_t numeric_data[], int categ_data[],
						  bool is_col_major,
						  real_t Xr[],
						  sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
						  size_t nrows, bool use_long_real_t, int nthreads,
						  IsoForest *model_outputs,
						  ExtIsoForest *model_outputs_ext, Imputer &imputer);
	template <class ImputedData>
	void
	add_from_impute_node(ImputeNode &imputer, ImputedData &imputed_data,
						 real_t w)
	{
		size_t col;
		for (size_t ix = 0; ix < imputed_data.n_missing_num; ix++)
		{
			col = imputed_data.missing_num[ix];
			imputed_data.num_sum[ix] +=
				(!is_na_or_inf(imputer.num_sum[col])) ? (w * imputer.num_sum[col]) : 0;
			imputed_data.num_weight[ix] += w * imputer.num_weight[ix];
		}

		for (size_t ix = 0; ix < imputed_data.n_missing_sp; ix++)
		{
			col = imputed_data.missing_sp[ix];
			imputed_data.sp_num_sum[ix] +=
				(!is_na_or_inf(imputer.num_sum[col])) ? (w * imputer.num_sum[col]) : 0;
			imputed_data.sp_num_weight[ix] += w * imputer.num_weight[ix];
		}

		for (size_t ix = 0; ix < imputed_data.n_missing_cat; ix++)
		{
			col = imputed_data.missing_cat[ix];
			for (size_t cat = 0; cat < imputer.cat_sum[col].size(); cat++)
				imputed_data.cat_sum[col][cat] += w * imputer.cat_sum[col][cat];
		}
	}
	template <class InputData, class WorkerMemory>
	void
	add_from_impute_node(ImputeNode &imputer, WorkerMemory &workspace,
						 InputData &input_data)
	{
		if (workspace.impute_vec.size())
		{
			if (!workspace.weights_arr.size() && !workspace.weights_map.size())
			{
				for (size_t row = workspace.st; row <= workspace.end; row++)
					if (input_data.has_missing[workspace.ix_arr[row]])
						add_from_impute_node(
							imputer, workspace.impute_vec[workspace.ix_arr[row]],
							(real_t)1);
			}

			else if (workspace.weights_arr.size())
			{
				for (size_t row = workspace.st; row <= workspace.end; row++)
					if (input_data.has_missing[workspace.ix_arr[row]])
						add_from_impute_node(
							imputer, workspace.impute_vec[workspace.ix_arr[row]],
							workspace.weights_arr[workspace.ix_arr[row]]);
			}

			else
			{
				for (size_t row = workspace.st; row <= workspace.end; row++)
					if (input_data.has_missing[workspace.ix_arr[row]])
						add_from_impute_node(
							imputer, workspace.impute_vec[workspace.ix_arr[row]],
							workspace.weights_map[workspace.ix_arr[row]]);
			}
		}
		else if (workspace.impute_map.size())
		{
			if (!workspace.weights_arr.size() && !workspace.weights_map.size())
			{
				for (size_t row = workspace.st; row <= workspace.end; row++)
					if (input_data.has_missing[workspace.ix_arr[row]])
						add_from_impute_node(
							imputer, workspace.impute_map[workspace.ix_arr[row]],
							(real_t)1);
			}

			else if (workspace.weights_arr.size())
			{
				for (size_t row = workspace.st; row <= workspace.end; row++)
					if (input_data.has_missing[workspace.ix_arr[row]])
						add_from_impute_node(
							imputer, workspace.impute_map[workspace.ix_arr[row]],
							workspace.weights_arr[workspace.ix_arr[row]]);
			}

			else
			{
				for (size_t row = workspace.st; row <= workspace.end; row++)
					if (input_data.has_missing[workspace.ix_arr[row]])
						add_from_impute_node(
							imputer, workspace.impute_map[workspace.ix_arr[row]],
							workspace.weights_map[workspace.ix_arr[row]]);
			}
		}
	}

	template <class imp_arr, class InputData>
	void
	apply_imputation_results(imp_arr &impute_vec, Imputer &imputer,
							 InputData &input_data, int nthreads)
	{
		
		size_t col;
		UNDEF_REFERENCE(nthreads);
		UNDEF_REFERENCE2(nthreads);

		if (input_data.Xc_indptr != NULL)
		{
			std::vector<size_t> row_pos(input_data.nrows, 0);
			size_t row;

			for (size_t col = 0; col < input_data.ncols_numeric; col++)
			{
				for (auto ix = input_data.Xc_indptr[col];
					 ix < input_data.Xc_indptr[col + 1]; ix++)
				{
					if (is_na_or_inf(input_data.Xc[ix]))
					{
						row = input_data.Xc_ind[ix];
						if (impute_vec[row].sp_num_weight[row_pos[row]] > 0 && !is_na_or_inf(impute_vec[row].sp_num_sum[row_pos[row]]))
							input_data.Xc[ix] =
								impute_vec[row].sp_num_sum[row_pos[row]] / impute_vec[row].sp_num_weight[row_pos[row]];
						else
							input_data.Xc[ix] = imputer.col_means[col];

						row_pos[row]++;
					}
				}
			}
		}
#ifdef OPENMP_
#pragma omp parallel for schedule(dynamic) num_threads(nthreads) shared(input_data, impute_vec, imputer) private(col)
#endif

		for (size_t row = 0; row < (decltype(row))input_data.nrows; row++)
		{
			if (input_data.has_missing[row])
			{
				for (size_t ix = 0; ix < impute_vec[row].n_missing_num; ix++)
				{
					col = impute_vec[row].missing_num[ix];
					if (impute_vec[row].num_weight[ix] > 0 && !is_na_or_inf(impute_vec[row].num_sum[ix]))
						input_data.numeric_data[row + col * input_data.nrows] =
							impute_vec[row].num_sum[ix] / impute_vec[row].num_weight[ix];
					else
						input_data.numeric_data[row + col * input_data.nrows] =
							imputer.col_means[col];
				}

				for (size_t ix = 0; ix < impute_vec[row].n_missing_cat; ix++)
				{
					col = impute_vec[row].missing_cat[ix];
					input_data.categ_data[row + col * input_data.nrows] =
						std::distance(
							impute_vec[row].cat_sum[col].begin(),
							std::max_element(
								impute_vec[row].cat_sum[col].begin(),
								impute_vec[row].cat_sum[col].end()));

					if (input_data.categ_data[row + col * input_data.nrows] == 0 && impute_vec[row].cat_sum[col][0] <= 0)
						input_data.categ_data[row + col * input_data.nrows] =
							imputer.col_modes[col];
				}
			}
		}
	}
	template <class ImputedData, class InputData>
	void
	apply_imputation_results(std::vector<ImputedData> &impute_vec,
							 hashed_map<size_t, ImputedData> &impute_map,
							 Imputer &imputer, InputData &input_data,
							 int nthreads)
	{
		if (impute_vec.size())
			apply_imputation_results(impute_vec, imputer, input_data, nthreads);
		else if (impute_map.size())
			apply_imputation_results(impute_map, imputer, input_data, nthreads);
	}
	template <class PredictionData, class ImputedData>
	void
	apply_imputation_results(PredictionData &data, ImputedData &imp,
							 Imputer &imputer, size_t row)
	{
		size_t col;
		size_t pos = 0;
		if (data.is_col_major)
		{
			for (size_t ix = 0; ix < imp.n_missing_num; ix++)
			{
				col = imp.missing_num[ix];
				if (imp.num_weight[ix] > 0 && !is_na_or_inf(imp.num_sum[ix]))
					data.numeric_data[row + col * data.nrows] = imp.num_sum[ix] / imp.num_weight[ix];
				else
					data.numeric_data[row + col * data.nrows] =
						imputer.col_means[col];
			}
		}

		else
		{
			for (size_t ix = 0; ix < imp.n_missing_num; ix++)
			{
				col = imp.missing_num[ix];
				if (imp.num_weight[ix] > 0 && !is_na_or_inf(imp.num_sum[ix]))
					data.numeric_data[col + row * imputer.ncols_numeric] =
						imp.num_sum[ix] / imp.num_weight[ix];
				else
					data.numeric_data[col + row * imputer.ncols_numeric] =
						imputer.col_means[col];
			}
		}
		if (data.Xr != NULL)
			for (auto ix = data.Xr_indptr[row]; ix < data.Xr_indptr[row + 1]; ix++)
			{
				if (is_na_or_inf(data.Xr[ix]))
				{
					if (imp.sp_num_weight[pos] > 0 && !is_na_or_inf(imp.sp_num_sum[pos]))
						data.Xr[ix] = imp.sp_num_sum[pos] / imp.sp_num_weight[pos];
					else
						data.Xr[ix] = imputer.col_means[imp.missing_sp[pos]];
					pos++;
				}
			}

		if (data.is_col_major)
		{
			for (size_t ix = 0; ix < imp.n_missing_cat; ix++)
			{
				col = imp.missing_cat[ix];
				data.categ_data[row + col * data.nrows] = std::distance(
					imp.cat_sum[col].begin(),
					std::max_element(imp.cat_sum[col].begin(),
									 imp.cat_sum[col].end()));

				if (data.categ_data[row + col * data.nrows] == 0 && imp.cat_sum[col][0] <= 0)
					data.categ_data[row + col * data.nrows] =
						imputer.col_modes[col];
			}
		}

		else
		{
			for (size_t ix = 0; ix < imp.n_missing_cat; ix++)
			{
				col = imp.missing_cat[ix];
				data.categ_data[col + row * imputer.ncols_categ] = std::distance(
					imp.cat_sum[col].begin(),
					std::max_element(imp.cat_sum[col].begin(),
									 imp.cat_sum[col].end()));

				if (data.categ_data[col + row * imputer.ncols_categ] == 0 && imp.cat_sum[col][0] <= 0)
					data.categ_data[col + row * imputer.ncols_categ] =
						imputer.col_modes[col];
			}
		}
	}
	/*
	 template <class ImputedData, class InputData>
	 void initialize_impute_calc(ImputedData &imp, InputData &input_data, size_t row)
	 {
	 imp.n_missing_num = 0;
	 imp.n_missing_cat = 0;
	 imp.n_missing_sp  = 0;

	 if (input_data.numeric_data != NULL)
	 {
	 imp.missing_num.resize(input_data.ncols_numeric);
	 for (size_t col = 0; col < input_data.ncols_numeric; col++)
	 if (is_na_or_inf(input_data.numeric_data[row + col * input_data.nrows]))
	 imp.missing_num[imp.n_missing_num++] = col;
	 imp.missing_num.resize(imp.n_missing_num);
	 imp.num_sum.assign(imp.n_missing_num,    0);
	 imp.num_weight.assign(imp.n_missing_num, 0);
	 }
	 else if (input_data.Xc_indptr != NULL)
	 {
	 imp.missing_sp.resize(input_data.ncols_numeric);
	 decltype(input_data.Xc_indptr) res;
	 for (size_t col = 0; col < input_data.ncols_numeric; col++)
	 {
	 res = std::lower_bound(input_data.Xc_ind + input_data.Xc_indptr[col],
	 input_data.Xc_ind + input_data.Xc_indptr[col + 1],
	 row);
	 if (
	 res != input_data.Xc_ind + input_data.Xc_indptr[col + 1] &&
	 *res == static_cast<typename std::remove_pointer<decltype(res)>::type>(row) &&
	 is_na_or_inf(input_data.Xc[res - input_data.Xc_ind])
	 )
	 {
	 imp.missing_sp[imp.n_missing_sp++] = col;
	 }
	 }
	 imp.sp_num_sum.assign(imp.n_missing_sp,    0);
	 imp.sp_num_weight.assign(imp.n_missing_sp, 0);
	 }

	 if (input_data.categ_data != NULL)
	 {
	 imp.missing_cat.resize(input_data.ncols_categ);
	 for (size_t col = 0; col < input_data.ncols_categ; col++)
	 if (input_data.categ_data[row + col * input_data.nrows] < 0)
	 imp.missing_cat[imp.n_missing_cat++] = col;
	 imp.missing_cat.resize(imp.n_missing_cat);
	 imp.cat_weight.assign(imp.n_missing_cat, 0);
	 imp.cat_sum.resize(input_data.ncols_categ);
	 for (size_t cat = 0; cat < imp.n_missing_cat; cat++)
	 imp.cat_sum[imp.missing_cat[cat]].assign(input_data.ncat[imp.missing_cat[cat]], 0);
	 }
	 }


	 template <class ImputedData, class PredictionData>
	 void initialize_impute_calc(ImputedData &imp, PredictionData &data, Imputer &imputer, size_t row)
	 {
	 imp.n_missing_num = 0;
	 imp.n_missing_cat = 0;
	 imp.n_missing_sp  = 0;

	 if (data.numeric_data != NULL)
	 {
	 if (!imp.missing_num.size())
	 imp.missing_num.resize(imputer.ncols_numeric);

	 if (data.is_col_major)
	 {
	 for (size_t col = 0; col < imputer.ncols_numeric; col++)
	 if (is_na_or_inf(data.numeric_data[row + col * data.nrows]))
	 imp.missing_num[imp.n_missing_num++] = col;
	 }

	 else
	 {
	 for (size_t col = 0; col < imputer.ncols_numeric; col++)
	 if (is_na_or_inf(data.numeric_data[col + row * imputer.ncols_numeric]))
	 imp.missing_num[imp.n_missing_num++] = col;
	 }

	 if (!imp.num_sum.size())
	 {
	 imp.num_sum.resize(imputer.ncols_numeric,    0);
	 imp.num_weight.resize(imputer.ncols_numeric, 0);
	 }

	 else
	 {
	 std::fill(imp.num_sum.begin(),     imp.num_sum.begin()    + imp.n_missing_num,  0);
	 std::fill(imp.num_weight.begin(),  imp.num_weight.begin() + imp.n_missing_num,  0);
	 }
	 }
	 else if (data.Xr != NULL)
	 {
	 if (!imp.missing_sp.size())
	 imp.missing_sp.resize(imputer.ncols_numeric);
	 for (auto ix = data.Xr_indptr[row]; ix < data.Xr_indptr[row + 1]; ix++)
	 if (is_na_or_inf(data.Xr[ix]))
	 imp.missing_sp[imp.n_missing_sp++] = data.Xr_ind[ix];

	 if (!imp.sp_num_sum.size())
	 {
	 imp.sp_num_sum.resize(imputer.ncols_numeric,    0);
	 imp.sp_num_weight.resize(imputer.ncols_numeric, 0);
	 }

	 else
	 {
	 std::fill(imp.sp_num_sum.begin(),     imp.sp_num_sum.begin()    + imp.n_missing_sp,  0);
	 std::fill(imp.sp_num_weight.begin(),  imp.sp_num_weight.begin() + imp.n_missing_sp,  0);
	 }
	 }
	 if (data.categ_data != NULL)
	 {
	 if (!imp.missing_cat.size())
	 imp.missing_cat.resize(imputer.ncols_categ);

	 if (data.is_col_major)
	 {
	 for (size_t col = 0; col < imputer.ncols_categ; col++)
	 {
	 if (data.categ_data[row + col * data.nrows] < 0)
	 imp.missing_cat[imp.n_missing_cat++] = col;
	 }
	 }

	 else
	 {
	 for (size_t col = 0; col < imputer.ncols_categ; col++)
	 {
	 if (data.categ_data[col + row * imputer.ncols_categ] < 0)
	 imp.missing_cat[imp.n_missing_cat++] = col;
	 }
	 }

	 if (!imp.cat_weight.size())
	 {
	 imp.cat_weight.resize(imputer.ncols_categ, 0);
	 imp.cat_sum.resize(imputer.ncols_categ);
	 for (size_t col = 0; col < imputer.ncols_categ; col++)
	 imp.cat_sum[col].resize(imputer.ncat[col], 0);
	 }

	 else
	 {
	 std::fill(imp.cat_weight.begin(), imp.cat_weight.begin() + imp.n_missing_cat, 0);
	 for (size_t col = 0; col < imp.n_missing_cat; col++)
	 std::fill(imp.cat_sum[imp.missing_cat[col]].begin(),
	 imp.cat_sum[imp.missing_cat[col]].end(),
	 0);
	 }
	 }
	 }

	 */
	template <class ImputedData, class InputData>
	void
	allocate_imp_vec(std::vector<ImputedData> &impute_vec,
					 InputData &input_data, int nthreads)
	{

		UNDEF_REFERENCE(nthreads)
		UNDEF_REFERENCE2(nthreads)

		impute_vec.resize(input_data.nrows);
#ifdef OPENMP_
#pragma omp parallel for schedule(dynamic) num_threads(nthreads) shared(impute_vec, input_data)
#endif
		for (size_t row = 0; row < (decltype(row))input_data.nrows; row++)
			if (input_data.has_missing[row])
				initialize_impute_calc(impute_vec[row], input_data, row);
	}

	template <class ImputedData, class InputData>
	void
	allocate_imp_map(hashed_map<size_t, ImputedData> &impute_map,
					 InputData &input_data)
	{
		for (size_t row = 0; row < input_data.nrows; row++)
			if (input_data.has_missing[row])
				impute_map[row] = ImputedData(input_data, row);
	}

	template <class ImputedData, class InputData>
	void
	allocate_imp(InputData &input_data, std::vector<ImputedData> &impute_vec,
				 hashed_map<size_t, ImputedData> &impute_map,
				 int nthreads)
	{
		if (input_data.n_missing == 0)
			return;
		else if (input_data.n_missing <= input_data.nrows / (nthreads * 10))
			allocate_imp_map(impute_map, input_data);
		else
			allocate_imp_vec(impute_vec, input_data, nthreads);
	}

	template <class ImputedData, class InputData>
	void
	check_for_missing(InputData &input_data,
					  std::vector<ImputedData> &impute_vec,
					  hashed_map<size_t, ImputedData> &impute_map,
					  int nthreads);
	template <class PredictionData>
	size_t
	check_for_missing(PredictionData &prediction_data, Imputer &imputer,
					  size_t ix_arr[], int nthreads)
	{
		UNDEF_REFERENCE(nthreads);

		std::vector<char> has_missing(prediction_data.nrows, false);
#ifdef OPENMP_
#pragma omp parallel for schedule(static) num_threads(nthreads) shared(has_missing, prediction_data, imputer)
#endif

		for (size_t row = 0; row < (decltype(row))prediction_data.nrows; row++)
		{
			if (prediction_data.numeric_data != NULL)
			{
				if (prediction_data.is_col_major)
				{
					for (size_t col = 0; col < imputer.ncols_numeric; col++)
					{
						if (is_na_or_inf(
								prediction_data.numeric_data[row + col * prediction_data.nrows]))
						{
							has_missing[row] = true;
							break;
						}
					}
				}

				else
				{
					for (size_t col = 0; col < imputer.ncols_numeric; col++)
					{
						if (is_na_or_inf(
								prediction_data.numeric_data[col + row * imputer.ncols_numeric]))
						{
							has_missing[row] = true;
							break;
						}
					}
				}
			}

			else if (prediction_data.Xr != NULL)
			{
				for (auto ix = prediction_data.Xr_indptr[row];
					 ix < prediction_data.Xr_indptr[row + 1]; ix++)
				{
					if (is_na_or_inf(prediction_data.Xr[ix]))
					{
						has_missing[row] = true;
						break;
					}
				}
			}

			if (!has_missing[row])
			{
				if (prediction_data.is_col_major)
				{
					for (size_t col = 0; col < imputer.ncols_categ; col++)
					{
						if (prediction_data.categ_data[row + col * prediction_data.nrows] < 0)
						{
							has_missing[row] = true;
							break;
						}
					}
				}

				else
				{
					for (size_t col = 0; col < imputer.ncols_categ; col++)
					{
						if (prediction_data.categ_data[col + row * imputer.ncols_categ] < 0)
						{
							has_missing[row] = true;
							break;
						}
					}
				}
			}
		}
		size_t st = 0;
		size_t temp;
		for (size_t row = 0; row < prediction_data.nrows; row++)
		{
			if (has_missing[row])
			{
				temp = ix_arr[st];
				ix_arr[st] = ix_arr[row];
				ix_arr[row] = temp;
				st++;
			}
		}
		if (st == 0)
			return 0;

		return st;
	}

	void
	traverse_hplane(std::vector<IsoHPlane> &hplane, ExtIsoForest &model_outputs,
					PredictionData &prediction_data, real_t &output_depth,
					std::vector<ImputeNode> *impute_nodes,
					ImputedData *imputed_data,
					sparse_ix *tree_num,
					real_t *tree_depth, size_t row) noexcept;
	void
	batched_csc_predict(prediction_data &prediction_data, int nthreads,
						IsoForest *model_outputs,
						ExtIsoForest *model_outputs_ext, real_t *output_depths,
						sparse_ix *tree_num,
						real_t *per_tree_depths);

	void
	add_csc_range_penalty(WorkerForPredictCSC &workspace, PredictionData &data,
						  real_t *weights_arr, size_t col_num, real_t range_low,
						  real_t range_high)
	{
		std::sort(workspace.ix_arr.begin() + workspace.st,
				  workspace.ix_arr.begin() + workspace.end + 1);

		size_t st_col = data.Xc_indptr[col_num];
		size_t end_col = data.Xc_indptr[col_num + 1] - 1;
		size_t curr_pos = st_col;
		size_t ind_end_col = data.Xc_ind[end_col];
		size_t *ptr_st = std::lower_bound(
			workspace.ix_arr.data() + workspace.st,
			workspace.ix_arr.data() + workspace.end + 1, data.Xc_ind[st_col]);
		if (range_low <= 0 && range_high >= 0)
		{
			for (size_t *row = ptr_st;
				 row != workspace.ix_arr.data() + workspace.end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;)
			{
				if (data.Xc_ind[curr_pos] == (decltype(*data.Xc_ind))(*row))
				{
					if (likely(
							!std::isnan(data.Xc[curr_pos]) && (data.Xc[curr_pos] < range_low || data.Xc[curr_pos] > range_high)))
					{
						workspace.depths[*row] -=
							(weights_arr == NULL) ? 1. : weights_arr[*row];
					}

					if (row == workspace.ix_arr.data() + workspace.end || curr_pos == end_col)
						break;
					curr_pos = std::lower_bound(data.Xc_ind + curr_pos + 1,
												data.Xc_ind + end_col + 1,
												*(++row)) -
							   data.Xc_ind;
				}

				else
				{
					if (data.Xc_ind[curr_pos] > (decltype(*data.Xc_ind))(*row))
						row = std::lower_bound(
							row + 1, workspace.ix_arr.data() + workspace.end + 1,
							data.Xc_ind[curr_pos]);
					else
						curr_pos = std::lower_bound(data.Xc_ind + curr_pos + 1,
													data.Xc_ind + end_col + 1, *row) -
								   data.Xc_ind;
				}
			}
		}
	}

	void
	traverse_itree_csc(WorkerForPredictCSC &workspace,
					   std::vector<IsoTree> &trees, IsoForest &model_outputs,
					   prediction_data &data,
					   sparse_ix *tree_num,
					   real_t *per_tree_depths, size_t curr_tree,
					   bool has_range_penalty)
	{
		if (unlikely(trees[curr_tree].tree_left == 0))
		{
			if (model_outputs.missing_action != Divide)
				for (size_t row = workspace.st; row <= workspace.end; row++)
					workspace.depths[workspace.ix_arr[row]] += trees[curr_tree].score;
			else
				for (size_t row = workspace.st; row <= workspace.end; row++)
					workspace.depths[workspace.ix_arr[row]] +=
						workspace.weights_arr[workspace.ix_arr[row]] * trees[curr_tree].score;
			if (unlikely(tree_num != NULL))
				for (size_t row = workspace.st; row <= workspace.end; row++)
					tree_num[workspace.ix_arr[row]] = curr_tree;
			if (unlikely(per_tree_depths != NULL))
				for (size_t row = workspace.st; row <= workspace.end; row++)
					per_tree_depths[workspace.ix_arr[row]] = trees[curr_tree].score;
			return;
		}

		/* in this case, the indices are sorted in the csc penalty function */
		if (!(has_range_penalty && model_outputs.missing_action != Divide && curr_tree > 0) && trees[curr_tree].col_type == Numeric)
			std::sort(workspace.ix_arr.begin() + workspace.st,
					  workspace.ix_arr.begin() + workspace.end + 1);
		/* TODO: should mix the splitting function with the range penalty */

		/* divide according to tree */
		size_t orig_end = workspace.end;
		size_t st_NA, end_NA, split_ix;
		switch (trees[curr_tree].col_type)
		{
		case Numeric:
		{
			divide_subset_split(workspace.ix_arr.data(), workspace.st,
								workspace.end, trees[curr_tree].col_num, data.Xc,
								data.Xc_ind, data.Xc_indptr,
								trees[curr_tree].num_split,
								model_outputs.missing_action, st_NA, end_NA,
								split_ix);
			break;
		}

		case Categorical:
		{
			switch (model_outputs.cat_split_type)
			{
			case SingleCateg:
			{
				divide_subset_split(
					workspace.ix_arr.data(),
					data.categ_data + data.nrows * trees[curr_tree].col_num,
					workspace.st, workspace.end, trees[curr_tree].chosen_cat,
					model_outputs.missing_action, st_NA, end_NA, split_ix);
				break;
			}

			case SubSet:
			{
				if (!trees[curr_tree].cat_split.size())
					divide_subset_split(
						workspace.ix_arr.data(),
						data.categ_data + (data.nrows * trees[curr_tree].col_num),
						workspace.st, workspace.end, model_outputs.missing_action,
						model_outputs.new_cat_action,
						(bool)trees[curr_tree].pct_tree_left < .5, st_NA, end_NA,
						split_ix);
				else
					divide_subset_split(
						workspace.ix_arr.data(),
						data.categ_data + data.nrows * trees[curr_tree].col_num,
						workspace.st, workspace.end,
						(signed char *)trees[curr_tree].cat_split.data(),
						(int)trees[curr_tree].cat_split.size(),
						model_outputs.missing_action,
						model_outputs.new_cat_action,
						(bool)(trees[curr_tree].pct_tree_left < .5), st_NA,
						end_NA, split_ix);
				break;
			}
			}
			break;
		}

		default:
		{
			assert(0);
			break;
		}
		}

		/* continue splitting recursively */
		if (unlikely(
				model_outputs.new_cat_action == Weighted && model_outputs.cat_split_type == SubSet && data.categ_data != NULL))
			goto missing_action_divide;
		switch (model_outputs.missing_action)
		{
		case Impute:
		{
			split_ix = (trees[curr_tree].pct_tree_left >= .5) ? end_NA : st_NA;
		}
		break;

		case Fail:
		{
			if (split_ix > workspace.st)
			{
				workspace.end = split_ix - 1;

				if (has_range_penalty && trees[curr_tree].col_type == Numeric)
					add_csc_range_penalty(workspace, data, (real_t *)NULL,
										  trees[curr_tree].col_num,
										  trees[curr_tree].range_low,
										  trees[curr_tree].range_high);

				traverse_itree_csc(workspace, trees, model_outputs, data,
								   tree_num, per_tree_depths,
								   trees[curr_tree].tree_left,
								   has_range_penalty);
			}
			if (split_ix <= orig_end)
			{
				workspace.st = split_ix;
				workspace.end = orig_end;

				if (has_range_penalty && trees[curr_tree].col_type == Numeric)
					add_csc_range_penalty(workspace, data, (real_t *)NULL,
										  trees[curr_tree].col_num,
										  trees[curr_tree].range_low,
										  trees[curr_tree].range_high);

				traverse_itree_csc(workspace, trees, model_outputs, data,
								   tree_num, per_tree_depths,
								   trees[curr_tree].tree_right,
								   has_range_penalty);
			}
			break;
		}
		case Divide:
		{
		missing_action_divide:
			/* TODO: maybe here it shouldn't copy the whole ix_arr,
			 but then it'd need to re-generate it from outside too */
			std::vector<real_t> weights_arr;
			std::vector<size_t> ix_arr;
			if (end_NA > workspace.st)
			{
				weights_arr.assign(workspace.weights_arr.begin(),
								   workspace.weights_arr.begin() + end_NA);
				ix_arr.assign(workspace.ix_arr.data(),
							  workspace.ix_arr.data() + end_NA);
			}

			if (has_range_penalty && trees[curr_tree].col_type == Numeric)
			{
				size_t st = workspace.st;
				size_t end = workspace.end;

				if (workspace.st < st_NA)
				{
					workspace.end = st_NA - 1;
					add_csc_range_penalty(workspace, data,
										  workspace.weights_arr.data(),
										  trees[curr_tree].col_num,
										  trees[curr_tree].range_low,
										  trees[curr_tree].range_high);
				}

				if (workspace.end >= end_NA)
				{
					workspace.st = end_NA;
					workspace.end = end;
					add_csc_range_penalty(workspace, data,
										  workspace.weights_arr.data(),
										  trees[curr_tree].col_num,
										  trees[curr_tree].range_low,
										  trees[curr_tree].range_high);
				}
				workspace.st = st;
				workspace.end = end;
			}

			if (end_NA > workspace.st)
			{
				workspace.end = end_NA - 1;
				for (size_t row = st_NA; row < end_NA; row++)
					workspace.weights_arr[workspace.ix_arr[row]] *=
						trees[curr_tree].pct_tree_left;
				traverse_itree_csc(workspace, trees, model_outputs, data,
								   tree_num, per_tree_depths,
								   trees[curr_tree].tree_left,
								   has_range_penalty);
			}
			if (st_NA <= orig_end)
			{
				workspace.st = st_NA;
				workspace.end = orig_end;
				if (weights_arr.size())
				{
					std::copy(weights_arr.begin(), weights_arr.end(),
							  workspace.weights_arr.begin());
					std::copy(ix_arr.begin(), ix_arr.end(),
							  workspace.ix_arr.begin());
					weights_arr.clear();
					weights_arr.shrink_to_fit();
					ix_arr.clear();
					ix_arr.shrink_to_fit();
				}

				for (size_t row = st_NA; row < end_NA; row++)
					workspace.weights_arr[workspace.ix_arr[row]] *= (1. - trees[curr_tree].pct_tree_left);
				traverse_itree_csc(workspace, trees, model_outputs, data,
								   tree_num, per_tree_depths,
								   trees[curr_tree].tree_right,
								   has_range_penalty);
			}
			break;
		}
		}
	}

	void
	traverse_hplane_csc(WorkerForPredictCSC &workspace,
						std::vector<IsoHPlane> &hplanes,
						ExtIsoForest &model_outputs,
						prediction_data &prediction_data,
						sparse_ix *tree_num,
						real_t *per_tree_depths, size_t curr_tree,
						bool has_range_penalty);

	void
	merge_models(IsoForest *model, IsoForest *other, ExtIsoForest *ext_model,
				 ExtIsoForest *ext_other, Imputer *imputer, Imputer *iother,
				 TreesIndexer *indexer, TreesIndexer *ind_other);

	template <class InputData, class lreal_t_safe>
	void
	initialize_imputer(Imputer &imputer, InputData &input_data, size_t ntrees,
					   int nthreads)
	{
		UNDEF_REFERENCE(nthreads);
		UNDEF_REFERENCE2(nthreads);

		imputer.ncols_numeric = input_data.ncols_numeric;
		imputer.ncols_categ = input_data.ncols_categ;
		imputer.ncat.assign(input_data.ncat,
							input_data.ncat + input_data.ncols_categ);
		if (imputer.col_means.size())
		{
			imputer.col_means.resize(input_data.ncols_numeric);
			std::fill(imputer.col_means.begin(), imputer.col_means.end(), 0);
		}

		else
		{
			imputer.col_means.resize(input_data.ncols_numeric, 0);
		}

		imputer.col_modes.resize(input_data.ncols_categ);
		imputer.imputer_tree = std::vector<std::vector<ImputeNode>>(ntrees);

		size_t offset, cnt;
		if (input_data.numeric_data != NULL)
		{
			for (size_t col = 0; col < (decltype(col))input_data.ncols_numeric;
				 col++)
			{
				cnt = input_data.nrows;
				offset = col * input_data.nrows;
				for (size_t row = 0; row < input_data.nrows; row++)
				{
					imputer.col_means[col] +=
						(!is_na_or_inf(input_data.numeric_data[row + offset])) ? input_data.numeric_data[row + offset] : 0;
					cnt -= is_na_or_inf(input_data.numeric_data[row + offset]);
				}
				imputer.col_means[col] /= (lreal_t_safe)cnt;
			}
		}
		else if (input_data.Xc_indptr != NULL)
		{
			for (size_t col = 0; col < (decltype(col))input_data.ncols_numeric;
				 col++)
			{
				cnt = input_data.nrows;
				for (auto ix = input_data.Xc_indptr[col];
					 ix < input_data.Xc_indptr[col + 1]; ix++)
				{
					imputer.col_means[col] +=
						(!is_na_or_inf(input_data.Xc[ix])) ? input_data.Xc[ix] : 0;
					cnt -= is_na_or_inf(input_data.Xc[ix]);
				}
				imputer.col_means[col] /= (lreal_t_safe)cnt;
			}
		}
		if (input_data.categ_data != NULL)
		{
			std::vector<size_t> cat_counts(input_data.max_categ);
			for (size_t col = 0; col < (decltype(col))input_data.ncols_categ;
				 col++)
			{
				std::fill(cat_counts.begin(), cat_counts.end(), 0);
				offset = col * input_data.nrows;
				for (size_t row = 0; row < input_data.nrows; row++)
				{
					if (input_data.categ_data[row + offset] >= 0)
						cat_counts[input_data.categ_data[row + offset]]++;
				}
				imputer.col_modes[col] = (int)std::distance(
					cat_counts.begin(),
					std::max_element(
						cat_counts.begin(),
						cat_counts.begin() + input_data.ncat[col]));
			}
		}

		else if (input_data.Xc_indptr != NULL)
		{
			std::vector<size_t> cat_counts(input_data.max_categ);
			for (size_t col = 0; col < (decltype(col))input_data.ncols_categ;
				 col++)
			{
				std::fill(cat_counts.begin(), cat_counts.end(), 0);
				for (auto ix = input_data.Xc_indptr[col];
					 ix < input_data.Xc_indptr[col + 1]; ix++)
				{
					if (input_data.Xc[ix] >= 0)
						cat_counts[input_data.Xc[ix]]++;
				}
				imputer.col_modes[col] = (int)std::distance(
					cat_counts.begin(),
					std::max_element(
						cat_counts.begin(),
						cat_counts.begin() + input_data.ncat
												 [col]));
			}
		}
	}
	//
	real_t
	extract_spC(PredictionData &data, size_t row, size_t col_num) noexcept
	{
		decltype(data.Xc_indptr) search_res = std::lower_bound(
			data.Xc_ind + data.Xc_indptr[col_num],
			data.Xc_ind + data.Xc_indptr[col_num + 1], row);
		if (search_res == (data.Xc_ind + data.Xc_indptr[col_num + 1]) || (*search_res) != static_cast<typename std::remove_pointer<decltype(search_res)>::type>(row))
			return 0.;
		else
			return data.Xc[search_res - data.Xc_ind];
	}
	static inline real_t
	extract_spR(PredictionData &data, sparse_ix *row_st, sparse_ix *row_end,
				size_t col_num, size_t lb, size_t ub) noexcept
	{
		if (row_end == row_st || col_num < lb || col_num > ub)
			return 0.;
		sparse_ix *search_res = std::lower_bound(row_st, row_end,
												 (sparse_ix)col_num);
		if (search_res == row_end || *search_res != (sparse_ix)col_num)
			return 0.;
		else
			return data.Xr[search_res - data.Xr_ind];
	}
	real_t
	extract_spR(PredictionData &data, sparse_ix *row_st, sparse_ix *row_end,
				size_t col_num) noexcept
	{
		if (row_end == row_st)
			return 0.;
		sparse_ix *search_res = std::lower_bound(row_st, row_end,
												 (sparse_ix)col_num);
		if (search_res == row_end || *search_res != (sparse_ix)col_num)
			return 0.;
		else
			return data.Xr[search_res - data.Xr_ind];
	}

	void
	traverse_itree_fast(std::vector<IsoTree> &tree, IsoForest &model_outputs,
						real_t *row_numeric_data,
						real_t &output_depth,
						sparse_ix *tree_num,
						real_t *tree_depth, size_t row) noexcept
	{
		size_t curr_lev = 0;
		real_t xval;
		while (true)
		{
			if (unlikely(tree[curr_lev].tree_left == 0))
			{
				output_depth += tree[curr_lev].score;
				if (unlikely(tree_num != NULL))
					tree_num[row] = curr_lev;
				if (unlikely(tree_depth != NULL))
					*tree_depth = tree[curr_lev].score;
				break;
			}

			else
			{
				xval = row_numeric_data[tree[curr_lev].col_num];
				curr_lev =
					(xval <= tree[curr_lev].num_split) ? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
			}
		}
		if (model_outputs.cat_split_type == SubSet && model_outputs.new_cat_action == Weighted && model_outputs.missing_action == Divide)
		{
			output_depth /= model_outputs.orig_sample_size;
		}
		else
		if ( model_outputs.new_cat_action == Weighted && model_outputs.missing_action == Divide)
		{
			output_depth /= model_outputs.orig_sample_size;
		}
		 
		
	}

	void
	traverse_itree_no_recurse(std::vector<IsoTree> &tree,
							  IsoForest &model_outputs, PredictionData &data,
							  real_t &output_depth,
							  sparse_ix *tree_num,
							  real_t *tree_depth, size_t row) noexcept
	{
		size_t curr_lev = 0;
		real_t xval;
		int cval;
		while (true)
		{
			// if (tree[curr_lev].score > 0)
			if (unlikely(tree[curr_lev].tree_left == 0))
			{
				output_depth += tree[curr_lev].score;
				if (unlikely(tree_num != NULL))
					tree_num[row] = curr_lev;
				if (unlikely(tree_depth != NULL))
					*tree_depth = tree[curr_lev].score;
				break;
			}

			else
			{
				switch (tree[curr_lev].col_type)
				{
				case Numeric:
				{
					xval = data.numeric_data[data.is_col_major ? (row + tree[curr_lev].col_num * data.nrows) : (tree[curr_lev].col_num + row * data.ncols_numeric)];
					curr_lev =
						(xval <= tree[curr_lev].num_split) ? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
					break;
				}
				case Categorical:
				{
					cval = data.categ_data[data.is_col_major ? (row + tree[curr_lev].col_num * data.nrows) : (tree[curr_lev].col_num + row * data.ncols_categ)];
					switch (model_outputs.cat_split_type)
					{
					case SubSet:
					{

						if (tree[curr_lev].cat_split.empty()) /* this is for binary columns */
						{
							if (cval <= 1)
							{
								curr_lev =
									(cval == 0) ? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
							}

							else /* can only work with 'Smallest' + no NAs if reaching this point */
							{
								curr_lev =
									(tree[curr_lev].pct_tree_left < .5) ? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
							}
						}

						else
						{

							switch (model_outputs.new_cat_action)
							{
							case Random:
							{
								cval =
									(cval >= (int)tree[curr_lev].cat_split.size()) ? (cval % (int)tree[curr_lev].cat_split.size()) : cval;
								curr_lev =
									(tree[curr_lev].cat_split[cval]) ? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
								break;
							}

							case Smallest:
							{
								if (unlikely(
										cval >= (int)tree[curr_lev].cat_split.size()))
								{
									curr_lev =
										(tree[curr_lev].pct_tree_left < .5) ? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
								}

								else
								{
									curr_lev =
										(tree[curr_lev].cat_split[cval]) ? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
								}
								break;
							}

							default:
							{
								assert(0);
								break;
							}
							}
						}
						break;
					}
					case SingleCateg:
					{
						curr_lev =
							(cval == tree[curr_lev].chosen_cat) ? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
						break;
					}
					}
					break;
				}

				default:
				{
					assert(0);
					break;
				}
				}
			}
		}
	}
	enum NumericConfig
	{
		DenseRowMajor,
		DenseColMajor,
		SparseCSR,
		SparseCSC
	};

	real_t
	traverse_itree(std::vector<IsoTree> &tree, IsoForest &model_outputs,
				   prediction_data &data, std::vector<ImputeNode> *impute_nodes,
				   ImputedData *imputed_data, real_t curr_weight, size_t row,
				   sparse_ix *tree_num,
				   real_t *tree_depth, size_t curr_lev) noexcept
	{
		real_t xval = 0.;
		int cval;
		real_t range_penalty = 0;

		NumericConfig numeric_config(NumericConfig::DenseRowMajor);

		if (data.Xr_indptr != NULL)
			numeric_config = SparseCSR;
		else if (data.Xc_indptr != NULL)
			numeric_config = SparseCSC;
		else if (data.is_col_major)
			numeric_config = DenseColMajor;
		else
			numeric_config = DenseRowMajor;

		sparse_ix *row_st = NULL, *row_end = NULL;
		if (numeric_config == SparseCSR)
		{
			row_st = data.Xr_ind + data.Xr_indptr[row];
			row_end = data.Xr_ind + data.Xr_indptr[row + 1];
		}
		while (true)
		{
			// if (tree[curr_lev].score >= 0.)
			if (unlikely(tree[curr_lev].tree_left == 0))
			{
				if (unlikely(tree_num != NULL))
					tree_num[row] = curr_lev;
				if (unlikely(tree_depth != NULL))
					*tree_depth = tree[curr_lev].score;
				if (unlikely(imputed_data != NULL))
					add_from_impute_node((*impute_nodes)[curr_lev], *imputed_data,
										 curr_weight);

				return tree[curr_lev].score - range_penalty;
			}

			else
			{
				switch (tree[curr_lev].col_type)
				{
				case Numeric:
				{
					switch (numeric_config)
					{
					case DenseRowMajor:
					{
						xval = data.numeric_data[tree[curr_lev].col_num + row * data.ncols_numeric];
						break;
					}

					case DenseColMajor:
					{
						xval = data.numeric_data[row + tree[curr_lev].col_num * data.nrows];
						break;
					}

					case SparseCSR:
					{
						xval = extract_spR(data, row_st, row_end,
										   tree[curr_lev].col_num);
						break;
					}

					case SparseCSC:
					{
						xval = extract_spC(data, row, tree[curr_lev].col_num);
						break;
					}
					}
					if (unlikely(std::isnan(xval)))
					{
						switch (model_outputs.missing_action)
						{
						case Divide:
						{
							return tree[curr_lev].pct_tree_left * traverse_itree(
																	  tree, model_outputs, data, impute_nodes,
																	  imputed_data,
																	  curr_weight * tree[curr_lev].pct_tree_left,
																	  row, (sparse_ix *)NULL, tree_depth,
																	  tree[curr_lev].tree_left) +
								   (1. - tree[curr_lev].pct_tree_left) * traverse_itree(
																			 tree,
																			 model_outputs,
																			 data,
																			 impute_nodes,
																			 imputed_data,
																			 curr_weight * (1 - tree[curr_lev].pct_tree_left),
																			 row, (sparse_ix *)NULL, tree_depth,
																			 tree[curr_lev].tree_right) -
								   range_penalty;
						}

						case Impute:
						{
							curr_lev =
								(tree[curr_lev].pct_tree_left >= .5) ? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
							break;
						}

						case Fail:
						{
							return NAN;
						}
						}
					}

					else
					{
						range_penalty += (xval < tree[curr_lev].range_low) || (xval > tree[curr_lev].range_high);
						curr_lev =
							(xval <= tree[curr_lev].num_split) ? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
					}
					break;
				}
				case Categorical:
				{
					cval = data.categ_data[data.is_col_major ? (row + tree[curr_lev].col_num * data.nrows) : (tree[curr_lev].col_num + row * data.ncols_categ)];
					if (unlikely(cval < 0))
					{
						switch (model_outputs.missing_action)
						{
						case Divide:
						{
							return tree[curr_lev].pct_tree_left * traverse_itree(
																	  tree, model_outputs, data, impute_nodes,
																	  imputed_data,
																	  curr_weight * tree[curr_lev].pct_tree_left,
																	  row, (sparse_ix *)NULL, tree_depth,
																	  tree[curr_lev].tree_left) +
								   (1. - tree[curr_lev].pct_tree_left) * traverse_itree(
																			 tree,
																			 model_outputs,
																			 data,
																			 impute_nodes,
																			 imputed_data,
																			 curr_weight * (1 - tree[curr_lev].pct_tree_left),
																			 row, (sparse_ix *)NULL, tree_depth,
																			 tree[curr_lev].tree_right) -
								   range_penalty;
						}

						case Impute:
						{
							curr_lev =
								(tree[curr_lev].pct_tree_left >= .5) ? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
							break;
						}

						case Fail:
						{
							return NAN;
						}
						}
					}

					else
					{
						switch (model_outputs.cat_split_type)
						{
						case SingleCateg:
						{
							curr_lev =
								(cval == tree[curr_lev].chosen_cat) ? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
							break;
						}

						case SubSet:
						{

							if (tree[curr_lev].cat_split.empty())
							{
								if (cval <= 1)
								{
									curr_lev =
										(cval == 0) ? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
								}

								else
								{
									switch (model_outputs.new_cat_action)
									{
									case Smallest:
									{
										curr_lev =
											(tree[curr_lev].pct_tree_left < .5) ? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
										break;
									}

									case Weighted:
									{
										return tree[curr_lev].pct_tree_left * traverse_itree(
																				  tree,
																				  model_outputs,
																				  data,
																				  impute_nodes,
																				  imputed_data,
																				  curr_weight * tree[curr_lev].pct_tree_left,
																				  row, (sparse_ix *)NULL,
																				  tree_depth,
																				  tree[curr_lev].tree_left) +
											   (1. - tree[curr_lev].pct_tree_left) * traverse_itree(
																						 tree,
																						 model_outputs,
																						 data,
																						 impute_nodes,
																						 imputed_data,
																						 curr_weight * (1 - tree[curr_lev].pct_tree_left),
																						 row, (sparse_ix *)NULL,
																						 tree_depth,
																						 tree[curr_lev].tree_right) -
											   range_penalty;
									}

									default:
									{
										assert(0);
										break;
									}
									}
								}
							}

							else
							{
								switch (model_outputs.new_cat_action)
								{
								case Random:
								{
									cval =
										(cval >= (int)tree[curr_lev].cat_split.size()) ? (cval % (int)tree[curr_lev].cat_split.size()) : cval;
									curr_lev =
										(tree[curr_lev].cat_split[cval]) ? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
									break;
								}

								case Smallest:
								{
									if (unlikely(
											cval >= (int)tree[curr_lev].cat_split.size()))
									{
										curr_lev =
											(tree[curr_lev].pct_tree_left < .5) ? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
									}

									else
									{
										curr_lev =
											(tree[curr_lev].cat_split[cval]) ? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
									}
									break;
								}
								case Weighted:
								{
									if (cval >= (int)tree[curr_lev].cat_split.size() || tree[curr_lev].cat_split[cval] == (-1))
									{
										return tree[curr_lev].pct_tree_left * traverse_itree(
																				  tree,
																				  model_outputs,
																				  data,
																				  impute_nodes,
																				  imputed_data,
																				  curr_weight * tree[curr_lev].pct_tree_left,
																				  row, (sparse_ix *)NULL,
																				  tree_depth,
																				  tree[curr_lev].tree_left) +
											   (1. - tree[curr_lev].pct_tree_left) * traverse_itree(
																						 tree,
																						 model_outputs,
																						 data,
																						 impute_nodes,
																						 imputed_data,
																						 curr_weight * (1 - tree[curr_lev].pct_tree_left),
																						 row, (sparse_ix *)NULL,
																						 tree_depth,
																						 tree[curr_lev].tree_right) -
											   range_penalty;
									}

									else
									{
										curr_lev =
											(tree[curr_lev].cat_split[cval]) ? tree[curr_lev].tree_left : tree[curr_lev].tree_right;
									}
									break;
								}
								}
							}
							break;
						}
						}
					}
					break;
				}

				default:
				{
					assert(0);
					break;
				}
				}
			}
		}
	}

	
	void
	traverse_hplane_fast_colmajor(std::vector<IsoHPlane> &hplane,
								  ExtIsoForest &model_outputs,
								  prediction_data &data, real_t &output_depth,
								  sparse_ix *tree_num,
								  real_t *tree_depth, size_t row) noexcept
	{
		size_t curr_lev = 0;
		real_t hval;

		UNDEF_REFERENCE(model_outputs)
		UNDEF_REFERENCE2(data)
		

		if( unlikely(hplane[curr_lev].score == 0) )
		{
			output_depth = 0.;
			if (unlikely(tree_num != NULL))
				tree_num[row] = curr_lev;
			if (unlikely(tree_depth != NULL))
				*tree_depth = 0.;
			return;
		}
		
		while (true)
		{
			// if (hplane[curr_lev].score > 0)
			if (unlikely(hplane[curr_lev].hplane_left == 0))
			{
				output_depth += hplane[curr_lev].score;
				if (unlikely(tree_num != NULL))
					tree_num[row] = curr_lev;
				if (unlikely(tree_depth != NULL))
					*tree_depth = hplane[curr_lev].score;
				return;
			}

			else
			{
				hval = 0;
				for (size_t col = 0; col < hplane[curr_lev].col_num.size(); col++)
					hval += (data.numeric_data[row + hplane[curr_lev].col_num[col] * data.nrows] - hplane[curr_lev].mean[col]) * hplane[curr_lev].coef[col];

				curr_lev =
					(hval <= hplane[curr_lev].split_point) ? hplane[curr_lev].hplane_left : hplane[curr_lev].hplane_right;
			}
		}
	}

	void
	traverse_hplane_fast_rowmajor(std::vector<IsoHPlane> &hplane,
								  ExtIsoForest &model_outputs,
								  real_t *row_numeric_data,
								  real_t &output_depth,
								  sparse_ix *tree_num,
								  real_t *tree_depth, size_t row) noexcept
	{
		size_t curr_lev = 0;
		real_t hval;

	UNDEF_REFERENCE(model_outputs)
	UNDEF_REFERENCE2(row_numeric_data)
		while (true)
		{
			// if (hplane[curr_lev].score > 0)
			if (unlikely(hplane[curr_lev].hplane_left == 0))
			{
				output_depth += hplane[curr_lev].score;
				if (unlikely(tree_num != NULL))
					tree_num[row] = curr_lev;
				if (unlikely(tree_depth != NULL))
					*tree_depth = hplane[curr_lev].score;
				return;
			}

			else
			{
				hval = 0;
				for (size_t col = 0; col < hplane[curr_lev].col_num.size(); col++)
					hval += (row_numeric_data[hplane[curr_lev].col_num[col]] - hplane[curr_lev].mean[col]) * hplane[curr_lev].coef[col];

				curr_lev =
					(hval <= hplane[curr_lev].split_point) ? hplane[curr_lev].hplane_left : hplane[curr_lev].hplane_right;
			}
		}
	}

	void
	traverse_hplane(std::vector<IsoHPlane> &hplane, ExtIsoForest &model_outputs,
					prediction_data &data, real_t &output_depth,
					std::vector<ImputeNode> *impute_nodes,
					ImputedData *imputed_data,
					sparse_ix *tree_num,
					real_t *tree_depth, size_t row) noexcept
	{
    size_t  curr_lev = 0;
    double  xval;
    int     cval;
    double  hval;

    size_t ncols_numeric, ncols_categ;

    NumericConfig numeric_config;
    if (data.Xr_indptr != NULL)
        numeric_config = SparseCSR;
    else if (data.Xc_indptr != NULL)
        numeric_config = SparseCSC;
    else if (data.is_col_major)
        numeric_config = DenseColMajor;
    else
        numeric_config = DenseRowMajor;

    sparse_ix *row_st = NULL, *row_end = NULL;
    size_t lb, ub;
    if (numeric_config == SparseCSR)
    {
        row_st  = data.Xr_ind + data.Xr_indptr[row];
        row_end = data.Xr_ind + data.Xr_indptr[row + 1];
        lb = *row_st;
        ub = *(row_end-1);
    }
		 
    while (true)
    {
        // if (hplane[curr_lev].score > 0)
        if (unlikely(hplane[curr_lev].hplane_left == 0))
        {
            output_depth += hplane[curr_lev].score;
            if (unlikely(tree_num != NULL))
                tree_num[row] = curr_lev;
            if (unlikely(tree_depth != NULL))
                *tree_depth = hplane[curr_lev].score;
            if (unlikely(imputed_data != NULL))
            {
                add_from_impute_node((*impute_nodes)[curr_lev], *imputed_data, (double)1);
            }
            return;
        }
        else
        {
            hval = 0;
            ncols_numeric = 0; ncols_categ = 0;
            for (size_t col = 0; col < hplane[curr_lev].col_num.size(); col++)
            {
				if( hplane[curr_lev].col_type.size() <=col ) 
				{

					xval = data.numeric_data[ row * data.ncols_numeric];

					break; 
				}
				switch(hplane[curr_lev].col_type[col])
                {
                    case Numeric:
                    {
                        switch(numeric_config)
                        {
                            case DenseRowMajor:
                            {
                                xval = data.numeric_data[hplane[curr_lev].col_num[col] + row * data.ncols_numeric];
                            }
                            break;

                            case DenseColMajor:
                            {
								size_t index = row +  hplane[curr_lev].col_num[col] *data.nrows; 
								if (data.ncols_numeric > index) 
									xval = data.numeric_data[index];
								else
									xval = data.numeric_data[row+ col*data.nrows]; 

                            }
							break;
                            case SparseCSR:
                            {
                                xval = extract_spR(data, row_st, row_end, hplane[curr_lev].col_num[col], lb, ub);
                            }
							break;

                            case SparseCSC:
                            {
                                xval = extract_spC(data, row, hplane[curr_lev].col_num[col]);
                            }
							break;
                            
                        }

                        if (unlikely(is_na_or_inf(xval)))
                        {
                            if (model_outputs.missing_action != Fail)
                            {
								real_t fill_val = hplane[curr_lev].fill_val.size()<col?0:hplane[curr_lev].fill_val[col]; 
                				hval += fill_val;
								//hval += hplane[curr_lev].fill_val[col];
                            }
                            else
                            {
                                output_depth = NAN;
                                return;
                            }
                        }

                        else
                        {
                            hval += (xval - hplane[curr_lev].mean[ncols_numeric]) * hplane[curr_lev].coef[ncols_numeric];
                        }

                        ncols_numeric++;
                        break;
                    }
					break;
                    case Categorical:
                    {
						
						cval =data.ncols_categ
>0? data.categ_data[
                            data.is_col_major?
                            (row +  hplane[curr_lev].col_num[col] * data.nrows)
                                :
                            (hplane[curr_lev].col_num[col] + row * data.ncols_categ)
                        ] : 0;
                        if (unlikely(cval < 0))
                        {
                            if (model_outputs.missing_action != Fail)
                            {
                                hval += hplane[curr_lev].fill_val[col];
                            }
                            
                            else
                            {
                                output_depth = NAN;
                                return;
                            }
                        }

                        else
                        {
                            switch(model_outputs.cat_split_type)
                            {
                                case SingleCateg:
                                {
                                    hval += (cval == hplane[curr_lev].chosen_cat[ncols_categ])? hplane[curr_lev].fill_new[ncols_categ] : 0;
                                    break;
                                }

                                case SubSet:
                                {
                                    if (unlikely(cval >= (int)hplane[curr_lev].cat_coef[ncols_categ].size()))
                                    {
                                        if (model_outputs.new_cat_action == Random) {
                                            cval = cval % (int)hplane[curr_lev].cat_coef[ncols_categ].size();
                                            hval += hplane[curr_lev].cat_coef[ncols_categ][cval];
                                        }

                                        else {
                                            hval += hplane[curr_lev].fill_new[ncols_categ];
                                        }
                                    }
                                    
                                    else
                                    {
                                        hval += hplane[curr_lev].cat_coef[ncols_categ][cval];
                                    }
                                    
                                    break;
                                }
                            }
                        }

                        ncols_categ++;
                        break;
                    }
					break;
					case NotUsed:
					{
						break;
					}
				 	break;


  			        /* default:
                    {
                        assert(0);
                        break;
                    }*/
                }

            }

            output_depth -= (hval < hplane[curr_lev].range_low) ||
                            (hval > hplane[curr_lev].range_high);
            curr_lev       = (hval <= hplane[curr_lev].split_point)?
                             hplane[curr_lev].hplane_left : hplane[curr_lev].hplane_right;
        }// end else
    }// end while

	}// end function

	template <class ImputedData, class InputData>
	void
	check_for_missing(InputData &input_data,
					  std::vector<ImputedData> &impute_vec,
					  hashed_map<size_t, ImputedData> &impute_map,
					  int nthreads);

	template <class real_t_ = real_t>
	void
	add_linear_comb(size_t ix_arr[], size_t st, size_t end, real_t *res,
					real_t_ *x, real_t &coef, real_t x_sd, real_t x_mean,
					real_t &fill_val, MissingAction missing_action,
					real_t *buffer_arr, size_t *buffer_NAs, bool first_run);
	/* for regular numerical */

	/* for sparse numerical */

	isolation_forest::isolation_forest(size_t ndim, size_t ntry, provallo::CoefType coef_type,
									   bool coef_by_prop, bool with_replacement,
									   bool weight_as_sample, size_t sample_size,
									   size_t ntrees, size_t max_depth,
									   size_t ncols_per_tree, bool limit_depth,
									   bool penalize_range,
									   bool standardize_data,
									   provallo::ScoringMetric scoring_metric,
									   bool fast_bratio, bool weigh_by_kurt,
									   real_t prob_pick_by_gain_pl,
									   real_t prob_pick_by_gain_avg,
									   real_t prob_pick_by_full_gain,
									   real_t prob_pick_by_dens,
									   real_t prob_pick_col_by_range,
									   real_t prob_pick_col_by_var,
									   real_t prob_pick_col_by_kurt,
									   real_t min_gain,
									   provallo::MissingAction missing_action,
									   provallo::CategSplit cat_split_type,
									   provallo::NewCategAction new_cat_action,
									   bool all_perm, bool build_imputer,
									   size_t min_imp_obs,
									   provallo::UseDepthImp depth_imp,
									   provallo::WeighImpRows weigh_imp_rows,
									   uint64_t random_seed, int nthreads):				
		ndim(ndim), ntry(ntry), coef_type(coef_type), coef_by_prop(coef_by_prop),
		with_replacement(with_replacement), weight_as_sample(weight_as_sample),
		sample_size(sample_size), ntrees(ntrees), max_depth(max_depth),
		ncols_per_tree(ncols_per_tree), limit_depth(limit_depth),
		penalize_range(penalize_range), standardize_data(standardize_data),
		scoring_metric(scoring_metric), fast_bratio(fast_bratio),
		weigh_by_kurt(weigh_by_kurt), prob_pick_by_gain_pl(prob_pick_by_gain_pl),
		prob_pick_by_gain_avg(prob_pick_by_gain_avg),
		prob_pick_by_full_gain(prob_pick_by_full_gain),
		prob_pick_by_dens(prob_pick_by_dens),

		prob_pick_col_by_range(prob_pick_col_by_range),
		prob_pick_col_by_var(prob_pick_col_by_var),
		prob_pick_col_by_kurt(prob_pick_col_by_kurt), min_gain(min_gain),
		missing_action(missing_action), cat_split_type(cat_split_type),
		new_cat_action(new_cat_action), all_perm(all_perm),
		build_imputer(build_imputer), min_imp_obs(min_imp_obs),
		depth_imp(depth_imp), weigh_imp_rows(weigh_imp_rows),
		random_seed(random_seed), nthreads(nthreads)
	{
		this->check_params();
		this->is_fitted = false;
		//model was already initialized with default 
		//    std::vector<std::vector<iso_tree_struct>> trees;
    	//NewCategAction new_cat_action;
    	//CategSplit cat_split_type;
    	//MissingAction missing_action;
    	//ScoringMetric scoring_metric;
    	//real_t exp_avg_depth;
    	//real_t exp_avg_sep;
    	//size_t orig_sample_size;
    	//bool has_range_penalty;	

		model.has_range_penalty = false;
		model.orig_sample_size = 0;
		model.exp_avg_sep = 0;
		model.exp_avg_depth = 0;
		model.scoring_metric = scoring_metric;
		model.missing_action = missing_action;
		model.cat_split_type = cat_split_type;
		model.new_cat_action = new_cat_action;
		model.trees = std::vector<std::vector<iso_tree_struct>>();	
		model.trees.reserve(ntrees);
		//initialize model_ext :
		model_ext.has_range_penalty = false;
		model_ext.orig_sample_size = 0;
		model_ext.exp_avg_sep = 0;
		model_ext.exp_avg_depth = 0;
		model_ext.scoring_metric = scoring_metric;
		model_ext.missing_action = missing_action;
		model_ext.cat_split_type = cat_split_type;
		model_ext.new_cat_action = new_cat_action;
		//    std::vector<std::vector<iso_hplane>> hplanes;
 

 		model_ext.hplanes = std::vector<std::vector<iso_hplane>>();
		model_ext.hplanes.reserve(ntrees);
		


		//model.ntrees = ntrees;
		//model.max_depth = max_depth;
		//model_ext was already initialized with default
		
	}	


	void
	isolation_forest::fit(real_t X[], size_t nrows, size_t ncols)
	{
		this->check_params();
		this->override_previous_fit();

		auto retcode = fit_iforest((this->ndim == 1) ? &this->model : nullptr,
								   (this->ndim != 1) ? &this->model_ext : nullptr,
								   X, ncols, (int *)nullptr, (size_t)0,
								   (int *)nullptr, (real_t *)nullptr,
								   (int64_t *)nullptr, (int64_t *)nullptr,
								   this->ndim, this->ntry, this->coef_type,
								   this->coef_by_prop, (real_t *)nullptr,
								   this->with_replacement, this->weight_as_sample,
								   nrows, this->sample_size, this->ntrees,
								   this->max_depth, this->ncols_per_tree,
								   this->limit_depth, this->penalize_range,
								   this->standardize_data, this->scoring_metric,
								   this->fast_bratio, false, (real_t *)nullptr,
								   (real_t *)nullptr, true, (real_t *)nullptr,
								   this->weigh_by_kurt, this->prob_pick_by_gain_pl,
								   this->prob_pick_by_gain_avg,
								   this->prob_pick_by_full_gain,
								   this->prob_pick_by_dens,
								   this->prob_pick_col_by_range,
								   this->prob_pick_col_by_var,
								   this->prob_pick_col_by_kurt, this->min_gain,
								   this->missing_action, this->cat_split_type,
								   this->new_cat_action, this->all_perm,
								   &this->imputer, this->min_imp_obs,
								   this->depth_imp, this->weigh_imp_rows, false,
								   this->random_seed, false, this->nthreads);
		if (retcode != 0)
			unexpected_error();
		else

		this->is_fitted = true;
	}

	void
	isolation_forest::fit(real_t numeric_data[], size_t ncols_numeric,
						  size_t nrows, int categ_data[], size_t ncols_categ,
						  int ncat[], real_t sample_weights[],
						  real_t col_weights[])
	{
		this->check_params();
		this->override_previous_fit();

		auto retcode = fit_iforest((this->ndim == 1) ? &this->model : nullptr,
								   (this->ndim != 1) ? &this->model_ext : nullptr,
								   numeric_data, ncols_numeric, categ_data,
								   ncols_categ, ncat, (real_t *)nullptr,
								   (sparse_ix *)nullptr, (sparse_ix *)nullptr,
								   this->ndim, this->ntry, this->coef_type,
								   this->coef_by_prop, sample_weights,
								   this->with_replacement, this->weight_as_sample,
								   nrows, this->sample_size, this->ntrees,
								   this->max_depth, this->ncols_per_tree,
								   this->limit_depth, this->penalize_range,
								   this->standardize_data, this->scoring_metric,
								   this->fast_bratio, false, (real_t *)nullptr,
								   (real_t *)nullptr, true, col_weights,
								   this->weigh_by_kurt, this->prob_pick_by_gain_pl,
								   this->prob_pick_by_gain_avg,
								   this->prob_pick_by_full_gain,
								   this->prob_pick_by_dens,
								   this->prob_pick_col_by_range,
								   this->prob_pick_col_by_var,
								   this->prob_pick_col_by_kurt, this->min_gain,
								   this->missing_action, this->cat_split_type,
								   this->new_cat_action, this->all_perm,
								   &this->imputer, this->min_imp_obs,
								   this->depth_imp, this->weigh_imp_rows, false,
								   this->random_seed, false, this->nthreads);
		if (retcode != EXIT_SUCCESS)
			unexpected_error();
		else
		this->is_fitted = true;
	}

	void
	isolation_forest::fit(real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
						  size_t ncols_numeric, size_t nrows, int categ_data[],
						  size_t ncols_categ, int ncat[],
						  real_t sample_weights[], real_t col_weights[])
	{
		this->check_params();
		this->override_previous_fit();

		auto retcode = fit_iforest((this->ndim == 1) ? &this->model : nullptr,
								   (this->ndim != 1) ? &this->model_ext : nullptr,
								   (real_t *)nullptr, ncols_numeric, categ_data,
								   ncols_categ, ncat, Xc, Xc_ind, Xc_indptr,
								   this->ndim, this->ntry, this->coef_type,
								   this->coef_by_prop, sample_weights,
								   this->with_replacement, this->weight_as_sample,
								   nrows, this->sample_size, this->ntrees,
								   this->max_depth, this->ncols_per_tree,
								   this->limit_depth, this->penalize_range,
								   this->standardize_data, this->scoring_metric,
								   this->fast_bratio, false, (real_t *)nullptr,
								   (real_t *)nullptr, true, col_weights,
								   this->weigh_by_kurt, this->prob_pick_by_gain_pl,
								   this->prob_pick_by_gain_avg,
								   this->prob_pick_by_full_gain,
								   this->prob_pick_by_dens,
								   this->prob_pick_col_by_range,
								   this->prob_pick_col_by_var,
								   this->prob_pick_col_by_kurt, this->min_gain,
								   this->missing_action, this->cat_split_type,
								   this->new_cat_action, this->all_perm,
								   &this->imputer, this->min_imp_obs,
								   this->depth_imp, this->weigh_imp_rows, false,
								   this->random_seed, false, this->nthreads);
		if (retcode != EXIT_SUCCESS)
			unexpected_error();
		else
		this->is_fitted = true;
	}

	std::vector<real_t>
	isolation_forest::predict(real_t X[], size_t nrows, bool standardize)
	{
		this->check_is_fitted();
		this->check_nthreads();
		std::vector<real_t> out(nrows, 0.);
		predict_iforest(
			X, (int *)nullptr, true, (size_t)0, (size_t)0, (real_t *)nullptr,
			(int64_t *)nullptr, (int64_t *)nullptr, (real_t *)nullptr,
			(int64_t *)nullptr, (int64_t *)nullptr, nrows, this->nthreads,
			standardize, &this->model  ,
			  &this->model_ext  ,
			out.data(), (int64_t *)nullptr, (real_t *)nullptr,
			(TreesIndexer *)nullptr);
		return out;
	}

	std::vector<real_t>
	isolation_forest::predict(const std::string &string_of_real_data_)
	{
		std::vector<real_t> ret;
		std::vector<std::string> token_values;

		real_t *token_ = nullptr;

		tokenize(string_of_real_data_, token_values, ",");
		token_ = new real_t[token_values.size()];

		for (size_t i = 0; i < token_values.size(); ++i)
		{

			real_t val = atof(token_values[i].c_str());

			if ((val == val) == false) // isNAN/-inf,+inf
				token_[i] = 0.;
			else
				token_[i] = val;
		}

		ret = this->predict(token_, 1, true);

		delete[] token_;
		return ret;
	}
	void
	isolation_forest::predict(real_t numeric_data[], int categ_data[],
							  bool is_col_major, size_t nrows, size_t ld_numeric,
							  size_t ld_categ, bool standardize,
							  real_t output_depths[], sparse_ix tree_num[],
							  real_t per_tree_depths[])
	{
		this->check_is_fitted();
		this->check_nthreads();
		if ((tree_num || per_tree_depths) && !this->check_can_predict_per_tree())
			throw std::runtime_error(
				"Cannot predict tree numbers/depths with this model.\n");
		predict_iforest(
			numeric_data, categ_data, is_col_major, ld_numeric, ld_categ,
			(real_t *)nullptr, (int64_t *)nullptr, (int64_t *)nullptr,
			(real_t *)nullptr, (int64_t *)nullptr, (int64_t *)nullptr, nrows,
			this->nthreads, standardize,
			  &this->model  ,
			  &this->model_ext ,
			output_depths, tree_num, per_tree_depths,
			 &this->indexer   );
			
	}

	void
	isolation_forest::predict(real_t X_sparse[], sparse_ix X_ind[],
							  sparse_ix X_indptr[],
							  bool is_csc, int categ_data[], bool is_col_major,
							  size_t ld_categ, size_t nrows, bool standardize,
							  real_t output_depths[], sparse_ix tree_num[],
							  real_t per_tree_depths[])
	{
		this->check_is_fitted();
		this->check_nthreads();
		if ((tree_num || per_tree_depths) && !this->check_can_predict_per_tree())
			throw std::runtime_error(
				"Cannot predict tree numbers/depths with this model.\n");
		std::vector<real_t> out(nrows);
		predict_iforest(
			(real_t *)nullptr, categ_data, is_col_major, (size_t)0, ld_categ,
			is_csc ? X_sparse : (real_t *)nullptr,
			is_csc ? X_ind : (sparse_ix *)nullptr,
			is_csc ? X_indptr : (sparse_ix *)nullptr,
			is_csc ? (real_t *)nullptr : X_sparse,
			is_csc ? (sparse_ix *)nullptr : X_ind,
			is_csc ? (sparse_ix *)nullptr : X_indptr, nrows, this->nthreads,
			standardize, (!this->model.trees.empty()) ? &this->model : nullptr,
			(!this->model_ext.hplanes.empty()) ? &this->model_ext : nullptr,
			output_depths, tree_num, per_tree_depths,
			(!this->indexer.indices.empty()) ? &this->indexer : nullptr);
	}

	std::vector<real_t>
	isolation_forest::predict_distance(real_t X[], size_t nrows, bool as_kernel,
									   bool assume_full_distr, bool standardize,
									   bool triangular)
	{
		this->check_is_fitted();
		this->check_nthreads();
		std::vector<real_t> tmat(calc_ncomb(nrows));
		std::vector<real_t> dmat(triangular ? square(nrows) : 0);

		calc_similarity(
			X, (int *)nullptr, (real_t *)nullptr, (sparse_ix *)nullptr, (sparse_ix *)nullptr,
			nrows, false, this->nthreads, assume_full_distr, standardize, as_kernel,
			(!this->model.trees.empty()) ? &this->model : nullptr,
			(!this->model_ext.hplanes.empty()) ? &this->model_ext : nullptr,
			tmat.data(), (real_t *)nullptr, (size_t)0, false,
			(!this->indexer.indices.empty()) ? &this->indexer : nullptr, true,
			(size_t)0, (size_t)0);
		if (!triangular)
		{
			real_t diag_filler;
			if (as_kernel)
			{
				if (standardize)
					diag_filler = 1.;
				else
					diag_filler = std::max(this->model.trees.size(),
										   this->model_ext.hplanes.size());
			}
			else
			{
				if (standardize)
					diag_filler = 0;
				else
					diag_filler = std::numeric_limits<real_t>::infinity();
			}
			tmat_to_dense(tmat.data(), dmat.data(), nrows, diag_filler);
		}
		return (triangular ? tmat : dmat);
	}

	void
	isolation_forest::predict_distance(real_t numeric_data[], int categ_data[],
									   size_t nrows, bool as_kernel,
									   bool assume_full_distr, bool standardize,
									   bool triangular, real_t dist_matrix[])
	{
		this->check_is_fitted();
		this->check_nthreads();
		std::vector<real_t> tmat(triangular ? 0 : calc_ncomb(nrows));

		calc_similarity(
			numeric_data, categ_data, (real_t *)nullptr, (sparse_ix *)nullptr,
			(sparse_ix *)nullptr, nrows, false, this->nthreads, assume_full_distr,
			standardize, as_kernel,
			(!this->model.trees.empty()) ? &this->model : nullptr,
			(!this->model_ext.hplanes.empty()) ? &this->model_ext : nullptr,
			triangular ? dist_matrix : tmat.data(), (real_t *)nullptr, (size_t)0,
			false, (!this->indexer.indices.empty()) ? &this->indexer : nullptr,
			true, (size_t)0, (size_t)0);
		if (!triangular)
		{
			real_t diag_filler;
			if (as_kernel)
			{
				if (standardize)
					diag_filler = 1.;
				else
					diag_filler = std::max(this->model.trees.size(),
										   this->model_ext.hplanes.size());
			}
			else
			{
				if (standardize)
					diag_filler = 0;
				else
					diag_filler = std::numeric_limits<real_t>::infinity();
			}
			tmat_to_dense(tmat.data(), dist_matrix, nrows, diag_filler);
		}
	}

	void
	isolation_forest::predict_distance(real_t Xc[], sparse_ix Xc_ind[],
									   sparse_ix Xc_indptr[], int categ_data[],
									   size_t nrows, bool as_kernel,
									   bool assume_full_distr, bool standardize,
									   bool triangular, real_t dist_matrix[])
	{
		this->check_is_fitted();
		this->check_nthreads();

		//avoid pedantic errors : 
		UNDEF_REFERENCE(categ_data);
		UNDEF_REFERENCE2(categ_data);
			
 		std::vector<real_t> tmat(triangular ? 0 : calc_ncomb(nrows));

		calc_similarity(
			(real_t *)nullptr, (int *)nullptr, Xc, Xc_ind, Xc_indptr, nrows, false,
			this->nthreads, assume_full_distr, standardize, as_kernel,
			(!this->model.trees.empty()) ? &this->model : nullptr,
			(!this->model_ext.hplanes.empty()) ? &this->model_ext : nullptr,
			triangular ? dist_matrix : tmat.data(), (real_t *)nullptr, (size_t)0,
			false, (!this->indexer.indices.empty()) ? &this->indexer : nullptr,
			true, (size_t)0, (size_t)0);
		if (!triangular)
		{
			real_t diag_filler;
			if (as_kernel)
			{
				if (standardize)
					diag_filler = 1.;
				else
					diag_filler = std::max(this->model.trees.size(),
										   this->model_ext.hplanes.size());
			}
			else
			{
				if (standardize)
					diag_filler = 0;
				else
					diag_filler = std::numeric_limits<real_t>::infinity();
			}

			tmat_to_dense(tmat.data(), dist_matrix, nrows, diag_filler);

		}
	}

	void
	isolation_forest::impute(real_t X[], size_t nrows)
	{

		this->check_is_fitted();
		this->check_nthreads();
		if (this->imputer.imputer_tree.empty())
			throw std::runtime_error(
				"Model was built without imputation capabilities.\n");
		impute_missing_values(
			X, (int *)nullptr, true, (real_t *)nullptr, (long int *)nullptr,
			(long int *)nullptr, nrows, false, this->nthreads,
			(!this->model.trees.empty()) ? &this->model : nullptr,
			(!this->model_ext.hplanes.empty()) ? &this->model_ext : nullptr,
			this->imputer);
	}

	void
	isolation_forest::impute(real_t numeric_data[], int categ_data[],
							 bool is_col_major, size_t nrows)
	{
		this->check_is_fitted();
		
		if (this->imputer.imputer_tree.empty())
			throw std::runtime_error(
				"Model was built without imputation capabilities.\n");

		//
		this->check_nthreads();


		// call impute missing values
		///
		///
		///
	        	
		provallo::impute_missing_values(
			numeric_data, categ_data, is_col_major, (real_t *)nullptr,
			(long int *)nullptr, (long int *)nullptr, nrows, false, this->nthreads,
			(!this->model.trees.empty()) ? &this->model : nullptr,
			(!this->model_ext.hplanes.empty()) ? &this->model_ext : nullptr,
			this->imputer);
		///
	}

	void
	isolation_forest::impute(real_t Xr[], sparse_ix Xr_ind[],
							 sparse_ix Xr_indptr[],
							 int categ_data[], bool is_col_major, size_t nrows)
	{
		this->check_is_fitted();
		if (this->imputer.imputer_tree.empty())
			throw std::runtime_error(
				"Model was built without imputation capabilities.\n");
		this->check_nthreads();
		impute_missing_values(
			(real_t *)nullptr, categ_data, is_col_major, Xr, Xr_ind, Xr_indptr,
			nrows, false, this->nthreads,
			(!this->model.trees.empty()) ? &this->model : nullptr,
			(!this->model_ext.hplanes.empty()) ? &this->model_ext : nullptr,
			this->imputer);
	}

	void
	isolation_forest::build_indexer(const bool with_distances)
	{
		this->check_is_fitted();
		if (!this->indexer.indices.empty())
			return;
		if (this->missing_action == Divide)
			throw std::runtime_error(
				"Cannot build tree indexer when using 'missing_action=Divide'.\n");
		if (!this->model.trees.empty() && this->new_cat_action == Weighted && this->cat_split_type == SubSet)
			throw std::runtime_error(
				"Cannot build tree indexer when using 'new_cat_action=Weighted' with single-variable model.\n");

		if (!this->model.trees.empty())
			build_tree_indices(this->indexer, this->model, this->nthreads,
							   with_distances);
		else if (!this->model_ext.hplanes.empty())
			build_tree_indices(this->indexer, this->model_ext, this->nthreads,
							   with_distances);
		else
			unexpected_error();
	}

	void
	isolation_forest::set_as_reference_points(real_t numeric_data[],
											  int categ_data[],
											  bool is_col_major, size_t nrows,
											  size_t ld_numeric, size_t ld_categ,
											  const bool with_distances)
	{
		this->check_is_fitted();
		if (!this->model.trees.empty())
			set_reference_points(&this->model, (ExtIsoForest *)NULL, &this->indexer,
								 with_distances, numeric_data, categ_data,
								 is_col_major, ld_numeric, ld_categ, (real_t *)NULL,
								 (sparse_ix *)NULL, (sparse_ix *)NULL,
								 (real_t *)NULL, (sparse_ix *)NULL,
								 (sparse_ix *)NULL, nrows, this->nthreads);
		else
			set_reference_points((iso_forest *)NULL, &this->model_ext,
								 &this->indexer, with_distances, numeric_data,
								 categ_data, is_col_major, ld_numeric, ld_categ,
								 (real_t *)NULL, (sparse_ix *)NULL,
								 (sparse_ix *)NULL, (real_t *)NULL,
								 (sparse_ix *)NULL, (sparse_ix *)NULL, nrows,
								 this->nthreads);
	}

	void
	isolation_forest::set_as_reference_points(real_t Xc[], sparse_ix Xc_ind[],
											  sparse_ix Xc_indptr[],
											  int categ_data[], size_t nrows,
											  const bool with_distances)
	{
		UNDEF_REFERENCE(categ_data);
		UNDEF_REFERENCE2(categ_data);
		UNDEF_REFERENCE2(Xc_indptr);

		this->check_is_fitted();
		if (!this->model.trees.empty())

			set_reference_points(&this->model, (ExtIsoForest *)NULL, &this->indexer,
								 with_distances, (real_t *)NULL, (int *)NULL, true,
								 (size_t)0, (size_t)0, Xc, Xc_ind, Xc_indptr,
								 (real_t *)NULL, (sparse_ix *)NULL,
								 (sparse_ix *)NULL, nrows, this->nthreads);
		else
			set_reference_points((iso_forest *)NULL, &this->model_ext,
								 &this->indexer, with_distances, nullptr, nullptr,
								 true, (size_t)0, (size_t)0, Xc, Xc_ind, Xc_indptr,
								 (real_t *)NULL, (sparse_ix *)NULL,
								 (sparse_ix *)NULL, nrows, this->nthreads);
	}

	size_t
	isolation_forest::get_num_reference_points() const noexcept
	{
		return provallo::get_number_of_reference_points(this->indexer);
	}

	void
	isolation_forest::predict_distance_to_ref_points(real_t numeric_data[],
													 int categ_data[],
													 real_t Xc[], sparse_ix Xc_ind[],
													 sparse_ix Xc_indptr[],
													 size_t nrows,
													 bool is_col_major,
													 size_t ld_numeric,
													 size_t ld_categ,
													 bool as_kernel,
													 bool standardize,
													 real_t dist_matrix[])
	{
		this->check_is_fitted();
		if (this->indexer.indices.empty())
			throw std::runtime_error(
				"Model has no indexer. Cannot predict distances to indexer.\n");
		if (!as_kernel && this->indexer.indices.front().node_distances.empty())
			throw std::runtime_error(
				"Model's indexer was built without distances. Cannot calculate distances to reference points.\n");
		if (this->indexer.indices.front().reference_points.empty())
			throw std::runtime_error(
				"Model's indexer has no reference points. Cannot calculate distances to reference points.\n");
		if (dist_matrix == NULL)
			throw std::runtime_error("Passed a NULL pointer for 'dist_matrix'.\n");

		provallo::calc_similarity(
			numeric_data, categ_data, Xc, Xc_ind, Xc_indptr, nrows, false,
			this->nthreads, true, standardize, as_kernel,
			(!this->model.trees.empty()) ? &this->model : NULL,
			(!this->model_ext.hplanes.empty()) ? &this->model_ext : NULL,
			(real_t *)NULL, dist_matrix, (size_t)0, true, &this->indexer,
			is_col_major, ld_numeric, ld_categ);
	}

	void
	isolation_forest::serialize(std::FILE *out) const
	{
		this->serialize_template(out);
	}

	void
	isolation_forest::serialize(std::ostream &out) const
	{
		this->serialize_template(out);
	}

	isolation_forest
	isolation_forest::deserialize(std::FILE *inp, int nthreads)
	{
		return deserialize_template(inp, nthreads);
	}

	isolation_forest
	isolation_forest::deserialize(std::istream &inp, int nthreads)
	{
		return deserialize_template(inp, nthreads);
	}

	std::ostream &
	operator<<(std::ostream &ost, const isolation_forest &model)
	{
		model.serialize(ost);
		return ost;
	}

#if 0

	std::ostream& isotree::operator<<(std::ostream &ost, const isolation_forest &model)
	{
		model.serialize(ost);
		return ost;
	}
#endif

	std::istream &
	operator>>(std::istream &ist, isolation_forest &model)
	{
		model = isolation_forest::deserialize(ist, -1);
		return ist;
	}
#if 0
std::istream& isotree::operator>>(std::istream &ist, isolation_forest &model)
{
	model = isolation_forest::deserialize(ist, -1);
	return ist;
}
#endif
	provallo::iso_forest &
	isolation_forest::get_model()
	{
		if (this->ndim != 1)
			throw std::runtime_error(
				"Error: class contains an 'ExtIsoForest' model only.\n");
		return this->model;
	}

	provallo::ExtIsoForest &
	isolation_forest::get_model_ext()
	{
		if (this->ndim == 1)
			throw std::runtime_error(
				"Error: class contains an 'IsoForest' model only.\n");
		return this->model_ext;
	}

	provallo::Imputer &
	isolation_forest::get_imputer()
	{
		if (!this->build_imputer)
			throw std::runtime_error("Error: model does not contain imputer.\n");
		return this->imputer;
	}

	provallo::TreesIndexer &
	isolation_forest::get_indexer()
	{
		if (this->indexer.indices.empty() && (!this->model.trees.empty() || !this->model_ext.hplanes.empty()))
			throw std::runtime_error("Error: model does not contain indexer.\n");
		return this->indexer;
	}

	void
	isolation_forest::check_nthreads()
	{
		if (this->nthreads < 0)
		{
#ifdef _OPENMP
			this->nthreads = omp_get_max_threads() + this->nthreads + 1;
#else
			this->nthreads = 1;
#endif
		}
		if (nthreads <= 0)
		{
			fprintf(stderr, "'isotree' got invalid 'nthreads', will set to 1.\n");
			this->nthreads = 1;
		}
#ifndef _OPENMP
		else if (nthreads > 1)
		{
			fprintf(stderr, "Passed nthreads:%d to 'isotree', add std::thread.\n",
					this->nthreads);
			this->nthreads = 1;
		}
#endif
	}

	size_t
	isolation_forest::get_ntrees() const
	{
		if (!this->model.trees.empty())
			return this->model.trees.size();
		else if (!this->model_ext.hplanes.empty())
			return this->model_ext.hplanes.size();
		else
			throw std::runtime_error("Model is not fitted or is corrupted.\n");
	}

	bool
	isolation_forest::check_can_predict_per_tree() const
	{
		if (!this->model.trees.empty())
		{
			if (this->model.missing_action == Divide)
				return false;
			if (this->model.new_cat_action == Weighted && this->cat_split_type != SingleCateg)
			{
				for (const std::vector<iso_tree_struct> &tree : this->model.trees)
					for (const iso_tree_struct &node : tree)
						if (node.col_type == Categorical)
							return false;
			}
		}

		return true;
	}

	void
	isolation_forest::override_previous_fit()
	{
		if (this->is_fitted)
		{

			this->model.trees.clear();

			this->model_ext.hplanes.clear();

			this->imputer.imputer_tree.clear();
			this->imputer.col_means.clear();
			this->imputer.col_modes.clear();
			indexer.indices.clear(); /* reset indices */

			this->is_fitted = false; /* reset flag */
		}
	}

	void
	isolation_forest::check_params()
	{
		this->check_nthreads();

		if (this->prob_pick_by_gain_avg < 0)
			throw std::runtime_error("'prob_pick_by_gain_avg' must be >= 0.\n");
		if (this->prob_pick_by_gain_pl < 0)
			throw std::runtime_error("'prob_pick_by_gain_pl' must be >= 0.\n");
		if (this->prob_pick_by_full_gain < 0)
			throw std::runtime_error("'prob_pick_by_full_gain' must be >= 0.\n");
		if (this->prob_pick_by_dens < 0)
			throw std::runtime_error("'prob_pick_by_dens' must be >= 0.\n");
		if (this->prob_pick_col_by_range < 0)
			throw std::runtime_error("'prob_pick_col_by_range' must be >= 0.\n");
		if (this->prob_pick_col_by_var < 0)
			throw std::runtime_error("'prob_pick_col_by_var' must be >= 0.\n");
		if (this->prob_pick_col_by_kurt < 0)
			throw std::runtime_error("'prob_pick_col_by_kurt' must be >= 0.\n");

		if (prob_pick_by_gain_avg + prob_pick_by_gain_pl + prob_pick_by_full_gain + prob_pick_by_dens > 1. + 2. * std::numeric_limits<real_t>::epsilon())
			throw std::runtime_error(
				"Probabilities for gain-based splits sum to more than 1.\n");

		if (prob_pick_col_by_var + prob_pick_col_by_var + prob_pick_col_by_kurt > 1. + 2. * std::numeric_limits<real_t>::epsilon())
			throw std::runtime_error(
				"Probabilities for column choices sum to more than 1.\n");

		if (min_gain < 0)
			throw std::runtime_error("'min_gain' cannot be negative.\n");

		if (this->ndim != 1)
		{
			if (this->missing_action == Divide)
				throw std::runtime_error(
					"'missing_action' = 'Divide' not supported in extended model.\n");
		}

		if (this->coef_type != Uniform && this->coef_type != Normal)
			throw std::runtime_error("Invalid 'coef_type'.\n");
		if (this->missing_action != Divide && this->missing_action != Impute && this->missing_action != Fail)
			throw std::runtime_error("Invalid 'missing_action'.\n");
		if (this->cat_split_type != SubSet && this->cat_split_type != SingleCateg)
			throw std::runtime_error("Invalid 'cat_split_type'.\n");
		if (this->new_cat_action != Weighted && this->new_cat_action != Smallest && this->new_cat_action != Random)
			throw std::runtime_error("Invalid 'new_cat_action'.\n");
		if (this->depth_imp != Lower && this->depth_imp != Higher && this->depth_imp != Same)
			throw std::runtime_error("Invalid 'depth_imp'.\n");
		if (this->weigh_imp_rows != Inverse && this->weigh_imp_rows != Prop && this->weigh_imp_rows != Flat)
			throw std::runtime_error("Invalid 'weigh_imp_rows'.\n");

		if (this->sample_size > 0 && this->sample_size <= 2)
			throw std::runtime_error("'sample_size' must be greater than 2.\n");

		if (this->penalize_range && (this->scoring_metric == provallo::Density || this->scoring_metric == provallo::AdjDensity))
			throw std::runtime_error(
				"'penalize_range' is incompatible with density scoring.\n");
	
	
	
	
		//allow for missing data
		if (this->missing_action == Divide && this->build_imputer)
			throw std::runtime_error(
				"Cannot build imputer when using 'missing_action=Divide'.\n"); 

		//initialize imputer if not already done
		if (this->build_imputer && this->imputer.imputer_tree.empty())
		{
			//this->imputer.imputer_tree.resize(this->ntrees);
			//call imputer.init_imputer(this->imputer, this->model, this->model_ext, this->nthreads); 
 			//this->imputer.col_means.resize( this->standardize_data.size1());
			//this->imputer.col_modes.resize( this->standardize_data.size1());
			//init imputer
			
		}
	}

	void
	isolation_forest::check_is_fitted() const
	{
		if (!this->is_fitted)
			throw std::runtime_error("Model has not been fitted.\n");
	}

	template <class otype>
	void
	isolation_forest::serialize_template(otype &out) const
	{
		this->check_is_fitted();

		serialize_combined(
			(!this->model.trees.empty()) ? &this->model : nullptr,
			(!this->model_ext.hplanes.empty()) ? &this->model_ext : nullptr,
			(!this->imputer.imputer_tree.empty()) ? &this->imputer : nullptr,
			(!this->indexer.indices.empty()) ? &this->indexer : nullptr,
			(char *)nullptr, (size_t)0, out);
	}

	void
	isolation_forest::predict_iforest(
		real_t *numeric_data, int *categ_data, bool is_col_major,
		size_t ld_numeric, size_t ld_categ,
		real_t *Xc,
		sparse_ix *Xc_ind, sparse_ix *Xc_indptr,
		real_t *Xr,
		sparse_ix *Xr_ind, sparse_ix *Xr_indptr, size_t nrows, int nthreads,
		bool standardize, provallo::IsoForest *model_outputs,
		provallo::ext_iso_forest *model_outputs_ext, real_t *output_depths,
		sparse_ix *tree_num,
		real_t *per_tree_depths, provallo::TreesIndexer *indexer)
	{
		if (nrows < 2)
			return;
		// if (unlikely(!nrows)) return;

		/* put data in a struct for passing it in fewer lines */
		prediction_data prediction =
			{numeric_data, categ_data, nrows, is_col_major, ld_numeric, ld_categ, Xc,
			 Xc_ind, Xc_indptr, Xr, Xr_ind, Xr_indptr};

		int nthreads_orig = nthreads;
		if ((size_t)nthreads > nrows)
			nthreads = nrows;

		/* For batch predictions of sparse CSC, will take a specialized route */
		if (prediction.Xc_indptr != NULL && (prediction.categ_data == NULL || prediction.is_col_major))
		{
			batched_csc_predict(prediction, nthreads_orig, model_outputs,
								model_outputs_ext, output_depths, tree_num,
								per_tree_depths);
		}

		/* Regular case (no specialized CSC route) */
		else if (model_outputs != NULL)
		{
			if (model_outputs->missing_action == Fail && (model_outputs->new_cat_action != Weighted || model_outputs->cat_split_type == SingleCateg || prediction.categ_data == NULL) && prediction.Xc_indptr == NULL && prediction.Xr_indptr == NULL && !model_outputs->has_range_penalty)
			{
				if (prediction.categ_data == NULL && (nrows == 1 || !prediction.is_col_major))
				{
					for (size_t row = 0; row < (decltype(row))nrows; row++)
					{
						real_t score = 0;
						for (size_t tree = 0; tree < model_outputs->trees.size();
							 tree++)
						{
							traverse_itree_fast(
								model_outputs->trees[tree],
								*model_outputs,
								prediction.numeric_data + row * prediction.ncols_numeric,
								score,
								(tree_num == NULL) ? NULL : (tree_num + nrows * tree),
								(per_tree_depths == NULL) ? NULL : (per_tree_depths + tree + row * model_outputs->trees.size()),
								(size_t)row);
						}
						output_depths[row] = score;
					}
				}

				else
				{
					for (size_t row = 0; row < (decltype(row))nrows; row++)
					{
						real_t score = 0;
						for (size_t tree = 0; tree < model_outputs->trees.size();
							 tree++)
						{
							traverse_itree_no_recurse(
								model_outputs->trees[tree],
								*model_outputs,
								prediction,
								score,
								(tree_num == NULL) ? NULL : (tree_num + nrows * tree),
								(per_tree_depths == NULL) ? NULL : (per_tree_depths + tree + row * model_outputs->trees.size()),
								(size_t)row);
						}
						output_depths[row] = score==score?score:0.0;
					}
				}
			}

			else
			{

				for (size_t row = 0; row < (decltype(row))nrows; row++)
				{
					real_t score = 0;
					for (size_t tree = 0; tree < model_outputs->trees.size();
						 tree++)
					{
						score += traverse_itree(
							model_outputs->trees[tree],
							*model_outputs,
							prediction,
							(std::vector<provallo::ImputeNode> *)NULL,
							(provallo::ImputedData *)NULL,
							(real_t)0,
							(size_t)row,
							(tree_num == NULL) ? NULL : (tree_num + nrows * tree),
							(per_tree_depths == NULL) ? NULL : (per_tree_depths + tree + row * model_outputs->trees.size()),
							(size_t)0);
					}
					output_depths[row] = score==score?score:0.0;
				}
			}
		} //end if

		else if (model_outputs_ext != NULL)
		{
			if (model_outputs_ext->missing_action == Fail && prediction.categ_data == NULL && prediction.Xc_indptr == NULL && prediction.Xr_indptr == NULL && !model_outputs_ext->has_range_penalty)
			{
				if (prediction.is_col_major && nrows > 1)
				{
					for (size_t row = 0; row < (decltype(row))nrows; row++)
					{
						real_t score = 0.;
						for (size_t tree = 0;
							 tree < model_outputs_ext->hplanes.size(); tree++)
						{
							traverse_hplane_fast_colmajor(
								model_outputs_ext->hplanes[tree],
								*model_outputs_ext,
								prediction,
								score,
								(tree_num == NULL) ? NULL : (tree_num + nrows * tree),
								(per_tree_depths == NULL) ? NULL : (per_tree_depths + tree + row * model_outputs_ext->hplanes.size()),
								(size_t)row);
						}

						output_depths[row] = score==score?score:0.0;
					}
				}

				else
				{
#ifdef OPENMP_
#pragma omp parallel for if (nrows > 1) schedule(static) num_threads(nthreads) \
	shared(nrows, model_outputs_ext, prediction_data, output_depths, tree_num, per_tree_depths)
#endif
					for (size_t row = 0; row < (decltype(row))nrows; row++)
					{
						real_t score = 0;
						for (size_t tree = 0;
							 tree < model_outputs_ext->hplanes.size(); tree++)
						{
							traverse_hplane_fast_rowmajor(
								model_outputs_ext->hplanes[tree],
								*model_outputs_ext,
								prediction.numeric_data + row * prediction.ncols_numeric,
								score,
								(tree_num == NULL) ? NULL : (tree_num + nrows * tree),
								(per_tree_depths == NULL) ? NULL : (per_tree_depths + tree + row * model_outputs_ext->hplanes.size()),
								(size_t)row);
						}
						output_depths[row] = score==score?score:0.0;
					}
				}
			}//end if

			else
			{
				for (size_t row = 0; row < (decltype(row))nrows; row++)
				{
					real_t score = 0;
					for (size_t tree = 0; tree < model_outputs_ext->hplanes.size();
						 tree++)
					{
						traverse_hplane(
							model_outputs_ext->hplanes[tree],
							*model_outputs_ext,
							prediction,
							score,
							(std::vector<ImputeNode> *)NULL,
							(ImputedData *)NULL,
							(tree_num == NULL) ? NULL : (tree_num + nrows * tree),
							(per_tree_depths == NULL) ? NULL : (per_tree_depths + tree + row * model_outputs_ext->hplanes.size()),
							(size_t)row);
					}
					output_depths[row] = score==score?score:0.0;
				}
			}//
		}

		/* translate sum-of-depths to outlier score */
		real_t ntrees=1., depth_divisor=0.000001;
		if (model_outputs != NULL)
		{
			ntrees = (real_t)model_outputs->trees.size();
			depth_divisor = ntrees * (model_outputs->exp_avg_depth);
		}

		else
		{
			ntrees = (real_t)model_outputs_ext->hplanes.size();
			depth_divisor = ntrees * (model_outputs_ext->exp_avg_depth);
		}

		/* for density and boxed_ratio, each tree will have 'log(d)'' instead of 'd' */
		bool is_density = (model_outputs != NULL && model_outputs->scoring_metric == Density) || (model_outputs_ext != NULL && model_outputs_ext->scoring_metric == Density);
		bool is_bratio = (model_outputs != NULL && model_outputs->scoring_metric == BoxedRatio) || (model_outputs_ext != NULL && model_outputs_ext->scoring_metric == BoxedRatio);
		bool is_bdens = (model_outputs != NULL && model_outputs->scoring_metric == BoxedDensity) || (model_outputs_ext != NULL && model_outputs_ext->scoring_metric == BoxedDensity);
		bool is_bdens2 = (model_outputs != NULL && model_outputs->scoring_metric == BoxedDensity2) || (model_outputs_ext != NULL && model_outputs_ext->scoring_metric == BoxedDensity2);

		if (standardize)
		{
			if (is_density || is_bdens2)
			{
				ntrees = -ntrees;
				for (size_t row = 0; row < nrows; row++)
					output_depths[row] /= ntrees;
			}

			else if (is_bdens)
			{
				for (size_t row = 0; row < nrows; row++)
					output_depths[row] = -std::exp(output_depths[row] / ntrees);
			}

			else if (is_bratio)
			{
				for (size_t row = 0; row < nrows; row++)
					output_depths[row] = output_depths[row] / ntrees;
			}

			else
			{
				for (size_t row = 0; row < nrows; row++)
					output_depths[row] = std::exp2(
						-output_depths[row] / depth_divisor);
			}
		}

		else
		{
			if (is_density || is_bdens || is_bdens2)
			{
				for (size_t row = 0; row < nrows; row++)
					output_depths[row] = std::exp(output_depths[row] / ntrees);
			}

			else if (is_bratio)
			{
				ntrees = -ntrees;
				for (size_t row = 0; row < nrows; row++)
					output_depths[row] /= ntrees;
			}

			else
			{
				for (size_t row = 0; row < nrows; row++)
					output_depths[row] /= ntrees;
			}
		}

		if (per_tree_depths != NULL && (is_density || is_bdens || is_bdens2))
		{
			size_t ntrees =
				(model_outputs != NULL) ? model_outputs->trees.size() : model_outputs_ext->hplanes.size();
			for (size_t ix = 0; ix < nrows * ntrees; ix++)
				per_tree_depths[ix] = std::exp(per_tree_depths[ix]);
		}

		/* re-map tree numbers to start at zero (if predicting tree numbers) */
		/* Note: usually this type of 'prediction' is not required,
		 thus this mapping is not stored in the model objects so as to
		 save memory */
		if (tree_num != NULL)
		{
			if (indexer != NULL && !indexer->indices.empty())
			{
				size_t ntrees =
					(model_outputs != NULL) ? model_outputs->trees.size() : model_outputs_ext->hplanes.size();
				if (model_outputs != NULL)
				{
					if (model_outputs->missing_action == Divide)
						goto manual_remap;
					if (model_outputs->new_cat_action == Weighted && model_outputs->cat_split_type == SubSet && categ_data != NULL)
						goto manual_remap;
				}

				for (size_t tree = 0; tree < ntrees; tree++)
				{
					size_t *mapping =
						indexer->indices[tree].terminal_node_mappings.data();
					for (size_t row = 0; row < nrows; row++)
					{
						tree_num[row + tree * nrows] = mapping[tree_num[row + tree * nrows]];
					}
				}
			}
 			else
			{
			manual_remap:
				remap_terminal_trees(model_outputs, model_outputs_ext,
									 prediction, tree_num, nthreads);
			}
		}
		//update output data
		
	}

	isolation_forest::isolation_forest(size_t ndim, size_t ntrees,
									   bool build_imputer, int nthreads_) : ndim(ndim), ntrees(ntrees), build_imputer(build_imputer), nthreads(nthreads_)
	{
		this->is_fitted = true;
	}

	template <class itype>
	isolation_forest
	isolation_forest::deserialize_template(itype &inp, int nthreads)
	{
		bool is_isotree_model = false;
		bool is_compatible = false;
		bool has_combined_objects = false;
		// bool has_IsoForest = false;
		// bool has_ExtIsoForest = false;
		// bool has_Imputer = false;
		// bool has_Indexer = false;
		// bool has_metadata = false;
		// size_t size_metadata = 0;
		//  TODO:Fix inspect serialized, replace with auto.
		/*inspect_serialized_object(
		 inp,
		 is_isotree_model,
		 is_compatible,
		 has_combined_objects,
		 has_IsoForest,
		 has_ExtIsoForest,
		 has_Imputer,
		 has_Indexer,
		 has_metadata,
		 size_metadata
		 );*/
		if (is_isotree_model && is_compatible && !has_combined_objects)
			throw std::runtime_error("Serialized model is not compatible.\n");

		provallo::iso_forest model = iso_forest();
		provallo::ExtIsoForest model_ext = ExtIsoForest();
		provallo::Imputer imputer = Imputer();
		provallo::TreesIndexer indexer = TreesIndexer();

		deserialize_combined(inp, &model, &model_ext, &imputer, &indexer,
							 (char *)nullptr);

		if (model.trees.empty() && model_ext.hplanes.empty())
			throw std::runtime_error("Error: model contains no trees.\n");

		size_t ntrees;
		size_t ndim = 3;
		bool build_imputer = false;

		if (!model.trees.empty())
		{
			ntrees = model.trees.size();
			ndim = 1;
		}
		else
		{
			ntrees = model_ext.hplanes.size();
		}
		if (!imputer.imputer_tree.empty())
		{
			if (imputer.imputer_tree.size() != ntrees)
				throw std::runtime_error(
					"Error: imputer has incorrect number of trees.\n");
			build_imputer = true;
		}
		if (!indexer.indices.empty())
		{
			if (indexer.indices.size() != ntrees)
				throw std::runtime_error(
					"Error: indexer has incorrect number of trees.\n");
		}

		isolation_forest out = isolation_forest(nthreads, ndim, ntrees,
												build_imputer);

		if (!model.trees.empty())
		{
			out.get_model() = std::move(model);
			out.penalize_range = out.get_model().has_range_penalty;
		}
		else
		{
			out.get_model_ext() = std::move(model_ext);
			out.penalize_range = out.get_model_ext().has_range_penalty;
		}
		if (!imputer.imputer_tree.empty())
			out.get_imputer() = std::move(imputer);
		if (!indexer.indices.empty())
			out.indexer = std::move(indexer);

		return out;
	}
	template <class real_t_>
	void
	add_linear_comb(size_t ix_arr[], size_t st, size_t end, real_t *res,
					real_t_ *x, real_t &coef, real_t x_sd, real_t x_mean,
					real_t &fill_val, MissingAction missing_action,
					real_t *buffer_arr, size_t *buffer_NAs, bool first_run);

	template <class real_t_, class mapping, class lreal_t_safe = long real_t>
	void
	add_linear_comb_weighted(size_t ix_arr[], size_t st, size_t end,
							 real_t *res, real_t_ *x, real_t &coef,
							 real_t x_sd, real_t x_mean, real_t &fill_val,
							 MissingAction missing_action, real_t *buffer_arr,
							 size_t *buffer_NAs, bool first_run, mapping &w);
	template <class real_t_ = real_t, class sparse_ix_ = sparse_ix>
	void
	add_linear_comb(size_t *ix_arr, size_t st, size_t end, size_t col_num,
					real_t *res, real_t_ *Xc, sparse_ix_ *Xc_ind,
					sparse_ix_ *Xc_indptr, real_t &coef, real_t x_sd,
					real_t x_mean, real_t &fill_val,
					MissingAction missing_action, real_t *buffer_arr,
					size_t *buffer_NAs, bool first_run);
	template <class real_t_, class sparse_ix, class mapping, class lreal_t_safe>
	void
	add_linear_comb_weighted(size_t *ix_arr, size_t st, size_t end,
							 size_t col_num, real_t *res, real_t_ *Xc,
							 sparse_ix *Xc_ind,
							 sparse_ix *Xc_indptr, real_t &coef, real_t x_sd,
							 real_t x_mean, real_t &fill_val,
							 MissingAction missing_action, real_t *buffer_arr,
							 size_t *buffer_NAs, bool first_run, mapping &w);
	template <class mapping>
	void
	add_linear_comb_weighted(size_t *ix_arr, size_t st, size_t end,
							 real_t *res, int x[], int ncat, real_t *cat_coef,
							 real_t single_cat_coef, int chosen_cat,
							 real_t &fill_val, real_t &fill_new,
							 size_t *buffer_pos, NewCategAction new_cat_action,
							 MissingAction missing_action,
							 CategSplit cat_split_type, bool first_run,
							 mapping &w);
	template <class lreal_t_safe>
	void
	add_linear_comb(size_t *ix_arr, size_t st, size_t end, real_t *res,
					int x[], int ncat, real_t *cat_coef,
					real_t single_cat_coef, int chosen_cat, real_t &fill_val,
					real_t &fill_new, size_t *buffer_cnt, size_t *buffer_pos,
					NewCategAction new_cat_action,
					MissingAction missing_action, CategSplit cat_split_type,
					bool first_run);
	template <class mapping, class lreal_t_safe>
	void
	add_linear_comb_weighted(size_t *ix_arr, size_t st, size_t end,
							 real_t *res, int x[], int ncat, real_t *cat_coef,
							 real_t single_cat_coef, int chosen_cat,
							 real_t &fill_val, real_t &fill_new,
							 size_t *buffer_pos, NewCategAction new_cat_action,
							 MissingAction missing_action,
							 CategSplit cat_split_type, bool first_run,
							 mapping &w);

	void
	traverse_hplane_csc(WorkerForPredictCSC &workspace,
						std::vector<IsoHPlane> &hplanes,
						ExtIsoForest &model_outputs, prediction_data &_data,
						sparse_ix *tree_num,
						real_t *per_tree_depths, size_t curr_tree,
						bool has_range_penalty)
	{
		// if (hplanes[curr_tree].score >= 0)
		if (unlikely(hplanes[curr_tree].hplane_left == 0))
		{
			for (size_t row = workspace.st; row <= workspace.end; row++)
				workspace.depths[workspace.ix_arr[row]] += hplanes[curr_tree].score;
			if (unlikely(tree_num != NULL))
				for (size_t row = workspace.st; row <= workspace.end; row++)
					tree_num[workspace.ix_arr[row]] = curr_tree;
			if (unlikely(per_tree_depths != NULL))
				for (size_t row = workspace.st; row <= workspace.end; row++)
					per_tree_depths[workspace.ix_arr[row]] = hplanes[curr_tree].score;
			return;
		}

		std::sort(workspace.ix_arr.begin() + workspace.st,
				  workspace.ix_arr.begin() + workspace.end + 1);
		std::fill(workspace.comb_val.begin(),
				  workspace.comb_val.begin() + (workspace.end - workspace.st + 1),
				  0.);
		real_t unused;

		if (likely(_data.categ_data == NULL))
		{
			for (size_t col = 0; col < hplanes[curr_tree].col_num.size(); col++)
				add_linear_comb(
					workspace.ix_arr.data(),
					workspace.st,
					workspace.end,
					hplanes[curr_tree].col_num[col],
					workspace.comb_val.data(),
					_data.Xc,
					_data.Xc_ind,
					_data.Xc_indptr,
					hplanes[curr_tree].coef[col],
					(real_t)0.,
					hplanes[curr_tree].mean[col],
					(model_outputs.missing_action == Fail) ? unused : hplanes[curr_tree].fill_val[col],
					model_outputs.missing_action, NULL, NULL, false);
		}

		else
		{
			size_t ncols_numeric = 0;
			size_t ncols_categ = 0;
			for (size_t col = 0; col < hplanes[curr_tree].col_num.size(); col++)
			{
				switch (hplanes[curr_tree].col_type[col])
				{
				case Numeric:
				{
					add_linear_comb(
						workspace.ix_arr.data(),
						workspace.st,
						workspace.end,
						hplanes[curr_tree].col_num[col],
						workspace.comb_val.data(),
						_data.Xc,
						_data.Xc_ind,
						_data.Xc_indptr,
						hplanes[curr_tree].coef[ncols_numeric],
						(real_t)0.0,
						hplanes[curr_tree].mean[ncols_numeric],
						(model_outputs.missing_action == Fail) ? unused : hplanes[curr_tree].fill_val[col],
						model_outputs.missing_action, NULL, NULL, false);
					ncols_numeric++;
					break;
				}

				case Categorical:
				{
					add_linear_comb<real_t>(
						workspace.ix_arr.data(),
						workspace.st,
						workspace.end,
						workspace.comb_val.data(),
						_data.categ_data + (hplanes[curr_tree].col_num[col] * _data.nrows),
						(model_outputs.cat_split_type == SubSet) ? (int)hplanes[curr_tree].cat_coef[ncols_categ].size() : 0,
						(model_outputs.cat_split_type == SubSet) ? hplanes[curr_tree].cat_coef[ncols_categ].data() : NULL,
						(model_outputs.cat_split_type == SingleCateg) ? hplanes[curr_tree].fill_new[ncols_categ] : 0.,
						(model_outputs.cat_split_type == SingleCateg) ? hplanes[curr_tree].chosen_cat[ncols_categ] : 0,
						hplanes[curr_tree].fill_val[col],
						hplanes[curr_tree].fill_new[ncols_categ], NULL, NULL,
						model_outputs.new_cat_action,
						model_outputs.missing_action,
						model_outputs.cat_split_type, false);
					ncols_categ++;
					break;
				}

				default:
				{
					assert(0);
					break;
				}
				}
			}
		}

		if (has_range_penalty)
		{
			for (size_t row = workspace.st; row <= workspace.end; row++)
				workspace.depths[workspace.ix_arr[row]] -= (workspace.comb_val[row - workspace.st] < hplanes[curr_tree].range_low) || (workspace.comb_val[row - workspace.st] > hplanes[curr_tree].range_high);
		}

		/* divide data */
		size_t split_ix = divide_subset_split(workspace.ix_arr.data(),
											  workspace.comb_val.data(),
											  workspace.st, workspace.end,
											  hplanes[curr_tree].split_point);

		/* continue splitting recursively */
		size_t orig_end = workspace.end;
		if (split_ix > workspace.st)
		{
			workspace.end = split_ix - 1;
			traverse_hplane_csc(workspace, hplanes, model_outputs, _data, tree_num,
								per_tree_depths, hplanes[curr_tree].hplane_left,
								has_range_penalty);
		}

		if (split_ix <= orig_end)
		{
			workspace.st = split_ix;
			workspace.end = orig_end;
			traverse_hplane_csc(workspace, hplanes, model_outputs, _data, tree_num,
								per_tree_depths, hplanes[curr_tree].hplane_right,
								has_range_penalty);
		}
	}

	classifier::~classifier()
	{	
		delete_factory();
		
 	}
	void
	sample_random_rows(std::vector<size_t> &ix_arr, size_t nrows,
					   bool with_replacement,
					   RNG_engine &rnd_generator,
					   std::vector<size_t> &ix_all,
					   real_t *sample_weights,
					   std::vector<real_t> &btree_weights, size_t log2_n,
					   size_t btree_offset, std::vector<bool> &is_repeated)
	{
		size_t ntake = ix_arr.size();

		/* if with replacement, just generate random uniform numbers */
		if (with_replacement)
		{
			if (sample_weights == NULL)
			{
				std::uniform_int_distribution<size_t> runif(0, nrows - 1);
				for (size_t &ix : ix_arr)
					ix = runif(rnd_generator);
			}

			else
			{
				std::discrete_distribution<size_t> runif(sample_weights,
														 sample_weights + nrows);
				for (size_t &ix : ix_arr)
					ix = runif(rnd_generator);
			}
		}

		/* if all the elements are needed, don't bother with any sampling */
		else if (ntake == nrows)
		{
			std::iota(ix_arr.begin(), ix_arr.end(), (size_t)0);
		}
		else if (sample_weights != NULL)
		{
			/* TODO: here could instead generate only 1 random number from zero to the full weight,
			 and then subtract from it as it goes down every level. Would have less precision
			 but should still work fine. */

			real_t rnd_subrange, w_left;
			real_t curr_subrange;
			size_t curr_ix;
			for (size_t &ix : ix_arr)
			{
				/* go down the tree by drawing a random number and
				 checking if it falls in the left or right ranges */
				curr_ix = 0;
				curr_subrange = btree_weights[0];
				for (size_t lev = 0; lev < log2_n; lev++)
				{
					rnd_subrange = std::uniform_real_distribution<real_t>(
						0., curr_subrange)(rnd_generator);
					w_left = btree_weights[ix_child(curr_ix)];
					curr_ix = ix_child(curr_ix) + (rnd_subrange >= w_left);
					curr_subrange = btree_weights[curr_ix];
				}

				/* finally, determine element to choose in this iteration */
				ix = curr_ix - btree_offset;

				/* now remove the weight of the chosen element */
				btree_weights[curr_ix] = 0;
				for (size_t lev = 0; lev < log2_n; lev++)
				{
					curr_ix = ix_parent(curr_ix);
					btree_weights[curr_ix] = btree_weights[ix_child(curr_ix)] + btree_weights[ix_child(curr_ix) + 1];
				}
			}
		}
		else
		{

			/* if sampling a larger fraction, fill an array enumerating the rows, shuffle, and take first N  */
			if (ntake >= (nrows / 2))
			{

				if (ix_all.empty())
					ix_all.resize(nrows);

				/* in order for random seeds to always be reproducible, don't re-use previous shuffles */
				std::iota(ix_all.begin(), ix_all.end(), (size_t)0);

				/* If the number of sampled elements is large, do a full shuffle, enjoy simd-instructs when copying over */
				if (ntake >= ((nrows * 3) / 4))
				{
					std::shuffle(ix_all.begin(), ix_all.end(), rnd_generator);
					ix_arr.assign(ix_all.begin(), ix_all.begin() + ntake);
				}

				/* otherwise, do only a partial shuffle (use Yates algorithm) and copy elements along the way */
				else
				{
					size_t chosen;
					for (size_t i = nrows - 1; i >= nrows - ntake; i--)
					{
						chosen = std::uniform_int_distribution<size_t>(0, i)(
							rnd_generator);
						ix_arr[nrows - i - 1] = ix_all[chosen];
						ix_all[chosen] = ix_all[i];
					}
				}
			}
			else
			{

				size_t candidate;

				/* if the sample size is relatively large, use a temporary boolean vector */
				if (((long real_t)ntake / (long real_t)nrows) > (1. / 50.))
				{

					if (is_repeated.empty())
						is_repeated.resize(nrows, false);
					else
						is_repeated.assign(is_repeated.size(), false);

					for (size_t rnd_ix = nrows - ntake; rnd_ix < nrows; rnd_ix++)
					{
						candidate = std::uniform_int_distribution<size_t>(0,
																		  rnd_ix)(
							rnd_generator);
						if (is_repeated[candidate])
						{
							ix_arr[ntake - (nrows - rnd_ix)] = rnd_ix;
							is_repeated[rnd_ix] = true;
						}

						else
						{
							ix_arr[ntake - (nrows - rnd_ix)] = candidate;
							is_repeated[candidate] = true;
						}
					}
				}
				/* if the sample size is very small, use an unordered set */
				else
				{

					hashed_set<size_t> repeated_set;
					repeated_set.reserve(ntake);
					for (size_t rnd_ix = nrows - ntake; rnd_ix < nrows; rnd_ix++)
					{
						candidate = std::uniform_int_distribution<size_t>(0,
																		  rnd_ix)(
							rnd_generator);
						if (repeated_set.find(candidate) == repeated_set.end()) /* TODO: switch to C++20 'contains' */
						{
							ix_arr[ntake - (nrows - rnd_ix)] = candidate;
							repeated_set.insert(candidate);
						}

						else
						{
							ix_arr[ntake - (nrows - rnd_ix)] = rnd_ix;
							repeated_set.insert(rnd_ix);
						}
					}
				}
			}
		}
	}

	template <typename lreal_t_safe>
	real_t
	calc_kurtosis_weighted(real_t *x, size_t n_, MissingAction missing_action,
						   real_t *w)
	{
		lreal_t_safe m = 0;
		lreal_t_safe M2 = 0, M3 = 0, M4 = 0;
		lreal_t_safe delta=0., delta_s=0., delta_div=0.;
		lreal_t_safe diff;
		lreal_t_safe n = 0;
		lreal_t_safe out =0.;
		lreal_t_safe n_prev = 0.;
		lreal_t_safe w_this;
		UNDEF_REFERENCE(missing_action);
		UNDEF_REFERENCE2(missing_action );
		UNDEF_REFERENCE2(w);

		for (size_t row = 0; row < n_; row++)
		{
			if (likely(!is_na_or_inf(x[row])))
			{
				w_this = w[row];
				n += w_this;

				delta = x[row] - m;
				delta_div = delta / n;
				delta_s = delta_div * delta_div;
				diff = delta * (delta_div * n_prev);
				n_prev = n;

				m += w_this * (delta_div);
				M4 += w_this * (diff * delta_s * (n * n - 3 * n + 3) + 6 * delta_s * M2 - 4 * delta_div * M3);
				M3 += w_this * (diff * delta_div * (n - 2) - 3 * delta_div * M2);
				M2 += w_this * (diff);
			}
		}
		if (unlikely(n <= 0))
			return -HUGE_VAL;
		out = (M4 / M2) * (n / M2);
		return (!is_na_or_inf(out)) ? std::fmax((real_t)out, 0.) : (-HUGE_VAL);
	}
	template <class real, class lreal_t_safe>
	real_t
	calc_kurtosis(size_t ix_arr[], size_t st, size_t end, real x[],
				  MissingAction missing_action)
	{
		lreal_t_safe m = 0;
		lreal_t_safe M2 = 0, M3 = 0, M4 = 0;
		lreal_t_safe delta, delta_s, delta_div;
		lreal_t_safe diff, n;
		lreal_t_safe out;

		if (missing_action == Fail)
		{
			for (size_t row = st; row <= end; row++)
			{
				n = (lreal_t_safe)(row - st + 1);

				delta = x[ix_arr[row]] - m;
				delta_div = delta / n;
				delta_s = delta_div * delta_div;
				diff = delta * (delta_div * (lreal_t_safe)(row - st));

				m += delta_div;
				M4 += diff * delta_s * (n * n - 3 * n + 3) + 6 * delta_s * M2 - 4 * delta_div * M3;
				M3 += diff * delta_div * (n - 2) - 3 * delta_div * M2;
				M2 += diff;
			}

			if (unlikely(!is_na_or_inf(M2) && M2 <= 0))
			{
				if (!check_more_than_two_unique_values(ix_arr, st, end, x,
													   missing_action))
					return -HUGE_VAL;
			}

			out = (M4 / M2) * ((lreal_t_safe)(end - st + 1) / M2);
			return (!is_na_or_inf(out)) ? std::fmax((real_t)out, 0.) : (-HUGE_VAL);
		}

		else
		{
			size_t cnt = 0;
			for (size_t row = st; row <= end; row++)
			{
				if (likely(!is_na_or_inf(x[ix_arr[row]])))
				{
					cnt++;
					n = (lreal_t_safe)cnt;

					delta = x[ix_arr[row]] - m;
					delta_div = delta / n;
					delta_s = delta_div * delta_div;
					diff = delta * (delta_div * (lreal_t_safe)(cnt - 1));

					m += delta_div;
					M4 += diff * delta_s * (n * n - 3 * n + 3) + 6 * delta_s * M2 - 4 * delta_div * M3;
					M3 += diff * delta_div * (n - 2) - 3 * delta_div * M2;
					M2 += diff;
				}
			}

			if (unlikely(cnt == 0))
				return -HUGE_VAL;
			if (unlikely(!is_na_or_inf(M2) && M2 <= 0))
			{
				if (!check_more_than_two_unique_values(ix_arr, st, end, x,
													   missing_action))
					return -HUGE_VAL;
			}

			out = (M4 / M2) * ((lreal_t_safe)cnt / M2);
			return (!is_na_or_inf(out)) ? std::fmax((real_t)out, 0.) : (-HUGE_VAL);
		}
	}

	template <class xreal, class lreal_t_safe>
	real_t
	calc_kurtosis(xreal x[], size_t n, MissingAction missing_action)
	{
		lreal_t_safe m = 0;
		lreal_t_safe M2 = 0, M3 = 0, M4 = 0;
		lreal_t_safe delta, delta_s, delta_div;
		lreal_t_safe diff, n_;
		lreal_t_safe out;

		if (missing_action == Fail)
		{
			for (size_t row = 0; row < n; row++)
			{
				n_ = (lreal_t_safe)(row + 1);

				delta = x[row] - m;
				delta_div = delta / n_;
				delta_s = delta_div * delta_div;
				diff = delta * (delta_div * (lreal_t_safe)row);

				m += delta_div;
				M4 += diff * delta_s * (n_ * n_ - 3 * n_ + 3) + 6 * delta_s * M2 - 4 * delta_div * M3;
				M3 += diff * delta_div * (n_ - 2) - 3 * delta_div * M2;
				M2 += diff;
			}

			out = (M4 / M2) * ((lreal_t_safe)n / M2);
			return (!is_na_or_inf(out)) ? std::fmax((real_t)out, 0.) : (-HUGE_VAL);
		}

		else
		{
			size_t cnt = 0;
			for (size_t row = 0; row < n; row++)
			{
				if (likely(!is_na_or_inf(x[row])))
				{
					cnt++;
					n_ = (lreal_t_safe)cnt;

					delta = x[row] - m;
					delta_div = delta / n_;
					delta_s = delta_div * delta_div;
					diff = delta * (delta_div * (lreal_t_safe)(cnt - 1));

					m += delta_div;
					M4 += diff * delta_s * (n_ * n_ - 3 * n_ + 3) + 6 * delta_s * M2 - 4 * delta_div * M3;
					M3 += diff * delta_div * (n_ - 2) - 3 * delta_div * M2;
					M2 += diff;
				}
			}

			if (unlikely(cnt == 0))
				return -HUGE_VAL;

			out = (M4 / M2) * ((lreal_t_safe)cnt / M2);
			return (!is_na_or_inf(out)) ? std::fmax((real_t)out, 0.) : (-HUGE_VAL);
		}
	}

	/* TODO: is this algorithm correct? */
	template <class xreal, class mapping, class lreal_t_safe>
	real_t
	calc_kurtosis_weighted(size_t ix_arr[], size_t st, size_t end, xreal x[],
						   MissingAction missing_action, mapping &w)
	{
		lreal_t_safe m = 0;
		lreal_t_safe M2 = 0, M3 = 0, M4 = 0;
		lreal_t_safe delta, delta_s, delta_div;
		lreal_t_safe diff;
		lreal_t_safe n = 0;
		lreal_t_safe out;
		lreal_t_safe n_prev = 0.;
		lreal_t_safe w_this;

		for (size_t row = st; row <= end; row++)
		{
			if (likely(!is_na_or_inf(x[ix_arr[row]])))
			{
				w_this = w[ix_arr[row]];
				n += w_this;

				delta = x[ix_arr[row]] - m;
				delta_div = delta / n;
				delta_s = delta_div * delta_div;
				diff = delta * (delta_div * n_prev);
				n_prev = n;

				m += w_this * (delta_div);
				M4 += w_this * (diff * delta_s * (n * n - 3 * n + 3) + 6 * delta_s * M2 - 4 * delta_div * M3);
				M3 += w_this * (diff * delta_div * (n - 2) - 3 * delta_div * M2);
				M2 += w_this * (diff);
			}
		}

		if (unlikely(n <= 0))
			return -HUGE_VAL;
		if (unlikely(
				!is_na_or_inf(M2) && M2 <= std::numeric_limits<real_t>::epsilon()))
		{
			if (!check_more_than_two_unique_values(ix_arr, st, end, x,
												   missing_action))
				return -HUGE_VAL;
		}

		out = (M4 / M2) * (n / M2);
		return (!is_na_or_inf(out)) ? std::fmax((real_t)out, 0.) : (-HUGE_VAL);
	}

	template <class xreal, class lreal_t_safe>
	real_t
	calc_kurtosis_weighted(xreal *x, size_t n_, MissingAction missing_action,
						   xreal *w)
	{
 		UNDEF_REFERENCE(missing_action)
 		UNDEF_REFERENCE2(missing_action)
	
		lreal_t_safe m = 0;
		lreal_t_safe M2 = 0, M3 = 0, M4 = 0;
		lreal_t_safe delta, delta_s, delta_div;
		lreal_t_safe diff;
		lreal_t_safe n = 0;
		lreal_t_safe out;
		lreal_t_safe n_prev = 0.;
		lreal_t_safe w_this =(lreal_t_safe) w[0];

		for (size_t row = 0; row < n_; row++)
		{
			if (likely(!is_na_or_inf(x[row])))
			{
				w_this = w[row];
				n += w_this;

				delta = x[row] - m;
				delta_div = delta / n;
				delta_s = delta_div * delta_div;
				diff = delta * (delta_div * n_prev);
				n_prev = n;

				m += w_this * (delta_div);
				M4 += w_this * (diff * delta_s * (n * n - 3 * n + 3) + 6 * delta_s * M2 - 4 * delta_div * M3);
				M3 += w_this * (diff * delta_div * (n - 2) - 3 * delta_div * M2);
				M2 += w_this * (diff);
			}
		}

		if (unlikely(n <= 0))
			return -HUGE_VAL;

		out = (M4 / M2) * (n / M2);
		return (!is_na_or_inf(out)) ? std::fmax((real_t)out, 0.) : (-HUGE_VAL);
	}

	/* TODO: make these compensated sums */
	/* TODO: can this use the same algorithm as above but with a correction at the end,
	 like it was done for the variance? */
	template <class xreal = real_t, class xsparse = sparse_ix,
			  class lreal_t_safe = long real_t>
	real_t
	calc_kurtosis(size_t *ix_arr, size_t st, size_t end, size_t col_num,
				  xreal Xc[], sparse_ix *Xc_ind, sparse_ix *Xc_indptr,
				  MissingAction missing_action)
	{
		/* ix_arr must be already sorted beforehand */
		if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
			return -HUGE_VAL;

		lreal_t_safe s1 = 0;
		lreal_t_safe s2 = 0;
		lreal_t_safe s3 = 0;
		lreal_t_safe s4 = 0;
		lreal_t_safe x_sq;
		size_t cnt = end - st + 1;

		if (unlikely(cnt <= 1))
			return -HUGE_VAL;

		size_t st_col = Xc_indptr[col_num];
		size_t end_col = Xc_indptr[col_num + 1] - 1;
		size_t curr_pos = st_col;
		size_t ind_end_col = Xc_ind[end_col];
		size_t *ptr_st = std::lower_bound(ix_arr + st, ix_arr + end + 1,
										  Xc_ind[st_col]);

		lreal_t_safe xval;

		if (missing_action != Fail)
		{
			for (size_t *row = ptr_st;
				 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;)
			{
				if (Xc_ind[curr_pos] == (sparse_ix)(*row))
				{
					xval = Xc[curr_pos];
					if (unlikely(is_na_or_inf(xval)))
					{
						cnt--;
					}

					else
					{
						/* TODO: is it safe to use FMA here? some calculations rely on assuming that
						 some of these 's' are larger than the others. Would this procedure be guaranteed
						 to preserve such differences if done with a mixture of sums and FMAs? */
						x_sq = square(xval);
						s1 += xval;
						s2 = std::fma(xval, xval, s2);
						s3 = std::fma(x_sq, xval, s3);
						s4 = std::fma(x_sq, x_sq, s4);
						// s2 += x_sq;
						s1 += pw1(xval);
						s2 += pw2(xval);
						s3 += pw3(xval);
						s4 += pw4(xval);
					}

					if (row == ix_arr + end || curr_pos == end_col)
						break;
					curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
												Xc_ind + end_col + 1, *(++row)) -
							   Xc_ind;
				}

				else
				{
					if (Xc_ind[curr_pos] > (sparse_ix)(*row))
						row = std::lower_bound(row + 1, ix_arr + end + 1,
											   Xc_ind[curr_pos]);
					else
						curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
													Xc_ind + end_col + 1, *row) -
								   Xc_ind;
				}
			}

			if (unlikely(
					cnt <= (end - st + 1) - (Xc_indptr[col_num + 1] - Xc_indptr[col_num])))
				return -HUGE_VAL;
		}

		else
		{
			for (size_t *row = ptr_st;
				 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;)
			{
				if (Xc_ind[curr_pos] == (sparse_ix)(*row))
				{
					xval = Xc[curr_pos];
					x_sq = square(xval);

					s1 += xval;
					s2 = std::fma(xval, xval, s2);
					s3 = std::fma(x_sq, xval, s3);
					s4 = std::fma(x_sq, x_sq, s4);
					// s1 += pw1(xval);
					// s2 += pw2(xval);
					// s3 += pw3(xval);
					// s4 += pw4(xval);

					if (row == ix_arr + end || curr_pos == end_col)
						break;
					curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
												Xc_ind + end_col + 1, *(++row)) -
							   Xc_ind;
				}

				else
				{
					if (Xc_ind[curr_pos] > (sparse_ix)(*row))
						row = std::lower_bound(row + 1, ix_arr + end + 1,
											   Xc_ind[curr_pos]);
					else
						curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
													Xc_ind + end_col + 1, *row) -
								   Xc_ind;
				}
			}
		}

		if (unlikely(cnt <= 1 || s2 == 0 || s2 == pw2(s1)))
			return -HUGE_VAL;
		lreal_t_safe cnt_l = (lreal_t_safe)cnt;
		lreal_t_safe sn = s1 / cnt_l;
		lreal_t_safe v = s2 / cnt_l - pw2(sn);
		if (unlikely(std::isnan(v)))
			return -HUGE_VAL;
		if (v <= std::numeric_limits<real_t>::epsilon() && !check_more_than_two_unique_values(ix_arr, st, end, col_num,
																							  Xc_indptr, Xc_ind, Xc,
																							  missing_action))
			return -HUGE_VAL;
		if (unlikely(v <= 0))
			return 0.;
		lreal_t_safe out = (s4 - 4 * s3 * sn + 6 * s2 * pw2(sn) - 4 * s1 * pw3(sn) + cnt_l * pw4(sn)) / (cnt_l * pw2(v));
		return (!is_na_or_inf(out)) ? std::fmax((real_t)out, 0.) : (-HUGE_VAL);
	}

	template <class xreal = real_t, class xsparse = sparse_ix,
			  class lreal_t_safe = long real_t>
	real_t
	calc_kurtosis(size_t col_num, size_t nrows, xreal Xc[], xsparse *Xc_ind,
				  xsparse *Xc_indptr, MissingAction missing_action)
	{
		if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
			return -HUGE_VAL;

		lreal_t_safe s1 = 0;
		lreal_t_safe s2 = 0;
		lreal_t_safe s3 = 0;
		lreal_t_safe s4 = 0;
		lreal_t_safe x_sq;
		size_t cnt = nrows;

		if (unlikely(cnt <= 1))
			return -HUGE_VAL;

		lreal_t_safe xval;

		if (missing_action != Fail)
		{
			for (auto ix = Xc_indptr[col_num]; ix < Xc_indptr[col_num + 1]; ix++)
			{
				xval = Xc[ix];
				if (unlikely(is_na_or_inf(xval)))
				{
					cnt--;
				}

				else
				{
					x_sq = square(xval);
					s1 += xval;
					s2 = std::fma(xval, xval, s2);
					s3 = std::fma(x_sq, xval, s3);
					s4 = std::fma(x_sq, x_sq, s4);
					// s1 += pw1(xval);
					// s2 += pw2(xval);
					// s3 += pw3(xval);
					// s4 += pw4(xval);
				}
			}

			if (cnt <= (nrows) - (Xc_indptr[col_num + 1] - Xc_indptr[col_num]))
				return -HUGE_VAL;
		}

		else
		{
			for (auto ix = Xc_indptr[col_num]; ix < Xc_indptr[col_num + 1]; ix++)
			{
				xval = Xc[ix];
				x_sq = square(xval);
				s1 += xval;
				s2 = std::fma(xval, xval, s2);
				s3 = std::fma(x_sq, xval, s3);
				s4 = std::fma(x_sq, x_sq, s4);
				// s1 += pw1(xval);
				// s2 += pw2(xval);
				// s3 += pw3(xval);
				// s4 += pw4(xval);
			}
		}

		if (unlikely(cnt <= 1 || s2 == 0 || s2 == pw2(s1)))
			return -HUGE_VAL;
		lreal_t_safe cnt_l = (lreal_t_safe)cnt;
		lreal_t_safe sn = s1 / cnt_l;
		lreal_t_safe v = s2 / cnt_l - pw2(sn);
		if (unlikely(std::isnan(v)))
			return -HUGE_VAL;
		if (v <= std::numeric_limits<real_t>::epsilon() && !check_more_than_two_unique_values(nrows, col_num, Xc_indptr,
																							  Xc_ind, Xc, missing_action))
			return -HUGE_VAL;
		if (unlikely(v <= 0))
			return 0.;
		lreal_t_safe out = (s4 - 4 * s3 * sn + 6 * s2 * pw2(sn) - 4 * s1 * pw3(sn) + cnt_l * pw4(sn)) / (cnt_l * pw2(v));
		return (!is_na_or_inf(out)) ? std::fmax((real_t)out, 0.) : (-HUGE_VAL);
	}

	template <class xreal = real_t, class xsparse = sparse_ix, class xmapping,
			  class lreal_t_safe>
	real_t
	calc_kurtosis_weighted(size_t *ix_arr, size_t st, size_t end,
						   size_t col_num,
						   real_t Xc[],
						   xsparse *Xc_ind, xsparse *Xc_indptr,
						   MissingAction missing_action, xmapping &w)
	{
		/* ix_arr must be already sorted beforehand */
		if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
			return -HUGE_VAL;

		lreal_t_safe s1 = 0;
		lreal_t_safe s2 = 0;
		lreal_t_safe s3 = 0;
		lreal_t_safe s4 = 0;
		lreal_t_safe x_sq;
		lreal_t_safe w_this;
		lreal_t_safe cnt = 0;
		for (size_t row = st; row <= end; row++)
			cnt += w[ix_arr[row]];

		if (unlikely(cnt <= 0))
			return -HUGE_VAL;

		size_t st_col = Xc_indptr[col_num];
		size_t end_col = Xc_indptr[col_num + 1] - 1;
		size_t curr_pos = st_col;
		size_t ind_end_col = Xc_ind[end_col];
		size_t *ptr_st = std::lower_bound(ix_arr + st, ix_arr + end + 1,
										  Xc_ind[st_col]);

		lreal_t_safe xval;

		if (missing_action != Fail)
		{
			for (size_t *row = ptr_st;
				 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;)
			{
				if (Xc_ind[curr_pos] == (sparse_ix)(*row))
				{
					w_this = w[*row];
					xval = Xc[curr_pos];

					if (unlikely(is_na_or_inf(xval)))
					{
						cnt -= w_this;
					}

					else
					{
						x_sq = xval * xval;
						s1 = std::fma(w_this, xval, s1);
						s2 = std::fma(w_this, x_sq, s2);
						s3 = std::fma(w_this, x_sq * xval, s3);
						s4 = std::fma(w_this, x_sq * x_sq, s4);
						// s1 += w_this * pw1(xval);
						// s2 += w_this * pw2(xval);
						// s3 += w_this * pw3(xval);
						// s4 += w_this * pw4(xval);
					}

					if (row == ix_arr + end || curr_pos == end_col)
						break;
					curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
												Xc_ind + end_col + 1, *(++row)) -
							   Xc_ind;
				}

				else
				{
					if (Xc_ind[curr_pos] > (sparse_ix)(*row))
						row = std::lower_bound(row + 1, ix_arr + end + 1,
											   Xc_ind[curr_pos]);
					else
						curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
													Xc_ind + end_col + 1, *row) -
								   Xc_ind;
				}
			}

			if (unlikely(cnt <= 0))
				return -HUGE_VAL;
		}

		else
		{
			for (size_t *row = ptr_st;
				 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;)
			{
				if (Xc_ind[curr_pos] == (sparse_ix)(*row))
				{
					w_this = w[*row];
					xval = Xc[curr_pos];

					x_sq = xval * xval;
					s1 = std::fma(w_this, xval, s1);
					s2 = std::fma(w_this, x_sq, s2);
					s3 = std::fma(w_this, x_sq * xval, s3);
					s4 = std::fma(w_this, x_sq * x_sq, s4);
					// s1 += w_this * pw1(xval);
					// s2 += w_this * pw2(xval);
					// s3 += w_this * pw3(xval);
					// s4 += w_this * pw4(xval);

					if (row == ix_arr + end || curr_pos == end_col)
						break;
					curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
												Xc_ind + end_col + 1, *(++row)) -
							   Xc_ind;
				}

				else
				{
					if (Xc_ind[curr_pos] > (sparse_ix)(*row))
						row = std::lower_bound(row + 1, ix_arr + end + 1,
											   Xc_ind[curr_pos]);
					else
						curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
													Xc_ind + end_col + 1, *row) -
								   Xc_ind;
				}
			}
		}

		if (unlikely(cnt <= 1 || s2 == 0 || s2 == pw2(s1)))
			return -HUGE_VAL;
		lreal_t_safe sn = s1 / cnt;
		lreal_t_safe v = s2 / cnt - pw2(sn);
		if (unlikely(std::isnan(v)))
			return -HUGE_VAL;
		if (v <= std::numeric_limits<real_t>::epsilon() && !check_more_than_two_unique_values(ix_arr, st, end, col_num,
																							  Xc_indptr, Xc_ind, Xc,
																							  missing_action))
			return -HUGE_VAL;
		if (v <= 0)
			return 0.;
		lreal_t_safe out = (s4 - 4 * s3 * sn + 6 * s2 * pw2(sn) - 4 * s1 * pw3(sn) + cnt * pw4(sn)) / (cnt * pw2(v));
		return (!is_na_or_inf(out)) ? std::fmax((real_t)out, 0.) : (-HUGE_VAL);
	}

	template <class xreal, class xsparse, class lreal_t_safe>
	real_t
	calc_kurtosis_weighted(size_t col_num, size_t nrows, xreal *Xc,
						   xsparse *Xc_ind, xsparse *Xc_indptr,
						   MissingAction missing_action, xreal *w)
	{
		if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
			return -HUGE_VAL;

		lreal_t_safe s1 = 0;
		lreal_t_safe s2 = 0;
		lreal_t_safe s3 = 0;
		lreal_t_safe s4 = 0;
		lreal_t_safe x_sq;
		lreal_t_safe w_this;
		lreal_t_safe cnt = nrows - (Xc_indptr[col_num + 1] - Xc_indptr[col_num]);
		for (auto ix = Xc_indptr[col_num]; ix < Xc_indptr[col_num + 1]; ix++)
			cnt += w[Xc_ind[ix]];

		if (unlikely(cnt <= 0))
			return -HUGE_VAL;

		lreal_t_safe xval;

		if (missing_action != Fail)
		{
			for (auto ix = Xc_indptr[col_num]; ix < Xc_indptr[col_num + 1]; ix++)
			{
				w_this = w[Xc_ind[ix]];
				xval = Xc[ix];

				if (unlikely(is_na_or_inf(xval)))
				{
					cnt -= w_this;
				}

				else
				{
					x_sq = xval * xval;
					s1 = std::fma(w_this, xval, s1);
					s2 = std::fma(w_this, x_sq, s2);
					s3 = std::fma(w_this, x_sq * xval, s3);
					s4 = std::fma(w_this, x_sq * x_sq, s4);
					// s1 += w_this * pw1(xval);
					// s2 += w_this * pw2(xval);
					// s3 += w_this * pw3(xval);
					// s4 += w_this * pw4(xval);
				}
			}

			if (cnt <= 0)
				return -HUGE_VAL;
		}

		else
		{
			for (auto ix = Xc_indptr[col_num]; ix < Xc_indptr[col_num + 1]; ix++)
			{
				w_this = w[Xc_ind[ix]];
				xval = Xc[ix];

				x_sq = xval * xval;
				s1 = std::fma(w_this, xval, s1);
				s2 = std::fma(w_this, x_sq, s2);
				s3 = std::fma(w_this, x_sq * xval, s3);
				s4 = std::fma(w_this, x_sq * x_sq, s4);
				// s1 += w_this * pw1(xval);
				// s2 += w_this * pw2(xval);
				// s3 += w_this * pw3(xval);
				// s4 += w_this * pw4(xval);
			}
		}

		if (unlikely(cnt <= 1 || s2 == 0 || s2 == pw2(s1)))
			return -HUGE_VAL;
		lreal_t_safe sn = s1 / cnt;
		lreal_t_safe v = s2 / cnt - pw2(sn);
		if (unlikely(std::isnan(v)))
			return -HUGE_VAL;
		if (v <= std::numeric_limits<real_t>::epsilon() && !check_more_than_two_unique_values(nrows, col_num, Xc_indptr,
																							  Xc_ind, Xc, missing_action))
			return -HUGE_VAL;
		if (unlikely(v <= 0))
			return -HUGE_VAL;
		lreal_t_safe out = (s4 - 4 * s3 * sn + 6 * s2 * pw2(sn) - 4 * s1 * pw3(sn) + cnt * pw4(sn)) / (cnt * pw2(v));
		return (!is_na_or_inf(out)) ? std::fmax((real_t)out, 0.) : (-HUGE_VAL);
	}

	template <class lreal_t_safe>
	real_t
	calc_kurtosis_internal(size_t cnt, int x[], int ncat, size_t buffer_cnt[],
						   real_t buffer_prob[], MissingAction missing_action,
						   CategSplit cat_split_type,
						   RNG_engine &rnd_generator)
	{
		/* This calculation proceeds as follows:
		 - If splitting by subsets, it will assign a random weight ~Unif(0,1) to
		 each category, and approximate kurtosis by sampling from such distribution
		 with the same probabilities as given by the current counts.
		 - If splitting by isolating one category, will binarize at each categorical level,
		 assume the values are zero or one, and output the average assuming each categorical
		 level has equal probability of being picked.
		 (Note that both are misleading heuristics, but might be better than random)
		 */
		UNDEF_REFERENCE(x)	
		UNDEF_REFERENCE2(missing_action)
		

		real_t sum_kurt = 0;

		cnt -= buffer_cnt[ncat];
		if (cnt <= 1)
			return -HUGE_VAL;
		lreal_t_safe cnt_l = (lreal_t_safe)cnt;
		for (int cat = 0; cat < ncat; cat++)
			buffer_prob[cat] = buffer_cnt[cat] / cnt_l;

		switch (cat_split_type)
		{
		case SubSet:
		{
			lreal_t_safe temp_v;
			lreal_t_safe s1, s2, s3, s4;
			lreal_t_safe coef;
			lreal_t_safe coef2;
			lreal_t_safe w_this;
			UniformUnitInterval runif(0, 1);
			size_t ntry = 50;
			for (size_t iternum = 0; iternum < 50; iternum++)
			{
				s1 = 0;
				s2 = 0;
				s3 = 0;
				s4 = 0;
				for (int cat = 0; cat < ncat; cat++)
				{
					coef = runif(rnd_generator);
					coef2 = coef * coef;
					w_this = buffer_prob[cat];
					s1 = std::fma(w_this, coef, s1);
					s2 = std::fma(w_this, coef2, s2);
					s3 = std::fma(w_this, coef2 * coef, s3);
					s4 = std::fma(w_this, coef2 * coef2, s4);
					// s1 += buffer_prob[cat] * pw1(coef);
					// s2 += buffer_prob[cat] * pw2(coef);
					// s3 += buffer_prob[cat] * pw3(coef);
					// s4 += buffer_prob[cat] * pw4(coef);
				}
				temp_v = s2 - pw2(s1);
				if (temp_v <= 0)
					ntry--;
				else
					sum_kurt += (s4 - 4 * s3 * pw1(s1) + 6 * s2 * pw2(s1) - 4 * s1 * pw3(s1) + pw4(s1)) / pw2(temp_v);
			}
			if (unlikely(!ntry))
				return -HUGE_VAL;
			else if (unlikely(is_na_or_inf(sum_kurt)))
				return -HUGE_VAL;
			else
				return std::fmax(sum_kurt, 0.) / (real_t)ntry;
		}

		case SingleCateg:
		{
			real_t p;
			int ncat_present = ncat;
			for (int cat = 0; cat < ncat; cat++)
			{
				p = buffer_prob[cat];
				if (p == 0)
					ncat_present--;
				else
					sum_kurt += (p - 4 * p * pw1(p) + 6 * p * pw2(p) - 4 * p * pw3(p) + pw4(p)) / pw2(p - pw2(p));
			}
			if (ncat_present <= 1)
				return -HUGE_VAL;
			else if (unlikely(is_na_or_inf(sum_kurt)))
				return -HUGE_VAL;
			else
				return std::fmax(sum_kurt, 0.) / (real_t)ncat_present;
		}
		}

		return -1; /* this will never be reached, but CRAN complains otherwise */
	}

	template <class lreal_t_safe>
	real_t
	calc_kurtosis(size_t *ix_arr, size_t st, size_t end, int x[], int ncat,
				  size_t *buffer_cnt, real_t buffer_prob[],
				  MissingAction missing_action, CategSplit cat_split_type,
				  RNG_engine &rnd_generator)
	{
		/* This calculation proceeds as follows:
		 - If splitting by subsets, it will assign a random weight ~Unif(0,1) to
		 each category, and approximate kurtosis by sampling from such distribution
		 with the same probabilities as given by the current counts.
		 - If splitting by isolating one category, will binarize at each categorical level,
		 assume the values are zero or one, and output the average assuming each categorical
		 level has equal probability of being picked.
		 (Note that both are misleading heuristics, but might be better than random)
		 */
		size_t cnt = end - st + 1;
		std::fill(buffer_cnt, buffer_cnt + ncat + 1, (size_t)0);

		if (missing_action == Fail)
		{
			for (size_t row = st; row <= end; row++)
				buffer_cnt[x[ix_arr[row]]]++;
		}

		else
		{
			for (size_t row = st; row <= end; row++)
			{
				if (likely(x[ix_arr[row]] >= 0))
					buffer_cnt[x[ix_arr[row]]]++;
				else
					buffer_cnt[ncat]++;
			}
		}

		return calc_kurtosis_internal<lreal_t_safe>(cnt, x, ncat, buffer_cnt,
													buffer_prob, missing_action,
													cat_split_type,
													rnd_generator);
	}

	template <class lreal_t_safe = long real_t>
	real_t
	calc_kurtosis(size_t nrows, int x[], int ncat, size_t buffer_cnt[],
				  real_t buffer_prob[], MissingAction missing_action,
				  CategSplit cat_split_type, RNG_engine &rnd_generator)
	{
		size_t cnt = nrows;
		std::fill(buffer_cnt, buffer_cnt + ncat + 1, (size_t)0);

		if (missing_action == Fail)
		{
			for (size_t row = 0; row < nrows; row++)
				buffer_cnt[x[row]]++;
		}

		else
		{
			for (size_t row = 0; row < nrows; row++)
			{
				if (likely(x[row] >= 0))
					buffer_cnt[x[row]]++;
				else
					buffer_cnt[ncat]++;
			}
		}

		return calc_kurtosis_internal<lreal_t_safe>(cnt, x, ncat, buffer_cnt,
													buffer_prob, missing_action,
													cat_split_type,
													rnd_generator);
	}

	/* TODO: this one should get a buffer preallocated from outside */
	template <class mapping, class lreal_t_safe>
	real_t
	calc_kurtosis_weighted_internal(std::vector<lreal_t_safe> &buffer_cnt,
									int x[], int ncat, real_t buffer_prob[],
									MissingAction missing_action,
									CategSplit cat_split_type,
									RNG_engine &rnd_generator,
									mapping &w)
	{
		UNDEF_REFERENCE(x)
		UNDEF_REFERENCE2(missing_action)
		UNDEF_REFERENCE2(w)

		real_t sum_kurt = 0;

		lreal_t_safe cnt = std::accumulate(buffer_cnt.begin(),
										   buffer_cnt.end(), (lreal_t_safe)0);

		cnt -= buffer_cnt[ncat];
		if (unlikely(cnt <= 1))
			return -HUGE_VAL;
		for (int cat = 0; cat < ncat; cat++)
			buffer_prob[cat] = buffer_cnt[cat] / cnt;

		switch (cat_split_type)
		{
		case SubSet:
		{
			lreal_t_safe temp_v;
			lreal_t_safe s1, s2, s3, s4;
			lreal_t_safe coef, coef2;
			lreal_t_safe w_this;
			UniformUnitInterval runif(0, 1);
			size_t ntry = 50;
			for (size_t iternum = 0; iternum < 50; iternum++)
			{
				s1 = 0;
				s2 = 0;
				s3 = 0;
				s4 = 0;
				for (int cat = 0; cat < ncat; cat++)
				{
					coef = runif(rnd_generator);
					coef2 = coef * coef;
					w_this = buffer_prob[cat];
					s1 = std::fma(w_this, coef, s1);
					s2 = std::fma(w_this, coef2, s2);
					s3 = std::fma(w_this, coef2 * coef, s3);
					s4 = std::fma(w_this, coef2 * coef2, s4);
					// s1 += buffer_prob[cat] * pw1(coef);
					// s2 += buffer_prob[cat] * pw2(coef);
					// s3 += buffer_prob[cat] * pw3(coef);
					// s4 += buffer_prob[cat] * pw4(coef);
				}
				temp_v = s2 - pw2(s1);
				if (unlikely(temp_v <= 0))
					ntry--;
				else
					sum_kurt += (s4 - 4 * s3 * pw1(s1) + 6 * s2 * pw2(s1) - 4 * s1 * pw3(s1) + pw4(s1)) / pw2(temp_v);
			}
			if (unlikely(!ntry))
				return -HUGE_VAL;
			else if (unlikely(is_na_or_inf(sum_kurt)))
				return -HUGE_VAL;
			else
				return std::fmax(sum_kurt, 0.) / (real_t)ntry;
		}

		case SingleCateg:
		{
			real_t p;
			int ncat_present = ncat;
			for (int cat = 0; cat < ncat; cat++)
			{
				p = buffer_prob[cat];
				if (p == 0)
					ncat_present--;
				else
					sum_kurt += (p - 4 * p * pw1(p) + 6 * p * pw2(p) - 4 * p * pw3(p) + pw4(p)) / pw2(p - pw2(p));
			}
			if (ncat_present <= 1)
				return -HUGE_VAL;
			else if (unlikely(is_na_or_inf(sum_kurt)))
				return -HUGE_VAL;
			else
				return std::fmax(sum_kurt, 0.) / (real_t)ncat_present;
		}
		}


		return -1; /* this will never be reached, but CRAN complains otherwise */
	}

	template <class mapping, class lreal_t_safe>
	real_t
	calc_kurtosis_weighted(size_t ix_arr[], size_t st, size_t end, int x[],
						   int ncat, real_t buffer_prob[],
						   MissingAction missing_action,
						   CategSplit cat_split_type,
						   RNG_engine &rnd_generator,
						   mapping &w)
	{
		std::vector<lreal_t_safe> buffer_cnt(ncat + 1, 0.);
		lreal_t_safe w_this;

		for (size_t row = st; row <= end; row++)
		{
			w_this = w[ix_arr[row]];
			if (likely(x[ix_arr[row]] >= 0))
				buffer_cnt[x[ix_arr[row]]] += w_this;
			else
				buffer_cnt[ncat] += w_this;
		}

		return calc_kurtosis_weighted_internal<mapping, lreal_t_safe>(
			buffer_cnt, x, ncat, buffer_prob, missing_action, cat_split_type,
			rnd_generator, w);
	}

	template <class xreal = real_t, class lreal_t_safe>
	real_t
	calc_kurtosis_weighted(size_t nrows, int x[], int ncat,
						   real_t *buffer_prob, MissingAction missing_action,
						   CategSplit cat_split_type,
						   RNG_engine &rnd_generator,
						   xreal *w)
	{
		std::vector<lreal_t_safe> buffer_cnt(ncat + 1, 0.);
		lreal_t_safe w_this;

		for (size_t row = 0; row < nrows; row++)
		{
			w_this = w[row];
			if (likely(x[row] >= 0))
				buffer_cnt[x[row]] += w_this;
			else
				buffer_cnt[ncat] += w_this;
		}

		return calc_kurtosis_weighted_internal<real_t *, lreal_t_safe>(
			buffer_cnt, x, ncat, buffer_prob, missing_action, cat_split_type,
			rnd_generator, w);
	}

	/* Note: this isn't exactly comparable to the pooled gain from numeric variables,
	 but among all the possible options, this is what happens to end up in the most
	 similar scale when considering standardized gain. */

	/*  A couple notes about gain calculation:

	 Here one wants to find the best split point, maximizing either:
	 (1/sigma) * (sigma - (1/n)*(n_left*sigma_left + n_right*sigma_right))
	 or:
	 (1/sigma) * (sigma - (1/2)*(sigma_left + sigma_right))

	 All the algorithms here use the sorted-indices approach, which is
	 an exact method (note that there's still room for optimization by adding the
	 unsorted approach for small sample sizes and for sparse matrices).

	 A naive approach would move observations one at a time from right
	 to left using this formula:
	 sigma = (ssq - s^2/n) / n
	 ssq = sum(x^2)
	 s = sum(x)
	 But such approach has poor numerical precision, and this library is
	 aimed precisely at cases in which there are outliers in the data.
	 It's possible to improve the numerical precision by standardizing the
	 data beforehand, but this library uses instead a more exact two-pass
	 sigma calculation observation-by-observation (from left to right and
	 from right to left, keeping the calculations of the first pass in an
	 array and calculating gain in the second pass), but there's
	 other methods too.

	 If one is aiming at maximizing the pooled gain, it's possible to
	 simplify either the gain or the increase in gain without involving
	 'ssq'. Assuming one already has 'ssq' and 's' calculated for the left and
	 right partitions, and one wants to move one ovservation from right to left,
	 the following hold:
	 s_right = s - s_left
	 ssq_right = ssq - ssq_left
	 n_right = n - n_left
	 If one then moves observation x, these are updated as follows:
	 s_left_new = s_left + x
	 s_right_new = s - s_left - x
	 ssq_left_new = ssq_left + x^2
	 ssq_right_new = ssq - ssq_left - x^2
	 n_left_new = n_left + 1
	 n_right_new = n - n_left - 1
	 Gain is then:
	 (1/sigma) * (sigma - (1/n)*({ssq_left_new - s_left_new^2/n_left_new} + {ssq_right_new - s_right_new^2/n_right_new}))
	 Which simplifies to:
	 1 - (1/(sigma*n))(ssq - ( (s_left + x)^2/(n_left+1)  +  (s - (s_left + x))^2/(n - (n_left+1)) ))
	 Since 'sigma', n', and 'ssq' are constant, they can be ignored when determining the
	 maximum gain - that is, one is interest in finding the point that maximizes:
	 (s_left+x)^2/(n_left+1) + (s-(s_left+x))^2/(n-(n_left+1))
	 And this calculation will be robust-enough when dealing with numbers that were
	 already standardized beforehand, as the extended model does at each step.
	 Note however that, when fitting this model, one is usually interested in evaluating
	 the actual gain, standardized by the standard deviation, as it will try different
	 linear combinations which will give different standard deviations, so this simpler
	 formula cannot be applied unless only one linear combination is probed.

	 One can also look at:
	 diff_gain = (1/sigma) * (gain_new - gain)
	 Which can be simplified to something that doesn't include sums of squares:
	 (1/(sigma*n))*(  -s_left^2/n_left  -  (s-s_left)^2/(n-n_left)  +  (s_left+x)^2/(n_left+1)  +  (s-(s_left+x))^2/(n-(n_left+1))  )
	 And this calculation would in theory allow getting the actual standardized gain.
	 In practice however, this calculation can have poor numerical precision when the
	 sample size is large, so the functions here do not even attempt at calculating it,
	 and this is the reason why the two-pass approach is preferred.

	 The averaged SD formula unfortunately doesn't reduce to something that would involve
	 only sums.
	 */

	/*  TODO: maybe it's not a good idea to use the two-pass approach with un-standardized
	 variables at large sample sizes (ndim=1), considering that they come in sorted order.
	 Maybe it should instead use sums of centered squares: sigma = sqrt((x-mean(x))^2/n)
	 The sums of centered squares method is also likely to be more precise. */

	template <class xreal = real_t>
	real_t
	midpoint(xreal x, xreal y)
	{
		real_t m = x + (y - x) / (real_t)2;
		if (likely((real_t)m < (real_t)y))
			return m;
		else
		{
			m = std::nextafter(m, y);
			if (m > x && m < y)
				return m;
			else
				return x;
		}
	}

	template <class xreal = real_t>
	real_t
	midpoint_with_reorder(xreal x, xreal y)
	{
		if (x < y)
			return midpoint(x, y);
		else
			return midpoint(y, x);
	}

	template <class xreal, class yreal>
	real_t
	find_split_rel_gain_t(xreal *x, size_t n, real_t &split_point)
	{
		yreal this_gain;
		yreal best_gain = -HUGE_VAL;
		yreal x1 = 0, x2 = 0;
		yreal sum_left = 0, sum_right = 0, sum_tot = 0;
		for (size_t row = 0; row < n; row++)
			sum_tot += x[row];
		for (size_t row = 0; row < n - 1; row++)
		{
			sum_left += x[row];
			if (x[row] == x[row + 1])
				continue;

			sum_right = sum_tot - sum_left;
			this_gain = sum_left * (sum_left / (real_t)(row + 1)) + sum_right * (sum_right / (real_t)(n - row - 1));
			if (this_gain > best_gain)
			{
				best_gain = this_gain;
				x1 = x[row];
				x2 = x[row + 1];
			}
		}

		if (best_gain <= -HUGE_VAL)
			return best_gain;
		split_point = midpoint(x1, x2);
		return std::fmax((real_t)best_gain,
						 std::numeric_limits<real_t>::epsilon());
	}

	template <class xreal = real_t, class lreal_t_safe>
	real_t
	find_split_rel_gain(xreal *x, size_t n, real_t &split_point)
	{
		if (n < THRESHOLD_LONG_DOUBLE)
			return find_split_rel_gain_t<real_t, xreal>(x, n, split_point);
		else
			return find_split_rel_gain_t<lreal_t_safe, xreal>((lreal_t_safe *)x, n,
															  split_point);
	}

	/* Note: there is no 'weighted' version of 'find_split_rel_gain' with unindexed 'x', because calling it would
	 imply having to argsort the 'x' values in order to sort the weights, which is less efficient. */

	template <class xreal_t, class yreal_t>
	real_t
	find_split_rel_gain_t(xreal_t *x, yreal_t xmean, size_t *ix_arr, size_t st,
						  size_t end, real_t &split_point, size_t &split_ix)
	{
		xreal_t this_gain;
		xreal_t best_gain = -HUGE_VAL;
		split_ix = 0; /* <- avoid out-of-bounds at the end */
		xreal_t sum_left = 0, sum_right = 0, sum_tot = 0;
		for (size_t row = st; row <= end; row++)
			sum_tot += x[ix_arr[row]] - xmean;
		for (size_t row = st; row < end; row++)
		{
			sum_left += x[ix_arr[row]] - xmean;
			if (x[ix_arr[row]] == x[ix_arr[row + 1]])
				continue;

			sum_right = sum_tot - sum_left;
			this_gain = sum_left * (sum_left / (real_t)(row - st + 1)) + sum_right * (sum_right / (real_t)(end - row));
			if (this_gain > best_gain)
			{
				best_gain = this_gain;
				split_ix = row;
			}
		}

		if (best_gain <= -HUGE_VAL)
			return best_gain;
		split_point = midpoint(x[ix_arr[split_ix]], x[ix_arr[split_ix + 1]]);
		return std::fmax((real_t)best_gain,
						 std::numeric_limits<real_t>::epsilon());
	}

	template <class xreal_t, class lreal_t_safe>
	real_t
	find_split_rel_gain(xreal_t *x, xreal_t xmean, size_t *ix_arr, size_t st,
						size_t end, real_t &split_point, size_t &split_ix)
	{
		if ((end - st + 1) < THRESHOLD_LONG_DOUBLE)
			return find_split_rel_gain_t<real_t, xreal_t>(x, xmean, ix_arr, st,
														  end, split_point,
														  split_ix);
		else
			return find_split_rel_gain_t<lreal_t_safe, xreal_t>((lreal_t_safe *)x,
																xmean, ix_arr, st,
																end, split_point,
																split_ix);
	}

	/*	template <class xreal_t, class mapping, class lreal_t_safe>
	 real_t find_split_rel_gain_weighted(xreal_t * x,
	 xreal_t  xmean,
	 size_t *ix_arr,
	 size_t st,
	 size_t end,
	 real_t &split_point,
	 size_t &split_ix,
	 mapping &w)
	 {
	 if ((end-st+1) < THRESHOLD_LONG_DOUBLE)
	 return find_split_rel_gain_weighted_t<real_t, xreal_t, mapping>(x, xmean, ix_arr, st, end, split_point, split_ix, w);
	 else
	 return find_split_rel_gain_weighted_t<lreal_t_safe, xreal_t, mapping>(x, xmean, ix_arr, st, end, split_point, split_ix, w);
	 }

	 */
	template <class xreal, class yral>
	real_t
	calc_sd_right_to_left(xreal *x, size_t n, real_t *sd_arr)
	{
		xreal running_mean = 0;
		xreal running_ssq = 0;
		yral mean_prev = x[n - 1];
		for (size_t row = 0; row < n - 1; row++)
		{
			running_mean += (x[n - row - 1] - running_mean) / (real_t)(row + 1);
			running_ssq += (x[n - row - 1] - running_mean) * (x[n - row - 1] - mean_prev);
			mean_prev = running_mean;
			sd_arr[n - row - 1] =
				(row == 0) ? 0. : std::sqrt(running_ssq / (real_t)(row + 1));
		}
		running_mean += (x[0] - running_mean) / (real_t)n;
		running_ssq += (x[0] - running_mean) * (x[0] - mean_prev);
		return std::sqrt(running_ssq / (real_t)n);
	}

	template <class xreal, class lreal_t_safe>
	lreal_t_safe
	calc_sd_right_to_left_weighted(xreal *x, size_t n, real_t *sd_arr,
								   real_t *w, lreal_t_safe &cumw,
								   size_t *sorted_ix)
	{
		lreal_t_safe running_mean = 0;
		lreal_t_safe running_ssq = 0;
		lreal_t_safe mean_prev = x[sorted_ix[n - 1]];
		lreal_t_safe cnt = 0;
		real_t w_this;
		for (size_t row = 0; row < n - 1; row++)
		{
			w_this = w[sorted_ix[n - row - 1]];
			cnt += w_this;
			running_mean += w_this * (x[sorted_ix[n - row - 1]] - running_mean) / cnt;
			running_ssq += w_this * ((x[sorted_ix[n - row - 1]] - running_mean) * (x[sorted_ix[n - row - 1]] - mean_prev));
			mean_prev = running_mean;
			sd_arr[n - row - 1] = (row == 0) ? 0. : std::sqrt(running_ssq / cnt);
		}
		w_this = w[sorted_ix[0]];
		cnt += w_this;
		running_mean += (x[sorted_ix[0]] - running_mean) / cnt;
		running_ssq += w_this * ((x[sorted_ix[0]] - running_mean) * (x[sorted_ix[0]] - mean_prev));
		cumw = cnt;
		return std::sqrt(running_ssq / cnt);
	}

	template <class xreal, class yreal>
	xreal
	calc_sd_right_to_left(xreal *x, yreal xmean, size_t ix_arr[], size_t st,
						  size_t end, real_t *sd_arr)
	{
		real_t running_mean = 0;
		real_t running_ssq = 0;
		real_t mean_prev = x[ix_arr[end]] - xmean;
		size_t n = end - st + 1;
		for (size_t row = 0; row < n - 1; row++)
		{
			running_mean += ((x[ix_arr[end - row]] - xmean) - running_mean) / (real_t)(row + 1);
			running_ssq += ((x[ix_arr[end - row]] - xmean) - running_mean) * ((x[ix_arr[end - row]] - xmean) - mean_prev);
			mean_prev = running_mean;
			sd_arr[n - row - 1] =
				(row == 0) ? 0. : std::sqrt(running_ssq / (real_t)(row + 1));
		}
		running_mean += ((x[ix_arr[st]] - xmean) - running_mean) / (real_t)n;
		running_ssq += ((x[ix_arr[st]] - xmean) - running_mean) * ((x[ix_arr[st]] - xmean) - mean_prev);
		return std::sqrt(running_ssq / (xreal)n);
	}

	template <class xreal, class mapping, class lreal_t_safe>
	lreal_t_safe
	calc_sd_right_to_left_weighted(xreal *x, xreal xmean, size_t ix_arr[],
								   size_t st, size_t end, real_t *sd_arr,
								   mapping &w, lreal_t_safe &cumw)
	{
		lreal_t_safe running_mean = 0;
		lreal_t_safe running_ssq = 0;
		xreal mean_prev = x[ix_arr[end]] - xmean;
		size_t n = end - st + 1;
		lreal_t_safe cnt = 0;
		real_t w_this;
		for (size_t row = 0; row < n - 1; row++)
		{
			w_this = w[ix_arr[end - row]];
			cnt += w_this;
			running_mean += w_this * ((x[ix_arr[end - row]] - xmean) - running_mean) / cnt;
			running_ssq += w_this * (((x[ix_arr[end - row]] - xmean) - running_mean) * ((x[ix_arr[end - row]] - xmean) - mean_prev));
			mean_prev = running_mean;
			sd_arr[n - row - 1] = (row == 0) ? 0. : std::sqrt(running_ssq / cnt);
		}
		w_this = w[ix_arr[st]];
		cnt += w_this;
		running_mean += ((x[ix_arr[st]] - xmean) - running_mean) / cnt;
		running_ssq += w_this * (((x[ix_arr[st]] - xmean) - running_mean) * ((x[ix_arr[st]] - xmean) - mean_prev));
		cumw = cnt;
		return std::sqrt(running_ssq / cnt);
	}

	template <class xreal = real_t, class yreal = real_t>
	real_t
	find_split_std_gain_t(xreal *x, size_t n, real_t *sd_arr,
						  GainCriterion criterion, real_t min_gain,
						  real_t &split_point)
	{
		yreal full_sd = calc_sd_right_to_left((yreal *)x, n, sd_arr);
		yreal running_mean = 0;
		yreal running_ssq = 0;
		yreal mean_prev = x[0];
		yreal best_gain = -HUGE_VAL;
		yreal this_sd, this_gain;
		yreal n_ = (real_t)n;
		size_t best_ix = 0;
		for (size_t row = 0; row < n - 1; row++)
		{
			running_mean += (x[row] - running_mean) / (real_t)(row + 1);
			running_ssq += (x[row] - running_mean) * (x[row] - mean_prev);
			mean_prev = running_mean;
			if (x[row] == x[row + 1])
				continue;

			this_sd =
				(row == 0) ? 0. : std::sqrt(running_ssq / (real_t)(row + 1));
			this_gain =
				(criterion == Pooled) ? pooled_gain(full_sd, n_, this_sd, sd_arr[row + 1], row + 1,
													n - row - 1)
									  : sd_gain(full_sd, this_sd, sd_arr[row + 1]);
			if (this_gain > best_gain && this_gain > min_gain)
			{
				best_gain = this_gain;
				best_ix = row;
			}
		}

		if (best_gain > -HUGE_VAL)
			split_point = midpoint(x[best_ix], x[best_ix + 1]);

		return best_gain;
	}

	template <class real_t_, class lreal_t_safe>
	real_t
	find_split_std_gain(real_t_ *x, size_t n, real_t *sd_arr,
						GainCriterion criterion, real_t min_gain,
						real_t &split_point)
	{
		if (n < THRESHOLD_LONG_DOUBLE)
			return find_split_std_gain_t<real_t, real_t_>(x, n, sd_arr, criterion,
														  min_gain, split_point);
		else
			return find_split_std_gain_t<lreal_t_safe, real_t_>((lreal_t_safe *)x,
																n, sd_arr,
																criterion,
																min_gain,
																split_point);
	}

	template <class xreal, class lreal_t_safe>
	real_t
	find_split_std_gain_weighted(xreal *x, size_t n, real_t *sd_arr,
								 GainCriterion criterion, real_t min_gain,
								 real_t &split_point, real_t *w,
								 size_t *sorted_ix)
	{
		lreal_t_safe cumw;
		real_t full_sd = calc_sd_right_to_left_weighted(x, n, sd_arr, w, cumw,
														sorted_ix);
		lreal_t_safe running_mean = 0;
		lreal_t_safe running_ssq = 0;
		lreal_t_safe mean_prev = x[sorted_ix[0]];
		real_t best_gain = -HUGE_VAL;
		real_t this_sd, this_gain;
		real_t w_this;
		lreal_t_safe currw = 0;
		size_t best_ix = 0;

		for (size_t row = 0; row < n - 1; row++)
		{
			w_this = w[sorted_ix[row]];
			currw += w_this;
			running_mean += w_this * (x[sorted_ix[row]] - running_mean) / currw;
			running_ssq += w_this * ((x[sorted_ix[row]] - running_mean) * (x[sorted_ix[row]] - mean_prev));
			mean_prev = running_mean;
			if (x[sorted_ix[row]] == x[sorted_ix[row + 1]])
				continue;

			this_sd = (row == 0) ? 0. : std::sqrt(running_ssq / currw);
			this_gain =
				(criterion == Pooled) ? pooled_gain(full_sd, cumw, this_sd, sd_arr[row + 1], currw,
													cumw - currw)
									  : sd_gain(full_sd, this_sd, sd_arr[row + 1]);
			if (this_gain > best_gain && this_gain > min_gain)
			{
				best_gain = this_gain;
				best_ix = row;
			}
		}

		if (best_gain > -HUGE_VAL)
			split_point = midpoint(x[sorted_ix[best_ix]],
								   x[sorted_ix[best_ix + 1]]);

		return best_gain;
	}

	template <class xreal = real_t, class yreal = real_t>
	real_t
	find_split_std_gain_t(xreal *x, yreal xmean, size_t ix_arr[], size_t st,
						  size_t end, real_t *sd_arr, GainCriterion criterion,
						  real_t min_gain, real_t &split_point,
						  size_t &split_ix)
	{
		xreal full_sd = calc_sd_right_to_left(x, xmean, ix_arr, st, end, sd_arr);
		xreal running_mean = 0;
		xreal running_ssq = 0;
		xreal mean_prev = x[ix_arr[st]] - xmean;
		xreal best_gain = -HUGE_VAL;
		xreal n = (real_t)(end - st + 1);
		xreal this_sd, this_gain;
		split_ix = st;
		for (size_t row = st; row < end; row++)
		{
			running_mean += ((x[ix_arr[row]] - xmean) - running_mean) / (real_t)(row - st + 1);
			running_ssq += ((x[ix_arr[row]] - xmean) - running_mean) * ((x[ix_arr[row]] - xmean) - mean_prev);
			mean_prev = running_mean;
			if (x[ix_arr[row]] == x[ix_arr[row + 1]])
				continue;

			this_sd =
				(row == st) ? 0. : std::sqrt(running_ssq / (real_t)(row - st + 1));
			this_gain =
				(criterion == Pooled) ? pooled_gain(full_sd, n, this_sd, sd_arr[row - st + 1],
													row - st + 1, end - row)
									  : sd_gain(full_sd, this_sd, sd_arr[row - st + 1]);
			if (this_gain > best_gain && this_gain > min_gain)
			{
				best_gain = this_gain;
				split_ix = row;
			}
		}

		if (best_gain > -HUGE_VAL)
			split_point = midpoint(x[ix_arr[split_ix]], x[ix_arr[split_ix + 1]]);

		return best_gain;
	}

	template <class xreal, class lreal_t_safe>
	real_t
	find_split_std_gain(xreal *x, xreal xmean, size_t ix_arr[], size_t st,
						size_t end, real_t *sd_arr, GainCriterion criterion,
						real_t min_gain, real_t &split_point, size_t &split_ix)
	{
		if ((end - st + 1) < THRESHOLD_LONG_DOUBLE)
			return find_split_std_gain_t<real_t, xreal>(x, xmean, ix_arr, st, end,
														sd_arr, criterion,
														min_gain, split_point,
														split_ix);
		else
			return find_split_std_gain_t<lreal_t_safe, xreal>((lreal_t_safe *)x,
															  xmean, ix_arr, st,
															  end, sd_arr,
															  criterion, min_gain,
															  split_point,
															  split_ix);
	}

	template <class xreal, class mapping, class lreal_t_safe>
	real_t
	find_split_std_gain_weighted(xreal *x, real_t xmean, size_t ix_arr[],
								 size_t st, size_t end, real_t *sd_arr,
								 GainCriterion criterion, real_t min_gain,
								 real_t &split_point, size_t &split_ix,
								 mapping &w)
	{
		lreal_t_safe cumw;
		real_t full_sd = calc_sd_right_to_left_weighted(x, xmean, ix_arr, st,
														end, sd_arr, w, cumw);
		lreal_t_safe running_mean = 0;
		lreal_t_safe running_ssq = 0;
		lreal_t_safe mean_prev = x[ix_arr[st]] - xmean;
		real_t best_gain = -HUGE_VAL;
		lreal_t_safe currw = 0;
		real_t this_sd, this_gain;
		real_t w_this;
		split_ix = st;

		for (size_t row = st; row < end; row++)
		{
			w_this = w[ix_arr[row]];
			currw += w_this;
			running_mean += w_this * ((x[ix_arr[row]] - xmean) - running_mean) / currw;
			running_ssq += w_this * (((x[ix_arr[row]] - xmean) - running_mean) * ((x[ix_arr[row]] - xmean) - mean_prev));
			mean_prev = running_mean;
			if (x[ix_arr[row]] == x[ix_arr[row + 1]])
				continue;

			this_sd = (row == st) ? 0. : std::sqrt(running_ssq / currw);
			this_gain =
				(criterion == Pooled) ? pooled_gain(full_sd, cumw, this_sd, sd_arr[row - st + 1],
													currw, cumw - currw)
									  : sd_gain(full_sd, this_sd, sd_arr[row - st + 1]);
			if (this_gain > best_gain && this_gain > min_gain)
			{
				best_gain = this_gain;
				split_ix = row;
			}
		}

		if (best_gain > -HUGE_VAL)
			split_point = midpoint(x[ix_arr[split_ix]], x[ix_arr[split_ix + 1]]);

		return best_gain;
	}

#ifndef _FOR_R
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#endif
#endif

#ifndef _FOR_R
	[[gnu::optimize("Ofast")]]
#endif
	static inline void
	xpy1(real_t *x, real_t *y, size_t n)
	{
		for (size_t ix = 0; ix < n; ix++)
			y[ix] += x[ix];
	}

#ifndef _FOR_R
	[[gnu::optimize("Ofast")]]
#endif
	static inline void
	axpy1(const real_t a, real_t *x, real_t *y, size_t n)
	{
		for (size_t ix = 0; ix < n; ix++)
			y[ix] = std::fma(a, x[ix], y[ix]);
	}

#ifndef _FOR_R
	[[gnu::optimize("Ofast")]]
#endif
	static inline void
	xpy1(real_t *xval, size_t ind[], size_t nnz, real_t *y)
	{
		for (size_t ix = 0; ix < nnz; ix++)
			y[ind[ix]] += xval[ix];
	}

#ifndef _FOR_R
	[[gnu::optimize("Ofast")]]
#endif
	static inline void
	axpy1(const real_t a, real_t *xval, size_t ind[], size_t nnz, real_t *y)
	{
		for (size_t ix = 0; ix < nnz; ix++)
			y[ind[ix]] = std::fma(a, xval[ix], y[ind[ix]]);
	}

#ifndef _FOR_R
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif

	template <class xreal = real_t, class lreal_t_safe = long real_t>
	real_t
	find_split_full_gain(xreal *x, size_t st, size_t end, size_t *ix_arr,
						 size_t *cols_use, size_t ncols_use,
						 bool force_cols_use, real_t *X_row_major,
						 size_t ncols, real_t *Xr, size_t *Xr_ind,
						 size_t *Xr_indptr, real_t *buffer_sum_left,
						 real_t *buffer_sum_tot, size_t &split_ix,
						 real_t &split_point, bool x_uses_ix_arr)
	{
		if (end <= st)
			return -HUGE_VAL;
		if (cols_use != NULL && ncols_use && (real_t)ncols_use / (real_t)ncols < 0.1)
			force_cols_use = true;

		memset(buffer_sum_tot, 0,
			   (force_cols_use ? ncols_use : ncols) * sizeof(real_t));
		if (Xr_indptr == NULL)
		{
			if (force_cols_use)
			{
				real_t *ptr_row;
				for (size_t row = st; row <= end; row++)
				{
					ptr_row = X_row_major + ix_arr[row] * ncols;
					for (size_t col = 0; col < ncols_use; col++)
						buffer_sum_tot[col] += ptr_row[cols_use[col]];
				}
			}

			else
			{
				for (size_t row = st; row <= end; row++)
					xpy1(X_row_major + ix_arr[row] * ncols, buffer_sum_tot, ncols);
			}
		}

		else
		{
			if (force_cols_use)
			{
				size_t *curr_begin;
				size_t *row_end;
				size_t *curr_col;
				real_t *Xr_this;
				size_t *cols_end = cols_use + ncols_use;
				for (size_t row = st; row <= end; row++)
				{
					curr_begin = Xr_ind + Xr_indptr[ix_arr[row]];
					row_end = Xr_ind + Xr_indptr[ix_arr[row] + 1];
					if (curr_begin == row_end)
						continue;
					curr_col = cols_use;
					Xr_this = Xr + Xr_indptr[ix_arr[row]];

					while (curr_col < cols_end && curr_begin < row_end)
					{
						if (*curr_begin == *curr_col)
						{
							buffer_sum_tot[std::distance(cols_use, curr_col)] +=
								Xr_this[std::distance(curr_begin, row_end)];
							curr_col++;
							curr_begin++;
						}

						else
						{
							if (*curr_begin > *curr_col)
								curr_col = std::lower_bound(curr_col, cols_end,
															*curr_begin);
							else
								curr_begin = std::lower_bound(curr_begin, row_end,
															  *curr_col);
						}
					}
				}
			}

			else
			{
				size_t ptr_this;
				for (size_t row = st; row <= end; row++)
				{
					ptr_this = Xr_indptr[ix_arr[row]];
					xpy1(Xr + ptr_this, Xr_ind + ptr_this,
						 Xr_indptr[ix_arr[row] + 1] - ptr_this, buffer_sum_tot);
				}
			}
		}

		real_t best_gain = -HUGE_VAL;
		real_t this_gain;
		real_t sl, sr;
		real_t dl, dr;
		real_t vleft, vright;
		memset(buffer_sum_left, 0,
			   (force_cols_use ? ncols_use : ncols) * sizeof(real_t));
		if (Xr_indptr == NULL)
		{
			if (!force_cols_use)
			{
				for (size_t row = st; row < end; row++)
				{
					xpy1(X_row_major + ix_arr[row] * ncols, buffer_sum_left,
						 ncols);
					if (x_uses_ix_arr)
					{
						if (unlikely(x[ix_arr[row]] == x[ix_arr[row + 1]]))
							continue;
					}
					else
					{
						if (unlikely(x[row] == x[row + 1]))
							continue;
					}

					vleft = 0;
					vright = 0;
					dl = (real_t)(row - st + 1);
					dr = (real_t)(end - row);
					for (size_t col = 0; col < ncols; col++)
					{
						sl = buffer_sum_left[col];
						vleft += sl * (sl / dl);
						sr = buffer_sum_tot[col] - sl;
						vright += sr * (sr / dr);
					}

					this_gain = vleft + vright;
					if (this_gain > best_gain)
					{
						best_gain = this_gain;
						split_ix = row;
					}
				}
			}

			else
			{
				real_t *ptr_row;
				for (size_t row = st; row < end; row++)
				{
					ptr_row = X_row_major + ix_arr[row] * ncols;
					for (size_t col = 0; col < ncols_use; col++)
						buffer_sum_left[col] += ptr_row[cols_use[col]];
					if (x_uses_ix_arr)
					{
						if (unlikely(x[ix_arr[row]] == x[ix_arr[row + 1]]))
							continue;
					}
					else
					{
						if (unlikely(x[row] == x[row + 1]))
							continue;
					}

					vleft = 0;
					vright = 0;
					dl = (real_t)(row - st + 1);
					dr = (real_t)(end - row);
					for (size_t col = 0; col < ncols_use; col++)
					{
						sl = buffer_sum_left[col];
						vleft += sl * (sl / dl);
						sr = buffer_sum_tot[col] - sl;
						vright += sr * (sr / dr);
					}

					this_gain = vleft + vright;
					if (this_gain > best_gain)
					{
						best_gain = this_gain;
						split_ix = row;
					}
				}
			}
		}

		else
		{
			if (!force_cols_use)
			{
				size_t ptr_this;
				for (size_t row = st; row < end; row++)
				{
					ptr_this = Xr_indptr[ix_arr[row]];
					xpy1(Xr + ptr_this, Xr_ind + ptr_this,
						 Xr_indptr[ix_arr[row] + 1] - ptr_this, buffer_sum_left);
					if (x_uses_ix_arr)
					{
						if (unlikely(x[ix_arr[row]] == x[ix_arr[row + 1]]))
							continue;
					}
					else
					{
						if (unlikely(x[row] == x[row + 1]))
							continue;
					}

					vleft = 0;
					vright = 0;
					dl = (real_t)(row - st + 1);
					dr = (real_t)(end - row);
					for (size_t col = 0; col < ncols; col++)
					{
						sl = buffer_sum_left[col];
						vleft += sl * (sl / dl);
						sr = buffer_sum_tot[col] - sl;
						vright += sr * (sr / dr);
					}

					this_gain = vleft + vright;
					if (this_gain > best_gain)
					{
						best_gain = this_gain;
						split_ix = row;
					}
				}
			}

			else
			{
				size_t *curr_begin;
				size_t *row_end;
				size_t *curr_col;
				real_t *Xr_this;
				size_t *cols_end = cols_use + ncols_use;
				for (size_t row = st; row < end; row++)
				{
					curr_begin = Xr_ind + Xr_indptr[ix_arr[row]];
					row_end = Xr_ind + Xr_indptr[ix_arr[row] + 1];
					if (curr_begin == row_end)
						goto skip_sum;
					curr_col = cols_use;
					Xr_this = Xr + Xr_indptr[ix_arr[row]];
					while (curr_col < cols_end && curr_begin < row_end)
					{
						if (*curr_begin == *curr_col)
						{
							buffer_sum_left[std::distance(cols_use, curr_col)] +=
								Xr_this[std::distance(curr_begin, row_end)];
							curr_col++;
							curr_begin++;
						}

						else
						{
							if (*curr_begin > *curr_col)
								curr_col = std::lower_bound(curr_col, cols_end,
															*curr_begin);
							else
								curr_begin = std::lower_bound(curr_begin, row_end,
															  *curr_col);
						}
					}

				skip_sum:
					if (x_uses_ix_arr)
					{
						if (unlikely(x[ix_arr[row]] == x[ix_arr[row + 1]]))
							continue;
					}
					else
					{
						if (unlikely(x[row] == x[row + 1]))
							continue;
					}

					vleft = 0;
					vright = 0;
					dl = (real_t)(row - st + 1);
					dr = (real_t)(end - row);
					for (size_t col = 0; col < ncols_use; col++)
					{
						sl = buffer_sum_left[col];
						vleft += sl * (sl / dl);
						sr = buffer_sum_tot[col] - sl;
						vright += sr * (sr / dr);
					}

					this_gain = vleft + vright;
					if (this_gain > best_gain)
					{
						best_gain = this_gain;
						split_ix = row;
					}
				}
			}
		}

		if (best_gain <= -HUGE_VAL)
			return best_gain;

		if (x_uses_ix_arr)
			split_point = midpoint(x[ix_arr[split_ix]], x[ix_arr[split_ix + 1]]);
		else
			split_point = midpoint(x[split_ix], x[split_ix + 1]);
		return best_gain / (lreal_t_safe)(end - st + 1);
	}

	template <class xreal = real_t, class mapping, class lreal_t_safe>
	real_t
	find_split_full_gain_weighted(xreal *x, size_t st, size_t end,
								  size_t *ix_arr, size_t *cols_use,
								  size_t ncols_use, bool force_cols_use,
								  real_t *X_row_major, size_t ncols,
								  real_t *Xr, size_t *Xr_ind,
								  size_t *Xr_indptr, real_t *buffer_sum_left,
								  real_t *buffer_sum_tot, size_t &split_ix,
								  real_t &split_point, bool x_uses_ix_arr,
								  mapping &w)
	{
		if (end <= st)
			return -HUGE_VAL;
		if (cols_use != NULL && ncols_use && (real_t)ncols_use / (real_t)ncols < 0.1)
			force_cols_use = true;

		real_t wtot = 0;
		if (x_uses_ix_arr)
		{
			for (size_t row = st; row <= end; row++)
				wtot += w[ix_arr[row]];
		}

		else
		{
			for (size_t row = st; row <= end; row++)
				wtot += w[row];
		}

		memset(buffer_sum_tot, 0,
			   (force_cols_use ? ncols_use : ncols) * sizeof(real_t));
		if (Xr_indptr == NULL)
		{
			if (!force_cols_use)
			{
				if (x_uses_ix_arr)
				{
					for (size_t row = st; row <= end; row++)
						axpy1(w[ix_arr[row]], X_row_major + ix_arr[row] * ncols,
							  buffer_sum_tot, ncols);
				}

				else
				{
					for (size_t row = st; row <= end; row++)
						axpy1(w[row], X_row_major + ix_arr[row] * ncols,
							  buffer_sum_tot, ncols);
				}
			}

			else
			{
				real_t *ptr_row;
				real_t w_row;

				if (x_uses_ix_arr)
				{
					for (size_t row = st; row <= end; row++)
					{
						ptr_row = X_row_major + ix_arr[row] * ncols;
						w_row = w[ix_arr[row]];
						for (size_t col = 0; col < ncols_use; col++)
							buffer_sum_tot[col] = std::fma(w_row,
														   ptr_row[cols_use[col]],
														   buffer_sum_tot[col]);
					}
				}

				else
				{
					for (size_t row = st; row <= end; row++)
					{
						ptr_row = X_row_major + ix_arr[row] * ncols;
						w_row = w[row];
						for (size_t col = 0; col < ncols_use; col++)
							buffer_sum_tot[col] = std::fma(w_row,
														   ptr_row[cols_use[col]],
														   buffer_sum_tot[col]);
					}
				}
			}
		}

		else
		{
			if (!force_cols_use)
			{
				size_t ptr_this;
				if (x_uses_ix_arr)
				{
					for (size_t row = st; row <= end; row++)
					{
						ptr_this = Xr_indptr[ix_arr[row]];
						axpy1(w[ix_arr[row]], Xr + ptr_this, Xr_ind + ptr_this,
							  Xr_indptr[ix_arr[row] + 1] - ptr_this,
							  buffer_sum_tot);
					}
				}

				else
				{
					for (size_t row = st; row <= end; row++)
					{
						ptr_this = Xr_indptr[ix_arr[row]];
						axpy1(w[row], Xr + ptr_this, Xr_ind + ptr_this,
							  Xr_indptr[ix_arr[row] + 1] - ptr_this,
							  buffer_sum_tot);
					}
				}
			}

			else
			{
				size_t *curr_begin;
				size_t *row_end;
				size_t *curr_col;
				real_t *Xr_this;
				size_t *cols_end = cols_use + ncols_use;
				real_t w_row;
				for (size_t row = st; row <= end; row++)
				{
					curr_begin = Xr_ind + Xr_indptr[ix_arr[row]];
					row_end = Xr_ind + Xr_indptr[ix_arr[row] + 1];
					if (curr_begin == row_end)
						continue;
					curr_col = cols_use;
					Xr_this = Xr + Xr_indptr[ix_arr[row]];
					w_row = w[x_uses_ix_arr ? ix_arr[row] : row];
					size_t dtemp;

					while (curr_col < cols_end && curr_begin < row_end)
					{
						if (*curr_begin == *curr_col)
						{
							dtemp = std::distance(cols_use, curr_col);
							buffer_sum_tot[dtemp] = std::fma(
								w_row,
								Xr_this[std::distance(curr_begin, row_end)],
								buffer_sum_tot[dtemp]);
							curr_col++;
							curr_begin++;
						}

						else
						{
							if (*curr_begin > *curr_col)
								curr_col = std::lower_bound(curr_col, cols_end,
															*curr_begin);
							else
								curr_begin = std::lower_bound(curr_begin, row_end,
															  *curr_col);
						}
					}
				}
			}
		}

		real_t best_gain = -HUGE_VAL;
		real_t this_gain;
		real_t sl, sr;
		real_t vleft, vright;
		real_t wleft = 0;
		real_t w_row;
		real_t wright;
		memset(buffer_sum_left, 0,
			   (force_cols_use ? ncols_use : ncols) * sizeof(real_t));
		if (Xr_indptr == NULL)
		{
			if (!force_cols_use)
			{
				for (size_t row = st; row < end; row++)
				{
					w_row = w[x_uses_ix_arr ? ix_arr[row] : row];
					wleft += w_row;
					axpy1(w_row, X_row_major + ix_arr[row] * ncols,
						  buffer_sum_left, ncols);
					if (x_uses_ix_arr)
					{
						if (unlikely(x[ix_arr[row]] == x[ix_arr[row + 1]]))
							continue;
					}
					else
					{
						if (unlikely(x[row] == x[row + 1]))
							continue;
					}

					vleft = 0;
					vright = 0;
					wright = wtot - wleft;
					for (size_t col = 0; col < ncols; col++)
					{
						sl = buffer_sum_left[col];
						vleft += sl * (sl / wleft);
						sr = buffer_sum_tot[col] - sl;
						vright += sr * (sr / wright);
					}

					this_gain = vleft + vright;
					if (this_gain > best_gain)
					{
						best_gain = this_gain;
						split_ix = row;
					}
				}
			}

			else
			{
				real_t *ptr_row;
				real_t w_row;
				for (size_t row = st; row < end; row++)
				{
					w_row = w[x_uses_ix_arr ? ix_arr[row] : row];
					wleft += w_row;

					ptr_row = X_row_major + ix_arr[row] * ncols;
					for (size_t col = 0; col < ncols_use; col++)
						buffer_sum_left[col] = std::fma(w_row,
														ptr_row[cols_use[col]],
														buffer_sum_left[col]);
					if (x_uses_ix_arr)
					{
						if (unlikely(x[ix_arr[row]] == x[ix_arr[row + 1]]))
							continue;
					}
					else
					{
						if (unlikely(x[row] == x[row + 1]))
							continue;
					}

					vleft = 0;
					vright = 0;
					wright = wtot - wleft;
					for (size_t col = 0; col < ncols_use; col++)
					{
						sl = buffer_sum_left[col];
						vleft += sl * (sl / wleft);
						sr = buffer_sum_tot[col] - sl;
						vright += sr * (sr / wright);
					}

					this_gain = vleft + vright;
					if (this_gain > best_gain)
					{
						best_gain = this_gain;
						split_ix = row;
					}
				}
			}
		}

		else
		{
			if (!force_cols_use)
			{
				size_t ptr_this;
				real_t w_row;
				for (size_t row = st; row < end; row++)
				{
					w_row = w[x_uses_ix_arr ? ix_arr[row] : row];
					wleft += w_row;
					ptr_this = Xr_indptr[ix_arr[row]];
					axpy1(w_row, Xr + ptr_this, Xr_ind + ptr_this,
						  Xr_indptr[ix_arr[row] + 1] - ptr_this,
						  buffer_sum_left);
					if (x_uses_ix_arr)
					{
						if (unlikely(x[ix_arr[row]] == x[ix_arr[row + 1]]))
							continue;
					}
					else
					{
						if (unlikely(x[row] == x[row + 1]))
							continue;
					}

					vleft = 0;
					vright = 0;
					wright = wtot - wleft;
					for (size_t col = 0; col < ncols; col++)
					{
						sl = buffer_sum_left[col];
						vleft += sl * (sl / wleft);
						sr = buffer_sum_tot[col] - sl;
						vright += sr * (sr / wright);
					}

					this_gain = vleft + vright;
					if (this_gain > best_gain)
					{
						best_gain = this_gain;
						split_ix = row;
					}
				}
			}

			else
			{
				size_t *curr_begin;
				size_t *row_end;
				size_t *curr_col;
				real_t *Xr_this;
				size_t *cols_end = cols_use + ncols_use;
				real_t w_row;
				size_t dtemp;
				for (size_t row = st; row < end; row++)
				{
					w_row = w[x_uses_ix_arr ? ix_arr[row] : row];
					wleft += w_row;

					curr_begin = Xr_ind + Xr_indptr[ix_arr[row]];
					row_end = Xr_ind + Xr_indptr[ix_arr[row] + 1];
					if (curr_begin == row_end)
						goto skip_sum;
					curr_col = cols_use;
					Xr_this = Xr + Xr_indptr[ix_arr[row]];
					while (curr_col < cols_end && curr_begin < row_end)
					{
						if (*curr_begin == *curr_col)
						{
							dtemp = std::distance(cols_use, curr_col);
							buffer_sum_left[dtemp] = std::fma(
								w_row,
								Xr_this[std::distance(curr_begin, row_end)],
								buffer_sum_left[dtemp]);
							curr_col++;
							curr_begin++;
						}

						else
						{
							if (*curr_begin > *curr_col)
								curr_col = std::lower_bound(curr_col, cols_end,
															*curr_begin);
							else
								curr_begin = std::lower_bound(curr_begin, row_end,
															  *curr_col);
						}
					}

				skip_sum:
					if (x_uses_ix_arr)
					{
						if (unlikely(x[ix_arr[row]] == x[ix_arr[row + 1]]))
							continue;
					}
					else
					{
						if (unlikely(x[row] == x[row + 1]))
							continue;
					}

					vleft = 0;
					vright = 0;
					wright = wtot - wleft;
					for (size_t col = 0; col < ncols_use; col++)
					{
						sl = buffer_sum_left[col];
						vleft += sl * (sl / wleft);
						sr = buffer_sum_tot[col] - sl;
						vright += sr * (sr / wright);
					}

					this_gain = vleft + vright;
					if (this_gain > best_gain)
					{
						best_gain = this_gain;
						split_ix = row;
					}
				}
			}
		}

		if (best_gain <= -HUGE_VAL)
			return best_gain;

		split_point = midpoint(x[ix_arr[split_ix]], x[ix_arr[split_ix + 1]]);
		return best_gain / wtot;
	}

	template <class xreal = real_t, class yreal = real_t>
	real_t
	find_split_dens_shortform_t(xreal *x, size_t n, real_t &split_point)
	{
		real_t best_gain = -HUGE_VAL;
		size_t n_minus_one = n - 1;
		yreal xmin = x[0];
		yreal xmax = x[n - 1];
		yreal xleft, xright;
		yreal xmid;
		real_t this_gain;
		size_t split_ix = 0;

		for (size_t ix = 0; ix < n_minus_one; ix++)
		{
			if (x[ix] == x[ix + 1])
				continue;
			xmid = (xreal)x[ix] + ((xreal)x[ix + 1] - (xreal)x[ix]) / (xreal)2;
			xleft = xmid - xmin;
			xright = xmax - xmid;
			if (unlikely(!xleft || !xright))
				continue;
			this_gain = (xreal)square(ix + 1) / xleft + (yreal)square(n_minus_one - ix) / xright;
			if (this_gain > best_gain)
			{
				best_gain = this_gain;
				split_ix = ix;
			}
		}

		if (best_gain <= -HUGE_VAL)
			return best_gain;

		yreal xtot = (yreal)xmax - (yreal)xmin;
		yreal nleft = (yreal)(split_ix + 1);
		yreal nright = (yreal)(n_minus_one - split_ix);
		split_point = midpoint(x[split_ix], x[split_ix + 1]);
		yreal rpct_left = split_point / xtot;
		rpct_left = std::fmax(rpct_left, std::numeric_limits<real_t>::min());
		yreal rpct_right = (yreal)1 - rpct_left;
		rpct_right = std::fmax(rpct_right, std::numeric_limits<real_t>::min());

		yreal nl_sq = nleft / (yreal)n;
		nl_sq = square(nl_sq);
		yreal nr_sq = nright / (yreal)n;
		nl_sq = square(nr_sq);

		return nl_sq / rpct_left + nr_sq / rpct_right;
	}

	template <class xreal = real_t, class lreal_t_safe>
	real_t
	find_split_dens_shortform(xreal *x, size_t n, real_t &split_point)
	{
		if (n < INT32_MAX)
			return find_split_dens_shortform_t<real_t, xreal>(x, n, split_point);
		else
			return find_split_dens_shortform_t<lreal_t_safe, xreal>(
				(lreal_t_safe *)x, n, split_point);
	}

	template <class xreal, class yreal, class mapping>
	real_t
	find_split_dens_shortform_weighted_t(xreal *x, size_t n,
										 real_t &split_point, mapping &w,
										 size_t *buffer_indices)
	{
		real_t best_gain = -HUGE_VAL;
		size_t n_minus_one = n - 1;
		yreal xmin = x[buffer_indices[0]];
		yreal xmax = x[buffer_indices[n - 1]];
		yreal xleft, xright;
		yreal xmid;
		real_t this_gain;

		xreal wtot = 0;
		for (size_t ix = 0; ix < n; ix++)
			wtot += w[buffer_indices[ix]];
		xreal w_left = 0;
		xreal w_right;
		xreal best_w = 0;
		size_t split_ix = 0;

		for (size_t ix = 0; ix < n_minus_one; ix++)
		{
			w_left += w[buffer_indices[ix]];
			if (x[buffer_indices[ix]] == x[buffer_indices[ix + 1]])
				continue;
			xmid = (xreal)x[buffer_indices[ix]] + ((yreal)x[buffer_indices[ix + 1]] - (yreal)x[buffer_indices[ix]]) / (yreal)2;
			xleft = xmid - xmin;
			xright = xmax - xmid;
			if (unlikely(!xleft || !xright))
				continue;

			w_right = wtot - w_left;
			this_gain = square(w_left) / xleft + square(w_right) / xright;
			if (this_gain > best_gain)
			{
				best_gain = this_gain;
				best_w = w_left;
				split_ix = xmid;
			}
		}

		if (best_gain <= -HUGE_VAL)
			return best_gain;

		xreal xtot = xmax - xmin;
		w_left = best_w;
		w_right = wtot - w_left;
		w_left = std::fmax(w_left, std::numeric_limits<real_t>::min());
		w_right = std::fmax(w_right, std::numeric_limits<real_t>::min());
		split_point = midpoint(x[buffer_indices[split_ix]],
							   x[buffer_indices[split_ix + 1]]);
		yreal rpct_left = split_point / xtot;
		rpct_left = std::fmax(rpct_left, std::numeric_limits<real_t>::min());
		yreal rpct_right = (yreal)1 - rpct_left;
		rpct_right = std::fmax(rpct_right, std::numeric_limits<real_t>::min());

		yreal wl_sq = w_left / wtot;
		wl_sq = square(wl_sq);
		yreal wr_sq = w_right / wtot;
		wl_sq = square(wr_sq);

		return wl_sq / rpct_left + wr_sq / rpct_right;
	}

	template <class xreal, class mapping, class lreal_t_safe = long real_t>
	real_t
	find_split_dens_shortform_weighted(xreal *x, size_t n, real_t &split_point,
									   mapping &w, size_t *buffer_indices)
	{
		if (n < INT32_MAX)
			return find_split_dens_shortform_weighted_t<real_t, xreal, mapping>(
				x, n, split_point, w, buffer_indices);
		else
			return find_split_dens_shortform_weighted_t<lreal_t_safe, xreal, mapping>(
				(lreal_t_safe *)x, n, split_point, w, buffer_indices);
	}

	template <class xreal = real_t>
	real_t
	find_split_dens_shortform(xreal *x, size_t *ix_arr, size_t st, size_t end,
							  real_t &split_point, size_t &split_ix)
	{
		real_t best_gain = -HUGE_VAL;
		xreal xmin = x[ix_arr[st]];
		xreal xmax = x[ix_arr[end]];
		xreal xleft, xright;
		xreal xmid;
		real_t this_gain;

		for (size_t row = st; row < end; row++)
		{
			if (x[ix_arr[row]] == x[ix_arr[row + 1]])
				continue;
			xmid = x[ix_arr[row]] + (x[ix_arr[row + 1]] - x[ix_arr[row]]) / (real_t)2;
			xleft = xmid - xmin;
			xright = xmax - xmid;
			if (unlikely(!xleft || !xright))
				continue;
			this_gain = square(row - st + 1) / xleft + square(end - row) / xright;
			if (this_gain > best_gain)
			{
				best_gain = this_gain;
				split_ix = row;
			}
		}

		if (best_gain <= -HUGE_VAL)
			return best_gain;

		real_t xtot = (real_t)xmax - (real_t)xmin;
		real_t nleft = (real_t)(split_ix - st + 1);
		real_t nright = (real_t)(end - split_ix);
		split_point = midpoint(x[ix_arr[split_ix]], x[ix_arr[split_ix + 1]]);
		real_t rpct_left = split_point / xtot;
		rpct_left = std::fmax(rpct_left, std::numeric_limits<real_t>::min());
		real_t rpct_right = 1. - rpct_left;
		rpct_right = std::fmax(rpct_right, std::numeric_limits<real_t>::min());
		real_t ntot = (real_t)(end - st + 1);

		real_t nl_sq = nleft / ntot;
		nl_sq = square(nl_sq);
		real_t nr_sq = nright / ntot;
		nl_sq = square(nr_sq);

		return nl_sq / rpct_left + nr_sq / rpct_right;
	}

	template <class xreal, class mapping>
	real_t
	find_split_dens_shortform_weighted(xreal *x, size_t *ix_arr, size_t st,
									   size_t end, real_t &split_point,
									   size_t &split_ix, mapping &w)
	{
		real_t best_gain = -HUGE_VAL;
		xreal xmin = x[ix_arr[st]];
		xreal xmax = x[ix_arr[end]];
		xreal xleft, xright;
		xreal xmid;
		real_t this_gain;

		real_t wtot = 0;
		for (size_t row = st; row <= end; row++)
			wtot += w[ix_arr[row]];
		real_t w_left = 0;
		real_t w_right;
		real_t best_w = 0;

		for (size_t row = st; row < end; row++)
		{
			w_left += w[ix_arr[row]];
			if (x[ix_arr[row]] == x[ix_arr[row + 1]])
				continue;
			xmid = x[ix_arr[row]] + (x[ix_arr[row + 1]] - x[ix_arr[row]]) / (real_t)2;
			xleft = xmid - xmin;
			xright = xmax - xmid;
			if (unlikely(!xleft || !xright))
				continue;

			w_right = wtot - w_left;
			this_gain = square(w_left) / xleft + square(w_right) / xright;
			if (this_gain > best_gain)
			{
				best_gain = this_gain;
				best_w = w_left;
				split_ix = row;
			}
		}

		if (best_gain <= -HUGE_VAL)
			return best_gain;

		real_t xtot = (real_t)xmax - (real_t)xmin;
		w_left = best_w;
		w_right = wtot - w_left;
		w_left = std::fmax(w_left, std::numeric_limits<real_t>::min());
		w_right = std::fmax(w_right, std::numeric_limits<real_t>::min());
		split_point = midpoint(x[split_ix], x[split_ix + 1]);
		real_t rpct_left = split_point / xtot;
		rpct_left = std::fmax(rpct_left, std::numeric_limits<real_t>::min());
		real_t rpct_right = 1. - rpct_left;
		rpct_right = std::fmax(rpct_right, std::numeric_limits<real_t>::min());

		real_t wl_sq = w_left / wtot;
		wl_sq = square(wl_sq);
		real_t wr_sq = w_right / wtot;
		wl_sq = square(wr_sq);

		return wl_sq / rpct_left + wr_sq / rpct_right;
	}

	/* This is a slower but more numerically-robust form */
	template <class xreal, class lreal_t_safe>
	real_t
	find_split_dens_longform(xreal *x, size_t *ix_arr, size_t st, size_t end,
							 real_t &split_point, size_t &split_ix)
	{
		real_t best_gain = -HUGE_VAL;
		xreal xmin = x[ix_arr[st]];
		xreal xmax = x[ix_arr[end]];
		xreal xleft, xright;
		xreal xmid;
		lreal_t_safe pct_left, pct_right;
		lreal_t_safe rpct_left, rpct_right;
		lreal_t_safe n_tot = end - st + 1;
		lreal_t_safe xtot = (lreal_t_safe)xmax - (lreal_t_safe)xmin;
		lreal_t_safe cnt_left;
		real_t this_gain;

		for (size_t row = st; row < end; row++)
		{
			if (x[ix_arr[row]] == x[ix_arr[row + 1]])
				continue;
			xmid = midpoint(x[ix_arr[row]], x[ix_arr[row + 1]]);
			xleft = xmid - xmin;
			xright = xmax - xmid;
			if (unlikely(!xleft || !xright))
				continue;

			cnt_left = (lreal_t_safe)(row - st + 1);

			xleft = std::fmax(xleft,
							  (real_t)std::numeric_limits<real_t>::min());
			xright = std::fmax(xright,
							   (real_t)std::numeric_limits<real_t>::min());
			pct_left = cnt_left / n_tot;
			pct_right = (lreal_t_safe)1 - pct_left;
			rpct_left = (lreal_t_safe)xleft / xtot;
			rpct_right = (lreal_t_safe)xright / xtot;

			this_gain = square(pct_left) / rpct_left + square(pct_right) / rpct_right;
			if (unlikely(is_na_or_inf(this_gain)))
				continue;
			if (this_gain > best_gain)
			{
				best_gain = this_gain;
				split_point = xmid;
				split_ix = row;
			}
		}

		return best_gain;
	}

	template <class xreal, class mapping, class lreal_t_safe>
	real_t
	find_split_dens_longform_weighted(xreal *x, size_t *ix_arr, size_t st,
									  size_t end, real_t &split_point,
									  size_t &split_ix, mapping &w)
	{
		real_t best_gain = -HUGE_VAL;
		xreal xmin = x[ix_arr[st]];
		xreal xmax = x[ix_arr[end]];
		xreal xleft, xright;
		xreal xmid;
		lreal_t_safe pct_left, pct_right;
		lreal_t_safe rpct_left, rpct_right;
		lreal_t_safe xtot = (lreal_t_safe)xmax - (lreal_t_safe)xmin;
		real_t this_gain;

		lreal_t_safe wtot = 0;
		for (size_t row = st; row <= end; row++)
			wtot += w[ix_arr[row]];
		lreal_t_safe w_left = 0;

		for (size_t row = st; row < end; row++)
		{
			w_left += w[ix_arr[row]];
			if (x[ix_arr[row]] == x[ix_arr[row + 1]])
				continue;
			xmid = midpoint(x[ix_arr[row]], x[ix_arr[row + 1]]);
			xleft = xmid - xmin;
			xright = xmax - xmid;
			if (unlikely(!xleft || !xright))
				continue;

			xleft = std::fmax(xleft,
							  (real_t)std::numeric_limits<real_t>::min());
			xright = std::fmax(xright,
							   (real_t)std::numeric_limits<real_t>::min());
			pct_left = w_left / wtot;
			pct_right = (lreal_t_safe)1 - pct_left;
			rpct_left = (lreal_t_safe)xleft / xtot;
			rpct_right = (lreal_t_safe)xright / xtot;

			this_gain = square(pct_left) / rpct_left + square(pct_right) / rpct_right;
			if (unlikely(is_na_or_inf(this_gain)))
				continue;
			if (this_gain > best_gain)
			{
				best_gain = this_gain;
				split_point = xmid;
				split_ix = row;
			}
		}

		return best_gain;
	}

	template <class xreal, class lreal_t_safe>
	real_t
	find_split_dens(xreal *x, size_t *ix_arr, size_t st, size_t end,
					real_t &split_point, size_t &split_ix)
	{
		if (end - st + 1 < INT32_MAX && x[ix_arr[end]] - x[ix_arr[st]] >= 1)
			return find_split_dens_shortform<real_t>(x, ix_arr, st, end,
													 split_point, split_ix);
		else
			return find_split_dens_longform<real_t, lreal_t_safe>(x, ix_arr, st,
																  end, split_point,
																  split_ix);
	}

	template <class xreal, class mapping, class lreal_t_safe>
	real_t
	find_split_dens_weighted(xreal *x, size_t *ix_arr, size_t st, size_t end,
							 real_t &split_point, size_t &split_ix, mapping &w)
	{
		if (end - st + 1 < INT32_MAX && x[ix_arr[end]] - x[ix_arr[st]] >= 1)
			return find_split_dens_shortform_weighted<xreal, mapping>(x, ix_arr,
																	  st, end,
																	  split_point,
																	  split_ix, w);
		else
			return find_split_dens_longform_weighted<xreal, mapping, lreal_t_safe>(
				x, ix_arr, st, end, split_point, split_ix, w);
	}

	template <class int_t, class lreal_t_safe>
	real_t
	find_split_dens_longform(int *x, int ncat, size_t *ix_arr, size_t st,
							 size_t end, CategSplit cat_split_type,
							 MissingAction missing_action, int &chosen_cat,
							 signed char *split_categ, int *saved_cat_mode,
							 size_t *buffer_cnt, int_t *buffer_indices)
	{
		if (st >= end || ncat <= 1)
			return -HUGE_VAL;
		size_t n_nas = 0;
		int xval;

		/* count categories */
		memset(buffer_cnt, 0, sizeof(size_t) * ncat);
		if (missing_action == Fail)
		{
			for (size_t row = st; row <= end; row++)
				if (likely(x[ix_arr[row]] >= 0))
					buffer_cnt[x[ix_arr[row]]]++;
		}

		else if (missing_action == Impute)
		{
			for (size_t row = st; row <= end; row++)
			{
				xval = x[ix_arr[row]];
				if (unlikely(xval < 0))
					n_nas++;
				else
					buffer_cnt[xval]++;
			}

			if (unlikely(n_nas >= end - st))
				return -HUGE_VAL;

			if (n_nas)
			{
				auto idxmax = std::max_element(buffer_cnt, buffer_cnt + ncat);
				*idxmax += n_nas;
				*saved_cat_mode = (int)std::distance(buffer_cnt, idxmax);
			}
		}

		else
		{
			for (size_t row = st; row <= end; row++)
			{
				xval = x[ix_arr[row]];
				if (likely(xval >= 0))
					buffer_cnt[xval]++;
			}
		}

		std::iota(buffer_indices, buffer_indices + ncat, (int_t)0);
		std::sort(buffer_indices, buffer_indices + ncat, [&buffer_cnt](const int_t a, const int_t b)
				  { return buffer_cnt[a] < buffer_cnt[b]; });

		int curr = 0;
		if (split_categ != NULL)
		{
			while (buffer_cnt[buffer_indices[curr]] == 0)
			{
				split_categ[buffer_indices[curr]] = -1;
				curr++;
			}
		}

		else
		{
			while (buffer_cnt[buffer_indices[curr]] == 0)
				curr++;
		}

		int ncat_present = ncat - curr;
		if (ncat_present <= 1)
			return -HUGE_VAL;
		if (ncat_present == 2)
		{
			switch (cat_split_type)
			{
			case SingleCateg:
			{
				chosen_cat = buffer_indices[curr];
				break;
			}

			case SubSet:
			{
				split_categ[buffer_indices[curr]] = 1;
				split_categ[buffer_indices[curr + 1]] = 0;
				break;
			}
			}

			lreal_t_safe pct_left =
				(lreal_t_safe)buffer_cnt[buffer_indices[curr]] / (lreal_t_safe)(buffer_cnt[buffer_indices[curr]] + buffer_cnt[buffer_indices[curr + 1]]);

			return ((lreal_t_safe)buffer_cnt[buffer_indices[curr]] * (2. * pct_left) + (lreal_t_safe)buffer_cnt[buffer_indices[curr + 1]] * (2. - 2. * pct_left)) / (lreal_t_safe)(buffer_cnt[buffer_indices[curr]] + buffer_cnt[buffer_indices[curr + 1]]);
		}

		size_t ntot;
		if (missing_action == Impute)
			ntot = end - st + 1;
		else
			ntot = std::accumulate(buffer_cnt, buffer_cnt + ncat, (size_t)0);
		if (unlikely(ntot <= 1))
			unexpected_error();
		lreal_t_safe ntot_ = (lreal_t_safe)ntot;

		switch (cat_split_type)
		{
		case SingleCateg:
		{
			real_t pct_one_cat = 1. / (real_t)ncat_present;
			real_t pct_left_smallest =
				(lreal_t_safe)buffer_cnt[buffer_indices[curr]] / ntot_;
			real_t gain_smallest =
				(lreal_t_safe)buffer_cnt[buffer_indices[curr]] * (pct_left_smallest / pct_one_cat) + (lreal_t_safe)(ntot - buffer_cnt[buffer_indices[curr]]) * ((1. - pct_left_smallest) / (1. - pct_one_cat));

			real_t pct_left_biggest =
				(lreal_t_safe)buffer_cnt[buffer_indices[ncat - 1]] / ntot_;
			real_t gain_biggest = (lreal_t_safe)buffer_cnt[buffer_indices[ncat - 1]] * (pct_left_biggest / pct_one_cat) + (lreal_t_safe)(ntot - buffer_cnt[buffer_indices[ncat - 1]]) * ((1. - pct_left_biggest) / (1. - pct_one_cat));

			if (gain_smallest >= gain_biggest)
			{
				chosen_cat = buffer_indices[curr];
				return gain_smallest / ntot_;
			}

			else
			{
				chosen_cat = buffer_indices[ncat - 1];
				return gain_biggest / ntot_;
			}
			break;
		}

		case SubSet:
		{
			size_t cnt_left = 0;
			size_t cnt_right;
			int st_cat = curr - 1;
			real_t this_gain;
			real_t best_gain = -HUGE_VAL;
			int best_cat = 0;
			lreal_t_safe pct_left;
			real_t pct_cat_left;
			real_t ncat_present_ = (real_t)ncat_present;
			for (; curr < ncat; curr++)
			{
				cnt_left += buffer_cnt[buffer_indices[curr]];
				cnt_right = ntot - cnt_left;
				pct_left = (lreal_t_safe)cnt_left / ntot_;
				pct_cat_left = (real_t)(curr - st_cat) / ncat_present_;
				this_gain = (lreal_t_safe)cnt_left * (pct_left / pct_cat_left) + (lreal_t_safe)cnt_right * (((lreal_t_safe)1 - pct_left) / (1. - pct_cat_left));
				if (this_gain > best_gain)
				{
					best_gain = this_gain;
					best_cat = curr;
				}
			}

			if (best_gain <= -HUGE_VAL)
				return best_gain;
			st_cat++;
			for (; st_cat <= best_cat; st_cat++)
				split_categ[buffer_indices[st_cat]] = 1;
			for (; st_cat < ncat; st_cat++)
				split_categ[buffer_indices[st_cat]] = 0;
			return best_gain / ntot_;
			break;
		}
		}

		/* This will not be reached, but CRAN might complain otherwise */
		return -HUGE_VAL;
	}

	template <class mapping, class int_t, class lreal_t_safe>
	real_t
	find_split_dens_longform_weighted(int *x, int ncat, size_t *ix_arr,
									  size_t st, size_t end,
									  CategSplit cat_split_type,
									  MissingAction missing_action,
									  int &chosen_cat,
									  signed char *split_categ,
									  int *saved_cat_mode,
									  int_t *buffer_indices, mapping &w)
	{
		if (st >= end || ncat <= 1)
			return -HUGE_VAL;
		lreal_t_safe w_missing = 0;
		int xval;
		size_t ix_;

		/* count categories */
		/* TODO: allocate this buffer externally */
		std::vector<lreal_t_safe> buffer_cnt(ncat, (lreal_t_safe)0);
		if (missing_action == Fail)
		{
			for (size_t row = st; row <= end; row++)
			{
				ix_ = ix_arr[row];
				if (unlikely(x[ix_]) < 0)
					continue;
				buffer_cnt[x[ix_]] += w[ix_];
			}
		}

		else if (missing_action == Impute)
		{
			for (size_t row = st; row <= end; row++)
			{
				ix_ = ix_arr[row];
				xval = x[ix_];
				if (unlikely(xval < 0))
					w_missing += w[ix_];
				else
					buffer_cnt[xval] += w[ix_];
			}

			if (w_missing)
			{
				auto idxmax = std::max_element(buffer_cnt.begin(),
											   buffer_cnt.end());
				*idxmax += w_missing;
				*saved_cat_mode = (int)std::distance(buffer_cnt.begin(),
													 idxmax);
			}
		}

		else
		{
			for (size_t row = st; row <= end; row++)
			{
				ix_ = ix_arr[row];
				xval = x[ix_];
				if (likely(xval >= 0))
					buffer_cnt[xval] += w[ix_];
			}
		}

		std::iota(buffer_indices, buffer_indices + ncat, (int_t)0);
		std::sort(buffer_indices, buffer_indices + ncat, [&buffer_cnt](const int_t a, const int_t b)
				  { return buffer_cnt[a] < buffer_cnt[b]; });

		int curr = 0;
		if (split_categ != NULL)
		{
			while (buffer_cnt[buffer_indices[curr]] == 0)
			{
				split_categ[buffer_indices[curr]] = -1;
				curr++;
			}
		}

		else
		{
			while (buffer_cnt[buffer_indices[curr]] == 0)
				curr++;
		}

		int ncat_present = ncat - curr;
		if (ncat_present <= 1)
			return -HUGE_VAL;
		if (ncat_present == 2)
		{
			switch (cat_split_type)
			{
			case SingleCateg:
			{
				chosen_cat = buffer_indices[curr];
				break;
			}

			case SubSet:
			{
				split_categ[buffer_indices[curr]] = 1;
				split_categ[buffer_indices[curr + 1]] = 0;
				break;
			}
			}

			lreal_t_safe pct_left = buffer_cnt[buffer_indices[curr]] / (buffer_cnt[buffer_indices[curr]] + buffer_cnt[buffer_indices[curr + 1]]);

			return (buffer_cnt[buffer_indices[curr]] * (pct_left * 2.) + buffer_cnt[buffer_indices[curr + 1]] * (2. - 2. * pct_left)) / (buffer_cnt[buffer_indices[curr]] + buffer_cnt[buffer_indices[curr + 1]]);
		}

		lreal_t_safe ntot = std::accumulate(buffer_cnt.begin(),
											buffer_cnt.end(), (lreal_t_safe)0);
		if (unlikely(ntot <= 0))
			unexpected_error();

		switch (cat_split_type)
		{
		case SingleCateg:
		{
			real_t pct_one_cat = 1. / (real_t)ncat_present;
			real_t pct_left_smallest = buffer_cnt[buffer_indices[curr]] / ntot;
			real_t gain_smallest = buffer_cnt[buffer_indices[curr]] * (pct_left_smallest / pct_one_cat) + (ntot - buffer_cnt[buffer_indices[curr]]) * ((1. - pct_left_smallest) / (1. - pct_one_cat));

			real_t pct_left_biggest = buffer_cnt[buffer_indices[ncat - 1]] / ntot;
			real_t gain_biggest = buffer_cnt[buffer_indices[ncat - 1]] * (pct_left_biggest / pct_one_cat) + (ntot - buffer_cnt[buffer_indices[ncat - 1]]) * ((1. - pct_left_biggest) / (1. - pct_one_cat));

			if (gain_smallest >= gain_biggest)
			{
				chosen_cat = buffer_indices[curr];
				return gain_smallest / ntot;
			}

			else
			{
				chosen_cat = buffer_indices[ncat - 1];
				return gain_biggest / ntot;
			}
			break;
		}

		case SubSet:
		{
			lreal_t_safe cnt_left = 0;
			lreal_t_safe cnt_right;
			int st_cat = curr - 1;
			real_t this_gain;
			real_t best_gain = -HUGE_VAL;
			int best_cat = 0;
			lreal_t_safe pct_left;
			real_t pct_cat_left;
			real_t ncat_present_ = (real_t)ncat_present;
			for (; curr < ncat; curr++)
			{
				cnt_left += buffer_cnt[buffer_indices[curr]];
				cnt_right = ntot - cnt_left;
				pct_left = cnt_left / ntot;
				pct_cat_left = (real_t)(curr - st_cat) / ncat_present_;
				this_gain = (lreal_t_safe)cnt_left * (pct_left / pct_cat_left) + (lreal_t_safe)cnt_right * (((lreal_t_safe)1 - pct_left) / (1. - pct_cat_left));
				if (this_gain > best_gain)
				{
					best_gain = this_gain;
					best_cat = curr;
				}
			}

			if (best_gain <= -HUGE_VAL)
				return best_gain;
			st_cat++;
			for (; st_cat <= best_cat; st_cat++)
				split_categ[buffer_indices[st_cat]] = 1;
			for (; st_cat < ncat; st_cat++)
				split_categ[buffer_indices[st_cat]] = 0;
			return best_gain / ntot;
			break;
		}
		}

		/* This will not be reached, but CRAN might complain otherwise */
		return -HUGE_VAL;
	}
#if 0
	/* for split-criterion in hyperplanes (see below for version aimed at single-variable splits) */
	template <class lreal_t_safe >
	real_t eval_guided_crit(real_t *  x, size_t n, GainCriterion criterion,
	                        real_t min_gain, bool as_relative_gain, real_t *  buffer_sd,
	                        real_t &  split_point, real_t &  xmin, real_t &  xmax,
	                        size_t *  ix_arr_plus_st,
	                        size_t *  cols_use, size_t ncols_use, bool force_cols_use,
	                        real_t *  X_row_major, size_t ncols,
	                        real_t *  Xr, size_t *  Xr_ind, size_t *  Xr_indptr)
	{
	    /* Note: the input 'x' is supposed to be a linear combination of standardized variables, so
	       all numbers are assumed to be small and in the same scale */
	    real_t gain = 0;
	    if (criterion == DensityCrit || criterion == FullGain) min_gain = 0;

	    /* here it's assumed the 'x' vector matches exactly with 'ix_arr' + 'st' */
	    if (unlikely(n == 2))
	    {
	        if (x[0] == x[1]) return -HUGE_VAL;
	        split_point = midpoint_with_reorder(x[0], x[1]);
	        gain        = 1.;
	        if (gain > min_gain)
	            return gain;
	        else
	            return 0.;
	    }

	    if (criterion == FullGain)
	    {
	        /* TODO: these buffers should be allocated externally */
	        std::vector<size_t> argsorted(n);
	        std::iota(argsorted.begin(), argsorted.end(), (size_t)0);
	        std::sort(argsorted.begin(), argsorted.end(),
	                  [&x](const size_t a, const size_t b){return x[a] < x[b];});
	        if (x[argsorted[0]] == x[argsorted[n-1]]) return -HUGE_VAL;
	        std::vector<real_t> temp_buffer(n + mult2(ncols));
	        for (size_t ix = 0; ix < n; ix++) temp_buffer[ix] = x[argsorted[ix]];
	        for (size_t ix = 0; ix < n; ix++)
	            argsorted[ix] = ix_arr_plus_st[argsorted[ix]];
	        size_t ignored;
	        return find_split_full_gain<real_t, lreal_t_safe>(
	                                    temp_buffer.data(), (size_t)0, n-1, argsorted.data(),
	                                    cols_use, ncols_use, force_cols_use,
	                                    X_row_major, ncols,
	                                    Xr, Xr_ind, Xr_indptr,
	                                    temp_buffer.data() + n, temp_buffer.data() + n + ncols,
	                                    ignored, split_point,
	                                    false);
	    }

	    /* sort in ascending order */
	    std::sort(x, x + n);
	    xmin = x[0]; xmax = x[n-1];
	    if (x[0] == x[n-1]) return -HUGE_VAL;

	    if (criterion == Pooled && as_relative_gain && min_gain <= 0)
	        gain = find_split_rel_gain<real_t, lreal_t_safe>(x, n, split_point);
	    else if (criterion == Pooled || criterion == Averaged)
	        gain = find_split_std_gain<real_t, lreal_t_safe>(x, n, buffer_sd, criterion, min_gain, split_point);
	    else if (criterion == DensityCrit)
	        gain = find_split_dens_shortform<real_t, lreal_t_safe>(x, n, split_point);
	    /* Note: a gain of -Inf signals that the data is unsplittable. Zero signals it's below the minimum. */
	    return std::fmax(0., gain);
	}

	template <class lreal_t_safe>
	real_t eval_guided_crit_weighted(real_t *  x, size_t n, GainCriterion criterion,
	                                 real_t min_gain, bool as_relative_gain, real_t *  buffer_sd,
	                                 real_t &  split_point, real_t &  xmin, real_t &  xmax,
	                                 real_t *  w, size_t *  buffer_indices,
	                                 size_t *  ix_arr_plus_st,
	                                 size_t *  cols_use, size_t ncols_use, bool force_cols_use,
	                                 real_t *  X_row_major, size_t ncols,
	                                 real_t *  Xr, size_t *  Xr_ind, size_t *  Xr_indptr)
	{
	    /* Note: the input 'x' is supposed to be a linear combination of standardized variables, so
	       all numbers are assumed to be small and in the same scale */
	    real_t gain = 0;
	    if (criterion == DensityCrit || criterion == FullGain) min_gain = 0;

	    /* here it's assumed the 'x' vector matches exactly with 'ix_arr' + 'st' */
	    if (unlikely(n == 2))
	    {
	        if (x[0] == x[1]) return -HUGE_VAL;
	        split_point = midpoint_with_reorder(x[0], x[1]);
	        gain        = 1.;
	        if (gain > min_gain)
	            return gain;
	        else
	            return 0.;
	    }

	    /* sort in ascending order */
	    std::iota(buffer_indices, buffer_indices + n, (size_t)0);
	    std::sort(buffer_indices, buffer_indices + n,
	              [&x](const size_t a, const size_t b){return x[a] < x[b];});
	    xmin = x[buffer_indices[0]]; xmax = x[buffer_indices[n-1]];
	    if (xmin == xmax) return -HUGE_VAL;

	    if (criterion == Pooled || criterion == Averaged)
	        gain = find_split_std_gain_weighted<real_t, lreal_t_safe>(x, n, buffer_sd, criterion, min_gain, split_point, w, buffer_indices);
	    else if (criterion == DensityCrit)
	        gain = find_split_dens_shortform_weighted<real_t, real_t * , lreal_t_safe>(x, n, split_point, w, buffer_indices);
	    else if (criterion == FullGain)
	    {
	        std::vector<size_t> argsorted(n);
	        std::iota(argsorted.begin(), argsorted.end(), (size_t)0);
	        std::sort(argsorted.begin(), argsorted.end(),
	                  [&x](const size_t a, const size_t b){return x[a] < x[b];});
	        if (x[argsorted[0]] == x[argsorted[n-1]]) return -HUGE_VAL;
	        std::vector<real_t> temp_buffer(n + mult2(ncols));
	        for (size_t ix = 0; ix < n; ix++) temp_buffer[ix] = x[argsorted[ix]];
	        for (size_t ix = 0; ix < n; ix++)
	            argsorted[ix] = ix_arr_plus_st[argsorted[ix]];
	        size_t ignored;
	        gain = find_split_full_gain_weighted<real_t, real_t * , lreal_t_safe>(
	                                             temp_buffer.data(), (size_t)0, n-1, argsorted.data(),
	                                             cols_use, ncols_use, force_cols_use,
	                                             X_row_major, ncols,
	                                             Xr, Xr_ind, Xr_indptr,
	                                             temp_buffer.data() + n, temp_buffer.data() + n + ncols,
	                                             ignored, split_point,
	                                             false,
	                                             w);
	    }
	    /* Note: a gain of -Inf signals that the data is unsplittable. Zero signals it's below the minimum. */
	    return std::fmax(0., gain);
	}

	/* for split-criterion in single-variable splits */
	template <class real_t_, class lreal_t_safe>
	real_t eval_guided_crit(size_t *  ix_arr, size_t st, size_t end, real_t_ *  x,
	                        real_t *  buffer_sd, bool as_relative_gain,
	                        real_t *  buffer_imputed_x, real_t *  saved_xmedian,
	                        size_t &split_ix, real_t &  split_point, real_t &  xmin, real_t &  xmax,
	                        GainCriterion criterion, real_t min_gain, MissingAction missing_action,
	                        size_t *  cols_use, size_t ncols_use, bool force_cols_use,
	                        real_t *  X_row_major, size_t ncols,
	                        real_t *  Xr, size_t *  Xr_ind, size_t *  Xr_indptr)
	{
	    size_t st_orig = st;
	    real_t gain = 0;
	    if (criterion == DensityCrit || criterion == FullGain) min_gain = 0;

	    /* move NAs to the front if there's any, exclude them from calculations */
	    if (missing_action != Fail)
	        st = move_NAs_to_front(ix_arr, st, end, x);

	    if (unlikely(st >= end)) return -HUGE_VAL;
	    else if (unlikely(st == (end-1)))
	    {
	        if (x[ix_arr[st]] == x[ix_arr[end]])
	            return -HUGE_VAL;
	        split_point = midpoint_with_reorder(x[ix_arr[st]], x[ix_arr[end]]);
	        split_ix    = st;
	        gain        = 1.;
	        if (gain > min_gain)
	            return gain;
	        else
	            return 0.;
	    }

	    /* sort in ascending order */
	    std::sort(ix_arr + st, ix_arr + end + 1, [&x](const size_t a, const size_t b){return x[a] < x[b];});
	    if (x[ix_arr[st]] == x[ix_arr[end]]) return -HUGE_VAL;
	    xmin = x[ix_arr[st]]; xmax = x[ix_arr[end]];

	    /* unlike the previous case for the extended model, the data here has not been centered,
	       which could make the standard deviations have poor precision. It's nevertheless not
	       necessary for this mean to have good precision, since it's only meant for centering,
	       so it can be calculated inexactly with simd instructions. */
	    real_t_ xmean = 0;
	    if (criterion == Pooled || criterion == Averaged)
	    {
	        for (size_t ix = st; ix <= end; ix++)
	            xmean += x[ix_arr[ix]];
	        xmean /= (real_t_)(end - st + 1);
	    }

	    if (missing_action == Impute && st > st_orig)
	    {
	        missing_action = Fail;
	        fill_NAs_with_median(ix_arr, st_orig, st, end, x, buffer_imputed_x, saved_xmedian);
	        if (criterion == Pooled && as_relative_gain && min_gain <= 0)
	            gain = find_split_rel_gain<real_t, lreal_t_safe>(buffer_imputed_x, (real_t)xmean, ix_arr, st_orig, end, split_point, split_ix);
	        else if (criterion == Pooled || criterion == Averaged)
	            gain = find_split_std_gain<real_t, lreal_t_safe>(buffer_imputed_x, (real_t)xmean, ix_arr, st_orig, end, buffer_sd, criterion, min_gain, split_point, split_ix);
	        else if (criterion == DensityCrit)
	            gain = find_split_dens<real_t, lreal_t_safe>(buffer_imputed_x, ix_arr, st_orig, end, split_point, split_ix);
	        else if (criterion == FullGain)
	        {
	            /* TODO: this buffer should be allocated from outside */
	            std::vector<real_t> temp_buffer(mult2(ncols));
	            gain = find_split_full_gain<real_t, lreal_t_safe>(
	                                        buffer_imputed_x, st_orig, end, ix_arr,
	                                        cols_use, ncols_use, force_cols_use,
	                                        X_row_major, ncols,
	                                        Xr, Xr_ind, Xr_indptr,
	                                        temp_buffer.data(), temp_buffer.data() + ncols,
	                                        split_ix, split_point, true);
	        }

	        /* Note: in theory, it should be possible to use a faster version assuming a contiguous array for 'x',
	           but such an approach might give inexact split points. Better to avoid such inexactness at the
	           expense of more computations. */
	    }
 	    else
	    {
	        if (criterion == Pooled && as_relative_gain && min_gain <= 0)
	            gain = find_split_rel_gain<real_t_, lreal_t_safe>(x, xmean, ix_arr, st, end, split_point, split_ix);
	        else if (criterion == Pooled || criterion == Averaged)
	            gain = find_split_std_gain<real_t_, lreal_t_safe>(x, xmean, ix_arr, st, end, buffer_sd, criterion, min_gain, split_point, split_ix);
	        else if (criterion == DensityCrit)
	            gain = find_split_dens<real_t_, lreal_t_safe>(x, ix_arr, st, end, split_point, split_ix);
	        else if (criterion == FullGain)
	        {
	            /* TODO: this buffer should be allocated from outside */
	            std::vector<real_t> temp_buffer(mult2(ncols));
	            gain = find_split_full_gain<real_t_, lreal_t_safe>(
	                                        x, st, end, ix_arr,
	                                        cols_use, ncols_use, force_cols_use,
	                                        X_row_major, ncols,
	                                        Xr, Xr_ind, Xr_indptr,
	                                        temp_buffer.data(), temp_buffer.data() + ncols,
	                                        split_ix, split_point, true);
	        }
	    }

	    /* Note: a gain of -Inf signals that the data is unsplittable. Zero signals it's below the minimum. */
	    return std::fmax(0., gain);
	}

	template <class real_t_, class mapping, class lreal_t_safe>
	real_t eval_guided_crit_weighted(size_t *  ix_arr, size_t st, size_t end, real_t_ *  x,
	                                 real_t *  buffer_sd, bool as_relative_gain,
	                                 real_t *  buffer_imputed_x, real_t *  saved_xmedian,
	                                 size_t &split_ix, real_t &  split_point, real_t &  xmin, real_t &  xmax,
	                                 GainCriterion criterion, real_t min_gain, MissingAction missing_action,
	                                 size_t *  cols_use, size_t ncols_use, bool force_cols_use,
	                                 real_t *  X_row_major, size_t ncols,
	                                 real_t *  Xr, size_t *  Xr_ind, size_t *  Xr_indptr,
	                                 mapping &  w)
	{
	    size_t st_orig = st;
	    real_t gain = 0;
	    if (criterion == DensityCrit || criterion == FullGain) min_gain = 0;

	    /* move NAs to the front if there's any, exclude them from calculations */
	    if (missing_action != Fail)
	        st = move_NAs_to_front(ix_arr, st, end, x);

	    if (unlikely(st >= end)) return -HUGE_VAL;
	    else if (unlikely(st == (end-1)))
	    {
	        if (x[ix_arr[st]] == x[ix_arr[end]])
	            return -HUGE_VAL;
	        split_point = midpoint_with_reorder(x[ix_arr[st]], x[ix_arr[end]]);
	        split_ix    = st;
	        gain        = 1.;
	        if (gain > min_gain)
	            return gain;
	        else
	            return 0.;
	    }

	    /* sort in ascending order */
	    std::sort(ix_arr + st, ix_arr + end + 1, [&x](const size_t a, const size_t b){return x[a] < x[b];});
	    if (x[ix_arr[st]] == x[ix_arr[end]]) return -HUGE_VAL;
	    xmin = x[ix_arr[st]]; xmax = x[ix_arr[end]];

	    /* unlike the previous case for the extended model, the data here has not been centered,
	       which could make the standard deviations have poor precision. It's nevertheless not
	       necessary for this mean to have good precision, since it's only meant for centering,
	       so it can be calculated inexactly with simd instructions. */
	    real_t_ xmean = 0;
	    real_t_ cnt = 0;
	    if (criterion == Pooled || criterion == Averaged)
	    {
	        for (size_t ix = st; ix <= end; ix++)
	        {
	            xmean += x[ix_arr[ix]];
	            cnt += w[ix_arr[ix]];
	        }
	        xmean /= cnt;
	    }

	    if (missing_action == Impute && st > st_orig)
	    {
	        missing_action = Fail;
	        fill_NAs_with_median(ix_arr, st_orig, st, end, x, buffer_imputed_x, saved_xmedian);
	        if (criterion == Pooled && as_relative_gain && min_gain <= 0)
	            gain = find_split_rel_gain_weighted<real_t, mapping, lreal_t_safe>(buffer_imputed_x, (real_t)xmean, ix_arr, st_orig, end, split_point, split_ix, w);
	        else if (criterion == Pooled || criterion == Averaged)
	            gain = find_split_std_gain_weighted<real_t, mapping, lreal_t_safe>(buffer_imputed_x, (real_t)xmean, ix_arr, st_orig, end, buffer_sd, criterion, min_gain, split_point, split_ix, w);
	        else if (criterion == DensityCrit)
	            gain = find_split_dens_weighted<real_t, mapping, lreal_t_safe>(buffer_imputed_x, ix_arr, st_orig, end, split_point, split_ix, w);
	        else if (criterion == FullGain)
	        {
	            std::vector<real_t> temp_buffer(mult2(ncols));
	            gain = find_split_full_gain_weighted<real_t, mapping, lreal_t_safe>(
	                                                 buffer_imputed_x, st_orig, end, ix_arr,
	                                                 cols_use, ncols_use, force_cols_use,
	                                                 X_row_major, ncols,
	                                                 Xr, Xr_ind, Xr_indptr,
	                                                 temp_buffer.data(), temp_buffer.data() + ncols,
	                                                 split_ix, split_point, true,
	                                                 w);
	        }
	    }

	    else
	    {
	        if (criterion == Pooled && as_relative_gain && min_gain <= 0)
	            gain = find_split_rel_gain_weighted<real_t_, mapping, lreal_t_safe>(x, xmean, ix_arr, st, end, split_point, split_ix, w);
	        else if (criterion == Pooled || criterion == Averaged)
	            gain = find_split_std_gain_weighted<real_t_, mapping, lreal_t_safe>(x, xmean, ix_arr, st, end, buffer_sd, criterion, min_gain, split_point, split_ix, w);
	        else if (criterion == DensityCrit)
	            gain = find_split_dens_weighted<real_t_, mapping, lreal_t_safe>(x, ix_arr, st, end, split_point, split_ix, w);
	        else if (criterion == FullGain)
	        {
	            std::vector<real_t> temp_buffer(mult2(ncols));
	            gain = find_split_full_gain_weighted<real_t_, mapping, lreal_t_safe>(
	                                                 x, st, end, ix_arr,
	                                                 cols_use, ncols_use, force_cols_use,
	                                                 X_row_major, ncols,
	                                                 Xr, Xr_ind, Xr_indptr,
	                                                 temp_buffer.data(), temp_buffer.data() + ncols,
	                                                 split_ix, split_point, true,
	                                                 w);
	        }
	    }

	    /* Note: a gain of -Inf signals that the data is unsplittable. Zero signals it's below the minimum. */
	    return std::fmax(0., gain);
	}

	/* TODO: here it should only need to look at the non-zero entries. It can then use the
	   same algorithm as above, but putting an extra check to see where do the zeros fit in
	   the sorted order of the non-zero entries while calculating gains and SDs, and then
	   call the 'divide_subset' function after-the-fact to reach the same end result.
	   It should be much faster than this if the non-zero entries are few. */
	template <class xreal/*=real_t*/,class sparse_ix_ /*= sparse_ix*/, class lreal_t_safe /*= long real_t*/>
	real_t eval_guided_crit(size_t ix_arr[], size_t st, size_t end,
	                        size_t col_num, xreal Xc[], sparse_ix Xc_ind[], sparse_ix_ Xc_indptr[],
	                        real_t buffer_arr[], size_t buffer_pos[], bool as_relative_gain,
	                        real_t *  saved_xmedian,
	                        real_t &split_point, real_t &xmin, real_t &xmax,
	                        GainCriterion criterion, real_t min_gain, MissingAction missing_action,
	                        size_t *  cols_use, size_t ncols_use, bool force_cols_use,
	                        real_t *  X_row_major, size_t ncols,
	                        real_t *  Xr, size_t *  Xr_ind, size_t *  Xr_indptr)
	{
	    size_t ignored;


	    todense(ix_arr, st, end,
	            col_num, Xc, Xc_ind, Xc_indptr,
	            buffer_arr);
	    size_t tot = end - st + 1;
	    std::iota(buffer_pos, buffer_pos + tot, (size_t)0);

	    if (missing_action == Impute)
	    {
	        missing_action = Fail;
	        for (size_t ix = 0; ix < tot; ix++)
	        {
	            if (unlikely(is_na_or_inf(buffer_arr[ix])))
	            {
	                goto fill_missing;
	            }
	        }
	        goto no_nas;

	        fill_missing:
	        {
	            size_t idx_half = div2(tot);
	            std::nth_element(buffer_pos, buffer_pos + idx_half, buffer_pos + tot,
	                             [&buffer_arr](const size_t a, const size_t b){return buffer_arr[a] < buffer_arr[b];});
	            *saved_xmedian = buffer_arr[buffer_pos[idx_half]];

	            if ((tot % 2) == 0)
	            {
	                real_t xlow = *std::max_element(buffer_pos, buffer_pos + idx_half);
	                *saved_xmedian = xlow + ((*saved_xmedian)-xlow)/2.;
	            }

	            for (size_t ix = 0; ix < tot; ix++)
	                buffer_arr[ix] = is_na_or_inf(buffer_arr[ix])? (*saved_xmedian) : buffer_arr[ix];
	            std::iota(buffer_pos, buffer_pos + tot, (size_t)0);
	        }
	    }

	    no_nas:
	    return eval_guided_crit<real_t, lreal_t_safe>(
	                            buffer_pos, 0, end - st, buffer_arr, buffer_arr + tot,
	                            as_relative_gain, saved_xmedian, (real_t*)NULL, ignored, split_point,
	                            xmin, xmax, criterion, min_gain, missing_action,
	                            cols_use, ncols_use, force_cols_use,
	                            X_row_major, ncols,
	                            Xr, Xr_ind, Xr_indptr);
	}

	template <class real_t_, class sparse_ix, class mapping, class lreal_t_safe>
	real_t eval_guided_crit_weighted(size_t ix_arr[], size_t st, size_t end,
	                                 size_t col_num, real_t_ Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
	                                 real_t buffer_arr[], size_t buffer_pos[], bool as_relative_gain,
	                                 real_t *restrict saved_xmedian,
	                                 real_t &restrict split_point, real_t &restrict xmin, real_t &restrict xmax,
	                                 GainCriterion criterion, real_t min_gain, MissingAction missing_action,
	                                 size_t *restrict cols_use, size_t ncols_use, bool force_cols_use,
	                                 real_t *restrict X_row_major, size_t ncols,
	                                 real_t *restrict Xr, size_t *restrict Xr_ind, size_t *restrict Xr_indptr,
	                                 mapping &restrict w)
	{
	    size_t ignored;


	    todense(ix_arr, st, end,
	            col_num, Xc, Xc_ind, Xc_indptr,
	            buffer_arr);
	    size_t tot = end - st + 1;
	    std::iota(buffer_pos, buffer_pos + tot, (size_t)0);


	    if (missing_action == Impute)
	    {
	        missing_action = Fail;
	        for (size_t ix = 0; ix < tot; ix++)
	        {
	            if (unlikely(is_na_or_inf(buffer_arr[ix])))
	            {
	                goto fill_missing;
	            }
	        }
	        goto no_nas;

	        fill_missing:
	        {
	            size_t idx_half = div2(tot);
	            std::nth_element(buffer_pos, buffer_pos + idx_half, buffer_pos + tot,
	                             [&buffer_arr](const size_t a, const size_t b){return buffer_arr[a] < buffer_arr[b];});
	            *saved_xmedian = buffer_arr[buffer_pos[idx_half]];

	            if ((tot % 2) == 0)
	            {
	                real_t xlow = *std::max_element(buffer_pos, buffer_pos + idx_half);
	                *saved_xmedian = xlow + ((*saved_xmedian)-xlow)/2.;
	            }

	            for (size_t ix = 0; ix < tot; ix++)
	                buffer_arr[ix] = is_na_or_inf(buffer_arr[ix])? (*saved_xmedian) : buffer_arr[ix];
	            std::iota(buffer_pos, buffer_pos + tot, (size_t)0);
	        }
	    }


	    no_nas:
	    /* TODO: allocate this buffer externally */
	    std::vector<real_t> buffer_w(tot);
	    for (size_t row = st; row <= end; row++)
	        buffer_w[row-st] = w[ix_arr[row]];
	    /* TODO: in this case, as the weights match with the order of the indices, could use a faster version
	       with a weighted rel_gain function instead (not yet implemented). */
	    return eval_guided_crit_weighted<real_t, std::vector<real_t>, lreal_t_safe>(
	                                     buffer_pos, 0, end - st, buffer_arr, buffer_arr + tot,
	                                     as_relative_gain, saved_xmedian, (real_t*)NULL, ignored, split_point,
	                                     xmin, xmax, criterion, min_gain, missing_action,
	                                     cols_use, ncols_use, force_cols_use,
	                                     X_row_major, ncols,
	                                     Xr, Xr_ind, Xr_indptr,
	                                     buffer_w);
	}

	/* How this works:
	   - For Averaged criterion, will take the expected standard deviation that would be gotten with the category counts
	     if each category got assigned a real number at random ~ Unif(0,1) and the data were thus converted to
	     numerical. In such case, the best split (highest sd gain) is always putting the second-highest count in one
	     branch, so there is no point in doing a full search over other permutations. In order to get more reasonable
	     splits, when using the option to split by subsets of categories, it will sort the counts and evaluate only
	     splits in which the categories are grouped in sorted order - in such cases it tends to pick either the
	     smallest or the largest category to assign to one branch, but sometimes picks groups too.
	   - For Pooled criterion, will take shannon entropy, which tends to make a more even split. In the case of splitting
	     by a single category, it always puts the largest category in a separate branch. In the case of subsets,
	     it can either evaluate possible splits over all permutations (not feasible if there are too many categories),
	     or look up for splits in sorted order just like for Averaged criterion.
	   Splitting by averaged Gini gain (like with Averaged) also selects always the second-largest category to put in one branch,
	   while splitting by weighted Gini (like with Pooled) usually selects the largest category to put in one branch. The
	   Gini gain is not easily comparable to that of numerical columns, so it's not offered as an option here.
	*/
	/* https://math.stackexchange.com/questions/3343384/expected-variance-and-kurtosis-from-pmf-in-which-possible-discrete-values-are-dr */
	/* TODO: 'buffer_pos' doesn't need to be 'size_t', 'int' would suffice */

#endif
	template <class lreal_t_safe>
	real_t
	eval_guided_crit(size_t *ix_arr, size_t st, size_t end, int *x, int ncat,
					 int *saved_cat_mode, size_t *buffer_cnt,
					 size_t *buffer_pos, real_t *buffer_prob, int &chosen_cat,
					 signed char *split_categ, signed char *buffer_split,
					 GainCriterion criterion, real_t min_gain, bool all_perm,
					 MissingAction missing_action, CategSplit cat_split_type)
	{
		if (criterion == DensityCrit)
			return find_split_dens_longform<size_t, lreal_t_safe>(x, ncat, ix_arr,
																  st, end,
																  cat_split_type,
																  missing_action,
																  chosen_cat,
																  split_categ,
																  saved_cat_mode,
																  buffer_cnt,
																  buffer_pos);
		if (st >= end)
			return -HUGE_VAL;
		size_t n_nas = 0;
		int xval;

		/* count categories */
		memset(buffer_cnt, 0, sizeof(size_t) * ncat);
		if (missing_action == Fail)
		{
			for (size_t row = st; row <= end; row++)
				if (likely(x[ix_arr[row]] >= 0))
					buffer_cnt[x[ix_arr[row]]]++;
		}

		else if (missing_action == Impute)
		{
			for (size_t row = st; row <= end; row++)
			{
				xval = x[ix_arr[row]];
				if (unlikely(xval < 0))
					n_nas++;
				else
					buffer_cnt[xval]++;
			}

			if (unlikely(n_nas >= end - st))
				return -HUGE_VAL;

			if (n_nas)
			{
				auto idxmax = std::max_element(buffer_cnt, buffer_cnt + ncat);
				*idxmax += n_nas;
				*saved_cat_mode = (int)std::distance(buffer_cnt, idxmax);
			}
		}

		else
		{
			for (size_t row = st; row <= end; row++)
			{
				xval = x[ix_arr[row]];
				if (likely(xval >= 0))
					buffer_cnt[xval]++;
			}
		}

		real_t this_gain = -HUGE_VAL;
		real_t best_gain = -HUGE_VAL;
		std::iota(buffer_pos, buffer_pos + ncat, (size_t)0);
		size_t st_pos = 0;

		switch (cat_split_type)
		{
		case SingleCateg:
		{
			size_t cnt = end - st + 1;
			lreal_t_safe cnt_l = (lreal_t_safe)cnt;
			size_t ncat_present = 0;

			switch (criterion)
			{
			case Averaged:
			{
				/* move zero-counts to the beginning */
				size_t temp;
				for (int cat = 0; cat < ncat; cat++)
				{
					if (buffer_cnt[cat])
					{
						ncat_present++;
						buffer_prob[cat] = (lreal_t_safe)buffer_cnt[cat] / cnt_l;
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
					return -HUGE_VAL;

				real_t sd_full = expected_sd_cat<size_t, lreal_t_safe>(
					buffer_prob, ncat_present, buffer_pos + st_pos);

				/* try isolating each category one at a time */
				for (size_t pos = st_pos; (int)pos < ncat; pos++)
				{
					this_gain =
						sd_gain(
							sd_full,
							0.0,
							(expected_sd_cat_single<size_t, size_t,
													lreal_t_safe>(buffer_cnt, buffer_prob,
																  ncat_present,
																  buffer_pos + st_pos,
																  pos - st_pos, cnt)));
					if (this_gain > min_gain && this_gain > best_gain)
					{
						best_gain = this_gain;
						chosen_cat = buffer_pos[pos];
					}
				}
				break;
			}

			case Pooled:
			{
				/* here it will always pick the largest one */
				size_t ncat_present = 0;
				size_t cnt_max = 0;
				for (int cat = 0; cat < ncat; cat++)
				{
					if (buffer_cnt[cat])
					{
						ncat_present++;
						if (cnt_max < buffer_cnt[cat])
						{
							cnt_max = buffer_cnt[cat];
							chosen_cat = cat;
						}
					}
				}

				if (ncat_present <= 1)
					return -HUGE_VAL;

				lreal_t_safe cnt_left = (lreal_t_safe)((end - st + 1) - cnt_max);
				this_gain = ((lreal_t_safe)cnt * std::log((lreal_t_safe)cnt) - cnt_left * std::log(cnt_left) - (lreal_t_safe)cnt_max * std::log((lreal_t_safe)cnt_max)) / cnt;
				best_gain = (this_gain > min_gain) ? this_gain : best_gain;
				break;
			}

			default:
			{
				unexpected_error();
				break;
			}
			}
			break;
		}

		case SubSet:
		{
			/* sort by counts */
			std::sort(buffer_pos, buffer_pos + ncat, [&buffer_cnt](const size_t a, const size_t b)
					  { return buffer_cnt[a] < buffer_cnt[b]; });

			/* set split as: (1):left (0):right (-1):not_present */
			memset(buffer_split, 0, ncat * sizeof(signed char));

			lreal_t_safe cnt = (lreal_t_safe)(end - st + 1);

			switch (criterion)
			{
			case Averaged:
			{
				/* determine first non-zero and convert to probabilities */
				real_t sd_full;
				for (int cat = 0; cat < ncat; cat++)
				{
					if (buffer_cnt[buffer_pos[cat]])
					{
						buffer_prob[buffer_pos[cat]] =
							(lreal_t_safe)buffer_cnt[buffer_pos[cat]] / cnt;
					}

					else
					{
						buffer_split[buffer_pos[cat]] = -1;
						st_pos++;
					}
				}

				if ((int)st_pos >= (ncat - 1))
					return -HUGE_VAL;

				/* calculate full SD assuming they take values randomly ~Unif(0, 1) */
				size_t ncat_present = (size_t)ncat - st_pos;
				sd_full = expected_sd_cat<size_t, lreal_t_safe>(
					buffer_prob, ncat_present, buffer_pos + st_pos);
				if (ncat_present >= log2ceil(SIZE_MAX))
					all_perm = false;

				/* move categories one at a time */
				for (size_t pos = st_pos; pos < ((size_t)ncat - st_pos - 1);
					 pos++)
				{
					buffer_split[buffer_pos[pos]] = 1;
					this_gain = sd_gain(
						sd_full,
						(expected_sd_cat<size_t, size_t, lreal_t_safe>(
							buffer_cnt, buffer_prob, pos - st_pos + 1,
							buffer_pos + st_pos)),
						(expected_sd_cat<size_t, size_t, lreal_t_safe>(
							buffer_cnt, buffer_prob, (size_t)ncat - pos - 1,
							buffer_pos + pos + 1)));
					if (this_gain > min_gain && this_gain > best_gain)
					{
						best_gain = this_gain;
						memcpy(split_categ, buffer_split,
							   ncat * sizeof(signed char));
					}
				}

				break;
			}

			case Pooled:
			{
				lreal_t_safe s = 0;

				/* determine first non-zero and get base info */
				for (int cat = 0; cat < ncat; cat++)
				{
					if (buffer_cnt[buffer_pos[cat]])
					{
						s +=
							(buffer_cnt[buffer_pos[cat]] <= 1) ? 0 : ((lreal_t_safe)buffer_cnt[buffer_pos[cat]] * std::log((lreal_t_safe)buffer_cnt[buffer_pos[cat]]));
					}

					else
					{
						buffer_split[buffer_pos[cat]] = -1;
						st_pos++;
					}
				}

				if ((int)st_pos >= (ncat - 1))
					return -HUGE_VAL;

				/* calculate base info */
				lreal_t_safe base_info = cnt * std::log(cnt) - s;

				if (all_perm)
				{
					size_t cnt_left = 0, cnt_right = 0;
					real_t s_left = 0., s_right = 0.;
					size_t ncat_present = (size_t)ncat - st_pos;
					size_t ncomb = pow2(ncat_present) - 1;
					size_t best_combin = 0;

					for (size_t combin = 1; combin < ncomb; combin++)
					{
						cnt_left = 0;
						cnt_right = 0;
						s_left = 0;
						s_right = 0;
						for (size_t pos = st_pos; (int)pos < ncat; pos++)
						{
							if (extract_bit(combin, pos))
							{
								cnt_left += buffer_cnt[buffer_pos[pos]];
								s_left +=
									(buffer_cnt[buffer_pos[pos]] <= 1) ? 0 : ((lreal_t_safe)buffer_cnt[buffer_pos[pos]] * std::log((lreal_t_safe)buffer_cnt[buffer_pos[pos]]));
							}

							else
							{
								cnt_right += buffer_cnt[buffer_pos[pos]];
								s_right +=
									(buffer_cnt[buffer_pos[pos]] <= 1) ? 0 : ((lreal_t_safe)buffer_cnt[buffer_pos[pos]] * std::log((lreal_t_safe)buffer_cnt[buffer_pos[pos]]));
							}
						}

						this_gain = categ_gain<size_t, lreal_t_safe>(
							cnt_left, cnt_right, s_left, s_right, base_info,
							cnt);

						if (this_gain > min_gain && this_gain > best_gain)
						{
							best_gain = this_gain;
							best_combin = combin;
						}
					}

					if (best_gain > min_gain)
						for (size_t pos = 0; pos < ncat_present; pos++)
							split_categ[buffer_pos[st_pos + pos]] = extract_bit(
								best_combin, pos);
				}

				else
				{
					/* try moving the categories one at a time */
					size_t cnt_left = 0;
					size_t cnt_right = end - st + 1;
					real_t s_left = 0;
					real_t s_right = s;

					for (size_t pos = st_pos; pos < (ncat - st_pos - 1);
						 pos++)
					{
						buffer_split[buffer_pos[pos]] = 1;
						s_left +=
							(buffer_cnt[buffer_pos[pos]] <= 1) ? 0 : ((lreal_t_safe)buffer_cnt[buffer_pos[pos]] * std::log((lreal_t_safe)buffer_cnt[buffer_pos[pos]]));
						s_right -=
							(buffer_cnt[buffer_pos[pos]] <= 1) ? 0 : ((lreal_t_safe)buffer_cnt[buffer_pos[pos]] * std::log((lreal_t_safe)buffer_cnt[buffer_pos[pos]]));
						cnt_left += buffer_cnt[buffer_pos[pos]];
						cnt_right -= buffer_cnt[buffer_pos[pos]];

						this_gain = categ_gain<size_t, lreal_t_safe>(
							cnt_left, cnt_right, s_left, s_right, base_info,
							cnt);

						if (this_gain > min_gain && this_gain > best_gain)
						{
							best_gain = this_gain;
							memcpy(split_categ, buffer_split,
								   ncat * sizeof(signed char));
						}
					}
				}

				break;
			}

			default:
			{
				unexpected_error();
				break;
			}
			}
		}
		}

		if (st == (end - 1))
			return 0;

		if (best_gain <= -HUGE_VAL && this_gain <= min_gain && this_gain > -HUGE_VAL)
			return 0;
		else
			return best_gain;
	}

	template <class mapping, class lreal_t_safe>
	real_t
	eval_guided_crit_weighted(size_t *ix_arr, size_t st, size_t end, int *x,
							  int ncat, int *saved_cat_mode,
							  size_t *buffer_pos, real_t *buffer_prob,
							  int &chosen_cat, signed char *split_categ,
							  signed char *buffer_split,
							  GainCriterion criterion, real_t min_gain,
							  bool all_perm, MissingAction missing_action,
							  CategSplit cat_split_type, mapping &w)
	{
		if (criterion == DensityCrit)
			return find_split_dens_longform_weighted<mapping, size_t, lreal_t_safe>(
				x, ncat, ix_arr, st, end, cat_split_type, missing_action,
				chosen_cat, split_categ, saved_cat_mode, buffer_pos, w);
		if (st >= end)
			return -HUGE_VAL;
		lreal_t_safe w_missing = 0;
		int xval;
		size_t ix_;

		/* count categories */
		/* TODO: allocate this buffer externally */
		std::vector<lreal_t_safe> buffer_cnt(ncat, (lreal_t_safe)0);
		if (missing_action == Fail)
		{
			for (size_t row = st; row <= end; row++)
			{
				ix_ = ix_arr[row];
				if (unlikely(x[ix_]) < 0)
					continue;
				buffer_cnt[x[ix_]] += w[ix_];
			}
		}

		else if (missing_action == Impute)
		{
			for (size_t row = st; row <= end; row++)
			{
				ix_ = ix_arr[row];
				xval = x[ix_];
				if (unlikely(xval < 0))
					w_missing += w[ix_];
				else
					buffer_cnt[xval] += w[ix_];
			}

			if (w_missing)
			{
				auto idxmax = std::max_element(buffer_cnt.begin(),
											   buffer_cnt.end());
				*idxmax += w_missing;
				*saved_cat_mode = (int)std::distance(buffer_cnt.begin(),
													 idxmax);
			}
		}

		else
		{
			for (size_t row = st; row <= end; row++)
			{
				ix_ = ix_arr[row];
				xval = x[ix_];
				if (likely(xval >= 0))
					buffer_cnt[xval] += w[ix_];
			}
		}

		lreal_t_safe cnt = std::accumulate(buffer_cnt.begin(),
										   buffer_cnt.end(), (lreal_t_safe)0);

		real_t this_gain = -HUGE_VAL;
		real_t best_gain = -HUGE_VAL;
		std::iota(buffer_pos, buffer_pos + ncat, (size_t)0);
		size_t st_pos = 0;

		switch (cat_split_type)
		{
		case SingleCateg:
		{
			size_t ncat_present = 0;

			switch (criterion)
			{
			case Averaged:
			{
				/* move zero-counts to the beginning */
				size_t temp;
				for (int cat = 0; cat < ncat; cat++)
				{
					if (buffer_cnt[cat])
					{
						ncat_present++;
						buffer_prob[cat] = buffer_cnt[cat] / cnt;
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
					return -HUGE_VAL;

				real_t sd_full = expected_sd_cat<size_t, lreal_t_safe>(
					buffer_prob, ncat_present, buffer_pos + st_pos);

				/* try isolating each category one at a time */
				for (size_t pos = st_pos; (int)pos < ncat; pos++)
				{
					this_gain = sd_gain(
						sd_full,
						0.0,
						(expected_sd_cat_single<lreal_t_safe, size_t,
												lreal_t_safe>(buffer_cnt.data(), buffer_prob,
															  ncat_present, buffer_pos + st_pos,
															  pos - st_pos, cnt)));
					if (this_gain > min_gain && this_gain > best_gain)
					{
						best_gain = this_gain;
						chosen_cat = buffer_pos[pos];
					}
				}
				break;
			}

			case Pooled:
			{
				/* here it will always pick the largest one */
				size_t ncat_present = 0;
				lreal_t_safe cnt_max = 0;
				for (int cat = 0; cat < ncat; cat++)
				{
					if (buffer_cnt[cat])
					{
						ncat_present++;
						if (cnt_max < buffer_cnt[cat])
						{
							cnt_max = buffer_cnt[cat];
							chosen_cat = cat;
						}
					}
				}

				if (ncat_present <= 1)
					return -HUGE_VAL;

				lreal_t_safe cnt_left = (lreal_t_safe)(cnt - cnt_max);

				/* TODO: think of a better way of dealing with numbers between zero and one */
				this_gain = (std::fmax((lreal_t_safe)1, cnt) * std::log(std::fmax((lreal_t_safe)1, cnt)) - std::fmax((lreal_t_safe)1, cnt_left) * std::log(std::fmax((lreal_t_safe)1, cnt_left)) - std::fmax((lreal_t_safe)1, cnt_max) * std::log(std::fmax((lreal_t_safe)1, cnt_max))) / std::fmax((lreal_t_safe)1, cnt);
				best_gain = (this_gain > min_gain) ? this_gain : best_gain;
				break;
			}

			default:
			{
				unexpected_error();
				break;
			}
			}
			break;
		}

		case SubSet:
		{
			/* sort by counts */
			std::sort(buffer_pos, buffer_pos + ncat, [&buffer_cnt](const size_t a, const size_t b)
					  { return buffer_cnt[a] < buffer_cnt[b]; });

			/* set split as: (1):left (0):right (-1):not_present */
			memset(buffer_split, 0, ncat * sizeof(signed char));

			switch (criterion)
			{
			case Averaged:
			{
				/* determine first non-zero and convert to probabilities */
				real_t sd_full;
				for (int cat = 0; cat < ncat; cat++)
				{
					if (buffer_cnt[buffer_pos[cat]])
					{
						buffer_prob[buffer_pos[cat]] =
							(lreal_t_safe)buffer_cnt[buffer_pos[cat]] / cnt;
					}

					else
					{
						buffer_split[buffer_pos[cat]] = -1;
						st_pos++;
					}
				}

				if ((int)st_pos >= (ncat - 1))
					return -HUGE_VAL;

				/* calculate full SD assuming they take values randomly ~Unif(0, 1) */
				size_t ncat_present = (size_t)ncat - st_pos;
				sd_full = expected_sd_cat<size_t, lreal_t_safe>(
					buffer_prob, ncat_present, buffer_pos + st_pos);
				if (ncat_present >= log2ceil(SIZE_MAX))
					all_perm = false;

				/* move categories one at a time */
				for (size_t pos = st_pos; pos < ((size_t)ncat - st_pos - 1);
					 pos++)
				{
					buffer_split[buffer_pos[pos]] = 1;
					/* TODO: is this correct? */
					this_gain = sd_gain(
						sd_full,
						(expected_sd_cat<lreal_t_safe, size_t, lreal_t_safe>(
							buffer_cnt.data(), buffer_prob, pos - st_pos + 1,
							buffer_pos + st_pos)),
						(expected_sd_cat<lreal_t_safe, size_t, lreal_t_safe>(
							buffer_cnt.data(), buffer_prob,
							(size_t)ncat - pos - 1, buffer_pos + pos + 1)));
					if (this_gain > min_gain && this_gain > best_gain)
					{
						best_gain = this_gain;
						memcpy(split_categ, buffer_split,
							   ncat * sizeof(signed char));
					}
				}

				break;
			}

			case Pooled:
			{
				lreal_t_safe s = 0;

				/* determine first non-zero and get base info */
				for (int cat = 0; cat < ncat; cat++)
				{
					if (buffer_cnt[buffer_pos[cat]])
					{
						s +=
							(buffer_cnt[buffer_pos[cat]] <= 1) ? (lreal_t_safe)0 : ((lreal_t_safe)buffer_cnt[buffer_pos[cat]] * std::log((lreal_t_safe)buffer_cnt[buffer_pos[cat]]));
					}

					else
					{
						buffer_split[buffer_pos[cat]] = -1;
						st_pos++;
					}
				}

				if ((int)st_pos >= (ncat - 1))
					return -HUGE_VAL;

				/* calculate base info */
				lreal_t_safe base_info = std::fmax((lreal_t_safe)1, cnt) * std::log(std::fmax((lreal_t_safe)1, cnt)) - s;

				if (all_perm)
				{
					size_t cnt_left, cnt_right;
					real_t s_left, s_right;
					size_t ncat_present = (size_t)ncat - st_pos;
					size_t ncomb = pow2(ncat_present) - 1;
					size_t best_combin = 0;

					for (size_t combin = 1; combin < ncomb; combin++)
					{
						cnt_left = 0;
						cnt_right = 0;
						s_left = 0;
						s_right = 0;
						for (size_t pos = st_pos; (int)pos < ncat; pos++)
						{
							if (extract_bit(combin, pos))
							{
								cnt_left += buffer_cnt[buffer_pos[pos]];
								s_left +=
									(buffer_cnt[buffer_pos[pos]] <= 1) ? (lreal_t_safe)0 : ((lreal_t_safe)buffer_cnt[buffer_pos[pos]] * std::log((lreal_t_safe)buffer_cnt[buffer_pos[pos]]));
							}

							else
							{
								cnt_right += buffer_cnt[buffer_pos[pos]];
								s_right +=
									(buffer_cnt[buffer_pos[pos]] <= 1) ? (lreal_t_safe)0 : ((lreal_t_safe)buffer_cnt[buffer_pos[pos]] * std::log((lreal_t_safe)buffer_cnt[buffer_pos[pos]]));
							}
						}

						this_gain = categ_gain<size_t, lreal_t_safe>(
							cnt_left, cnt_right, s_left, s_right, base_info,
							cnt);

						if (this_gain > min_gain && this_gain > best_gain)
						{
							best_gain = this_gain;
							best_combin = combin;
						}
					}

					if (best_gain > min_gain)
						for (size_t pos = 0; pos < ncat_present; pos++)
							split_categ[buffer_pos[st_pos + pos]] = extract_bit(
								best_combin, pos);
				}

				else
				{
					/* try moving the categories one at a time */
					size_t cnt_left = 0;
					size_t cnt_right = end - st + 1;
					real_t s_left = 0;
					real_t s_right = s;

					for (size_t pos = st_pos; pos < (ncat - st_pos - 1);
						 pos++)
					{
						buffer_split[buffer_pos[pos]] = 1;
						s_left +=
							(buffer_cnt[buffer_pos[pos]] <= 1) ? (lreal_t_safe)0 : ((lreal_t_safe)buffer_cnt[buffer_pos[pos]] * std::log((lreal_t_safe)buffer_cnt[buffer_pos[pos]]));
						s_right -=
							(buffer_cnt[buffer_pos[pos]] <= 1) ? (lreal_t_safe)0 : ((lreal_t_safe)buffer_cnt[buffer_pos[pos]] * std::log((lreal_t_safe)buffer_cnt[buffer_pos[pos]]));
						cnt_left += buffer_cnt[buffer_pos[pos]];
						cnt_right -= buffer_cnt[buffer_pos[pos]];

						this_gain = categ_gain<size_t, lreal_t_safe>(
							cnt_left, cnt_right, s_left, s_right, base_info,
							cnt);

						if (this_gain > min_gain && this_gain > best_gain)
						{
							best_gain = this_gain;
							memcpy(split_categ, buffer_split,
								   ncat * sizeof(signed char));
						}
					}
				}

				break;
			}

			default:
			{
				unexpected_error();
				break;
			}
			}
		}
		}

		if (st == (end - 1))
			return 0;

		if (best_gain <= -HUGE_VAL && this_gain <= min_gain && this_gain > -HUGE_VAL)
			return 0;
		else
			return best_gain;
	}

	template <class PredictionData, class InputData, class WorkerMemory>
	void
	gather_sim_result(
		std::vector<provallo::WorkerForSimilarity> *worker_memory,
		std::vector<WorkerMemory> *worker_memory_m, PredictionData *data,
		InputData *input_data, provallo::IsoForest *model_outputs,
		ExtIsoForest *model_outputs_ext, real_t *tmat, real_t *rmat,
		size_t n_from, size_t ntrees, bool assume_full_distr,
		bool standardize_dist, bool as_kernel, int nthreads)
	{
		UNDEF_REFERENCE(nthreads)
		UNDEF_REFERENCE2(nthreads)

		if (interrupt_switch)
			return;

		size_t nrows = (data != NULL) ? data->nrows : input_data->nrows;
		size_t ncomb = calc_ncomb(nrows);
		size_t n_to = (data != NULL) ? (data->nrows - n_from) : 0;

		if (worker_memory != NULL)
		{
			if (!(*worker_memory)[0].tmat_sep.empty())
				std::copy((*worker_memory)[0].tmat_sep.begin(),
						  (*worker_memory)[0].tmat_sep.end(), tmat);
			else
				std::copy((*worker_memory)[0].rmat.begin(),
						  (*worker_memory)[0].rmat.end(), rmat);
		}

		else
		{
			std::copy((*worker_memory_m)[0].tmat_sep.begin(),
					  (*worker_memory_m)[0].tmat_sep.end(), tmat);
		}
		real_t ntrees_dbl = (real_t)ntrees;
		if (standardize_dist)
		{
			if (as_kernel)
			{
				if (tmat != NULL)
					for (size_t ix = 0; ix < ncomb; ix++)
						tmat[ix] /= ntrees_dbl;
				else
					for (size_t ix = 0; ix < (n_from * n_to); ix++)
						rmat[ix] /= ntrees_dbl;
				return;
			}
			/* Note: the separation distances up this point are missing the first hop, which is always
			 a +1 to every combination. Thus, it needs to be added back for the average separation depth.
			 For the standardized metric, it takes the expected divisor as 2(=3-1) instead of 3, given
			 that every combination will always get a +1 at the beginning. Since what's obtained here
			 is a sum across all trees, adding this +1 means adding the number of trees. */
			real_t div_trees = ntrees_dbl;
			if (assume_full_distr)
			{
				div_trees *= 2;
			}

			else if (input_data != NULL)
			{
				div_trees *= (expected_separation_depth(input_data->nrows) - 1);
			}

			else
			{
				div_trees *= ((
								  (model_outputs != NULL) ? expected_separation_depth_hotstart(
																model_outputs->exp_avg_sep,
																model_outputs->orig_sample_size,
																model_outputs->orig_sample_size + data->nrows)
														  : expected_separation_depth_hotstart(
																model_outputs_ext->exp_avg_sep,
																model_outputs_ext->orig_sample_size,
																model_outputs_ext->orig_sample_size + data->nrows)) -
							  1);
			}

			if (tmat != NULL)

				for (size_t ix = 0; ix < ncomb; ix++)
					tmat[ix] = std::exp2(-tmat[ix] / div_trees);
			else
				for (size_t ix = 0; ix < (n_from * n_to); ix++)
					rmat[ix] = std::exp2(-rmat[ix] / div_trees);
		}
		else
		{
			if (as_kernel)
				return;

			if (tmat != NULL)
				for (size_t ix = 0; ix < ncomb; ix++)
					tmat[ix] = (tmat[ix] + ntrees) / ntrees_dbl;
			else
				for (size_t ix = 0; ix < (n_from * n_to); ix++)
					rmat[ix] = (rmat[ix] + ntrees) / ntrees_dbl;
		}
	}

	template <class InputData = InputData, typename lreal_t_safe = long real_t>
	std::vector<real_t>
	calc_kurtosis_all_data(InputData &input_data, ModelParams &model_params,
						   RNG_engine &rnd_generator);

	template <class real_t_, class sparse_ix_,
			  typename lreal_t_safe>
	int
	fit_iforest_internal(IsoForest *model_outputs,
						 ExtIsoForest *model_outputs_ext,
						 real_t numeric_data[],
						 size_t ncols_numeric, int categ_data[],
						 size_t ncols_categ, int ncat[],
						 real_t Xc[],
						 sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
						 size_t ndim, size_t ntry, CoefType coef_type,
						 bool coef_by_prop,
						 real_t sample_weights[],
						 bool with_replacement, bool weight_as_sample,
						 size_t nrows, size_t sample_size, size_t ntrees,
						 size_t max_depth, size_t ncols_per_tree,
						 bool limit_depth, bool penalize_range,
						 bool standardize_data, ScoringMetric scoring_metric,
						 bool fast_bratio, bool standardize_dist,
						 real_t tmat[], real_t output_depths[],
						 bool standardize_depth,
						 real_t col_weights[],
						 bool weigh_by_kurt, real_t prob_pick_by_gain_pl,
						 real_t prob_pick_by_gain_avg,
						 real_t prob_pick_by_full_gain,
						 real_t prob_pick_by_dens,
						 real_t prob_pick_col_by_range,
						 real_t prob_pick_col_by_var,
						 real_t prob_pick_col_by_kurt, real_t min_gain,
						 MissingAction missing_action,
						 CategSplit cat_split_type,
						 NewCategAction new_cat_action, bool all_perm,
						 Imputer *imputer, size_t min_imp_obs,
						 UseDepthImp depth_imp, WeighImpRows weigh_imp_rows,
						 bool impute_at_fit, uint64_t random_seed,
						 int nthreads)
	{
		// sanity checks:
		if (prob_pick_by_gain_avg < 0 || prob_pick_by_gain_pl < 0 || prob_pick_by_full_gain < 0 || prob_pick_by_dens < 0 || prob_pick_col_by_range < 0 || prob_pick_col_by_var < 0 || prob_pick_col_by_kurt < 0)
		{
			throw std::runtime_error("Cannot pass negative probabilities.\n");
		}
		if (prob_pick_col_by_range && ncols_categ)
			throw std::runtime_error(
				"'prob_pick_col_by_range' is not compatible with categorical data.\n");
		if (prob_pick_by_full_gain && ncols_categ)
			throw std::runtime_error(
				"'prob_pick_by_full_gain' is not compatible with categorical data.\n");
		if (prob_pick_col_by_kurt && weigh_by_kurt)
			throw std::runtime_error(
				"'weigh_by_kurt' and 'prob_pick_col_by_kurt' cannot be used together.\n");
		if (ndim == 0 && model_outputs == NULL)
			throw std::runtime_error(
				"Must pass 'ndim>0' in the extended model.\n");
		if (penalize_range && (scoring_metric == Density || scoring_metric == AdjDensity || is_boxed_metric(scoring_metric)))
			throw std::runtime_error(
				"'penalize_range' is incompatible with density scoring.\n");
		if (with_replacement)
		{
			if (tmat != NULL)
				throw std::runtime_error(
					"Cannot calculate distance while sampling with replacement.\n");
			if (output_depths != NULL)
				throw std::runtime_error(
					"Cannot make predictions at fit time when sampling with replacement.\n");
			if (impute_at_fit)
				throw std::runtime_error(
					"Cannot impute at fit time when sampling with replacement.\n");
		}
		if (sample_size != 0 && sample_size < nrows)
		{
			if (output_depths != NULL)
				throw std::runtime_error(
					"Cannot produce outlier scores at fit time when using sub-sampling.\n");
			if (tmat != NULL)
				throw std::runtime_error(
					"Cannot calculate distances at fit time when using sub-sampling.\n");
			if (impute_at_fit)
				throw std::runtime_error(
					"Cannot produce missing data imputations at fit time when using sub-sampling.\n");
		}

		/* TODO: this function should also accept the array as a memoryview with a
		 leading dimension that might not correspond to the number of columns,
		 so as to avoid having to make deep copies of memoryviews in python and to
		 allow using pointers to columns of dataframes in R and Python. */

		/* calculate maximum number of categories to use later */
		int max_categ = 0;
		for (size_t col = 0; col < ncols_categ; col++)
			max_categ = (ncat[col] > max_categ) ? ncat[col] : max_categ;

		bool calc_dist = tmat != NULL;

		if (sample_size == 0)
			sample_size = nrows;

		if (model_outputs != NULL)
			ntry = std::min(ntry, ncols_numeric + ncols_categ);

		if (ncols_per_tree == 0)
			ncols_per_tree = ncols_numeric + ncols_categ;

		/* put data in structs to shorten function calls */
		InputData input_data =
			{numeric_data, ncols_numeric, categ_data, ncat, max_categ, ncols_categ,
			 nrows, ncols_numeric + ncols_categ, sample_weights,
			 weight_as_sample, col_weights, Xc, Xc_ind, Xc_indptr, 0, 0,
			 std::vector<real_t>(), std::vector<char>(), 0, NULL,
			 (real_t *)NULL, (real_t *)NULL, (int *)NULL, std::vector<real_t>(),
			 std::vector<real_t>(), std::vector<real_t>(),
			 std::vector<size_t>(), std::vector<size_t>()};
		ModelParams model_params =
			{with_replacement, sample_size, ntrees, ncols_per_tree,
			 limit_depth ? log2ceil(sample_size) : max_depth ? max_depth
															 : (sample_size - 1),
			 penalize_range,
			 standardize_data, random_seed, weigh_by_kurt, prob_pick_by_gain_avg,
			 prob_pick_by_gain_pl, prob_pick_by_full_gain, prob_pick_by_dens,
			 prob_pick_col_by_range, prob_pick_col_by_var, prob_pick_col_by_kurt,
			 min_gain, cat_split_type, new_cat_action, missing_action,
			 scoring_metric, fast_bratio, all_perm,
			 (model_outputs != NULL) ? 0 : ndim, ntry, coef_type,
			 coef_by_prop, calc_dist, (bool)(output_depths != NULL),
			 impute_at_fit, depth_imp, weigh_imp_rows, min_imp_obs};

		/* if calculating full gain, need to produce copies of the data in row-major order */
		if (prob_pick_by_full_gain)
		{
			if (input_data.Xc_indptr == NULL)
				colmajor_to_rowmajor<real_t>(input_data.numeric_data,
											 input_data.nrows,
											 input_data.ncols_numeric,
											 input_data.X_row_major);
			else
				colmajor_to_rowmajor<real_t, sparse_ix>(input_data.Xc,
														input_data.Xc_ind,
														input_data.Xc_indptr,
														input_data.nrows,
														input_data.ncols_numeric,
														input_data.Xr,
														input_data.Xr_ind,
														input_data.Xr_indptr);
		}

		/* if using weights as sampling probability, build a binary tree for faster sampling */
		if (input_data.weight_as_sample && input_data.sample_weights != NULL)
		{
			build_btree_sampler(input_data.btree_weights_init,
								input_data.sample_weights, input_data.nrows,
								input_data.log2_n, input_data.btree_offset);
		}

		/* same for column weights */
		/* TODO: this should also save the kurtoses when using 'prob_pick_col_by_kurt' */
		column_sampler<lreal_t_safe> base_col_sampler;
		if (col_weights != NULL || (model_params.weigh_by_kurt && model_params.sample_size == input_data.nrows && !model_params.with_replacement && (model_params.ncols_per_tree >= input_data.ncols_tot / (model_params.ntrees * 2))))
		{
			bool avoid_col_weights = (model_outputs != NULL && model_params.ntry >= model_params.ncols_per_tree && model_params.prob_pick_by_gain_avg + model_params.prob_pick_by_gain_pl + model_params.prob_pick_by_full_gain + model_params.prob_pick_by_dens >= 1) || (model_outputs == NULL && model_params.ndim >= model_params.ncols_per_tree) || (model_params.ncols_per_tree == 1);
			if (!avoid_col_weights)
			{
				if (model_params.weigh_by_kurt && model_params.sample_size == input_data.nrows && !model_params.with_replacement)
				{
					RNG_engine rnd_generator(random_seed);
					std::vector<real_t> kurt_weights = calc_kurtosis_all_data<
						InputData, lreal_t_safe>(input_data, model_params,
												 rnd_generator);
					if (col_weights != NULL)
					{
						for (size_t col = 0; col < input_data.ncols_tot; col++)
						{
							if (kurt_weights[col] <= 0)
								continue;
							kurt_weights[col] *= col_weights[col];
							kurt_weights[col] = std::fmax(kurt_weights[col],
														  1e-100);
						}
					}
					base_col_sampler.initialize(kurt_weights.data(),
												input_data.ncols_tot);

					if (model_params.prob_pick_col_by_range || model_params.prob_pick_col_by_var)
					{
						input_data.all_kurtoses = std::move(kurt_weights);
					}
				}

				else
				{
					base_col_sampler.initialize(input_data.col_weights,
												input_data.ncols_tot);
				}

				input_data.preinitialized_col_sampler = &base_col_sampler;
			}
		}

		/* in some cases, all trees will need to calculate variable ranges for all columns */
		/* TODO: the model might use 'leave_m_cols', or have 'prob_pick_col_by_range<1', in which
		 case it might not be beneficial to do this beforehand. Find out when the expected gain
		 from doing this here is not beneficial. */
		/* TODO: move this to a different file, it doesn't belong here */
		std::vector<real_t> variable_ranges_low;
		std::vector<real_t> variable_ranges_high;
		std::vector<int> variable_ncats;
		if (model_params.sample_size == input_data.nrows && !model_params.with_replacement && (model_params.ncols_per_tree >= input_data.ncols_numeric) && ((model_params.prob_pick_col_by_range && input_data.ncols_numeric) || is_boxed_metric(model_params.scoring_metric)))
		{
			variable_ranges_low.resize(input_data.ncols_numeric);
			variable_ranges_high.resize(input_data.ncols_numeric);

			std::unique_ptr<unsigned char[]> buffer_cats;
			size_t adj_col;
			if (is_boxed_metric(model_params.scoring_metric))
			{
				variable_ncats.resize(input_data.ncols_categ);
				buffer_cats = std::unique_ptr<unsigned char[]>(
					new unsigned char[input_data.max_categ]);
			}

			if (base_col_sampler.col_indices.empty())
				base_col_sampler.initialize(input_data.ncols_tot);

			bool unsplittable;
			size_t n_tried_numeric = 0;
			size_t col;
			base_col_sampler.prepare_full_pass();
			while (base_col_sampler.sample_col(col))
			{
				if (col < input_data.ncols_numeric)
				{
					if (input_data.Xc_indptr == NULL)
					{
						get_range(input_data.numeric_data + nrows * col,
								  input_data.nrows, model_params.missing_action,
								  variable_ranges_low[col],
								  variable_ranges_high[col], unsplittable);
					}

					else
					{
						get_range(col, input_data.nrows, input_data.Xc,
								  input_data.Xc_ind, input_data.Xc_indptr,
								  model_params.missing_action,
								  variable_ranges_low[col],
								  variable_ranges_high[col], unsplittable);
					}

					n_tried_numeric++;

					if (unsplittable)
					{
						variable_ranges_low[col] = 0;
						variable_ranges_high[col] = 0;
						base_col_sampler.drop_col(col);
					}
				}

				else
				{
					if (!is_boxed_metric(model_params.scoring_metric))
					{
						if (n_tried_numeric >= input_data.ncols_numeric)
							break;
						else
							continue;
					}
					adj_col = col - input_data.ncols_numeric;

					variable_ncats[adj_col] = count_ncateg_in_col(
						input_data.categ_data + input_data.nrows * adj_col,
						input_data.nrows, input_data.ncat[adj_col],
						buffer_cats.get());
					if (variable_ncats[adj_col] <= 1)
						base_col_sampler.drop_col(col);
				}
			}

			input_data.preinitialized_col_sampler = &base_col_sampler;
			if (input_data.ncols_numeric)
			{
				input_data.range_low = variable_ranges_low.data();
				input_data.range_high = variable_ranges_high.data();
			}
			if (input_data.ncols_categ)
			{
				input_data.ncat_ = variable_ncats.data();
			}
		}

		/* if imputing missing values on-the-fly, need to determine which are missing */
		std::vector<ImputedData> impute_vec;
		hashed_map<size_t, ImputedData> impute_map;
		if (model_params.impute_at_fit)
			check_for_missing(input_data, impute_vec, impute_map, nthreads);

		/* store model data */
		if (model_outputs != NULL)
		{
			model_outputs->trees.resize(ntrees);
			model_outputs->trees.shrink_to_fit();
			model_outputs->new_cat_action = new_cat_action;
			model_outputs->cat_split_type = cat_split_type;
			model_outputs->missing_action = missing_action;
			model_outputs->scoring_metric = scoring_metric;
			if (model_outputs->scoring_metric != Density && model_outputs->scoring_metric != BoxedDensity && model_outputs->scoring_metric != BoxedDensity2 && model_outputs->scoring_metric != BoxedRatio)
				model_outputs->exp_avg_depth = expected_avg_depth<lreal_t_safe>(
					sample_size);
			else
				model_outputs->exp_avg_depth = 1;
			model_outputs->exp_avg_sep = expected_separation_depth(
				model_params.sample_size);
			model_outputs->orig_sample_size = input_data.nrows;
			model_outputs->has_range_penalty = penalize_range;
		}

		else if( model_outputs_ext != NULL )
			
		{
				
			model_outputs_ext->hplanes.resize(ntrees);
			model_outputs_ext->hplanes.shrink_to_fit();
			model_outputs_ext->new_cat_action = new_cat_action;
			model_outputs_ext->cat_split_type = cat_split_type;
			model_outputs_ext->missing_action = missing_action;
			model_outputs_ext->scoring_metric = scoring_metric;
			if (model_outputs_ext->scoring_metric != Density && model_outputs_ext->scoring_metric != BoxedDensity && model_outputs_ext->scoring_metric != BoxedDensity2 && model_outputs_ext->scoring_metric != BoxedRatio)
				model_outputs_ext->exp_avg_depth =
					expected_avg_depth<lreal_t_safe>(sample_size);
			else
				model_outputs_ext->exp_avg_depth = 1;
			model_outputs_ext->exp_avg_sep = expected_separation_depth(
				model_params.sample_size);
			model_outputs_ext->orig_sample_size = input_data.nrows;
			model_outputs_ext->has_range_penalty = penalize_range;
		}

		if (imputer != NULL)
			initialize_imputer<decltype(input_data), lreal_t_safe>(*imputer,
																   input_data,
																   ntrees,
																   nthreads);

		/* initialize thread-private memory */
		if ((size_t)nthreads > ntrees)
			nthreads = (int)ntrees;
#ifdef _OPENMP
		std::vector<WorkerMemory<ImputedData, lreal_t_safe, real_t>> worker_memory(nthreads);
#else
		std::vector<WorkerMemory<ImputedData, lreal_t_safe, real_t>> worker_memory(
			1);
#endif

		/* Global variable that determines if the procedure receives a stop signal */
		signal_switcher ss = signal_switcher();

		/* For exception handling */
		bool threw_exception = false;
		std::exception_ptr ex = NULL;

		/* grow trees */
		for (size_t tree = 0; tree < (decltype(tree))ntrees; tree++)
		{
			if (interrupt_switch || threw_exception)
				continue;

			try
			{
				if (model_params.impute_at_fit && input_data.n_missing && !worker_memory[omp_get_thread_num()].impute_vec.size() && !worker_memory[omp_get_thread_num()].impute_map.size())
				{
#ifdef _OPENMP
					if (nthreads > 1)
					{
						worker_memory[omp_get_thread_num()].impute_vec = impute_vec;
						worker_memory[omp_get_thread_num()].impute_map = impute_map;
					}

					else
#endif
					{
						worker_memory[0].impute_vec = std::move(impute_vec);
						worker_memory[0].impute_map = std::move(impute_map);
					}
				}

				//
				fit_itree<decltype(input_data),
						  typename std::remove_pointer<decltype(worker_memory.data())>::type,
						  lreal_t_safe>(
					(model_outputs != NULL) ? &model_outputs->trees[tree] : NULL,
					(model_outputs_ext != NULL) ? &model_outputs_ext->hplanes[tree] : NULL,
					worker_memory[omp_get_thread_num()], input_data, model_params,
					(imputer != NULL) ? &(imputer->imputer_tree[tree]) : NULL,
					tree);

				if ((model_outputs != NULL))
					model_outputs->trees[tree].shrink_to_fit();
				else
					model_outputs_ext->hplanes[tree].shrink_to_fit();
			}

			catch (...)
			{
				// #pragma omp critical section
				{
					if (!threw_exception)
					{
						threw_exception = true;
						ex = std::current_exception();
					}
				}
			}
		}

		/* check if the procedure got interrupted */
		check_interrupt_switch(ss);
#if defined(DONT_THROW_ON_INTERRUPT)
		if (interrupt_switch)
			return EXIT_FAILURE;
#endif

		/* check if some exception was thrown */
		if (threw_exception)
			std::rethrow_exception(ex);

		if ((model_outputs != NULL))
			model_outputs->trees.shrink_to_fit();
		else
			model_outputs_ext->hplanes.shrink_to_fit();

		/* if calculating similarity/distance, now need to reduce and average */
		if (calc_dist)
			gather_sim_result<PredictionData, InputData>(NULL, &worker_memory,
														 NULL,
														 &input_data,
														 model_outputs,
														 model_outputs_ext, tmat,
														 NULL,
														 0, model_params.ntrees,
														 false, standardize_dist,
														 false, nthreads);

		check_interrupt_switch(ss);
#if defined(DONT_THROW_ON_INTERRUPT)
		if (interrupt_switch)
			return EXIT_FAILURE;
#endif

		/* same for depths */
		if (output_depths != NULL)
		{
#ifdef _OPENMP
			if (nthreads > 1)
			{
				for (auto &w : worker_memory)
				{
					if (w.row_depths.size())
					{
#pragma omp parallel for schedule(static) num_threads(nthreads) shared(input_data, output_depths, w, worker_memory)
						for (size_t_for row = 0; row < (decltype(row))input_data.nrows; row++)
							output_depths[row] += w.row_depths[row];
					}
				}
			}
			else
#endif
			{
				std::copy(worker_memory[0].row_depths.begin(),
						  worker_memory[0].row_depths.end(), output_depths);
			}

			if (standardize_depth)
			{
				real_t depth_divisor = (real_t)ntrees * ((model_outputs != NULL) ? model_outputs->exp_avg_depth : model_outputs_ext->exp_avg_depth);
				for (size_t row = 0; row < nrows; row++)
					output_depths[row] = std::exp2(
						-output_depths[row] / depth_divisor);
			}

			else
			{
				real_t ntrees_dbl = (real_t)ntrees;
				for (size_t row = 0; row < nrows; row++)
					output_depths[row] /= ntrees_dbl;
			}
		}

		check_interrupt_switch(ss);
#if defined(DONT_THROW_ON_INTERRUPT)
		if (interrupt_switch)
			return EXIT_FAILURE;
#endif

		/* if imputing missing values, now need to reduce and write final values */
		if (model_params.impute_at_fit)
		{
#ifdef _OPENMP
			if (nthreads > 1)
			{
				for (auto &w : worker_memory)
					combine_tree_imputations(w, impute_vec, impute_map, input_data.has_missing, nthreads);
			}

			else
#endif
			{
				impute_vec = std::move(worker_memory[0].impute_vec);
				impute_map = std::move(worker_memory[0].impute_map);
			}

			apply_imputation_results(impute_vec, impute_map, *imputer,
									 input_data, nthreads);
		}

		check_interrupt_switch(ss);
#if defined(DONT_THROW_ON_INTERRUPT)
		if (interrupt_switch)
			return EXIT_FAILURE;
#endif

		return EXIT_SUCCESS;
	}

	template <class real_t_ = real_t, class lreal_t_safe = long real_t>
	real_t
	calc_kurtosis(size_t ix_arr[], size_t st, size_t end, real_t_ x[],
				  MissingAction missing_action);

	template <class real_t_ = real_t, class lreal_t_safe = long real_t>
	real_t
	calc_kurtosis(real_t_ x[], size_t n, MissingAction missing_action);

	template <class real_t_, class mapping, class lreal_t_safe = long real_t>
	real_t
	calc_kurtosis_weighted(size_t ix_arr[], size_t st, size_t end, real_t_ x[],
						   MissingAction missing_action, mapping &w);

	template <class real_t_ = real_t, class lreal_t_safe = long real_t>
	real_t
	calc_kurtosis_weighted(real_t_ *x, size_t n_, MissingAction missing_action,
						   real_t_ *w);

	template <class real_t_, class sparse_ix_, class lreal_t_safe>
	real_t
	calc_kurtosis(size_t col_num, size_t nrows, real_t_ Xc[],
				  sparse_ix_ *Xc_ind, sparse_ix_ *Xc_indptr,
				  MissingAction missing_action);
	template <class real_t_, class sparse_ix_, class mapping, class lreal_t_safe>
	real_t
	calc_kurtosis_weighted(size_t *ix_arr, size_t st, size_t end,
						   size_t col_num, real_t_ Xc[], sparse_ix_ *Xc_ind,
						   sparse_ix_ *Xc_indptr, MissingAction missing_action,
						   mapping &w);

	template <class real_t_ = real_t, class sparse_ix_ = sparse_ix,
			  class lreal_t_safe = long real_t>
	real_t
	calc_kurtosis_weighted(size_t col_num, size_t nrows, real_t_ *Xc,
						   sparse_ix_ *Xc_ind, sparse_ix_ *Xc_indptr,
						   MissingAction missing_action, real_t_ *w);

	template <class lreal_t_safe>
	real_t
	calc_kurtosis_internal(size_t cnt, int x[], int ncat, size_t buffer_cnt[],
						   real_t buffer_prob[], MissingAction missing_action,
						   CategSplit cat_split_type,
						   RNG_engine &rnd_generator);
	template <class lreal_t_safe>
	real_t
	calc_kurtosis(size_t *ix_arr, size_t st, size_t end, int x[], int ncat,
				  size_t *buffer_cnt, real_t buffer_prob[],
				  MissingAction missing_action, CategSplit cat_split_type,
				  RNG_engine &rnd_generator);
	template <class lreal_t_safe>
	real_t
	calc_kurtosis(size_t nrows, int x[], int ncat, size_t buffer_cnt[],
				  real_t buffer_prob[], MissingAction missing_action,
				  CategSplit cat_split_type, RNG_engine &rnd_generator);
	template <class mapping, class lreal_t_safe>
	real_t
	calc_kurtosis_weighted_internal(std::vector<lreal_t_safe> &buffer_cnt,
									int x[], int ncat, real_t buffer_prob[],
									MissingAction missing_action,
									CategSplit cat_split_type,
									RNG_engine &rnd_generator,
									mapping &w);
	template <class mapping, class lreal_t_safe>
	real_t
	calc_kurtosis_weighted(size_t ix_arr[], size_t st, size_t end, int x[],
						   int ncat, real_t buffer_prob[],
						   MissingAction missing_action,
						   CategSplit cat_split_type,
						   RNG_engine &rnd_generator,
						   mapping &w);

	template <class real_t_, class lreal_t_safe>
	real_t
	calc_kurtosis_weighted(size_t nrows, int x[], int ncat,
						   real_t *buffer_prob, MissingAction missing_action,
						   CategSplit cat_split_type,
						   RNG_engine &rnd_generator,
						   real_t_ *w);

	template <class InputData, class WorkerMemory, class lreal_t_safe>
	void
	calc_kurt_all_cols(InputData &input_data, WorkerMemory &workspace,
					   ModelParams &model_params, real_t *kurtosis,
					   real_t *saved_xmin, real_t *saved_xmax)
	{
		workspace.col_sampler.prepare_full_pass();
		while (workspace.col_sampler.sample_col(workspace.col_chosen))
		{
			if (saved_xmin != NULL)
			{
				get_split_range(workspace, input_data, model_params);
				if (workspace.unsplittable)
				{
					workspace.col_sampler.drop_col(workspace.col_chosen);
					continue;
				}

				if (saved_xmin != NULL)
				{
					saved_xmin[workspace.col_chosen] = workspace.xmin;
					saved_xmax[workspace.col_chosen] = workspace.xmax;
				}
			}

			if (workspace.col_chosen < input_data.ncols_numeric)
			{
				if (input_data.Xc_indptr == NULL)
				{
					if (workspace.weights_arr.empty() && workspace.weights_map.empty())
					{
						kurtosis[workspace.col_chosen] = calc_kurtosis<
							typename std::remove_pointer<
								decltype(input_data.numeric_data)>::type,
							lreal_t_safe>(
							workspace.ix_arr.data(),
							workspace.st,
							workspace.end,
							input_data.numeric_data + workspace.col_chosen * input_data.nrows,
							model_params.missing_action);
					}

					else if (!workspace.weights_arr.empty())
					{
						kurtosis[workspace.col_chosen] = calc_kurtosis_weighted<
							typename std::remove_pointer<
								decltype(input_data.numeric_data)>::type,
							decltype(workspace.weights_arr), lreal_t_safe>(
							workspace.ix_arr.data(),
							workspace.st,
							workspace.end,
							input_data.numeric_data + workspace.col_chosen * input_data.nrows,
							model_params.missing_action, workspace.weights_arr);
					}

					else
					{
						kurtosis[workspace.col_chosen] = calc_kurtosis_weighted<
							typename std::remove_pointer<
								decltype(input_data.numeric_data)>::type,
							decltype(workspace.weights_map), lreal_t_safe>(
							workspace.ix_arr.data(),
							workspace.st,
							workspace.end,
							input_data.numeric_data + workspace.col_chosen * input_data.nrows,
							model_params.missing_action, workspace.weights_map);
					}
				}

				else
				{
					if (workspace.weights_arr.empty() && workspace.weights_map.empty())
					{
						kurtosis[workspace.col_chosen] =
							calc_kurtosis<
								typename std::remove_pointer<
									decltype(input_data.Xc)>::type,
								typename std::remove_pointer<
									decltype(input_data.Xc_indptr)>::type,
								lreal_t_safe>(workspace.ix_arr.data(),
											  workspace.st, workspace.end,
											  workspace.col_chosen,
											  input_data.Xc, input_data.Xc_ind,
											  input_data.Xc_indptr,
											  model_params.missing_action);
					}

					else if (!workspace.weights_arr.empty())
					{
						kurtosis[workspace.col_chosen] =
							calc_kurtosis_weighted<
								typename std::remove_pointer<
									decltype(input_data.Xc)>::type,
								typename std::remove_pointer<
									decltype(input_data.Xc_indptr)>::type,
								decltype(workspace.weights_arr), lreal_t_safe>(
								workspace.ix_arr.data(), workspace.st,
								workspace.end, workspace.col_chosen,
								input_data.Xc, input_data.Xc_ind,
								input_data.Xc_indptr, model_params.missing_action,
								workspace.weights_arr);
					}

					else
					{
						kurtosis[workspace.col_chosen] =
							calc_kurtosis_weighted<
								typename std::remove_pointer<
									decltype(input_data.Xc)>::type,
								typename std::remove_pointer<
									decltype(input_data.Xc_indptr)>::type,
								decltype(workspace.weights_map), lreal_t_safe>(
								workspace.ix_arr.data(), workspace.st,
								workspace.end, workspace.col_chosen,
								input_data.Xc, input_data.Xc_ind,
								input_data.Xc_indptr, model_params.missing_action,
								workspace.weights_map);
					}
				}
				if (kurtosis[workspace.col_chosen] == -HUGE_VAL)
					workspace.col_sampler.drop_col(workspace.col_chosen);

				kurtosis[workspace.col_chosen] =
					(kurtosis[workspace.col_chosen] == -HUGE_VAL) ? 0. : std::fmax(1e-8, -1. + kurtosis[workspace.col_chosen]);
				if (input_data.col_weights != NULL && kurtosis[workspace.col_chosen] > 0)
				{
					kurtosis[workspace.col_chosen] *=
						input_data.col_weights[workspace.col_chosen];
					kurtosis[workspace.col_chosen] = std::fmax(
						kurtosis[workspace.col_chosen], 1e-100);
				}
			}
		}
	}

	int
	isolation_forest::fit_iforest(IsoForest *model_outputs,
								  ExtIsoForest *model_outputs_ext,
								  real_t numeric_data[],
								  size_t ncols_numeric, int categ_data[],
								  size_t ncols_categ, int ncat[],
								  real_t Xc[],
								  sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
								  size_t ndim, size_t ntry, CoefType coef_type,
								  bool coef_by_prop,
								  real_t sample_weights[],
								  bool with_replacement, bool weight_as_sample,
								  size_t nrows, size_t sample_size,
								  size_t ntrees, size_t max_depth,
								  size_t ncols_per_tree, bool limit_depth,
								  bool penalize_range, bool standardize_data,
								  ScoringMetric scoring_metric, bool fast_bratio,
								  bool standardize_dist, real_t tmat[],
								  real_t output_depths[], bool standardize_depth,
								  real_t col_weights[],
								  bool weigh_by_kurt,
								  real_t prob_pick_by_gain_pl,
								  real_t prob_pick_by_gain_avg,
								  real_t prob_pick_by_full_gain,
								  real_t prob_pick_by_dens,
								  real_t prob_pick_col_by_range,
								  real_t prob_pick_col_by_var,
								  real_t prob_pick_col_by_kurt, real_t min_gain,
								  MissingAction missing_action,
								  CategSplit cat_split_type,
								  NewCategAction new_cat_action, bool all_perm,
								  Imputer *imputer, size_t min_imp_obs,
								  UseDepthImp depth_imp,
								  WeighImpRows weigh_imp_rows,
								  bool impute_at_fit, uint64_t random_seed,
								  bool use_long_real_t, int nthreads)
	{
		if (!use_long_real_t)
			return fit_iforest_internal<real_t, sparse_ix, real_t>(
				model_outputs, model_outputs_ext, numeric_data, ncols_numeric,
				categ_data, ncols_categ, ncat, Xc, Xc_ind, Xc_indptr, ndim, ntry,
				coef_type, coef_by_prop, sample_weights, with_replacement,
				weight_as_sample, nrows, sample_size, ntrees, max_depth,
				ncols_per_tree, limit_depth, penalize_range, standardize_data,
				scoring_metric, fast_bratio, standardize_dist, tmat, output_depths,
				standardize_depth, col_weights, weigh_by_kurt, prob_pick_by_gain_pl,
				prob_pick_by_gain_avg, prob_pick_by_full_gain, prob_pick_by_dens,
				prob_pick_col_by_range, prob_pick_col_by_var, prob_pick_col_by_kurt,
				min_gain, missing_action, cat_split_type, new_cat_action, all_perm,
				imputer, min_imp_obs, depth_imp, weigh_imp_rows, impute_at_fit,
				random_seed, nthreads);

		return fit_iforest_internal<real_t, sparse_ix, long real_t>(
			model_outputs, model_outputs_ext, numeric_data, ncols_numeric,
			categ_data, ncols_categ, ncat, Xc, Xc_ind, Xc_indptr, ndim, ntry,
			coef_type, coef_by_prop, sample_weights, with_replacement,
			weight_as_sample, nrows, sample_size, ntrees, max_depth, ncols_per_tree,
			limit_depth, penalize_range, standardize_data, scoring_metric,
			fast_bratio, standardize_dist, tmat, output_depths, standardize_depth,
			col_weights, weigh_by_kurt, prob_pick_by_gain_pl, prob_pick_by_gain_avg,
			prob_pick_by_full_gain, prob_pick_by_dens, prob_pick_col_by_range,
			prob_pick_col_by_var, prob_pick_col_by_kurt, min_gain, missing_action,
			cat_split_type, new_cat_action, all_perm, imputer, min_imp_obs,
			depth_imp, weigh_imp_rows, impute_at_fit, random_seed, nthreads);
	}
	template <class ImputedData, class InputData>
	void
	check_for_missing(InputData &input_data,
					  std::vector<ImputedData> &impute_vec,
					  hashed_map<size_t, ImputedData> &impute_map,
					  int nthreads)

	{
		input_data.has_missing.assign(input_data.nrows, false);

		if (input_data.Xc_indptr != NULL)
		{
			for (size_t col = 0; col < input_data.ncols_numeric; col++)
				for (size_t ix = input_data.Xc_indptr[col];
					 ix < (decltype(ix))input_data.Xc_indptr[col + 1]; ix++)
					if (is_na_or_inf(input_data.Xc[ix]))
						input_data.has_missing[input_data.Xc_ind[ix]] = true;
		}

		if (input_data.numeric_data != NULL || input_data.categ_data != NULL)
		{
			for (size_t row = 0; row < (decltype(row))input_data.nrows; row++)
			{
				if (input_data.Xc_indptr == NULL)
				{
					for (size_t col = 0; col < input_data.ncols_numeric; col++)
					{
						if (is_na_or_inf(
								input_data.numeric_data[row + col * input_data.nrows]))
						{
							input_data.has_missing[row] = true;
							break;
						}
					}
				}

				if (!input_data.has_missing[row])
					for (size_t col = 0; col < input_data.ncols_categ; col++)
					{
						if (input_data.categ_data[row + col * input_data.nrows] < 0)
						{
							input_data.has_missing[row] = true;
							break;
						}
					}
			}
		}

		input_data.n_missing = std::accumulate(input_data.has_missing.begin(),
											   input_data.has_missing.end(),
											   (size_t)0);
		allocate_imp(input_data, impute_vec, impute_map, nthreads);
	}

	int
	fit_iforest(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
				real_t numeric_data[],
				size_t ncols_numeric, int categ_data[], size_t ncols_categ,
				int ncat[],
				real_t Xc[],
				sparse_ix Xc_ind[], sparse_ix Xc_indptr[], size_t ndim,
				size_t ntry, CoefType coef_type, bool coef_by_prop,
				real_t sample_weights[],
				bool with_replacement, bool weight_as_sample, size_t nrows,
				size_t sample_size, size_t ntrees, size_t max_depth,
				size_t ncols_per_tree, bool limit_depth, bool penalize_range,
				bool standardize_data, ScoringMetric scoring_metric,
				bool fast_bratio, bool standardize_dist, real_t tmat[],
				real_t output_depths[], bool standardize_depth,
				real_t col_weights[],
				bool weigh_by_kurt, real_t prob_pick_by_gain_pl,
				real_t prob_pick_by_gain_avg, real_t prob_pick_by_full_gain,
				real_t prob_pick_by_dens, real_t prob_pick_col_by_range,
				real_t prob_pick_col_by_var, real_t prob_pick_col_by_kurt,
				real_t min_gain, MissingAction missing_action,
				CategSplit cat_split_type, NewCategAction new_cat_action,
				bool all_perm, Imputer *imputer, size_t min_imp_obs,
				UseDepthImp depth_imp, WeighImpRows weigh_imp_rows,
				bool impute_at_fit, uint64_t random_seed, bool use_long_real_t,
				int nthreads)
	{
		return fit_iforest(model_outputs, model_outputs_ext, numeric_data,
						   ncols_numeric, categ_data, ncols_categ, ncat, Xc,
						   Xc_ind, Xc_indptr, ndim, ntry, coef_type, coef_by_prop,
						   sample_weights, with_replacement, weight_as_sample,
						   nrows, sample_size, ntrees, max_depth, ncols_per_tree,
						   limit_depth, penalize_range, standardize_data,
						   scoring_metric, fast_bratio, standardize_dist, tmat,
						   output_depths, standardize_depth, col_weights,
						   weigh_by_kurt, prob_pick_by_gain_pl,
						   prob_pick_by_gain_avg, prob_pick_by_full_gain,
						   prob_pick_by_dens, prob_pick_col_by_range,
						   prob_pick_col_by_var, prob_pick_col_by_kurt, min_gain,
						   missing_action, cat_split_type, new_cat_action,
						   all_perm, imputer, min_imp_obs, depth_imp,
						   weigh_imp_rows, impute_at_fit, random_seed,
						   use_long_real_t, nthreads);
	}
	int
	add_tree(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
			 real_t numeric_data[],
			 size_t ncols_numeric, int categ_data[], size_t ncols_categ,
			 int ncat[],
			 real_t Xc[],
			 sparse_ix Xc_ind[], sparse_ix Xc_indptr[], size_t ndim, size_t ntry,
			 CoefType coef_type, bool coef_by_prop,
			 real_t sample_weights[],
			 size_t nrows, size_t max_depth, size_t ncols_per_tree,
			 bool limit_depth, bool penalize_range, bool standardize_data,
			 bool fast_bratio,
			 real_t col_weights[],
			 bool weigh_by_kurt, real_t prob_pick_by_gain_pl,
			 real_t prob_pick_by_gain_avg, real_t prob_pick_by_full_gain,
			 real_t prob_pick_by_dens, real_t prob_pick_col_by_range,
			 real_t prob_pick_col_by_var, real_t prob_pick_col_by_kurt,
			 real_t min_gain, MissingAction missing_action,
			 CategSplit cat_split_type, NewCategAction new_cat_action,
			 UseDepthImp depth_imp, WeighImpRows weigh_imp_rows, bool all_perm,
			 Imputer *imputer, size_t min_imp_obs, TreesIndexer *indexer,
			 real_t ref_numeric_data[],
			 int ref_categ_data[], bool ref_is_col_major, size_t ref_ld_numeric,
			 size_t ref_ld_categ,
			 real_t ref_Xc[],
			 sparse_ix ref_Xc_ind[], sparse_ix ref_Xc_indptr[],
			 uint64_t random_seed, bool use_long_real_t)
	{
		return add_tree(model_outputs, model_outputs_ext, numeric_data,
						ncols_numeric, categ_data, ncols_categ, ncat, Xc, Xc_ind,
						Xc_indptr, ndim, ntry, coef_type, coef_by_prop,
						sample_weights, nrows, max_depth, ncols_per_tree,
						limit_depth, penalize_range, standardize_data, fast_bratio,
						col_weights, weigh_by_kurt, prob_pick_by_gain_pl,
						prob_pick_by_gain_avg, prob_pick_by_full_gain,
						prob_pick_by_dens, prob_pick_col_by_range,
						prob_pick_col_by_var, prob_pick_col_by_kurt, min_gain,
						missing_action, cat_split_type, new_cat_action, depth_imp,
						weigh_imp_rows, all_perm, imputer, min_imp_obs, indexer,
						ref_numeric_data, ref_categ_data, ref_is_col_major,
						ref_ld_numeric, ref_ld_categ, ref_Xc, ref_Xc_ind,
						ref_Xc_indptr, random_seed, use_long_real_t);
	}
	void
	remap_terminal_trees(IsoForest *model_outputs,
						 ExtIsoForest *model_outputs_ext, PredictionData &data,
						 sparse_ix *tree_num,
						 int nthreads)
	{


		UNDEF_REFERENCE(nthreads)
		UNDEF_REFERENCE2(nthreads)
		UNDEF_REFERENCE2(nthreads)
		
		
		size_t ntrees =(model_outputs != NULL) ? model_outputs->trees.size() : model_outputs_ext->hplanes.size();
		size_t max_tree, curr_term;
		std::vector<sparse_ix> tree_mapping;
 		if (model_outputs != NULL)

		{
			max_tree = std::accumulate(
				model_outputs->trees.begin(), model_outputs->trees.end(),
				(size_t)0, [](const size_t curr_max, const std::vector<IsoTree> &tr)
				{ return std::max(curr_max, tr.size()); });
			tree_mapping.resize(max_tree);
			for (size_t tree = 0; tree < ntrees; tree++)
			{
				std::fill(tree_mapping.begin(), tree_mapping.end(), (size_t)0);
				curr_term = 0;
				for (size_t node = 0; node < model_outputs->trees[tree].size();
					 node++)
					if (model_outputs->trees[tree][node].tree_left == 0)
						tree_mapping[node] = curr_term++;
				for (size_t row = 0; row < (decltype(row))data.nrows; row++)
					tree_num[row + tree * data.nrows] = tree_mapping[tree_num[row + tree * data.nrows]];
			}
		}

		else
		{
			max_tree = std::accumulate(
				model_outputs_ext->hplanes.begin(),
				model_outputs_ext->hplanes.end(), (size_t)0, [](const size_t curr_max, const std::vector<IsoHPlane> &tr)
				{ return std::max(curr_max, tr.size()); });
			tree_mapping.resize(max_tree);
			for (size_t tree = 0; tree < ntrees; tree++)
			{
				std::fill(tree_mapping.begin(), tree_mapping.end(), (size_t)0);
				curr_term = 0;
				for (size_t node = 0;
					 node < model_outputs_ext->hplanes[tree].size(); node++)
					if (model_outputs_ext->hplanes[tree][node].hplane_left == 0)
						tree_mapping[node] = curr_term++;

				for (size_t row = 0; row < (decltype(row))data.nrows; row++)
					tree_num[row + tree * data.nrows] = tree_mapping[tree_num[row + tree * data.nrows]];
			}
		}
	}

	void
	predict_iforest(real_t *numeric_data, int *categ_data, bool is_col_major,
					size_t ld_numeric, size_t ld_categ,
					real_t *Xc,
					sparse_ix *Xc_ind, sparse_ix *Xc_indptr,
					real_t *Xr,
					sparse_ix *Xr_ind, sparse_ix *Xr_indptr, size_t nrows,
					int nthreads, bool standardize, IsoForest *model_outputs,
					ExtIsoForest *model_outputs_ext, real_t *output_depths,
					sparse_ix *tree_num,
					real_t *per_tree_depths, TreesIndexer *indexer)
	{
		if (unlikely(!nrows))
			return;

		/* put data in a struct for passing it in fewer lines */
		PredictionData data =
			{numeric_data, categ_data, nrows, is_col_major, ld_numeric, ld_categ, Xc,
			 Xc_ind, Xc_indptr, Xr, Xr_ind, Xr_indptr};

		int nthreads_orig = nthreads;
		if ((size_t)nthreads > nrows)
			nthreads = nrows;

		/* For batch predictions of sparse CSC, will take a specialized route */
		if (data.Xc_indptr != NULL && (data.categ_data == NULL || data.is_col_major))
		{
			batched_csc_predict(data, nthreads_orig, model_outputs,
								model_outputs_ext, output_depths, tree_num,
								per_tree_depths);
		}

		/* Regular case (no specialized CSC route) */
		else if (model_outputs != NULL)
		{
			if (model_outputs->missing_action == Fail && (model_outputs->new_cat_action != Weighted || model_outputs->cat_split_type == SingleCateg || data.categ_data == NULL) && data.Xc_indptr == NULL && data.Xr_indptr == NULL && !model_outputs->has_range_penalty)
			{
				if (data.categ_data == NULL && (nrows == 1 || !data.is_col_major))
				{
#ifdef OPENMP_
#pragma omp parallel for if (nrows > 1) schedule(static) num_threads(nthreads) \
	shared(nrows, model_outputs, prediction_data, output_depths, tree_num, per_tree_depths)
#endif
					for (size_t row = 0; row < (decltype(row))nrows; row++)
					{
						real_t score = 0;
						for (size_t tree = 0; tree < model_outputs->trees.size();
							 tree++)
						{
							traverse_itree_fast(
								model_outputs->trees[tree],
								*model_outputs,
								data.numeric_data + row * data.ncols_numeric,
								score,
								(tree_num == NULL) ? NULL : (tree_num + nrows * tree),
								(per_tree_depths == NULL) ? NULL : (per_tree_depths + tree + row * model_outputs->trees.size()),
								(size_t)row);
						}
						output_depths[row] = score;
					}
				}
				else
				{
#ifdef OPENMP_
#pragma omp parallel for if (nrows > 1) schedule(static) num_threads(nthreads) \
	shared(nrows, model_outputs, prediction_data, output_depths, tree_num, per_tree_depths)
#endif
					for (size_t row = 0; row < (decltype(row))nrows; row++)
					{
						real_t score = 0;
						for (size_t tree = 0; tree < model_outputs->trees.size();
							 tree++)
						{
							traverse_itree_no_recurse(
								model_outputs->trees[tree],
								*model_outputs,
								data,
								score,
								(tree_num == NULL) ? NULL : (tree_num + nrows * tree),
								(per_tree_depths == NULL) ? NULL : (per_tree_depths + tree + row * model_outputs->trees.size()),
								(size_t)row);
						}
						output_depths[row] = score;
					}
				}
			}

			else
			{
#ifdef OPENMP_
#pragma omp parallel for if (nrows > 1) schedule(static) num_threads(nthreads) \
	shared(nrows, model_outputs, prediction_data, output_depths, tree_num, per_tree_depths)
#endif
				for (size_t row = 0; row < (decltype(row))nrows; row++)
				{
					real_t score = 0;
					for (size_t tree = 0; tree < model_outputs->trees.size();
						 tree++)
					{
						score += traverse_itree(
							model_outputs->trees[tree],
							*model_outputs,
							data,
							(std::vector<ImputeNode> *)NULL,
							(ImputedData *)NULL,
							(real_t)0,
							(size_t)row,
							(tree_num == NULL) ? NULL : (tree_num + nrows * tree),
							(per_tree_depths == NULL) ? NULL : (per_tree_depths + tree + row * model_outputs->trees.size()),
							(size_t)0);
					}
					output_depths[row] = score;
				}
			}
		}

		else
		{
			if (model_outputs_ext->missing_action == Fail && data.categ_data == NULL && data.Xc_indptr == NULL && data.Xr_indptr == NULL && !model_outputs_ext->has_range_penalty)
			{
				if (data.is_col_major && nrows > 1)
				{
#ifdef OPENMP_
#pragma omp parallel for if (nrows > 1) schedule(static) num_threads(nthreads) \
	shared(nrows, model_outputs_ext, prediction_data, output_depths, tree_num, per_tree_depths)
#endif
					for (size_t row = 0; row < (decltype(row))nrows; row++)
					{
						real_t score = 0;
						for (size_t tree = 0;
							 tree < model_outputs_ext->hplanes.size(); tree++)
						{
							traverse_hplane_fast_colmajor(
								model_outputs_ext->hplanes[tree],
								*model_outputs_ext,
								data,
								score,
								(tree_num == NULL) ? NULL : (tree_num + nrows * tree),
								(per_tree_depths == NULL) ? NULL : (per_tree_depths + tree + row * model_outputs_ext->hplanes.size()),
								(size_t)row);
						}
						output_depths[row] = score;
					}
				}

				else
				{
#ifdef OPENMP_
#pragma omp parallel for if (nrows > 1) schedule(static) num_threads(nthreads) \
	shared(nrows, model_outputs_ext, prediction_data, output_depths, tree_num, per_tree_depths)
#endif
					for (size_t row = 0; row < (decltype(row))nrows; row++)
					{
						real_t score = 0;
						for (size_t tree = 0;
							 tree < model_outputs_ext->hplanes.size(); tree++)
						{
							traverse_hplane_fast_rowmajor(
								model_outputs_ext->hplanes[tree],
								*model_outputs_ext,
								data.numeric_data + row * data.ncols_numeric,
								score,
								(tree_num == NULL) ? NULL : (tree_num + nrows * tree),
								(per_tree_depths == NULL) ? NULL : (per_tree_depths + tree + row * model_outputs_ext->hplanes.size()),
								(size_t)row);
						}
						output_depths[row] = score;
					}
				}
			}
			else
			{
#ifdef OPENMP_
#pragma omp parallel for if (nrows > 1) schedule(static) num_threads(nthreads) \
	shared(nrows, model_outputs_ext, prediction_data, output_depths, tree_num, per_tree_depths)
#endif
				for (size_t row = 0; row < (decltype(row))nrows; row++)
				{
					real_t score = 0;
					for (size_t tree = 0; tree < model_outputs_ext->hplanes.size();
						 tree++)
					{
						traverse_hplane(
							model_outputs_ext->hplanes[tree],
							*model_outputs_ext,
							data,
							score,
							(std::vector<ImputeNode> *)NULL,
							(ImputedData *)NULL,
							(tree_num == NULL) ? NULL : (tree_num + nrows * tree),
							(per_tree_depths == NULL) ? NULL : (per_tree_depths + tree + row * model_outputs_ext->hplanes.size()),
							(size_t)row);
					}
					output_depths[row] = score;
				}
			}
		}
		/* translate sum-of-depths to outlier score */
		real_t ntrees, depth_divisor;
		if (model_outputs != NULL)
		{
			ntrees = (real_t)model_outputs->trees.size();
			depth_divisor = ntrees * (model_outputs->exp_avg_depth);
		}

		else
		{
			ntrees = (real_t)model_outputs_ext->hplanes.size();
			depth_divisor = ntrees * (model_outputs_ext->exp_avg_depth);
		}

		/* for density and boxed_ratio, each tree will have 'log(d)'' instead of 'd' */
		bool is_density = (model_outputs != NULL && model_outputs->scoring_metric == Density) || (model_outputs_ext != NULL && model_outputs_ext->scoring_metric == Density);
		bool is_bratio = (model_outputs != NULL && model_outputs->scoring_metric == BoxedRatio) || (model_outputs_ext != NULL && model_outputs_ext->scoring_metric == BoxedRatio);
		bool is_bdens = (model_outputs != NULL && model_outputs->scoring_metric == BoxedDensity) || (model_outputs_ext != NULL && model_outputs_ext->scoring_metric == BoxedDensity);
		bool is_bdens2 = (model_outputs != NULL && model_outputs->scoring_metric == BoxedDensity2) || (model_outputs_ext != NULL && model_outputs_ext->scoring_metric == BoxedDensity2);

		if (standardize)
		{
			if (is_density || is_bdens2)
			{
				ntrees = -ntrees;
				for (size_t row = 0; row < nrows; row++)
					output_depths[row] /= ntrees;
			}

			else if (is_bdens)
			{
#ifdef OPENMP_
#pragma omp simd

#endif
				for (size_t row = 0; row < nrows; row++)
					output_depths[row] = -std::exp(output_depths[row] / ntrees);
			}

			else if (is_bratio)
			{
				for (size_t row = 0; row < nrows; row++)
					output_depths[row] = output_depths[row] / ntrees;
			}

			else
			{
#ifdef OPENMP_
#pragma omp simd
#endif
				for (size_t row = 0; row < nrows; row++)
					output_depths[row] = std::exp2(
						-output_depths[row] / depth_divisor);
			}
		}

		else
		{
			if (is_density || is_bdens || is_bdens2)
			{
#ifdef OPENMP_
#pragma omp simd
#endif
				for (size_t row = 0; row < nrows; row++)
					output_depths[row] = std::exp(output_depths[row] / ntrees);
			}

			else if (is_bratio)
			{
				ntrees = -ntrees;
				for (size_t row = 0; row < nrows; row++)
					output_depths[row] /= ntrees;
			}

			else
			{
				for (size_t row = 0; row < nrows; row++)
					output_depths[row] /= ntrees;
			}
		}

		if (per_tree_depths != NULL && (is_density || is_bdens || is_bdens2))
		{
			size_t ntrees =
				(model_outputs != NULL) ? model_outputs->trees.size() : model_outputs_ext->hplanes.size();
#ifdef OPENMP_
#pragma omp simd
#endif
			for (size_t ix = 0; ix < nrows * ntrees; ix++)
				per_tree_depths[ix] = std::exp(per_tree_depths[ix]);
		}
		/* re-map tree numbers to start at zero (if predicting tree numbers) */
		/* Note: usually this type of 'prediction' is not required,
		 thus this mapping is not stored in the model objects so as to
		 save memory */
		if (tree_num != NULL)
		{
			if (indexer != NULL && !indexer->indices.empty())
			{
				size_t ntrees =
					(model_outputs != NULL) ? model_outputs->trees.size() : model_outputs_ext->hplanes.size();
				if (model_outputs != NULL)
				{
					if (model_outputs->missing_action == Divide)
						goto manual_remap;
					if (model_outputs->new_cat_action == Weighted && model_outputs->cat_split_type == SubSet && categ_data != NULL)
						goto manual_remap;
				}

				for (size_t tree = 0; tree < ntrees; tree++)
				{
					size_t *mapping =
						indexer->indices[tree].terminal_node_mappings.data();
					for (size_t row = 0; row < nrows; row++)
					{
						tree_num[row + tree * nrows] = mapping[tree_num[row + tree * nrows]];
					}
				}
			}

			else
			{
			manual_remap:
				remap_terminal_trees(model_outputs, model_outputs_ext, data,
									 tree_num, nthreads);
			}
		}
	}
	// #if 0
	void
	check_interrupt_switch(signal_switcher &ss)
	{
		if (interrupt_switch)
		{
			ss.restore_handle();
			fprintf(stderr, "Error: procedure was interrupted\n");
			raise(SIGINT);
#ifdef _FOR_R
			// Rcpp::checkUserInterrupt();

#elif !defined(DONT_THROW_ON_INTERRUPT)
			throw std::runtime_error("Error: procedure was interrupted.\n");
#endif
		}
	}
	// #endif
	void
	kernel_to_references(TreesIndexer &indexer, IsoForest *model_outputs,
						 ExtIsoForest *model_outputs_ext,
						 real_t *numeric_data,
						 int *categ_data,
						 real_t *Xc,
						 sparse_ix *Xc_ind, sparse_ix *Xc_indptr,
						 bool is_col_major, size_t ld_numeric, size_t ld_categ,
						 size_t nrows, int nthreads, real_t *rmat,
						 bool standardize)
	{
		size_t ntrees = indexer.indices.size();
		size_t n_ref = indexer.indices.front().reference_points.size();

		signal_switcher ss;

		std::unique_ptr<sparse_ix[]> terminal_indices(
			new sparse_ix[nrows * ntrees]);
		std::unique_ptr<real_t[]> ignored(new real_t[nrows]);
		predict_iforest(numeric_data, categ_data, is_col_major, ld_numeric,
						ld_categ, is_col_major ? Xc : nullptr,
						is_col_major ? Xc_ind : nullptr,
						is_col_major ? Xc_indptr : nullptr,
						is_col_major ? (real_t *)nullptr : Xc,
						is_col_major ? (sparse_ix *)nullptr : Xc_ind,
						is_col_major ? (sparse_ix *)nullptr : Xc_indptr, nrows,
						nthreads, false, model_outputs, model_outputs_ext,
						ignored.get(), terminal_indices.get(), (real_t *)NULL,
						&indexer);
		ignored.reset();
		check_interrupt_switch(ss);
#ifdef OPENMP_
#pragma omp parallel for schedule(static) num_threads(nthreads) \
	shared(indexer, terminal_indices, nrows, ntrees, n_ref, rmat)
#endif
		for (size_t row = 0; row < (decltype(row))nrows; row++)
		{
			if (interrupt_switch)
				continue;

			SingleTreeIndex *index_node;
			size_t idx_this;
			sparse_ix *terminal_indices_this = terminal_indices.get() + row;
			real_t *rmat_this = rmat + row * n_ref;
			memset(rmat_this, 0, n_ref * sizeof(real_t));

			for (size_t tree = 0; tree < ntrees; tree++)
			{
				idx_this = terminal_indices_this[tree * nrows];
				index_node = &indexer.indices[tree];
				for (size_t ind = index_node->reference_indptr[idx_this];
					 ind < index_node->reference_indptr[idx_this + 1]; ind++)
				{
					rmat_this[index_node->reference_mapping[ind]]++;
				}
			}
		}

		check_interrupt_switch(ss);

		if (standardize)
		{
			real_t ntrees_dbl = (real_t)ntrees;
			for (size_t ix = 0; ix < nrows * n_ref; ix++)
				rmat[ix] /= ntrees_dbl;
		}

		check_interrupt_switch(ss);
	}

	template <class PredictionData>
	void
	initialize_worker_for_sim(WorkerForSimilarity &workspace,
							  PredictionData &data, IsoForest *model_outputs,
							  ExtIsoForest *model_outputs_ext, size_t n_from,
							  bool assume_full_distr)
	{
		workspace.st = 0;
		workspace.end = data.nrows - 1;
		workspace.n_from = n_from;
		workspace.assume_full_distr = assume_full_distr; /* doesn't need to have one copy per worker */

		if (workspace.ix_arr.empty())
		{
			workspace.ix_arr.resize(data.nrows);
			std::iota(workspace.ix_arr.begin(), workspace.ix_arr.end(),
					  (size_t)0);
			if (!n_from)
				workspace.tmat_sep.resize(calc_ncomb(data.nrows), 0);
			else
				workspace.rmat.resize((data.nrows - n_from) * n_from, 0);
		}

		if (model_outputs != NULL && (model_outputs->missing_action == Divide || (model_outputs->new_cat_action == Weighted && model_outputs->cat_split_type == SubSet && data.categ_data != NULL)))
		{
			if (workspace.weights_arr.empty())
				workspace.weights_arr.resize(data.nrows, 1.);
			else
				std::fill(workspace.weights_arr.begin(),
						  workspace.weights_arr.end(), 1.);
		}

		if (model_outputs_ext != NULL)
		{
			if (workspace.comb_val.empty())
				workspace.comb_val.resize(data.nrows, 0);
			else
				std::fill(workspace.comb_val.begin(), workspace.comb_val.end(),
						  0);
		}
	}
	template <class PredictionData = PredictionData,
			  class lreal_t_safe = long real_t>
	void
	traverse_hplane_sim(WorkerForSimilarity &workspace,
						PredictionData &prediction_data,
						ExtIsoForest &model_outputs,
						std::vector<IsoHPlane> &hplanes, size_t curr_tree,
						const bool as_kernel)
	{
		if (interrupt_switch)
			return;

		if (workspace.st == workspace.end)
			return;

		if (workspace.tmat_sep.empty())
		{
			std::sort(workspace.ix_arr.begin() + workspace.st,
					  workspace.ix_arr.begin() + workspace.end + 1);
			if (workspace.ix_arr[workspace.st] >= workspace.n_from)
				return;
			if (workspace.ix_arr[workspace.end] < workspace.n_from)
				return;
		}
		if (hplanes[curr_tree].hplane_left == 0)
		{
			if (!as_kernel)
			{
				if (!workspace.tmat_sep.empty())
					increase_comb_counter(
						workspace.ix_arr.data(),
						workspace.st,
						workspace.end,
						prediction_data.nrows,
						workspace.tmat_sep.data(),
						workspace.assume_full_distr ? 3. : expected_separation_depth((lreal_t_safe)hplanes[curr_tree].remainder + (lreal_t_safe)(workspace.end - workspace.st + 1)));
				else if (!workspace.rmat.empty())
					increase_comb_counter_in_groups(
						workspace.ix_arr.data(),
						workspace.st,
						workspace.end,
						workspace.n_from,
						prediction_data.nrows,
						workspace.rmat.data(),
						workspace.assume_full_distr ? 3. : expected_separation_depth((lreal_t_safe)hplanes[curr_tree].remainder + (lreal_t_safe)(workspace.end - workspace.st + 1)));
			}
			else
			{
				if (!workspace.tmat_sep.empty())
				{
					size_t i_, j_;
					for (size_t i = workspace.st; i < workspace.end; i++)
					{
						i_ = workspace.ix_arr[i];
						for (size_t j = i + 1; j <= workspace.end; j++)
						{
							j_ = workspace.ix_arr[j];
							workspace.tmat_sep[ix_comb(i_, j_,
													   prediction_data.nrows,
													   workspace.tmat_sep.size())]++;
						}
					}
				}

				else if (!workspace.rmat.empty())
				{
					size_t n_group = std::distance(
						workspace.ix_arr.begin() + workspace.st,
						std::lower_bound(
							workspace.ix_arr.begin() + workspace.st,
							workspace.ix_arr.begin() + workspace.end + 1,
							workspace.n_from));
					real_t *rmat_this;
					for (size_t i = workspace.st; i < workspace.st + n_group; i++)
					{
						rmat_this = workspace.rmat.data() + workspace.ix_arr[i] * workspace.n_from;
						for (size_t j = workspace.st + n_group;
							 j <= workspace.end; j++)
						{
							rmat_this[workspace.ix_arr[j] - workspace.n_from]++;
						}
					}
				}
			}
			return;
		}

		else if (curr_tree > 0 && !as_kernel)
		{
			if (!workspace.tmat_sep.empty())
				increase_comb_counter(workspace.ix_arr.data(), workspace.st,
									  workspace.end, prediction_data.nrows,
									  workspace.tmat_sep.data(), -1.);
			else if (!workspace.rmat.empty())
				increase_comb_counter_in_groups(workspace.ix_arr.data(),
												workspace.st, workspace.end,
												workspace.n_from,
												prediction_data.nrows,
												workspace.rmat.data(), -1.);
		}

		if (prediction_data.Xc_indptr != NULL && workspace.tmat_sep.size())
			std::sort(workspace.ix_arr.begin() + workspace.st,
					  workspace.ix_arr.begin() + workspace.end + 1);

		/* reconstruct linear combination */
		size_t ncols_numeric = 0;
		size_t ncols_categ = 0;
		std::fill(
			workspace.comb_val.begin(),
			workspace.comb_val.begin() + (workspace.end - workspace.st + 1), 0);
		real_t unused;
		if (prediction_data.categ_data != NULL || prediction_data.Xc_indptr != NULL)
		{
			for (size_t col = 0; col < hplanes[curr_tree].col_num.size(); col++)
			{
				switch (hplanes[curr_tree].col_type[col])
				{
				case Numeric:
				{
					if (prediction_data.Xc_indptr == NULL)
						add_linear_comb(
							workspace.ix_arr.data(),
							workspace.st,
							workspace.end,
							workspace.comb_val.data(),
							prediction_data.numeric_data + prediction_data.nrows * hplanes[curr_tree].col_num[col],
							hplanes[curr_tree].coef[ncols_numeric],
							(real_t)0,
							hplanes[curr_tree].mean[ncols_numeric],
							(model_outputs.missing_action == Fail) ? unused : hplanes[curr_tree].fill_val[col],
							model_outputs.missing_action, NULL, NULL, false);
					else
						add_linear_comb(
							workspace.ix_arr.data(),
							workspace.st,
							workspace.end,
							hplanes[curr_tree].col_num[col],
							workspace.comb_val.data(),
							prediction_data.Xc,
							prediction_data.Xc_ind,
							prediction_data.Xc_indptr,
							hplanes[curr_tree].coef[ncols_numeric],
							(real_t)0,
							hplanes[curr_tree].mean[ncols_numeric],
							(model_outputs.missing_action == Fail) ? unused : hplanes[curr_tree].fill_val[col],
							model_outputs.missing_action, NULL, NULL, false);
					ncols_numeric++;
					break;
				}

				case Categorical:
				{
					switch (model_outputs.cat_split_type)
					{
					case SingleCateg:
					{
						add_linear_comb<lreal_t_safe>(
							workspace.ix_arr.data(),
							workspace.st,
							workspace.end,
							workspace.comb_val.data(),
							prediction_data.categ_data + prediction_data.nrows * hplanes[curr_tree].col_num[col],
							(int)0,
							NULL,
							hplanes[curr_tree].fill_new[ncols_categ],
							hplanes[curr_tree].chosen_cat[ncols_categ],
							(model_outputs.missing_action == Fail) ? unused : hplanes[curr_tree].fill_val[col],
							workspace.comb_val[0], NULL, NULL,
							model_outputs.new_cat_action,
							model_outputs.missing_action, SingleCateg, false);
						break;
					}

					case SubSet:
					{
						add_linear_comb<lreal_t_safe>(
							workspace.ix_arr.data(),
							workspace.st,
							workspace.end,
							workspace.comb_val.data(),
							prediction_data.categ_data + prediction_data.nrows * hplanes[curr_tree].col_num[col],
							(int)hplanes[curr_tree].cat_coef[ncols_categ].size(),
							hplanes[curr_tree].cat_coef[ncols_categ].data(),
							(real_t)0,
							(int)0,
							(model_outputs.missing_action == Fail) ? unused : hplanes[curr_tree].fill_val[col],
							hplanes[curr_tree].fill_new[ncols_categ], NULL,
							NULL,
							model_outputs.new_cat_action,
							model_outputs.missing_action, SubSet, false);
						break;
					}
					}
					ncols_categ++;
					break;
				}

				default:
				{
					assert(0);
					break;
				}
				}
			}
		}
		else /* faster version for numerical-only */
		{
			for (size_t col = 0; col < hplanes[curr_tree].col_num.size(); col++)
				add_linear_comb(
					workspace.ix_arr.data(),
					workspace.st,
					workspace.end,
					workspace.comb_val.data(),
					prediction_data.numeric_data + prediction_data.nrows * hplanes[curr_tree].col_num[col],
					hplanes[curr_tree].coef[col],
					(real_t)0,
					hplanes[curr_tree].mean[col],
					(model_outputs.missing_action == Fail) ? unused : hplanes[curr_tree].fill_val[col],
					model_outputs.missing_action, NULL, NULL, false);
		}

		/* divide data */
		size_t split_ix = divide_subset_split(workspace.ix_arr.data(),
											  workspace.comb_val.data(),
											  workspace.st, workspace.end,
											  hplanes[curr_tree].split_point);

		/* continue splitting recursively */
		size_t orig_end = workspace.end;
		if (split_ix > workspace.st)
		{
			workspace.end = split_ix - 1;
			traverse_hplane_sim<PredictionData, lreal_t_safe>(
				workspace, prediction_data, model_outputs, hplanes,
				hplanes[curr_tree].hplane_left, as_kernel);
		}

		if (split_ix <= orig_end)
		{
			workspace.st = split_ix;
			workspace.end = orig_end;
			traverse_hplane_sim<PredictionData, lreal_t_safe>(
				workspace, prediction_data, model_outputs, hplanes,
				hplanes[curr_tree].hplane_right, as_kernel);
		}
	}

	void
	calc_similarity_from_indexer(
		real_t *numeric_data,
		int *categ_data,
		real_t *Xc,
		sparse_ix *Xc_ind, sparse_ix *Xc_indptr,
		size_t nrows, int nthreads,
		bool assume_full_distr, bool standardize_dist,
		IsoForest *model_outputs,
		ExtIsoForest *model_outputs_ext, real_t *tmat,
		real_t *rmat, size_t n_from,
		TreesIndexer *indexer, bool is_col_major,
		size_t ld_numeric, size_t ld_categ)
	{
		UNDEF_REFERENCE(numeric_data)
		UNDEF_REFERENCE2(categ_data)
		UNDEF_REFERENCE2(Xc)
		UNDEF_REFERENCE2(Xc_indptr)
		UNDEF_REFERENCE2(nrows)
		UNDEF_REFERENCE2(nthreads)

		UNDEF_REFERENCE2(model_outputs)
		UNDEF_REFERENCE2(n_from)
		UNDEF_REFERENCE2(is_col_major)
		UNDEF_REFERENCE2(ld_categ)
		UNDEF_REFERENCE2(standardize_dist)
		UNDEF_REFERENCE2(assume_full_distr)
		UNDEF_REFERENCE2(Xc_ind)

		UNDEF_REFERENCE2(model_outputs_ext)
		UNDEF_REFERENCE2(indexer)
		UNDEF_REFERENCE2(ld_numeric)
		UNDEF_REFERENCE2(rmat)
		UNDEF_REFERENCE2(tmat)
		
		signal_switcher ss;
		//start:
		if(indexer!=nullptr)
		{
			//

		}

	}

	template <class PredictionData, class lreal_t_safe>
	void
	traverse_tree_sim(WorkerForSimilarity &workspace,
					  PredictionData &prediction_data,
					  IsoForest &model_outputs, std::vector<IsoTree> &trees,
					  size_t curr_tree, const bool as_kernel)
	{
		
		UNDEF_REFERENCE(prediction_data)
		UNDEF_REFERENCE2(model_outputs)
		UNDEF_REFERENCE2(trees)
		UNDEF_REFERENCE2(curr_tree)
		UNDEF_REFERENCE2(as_kernel)
	
		
		if (interrupt_switch)
			return;

		if (workspace.st == workspace.end)
			return;

		if (workspace.tmat_sep.empty())
		{
			std::sort(workspace.ix_arr.begin() + workspace.st,
					  workspace.ix_arr.begin() + workspace.end + 1);
			if (workspace.ix_arr[workspace.st] >= workspace.n_from)
				return;
			if (workspace.ix_arr[workspace.end] < workspace.n_from)
				return;
		}
		// TODO: IMPL.
	}

	void
	calc_similarity_from_indexer_with_references(
		real_t *numeric_data,
		int *categ_data,
		real_t *Xc,
		sparse_ix *Xc_ind,
		sparse_ix *Xc_indptr,
		size_t nrows, int nthreads,
		bool standardize_dist,
		IsoForest *model_outputs,
		ExtIsoForest *model_outputs_ext,
		real_t *rmat,
		TreesIndexer *indexer,
		bool is_col_major,
		size_t ld_numeric,
		size_t ld_categ)
	{
		size_t n_ref = get_number_of_reference_points(*indexer);
		if (unlikely(!n_ref))
			unexpected_error();
		signal_switcher ss;

		size_t ntrees =
			(model_outputs != NULL) ? model_outputs->trees.size() : model_outputs_ext->hplanes.size();
		std::vector<sparse_ix> terminal_indices(nrows * ntrees);
		std::unique_ptr<real_t[]> ignored(new real_t[nrows]);
		predict_iforest(numeric_data, categ_data, is_col_major, ld_numeric,
						ld_categ, is_col_major ? Xc : nullptr,
						is_col_major ? Xc_ind : nullptr,
						is_col_major ? Xc_indptr : nullptr,
						is_col_major ? (real_t *)nullptr : Xc,
						is_col_major ? (sparse_ix *)nullptr : Xc_ind,
						is_col_major ? (sparse_ix *)nullptr : Xc_indptr, nrows,
						nthreads, false, model_outputs, model_outputs_ext,
						ignored.get(), terminal_indices.data(), (real_t *)NULL,
						indexer);
		ignored.reset();

#ifndef _OPENMP
		nthreads = 1;
#endif

		check_interrupt_switch(ss);

		for (size_t row = 0; row < (decltype(row))nrows; row++)
		{
			if (interrupt_switch)
				continue;

			size_t i, j;
			size_t n_terminal_this;
			size_t ncomb_this;
			size_t *ref_this;
			sparse_ix *ind_this;
			real_t *node_depths_this;
			real_t *node_dist_this;
			real_t *rmat_this = rmat + row * n_ref;
			memset(rmat_this, 0, n_ref * sizeof(real_t));
			for (size_t tree = 0; tree < ntrees; tree++)
			{
				ref_this = indexer->indices[tree].reference_points.data();
				ind_this = terminal_indices.data() + tree * nrows;
				node_depths_this = indexer->indices[tree].node_depths.data();
				n_terminal_this = indexer->indices[tree].n_terminal;
				node_dist_this = indexer->indices[tree].node_distances.data();
				ncomb_this = calc_ncomb(n_terminal_this);
				for (size_t ref = 0; ref < n_ref; ref++)
				{
					i = ind_this[row];
					j = ref_this[ref];

					if (unlikely(i == j))
						rmat_this[ref] += node_depths_this[i] + 3.;
					else
						rmat_this[ref] += node_dist_this[ix_comb(i, j,
																 n_terminal_this,
																 ncomb_this)];
				}
			}
		}
		check_interrupt_switch(ss);

		size_t size_rmat = nrows * n_ref;
		if (standardize_dist)
		{
			real_t ntrees_dbl = (real_t)ntrees;
			real_t div_trees = (real_t)(mult2(ntrees));
			for (size_t ix = 0; ix < size_rmat; ix++)
				rmat[ix] = std::exp2(-(rmat[ix] - ntrees_dbl) / div_trees);
		}

		else
		{
			real_t div_trees = (real_t)ntrees;
			for (size_t ix = 0; ix < size_rmat; ix++)
				rmat[ix] /= div_trees;
		}

		check_interrupt_switch(ss);
	}

	template <typename lreal_t_safe>
	void
	calc_similarity_internal(
		real_t numeric_data[],
		int categ_data[],
		real_t Xc[],
		sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
		size_t nrows, int nthreads,
		bool assume_full_distr, bool standardize_dist,
		bool as_kernel, IsoForest *model_outputs,
		ExtIsoForest *model_outputs_ext, real_t tmat[],
		real_t rmat[], size_t n_from,
		bool use_indexed_references,
		TreesIndexer *indexer, bool is_col_major,
		size_t ld_numeric, size_t ld_categ)
	{
		if (nrows < 2 && (!use_indexed_references || indexer == NULL || indexer->indices.empty() || indexer->indices.front().reference_points.empty()))
			throw std::runtime_error(
				"Cannot calculate distances from less than 2 rows.\n");
		if (as_kernel && (tmat != NULL || !use_indexed_references || (indexer != NULL && !indexer->indices.empty() && indexer->indices.front().reference_points.empty())))
			indexer = NULL;

		if (indexer != NULL && model_outputs != NULL)
		{
			if (model_outputs->missing_action == Divide)
			{
				indexer = NULL;
				if (use_indexed_references)
					throw std::runtime_error(
						"Invalid indexer - cannot use references from it.\n");
			}
			if (model_outputs->new_cat_action == Weighted && model_outputs->cat_split_type == SubSet && categ_data != NULL)
			{
				indexer = NULL;
				if (use_indexed_references)
					throw std::runtime_error(
						"Invalid indexer - cannot use references from it.\n");
			}
		}
		if (!as_kernel && indexer != NULL && (indexer->indices.empty() || indexer->indices.front().node_distances.empty()))
		{
			if (use_indexed_references && !indexer->indices.empty() && !indexer->indices.front().reference_points.empty())
				throw std::runtime_error(
					"Indexer was built without distances. Cannot use references from it.\n");
			else
			{
				indexer = NULL;
				fprintf(
					stderr,
					"Indexer has no pre-computed distances, will not be used for distance calculations.\n");
			}
		}
		if (!is_col_major && indexer == NULL && (Xc_indptr != NULL || (nrows != 1 && ((numeric_data != NULL && ld_numeric > 1) || (categ_data != NULL && ld_categ > 1)))))
			throw std::runtime_error(
				"Cannot calculate distances with row-major data without indexer.\n");
		if (indexer != NULL)
		{
			if (use_indexed_references && tmat == NULL && !indexer->indices.empty() && !indexer->indices.front().reference_points.empty())
			{
				if (unlikely(!assume_full_distr))
					throw std::runtime_error(
						"Cannot calculate distances to reference points in indexer with 'assume_full_distr=false'.\n");

				if (!as_kernel)
				{
					calc_similarity_from_indexer_with_references(
						numeric_data, categ_data, Xc, Xc_ind, Xc_indptr, nrows,
						nthreads, standardize_dist, model_outputs,
						model_outputs_ext, rmat, indexer, is_col_major,
						ld_numeric, ld_categ);
				}

				else
				{
					kernel_to_references(*indexer, model_outputs,
										 model_outputs_ext, numeric_data,
										 categ_data, Xc, Xc_ind, Xc_indptr,
										 is_col_major, ld_numeric, ld_categ,
										 nrows, nthreads, rmat,
										 standardize_dist);
				}
			}

			else
			{
				if (as_kernel)
					goto skip_indexer_if_kernel;
				calc_similarity_from_indexer(numeric_data, categ_data, Xc,
											 Xc_ind, Xc_indptr, nrows, nthreads,
											 assume_full_distr, standardize_dist,
											 model_outputs, model_outputs_ext,
											 tmat, rmat, n_from, indexer,
											 is_col_major, ld_numeric, ld_categ);
			}

			return;
		}
	skip_indexer_if_kernel:

		PredictionData data =
			{numeric_data, categ_data, nrows, false, 0, 0, Xc, Xc_ind, Xc_indptr,
			 NULL, NULL, NULL};

		size_t ntrees =
			(model_outputs != NULL) ? model_outputs->trees.size() : model_outputs_ext->hplanes.size();

		if (tmat != NULL)
			n_from = 0;

		if (n_from == 0)
		{
#if SIZE_MAX == UINT32_MAX
			size_t lim_rows = (size_t)UINT16_MAX - (size_t)1;
#elif SIZE_MAX == UINT64_MAX
			size_t lim_rows = (size_t)UINT32_MAX - (size_t)1;
#else
			size_t lim_rows = (size_t)std::ceil(std::sqrt((lreal_t_safe)SIZE_MAX));
#endif
			if (nrows > lim_rows)
				throw std::runtime_error(
					"Number of rows implies too large distance matrix (integer overflow).");
		}

		if ((size_t)nthreads > ntrees)
			nthreads = (int)ntrees;
#ifdef _OPENMP
		std::vector<WorkerForSimilarity> worker_memory(nthreads);
#else
		std::vector<WorkerForSimilarity> worker_memory(1);
		nthreads = 1;
#endif
		/* Global variable that determines if the procedure receives a stop signal */
		signal_switcher ss = signal_switcher();
		check_interrupt_switch(ss);
#if defined(DONT_THROW_ON_INTERRUPT)
		if (interrupt_switch)
			return;
#endif
		/* For handling exceptions */
		bool threw_exception = false;
		std::exception_ptr ex = NULL;

		if (tmat == NULL && use_indexed_references && indexer != NULL && !indexer->indices.empty() && !indexer->indices.front().reference_points.empty() && (as_kernel || !indexer->indices.front().node_distances.empty()))
		{
			n_from = indexer->indices.front().reference_points.size();
		}

		if (model_outputs != NULL)
		{
			for (size_t tree = 0; tree < (decltype(tree))ntrees; tree++)
			{
				if (threw_exception || interrupt_switch)
					continue;
				try
				{
					initialize_worker_for_sim(
						worker_memory[omp_get_thread_num()], data, model_outputs,
						NULL,
						n_from, assume_full_distr);
					traverse_tree_sim<PredictionData, lreal_t_safe>(
						worker_memory[omp_get_thread_num()], data, *model_outputs,
						model_outputs->trees[tree], (size_t)0, as_kernel);
				}

				catch (...)
				{

					{
						if (!threw_exception)
						{
							threw_exception = true;
							ex = std::current_exception();
						}
					}
				}
			}
		}
		else
		{
			for (size_t hplane = 0; hplane < (decltype(hplane))ntrees; hplane++)
			{
				if (threw_exception || interrupt_switch)
					continue;
				try
				{
					initialize_worker_for_sim(
						worker_memory[omp_get_thread_num()], data,
						NULL,
						model_outputs_ext, n_from, assume_full_distr);
					traverse_hplane_sim<PredictionData, lreal_t_safe>(
						worker_memory[omp_get_thread_num()], data,
						*model_outputs_ext, model_outputs_ext->hplanes[hplane],
						(size_t)0, as_kernel);
				}

				catch (...)
				{
					{
						if (!threw_exception)
						{
							threw_exception = true;
							ex = std::current_exception();
						}
					}
				}
			}
		}

		check_interrupt_switch(ss);
#if defined(DONT_THROW_ON_INTERRUPT)
		if (interrupt_switch)
			return;
#endif

		if (threw_exception)
			std::rethrow_exception(ex);

		/* gather and transform the results */
		gather_sim_result<PredictionData, InputData,
						  WorkerMemory<ImputedData, lreal_t_safe, real_t>>(&worker_memory,
																		   NULL,
																		   &data, NULL,
																		   model_outputs,
																		   model_outputs_ext,
																		   tmat, rmat, n_from,
																		   ntrees,
																		   assume_full_distr,
																		   standardize_dist,
																		   as_kernel,
																		   nthreads);

		check_interrupt_switch(ss);
#if defined(DONT_THROW_ON_INTERRUPT)
		if (interrupt_switch)
			return;
#endif
	}

	void
	calc_similarity(real_t numeric_data[], int categ_data[],
					real_t Xc[],
					sparse_ix Xc_ind[], sparse_ix Xc_indptr[], size_t nrows,
					bool use_long_real_t, int nthreads, bool assume_full_distr,
					bool standardize_dist, bool as_kernel,
					IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
					real_t tmat[], real_t rmat[], size_t n_from,
					bool use_indexed_references, TreesIndexer *indexer,
					bool is_col_major, size_t ld_numeric, size_t ld_categ)
	{
		if (use_long_real_t && !has_long_real_t())
		{
			use_long_real_t = false;
			fprintf(
				stderr,
				"Passed 'use_long_real_t=true', but library was compiled without long real_t support.\n");
		}
#ifndef NO_LONG_DOUBLE
		if (likely(!use_long_real_t))
#endif
			calc_similarity_internal<real_t>(numeric_data, categ_data, Xc, Xc_ind,
											 Xc_indptr, nrows, nthreads,
											 assume_full_distr, standardize_dist,
											 as_kernel, model_outputs,
											 model_outputs_ext, tmat, rmat, n_from,
											 use_indexed_references, indexer,
											 is_col_major, ld_numeric, ld_categ);
#ifndef NO_LONG_DOUBLE
		else
			calc_similarity_internal<long real_t>(numeric_data, categ_data, Xc,
												  Xc_ind, Xc_indptr, nrows, nthreads,
												  assume_full_distr,
												  standardize_dist, as_kernel,
												  model_outputs, model_outputs_ext,
												  tmat, rmat, n_from,
												  use_indexed_references, indexer,
												  is_col_major, ld_numeric,
												  ld_categ);
#endif
	}
	void
	set_reference_points(IsoForest *model_outputs,
						 ExtIsoForest *model_outputs_ext, TreesIndexer *indexer,
						 const bool with_distances,
						 real_t *numeric_data,
						 int *categ_data, bool is_col_major, size_t ld_numeric,
						 size_t ld_categ,
						 real_t *Xc,
						 sparse_ix *Xc_ind, sparse_ix *Xc_indptr,
						 real_t *Xr,
						 sparse_ix *Xr_ind, sparse_ix *Xr_indptr, size_t nrows,
						 int nthreads)
	{
		set_reference_points(model_outputs, model_outputs_ext, indexer,
							 with_distances, numeric_data, categ_data,
							 is_col_major, ld_numeric, ld_categ, Xc, Xc_ind,
							 Xc_indptr, Xr, Xr_ind, Xr_indptr, nrows, nthreads);
	}

	static inline bool
	is_terminal_node(const IsoHPlane &node)
	{
		return node.hplane_left == 0;
	}
	static inline bool
	is_terminal_node(const IsoTree &node)
	{
		return node.tree_left == 0;
	}

	void
	build_tree_indices(TreesIndexer &indexer, const IsoForest &model,
					   int nthreads, const bool with_distances)
	{
		if (model.trees.empty())
			throw std::runtime_error("Cannot build indexed for unfitted model.\n");
		if (model.missing_action == Divide)
			throw std::runtime_error(
				"Cannot build tree indexer with 'missing_action=Divide'.\n");
		if (model.new_cat_action == Weighted && model.cat_split_type == SubSet)
		{
			for (const std::vector<IsoTree> &tree : model.trees)
			{
				for (const IsoTree &node : tree)
				{
					if (!is_terminal_node(node) && node.col_type == Categorical)
						throw std::runtime_error(
							"Cannot build tree indexer with 'new_cat_action=Weighted'.\n");
				}
			}
		}

		build_tree_indices<IsoForest>(indexer, model, nthreads, with_distances);
	}

	void
	build_tree_indices(TreesIndexer &indexer, const ExtIsoForest &model,
					   int nthreads, const bool with_distances)
	{
		if (model.hplanes.empty())
			throw std::runtime_error("Cannot build indexed for unfitted model.\n");
		build_tree_indices<ExtIsoForest>(indexer, model, nthreads, with_distances);
	}
	/* comment out similarity
	inline void
	calc_similarity(real_t numeric_data[], int categ_data[], real_t Xc[],
					int Xc_ind[], int Xc_indptr[], size_t nrows,
					bool use_long_real_t, int nthreads, bool assume_full_distr,
					bool standardize_dist, bool as_kernel,
					IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
					real_t tmat[], real_t rmat[], size_t n_from,
					bool use_indexed_references, TreesIndexer *indexer,
					bool is_col_major, size_t ld_numeric, size_t ld_categ)
	{

		if( model_outputs == NULL && model_outputs_ext == NULL )
			throw std::runtime_error("Must pass at least one model.\n");
		
		sparse_ix *Xc_indptr_ = (sparse_ix *)Xc_indptr; 
		sparse_ix *Xc_ind_ = (sparse_ix *)Xc_ind;


		if( model_outputs != NULL )
			{

				if( as_kernel )
				{
					if( standardize_dist )
						throw std::runtime_error("Cannot standardize kernel distances.\n");
					if( assume_full_distr )
						throw std::runtime_error("Cannot assume full distribution for kernel distances.\n");
				}
			}	
		if( model_outputs_ext != NULL )
			{
				if( as_kernel )
					throw std::runtime_error("Cannot use kernel distances with extended model.\n");
				if( standardize_dist )
					throw std::runtime_error("Cannot standardize extended model distances.\n");
				if( assume_full_distr )
					throw std::runtime_error("Cannot assume full distribution for extended model distances.\n");
			}
		
		if( use_indexed_references && indexer == NULL )
			throw std::runtime_error("Must pass a reference to a TreesIndexer object.\n");
		
		if(use_long_real_t)
		//instantiate similarity template parameters 
 		calc_similarity_internal<long real_t>(numeric_data, categ_data, Xc, Xc_ind_,
											 Xc_indptr_, nrows, nthreads,
											 assume_full_distr, standardize_dist,
											 as_kernel, model_outputs,
											 model_outputs_ext, tmat, rmat, n_from,
											 use_indexed_references, indexer,
											 is_col_major, ld_numeric, ld_categ);
 		else
 		calc_similarity_internal<real_t>(numeric_data, categ_data, Xc, Xc_ind_,
											 Xc_indptr_, nrows, nthreads,
											 assume_full_distr, standardize_dist,
											 as_kernel, model_outputs,
											 model_outputs_ext, tmat, rmat, n_from,
											 use_indexed_references, indexer,
											 is_col_major, ld_numeric, ld_categ);



 	}// calc_similarity
	comment out similarity*/
	
	size_t
	get_number_of_reference_points(const TreesIndexer &indexer) noexcept
	{
		if (indexer.indices.empty())
			return 0;
		return indexer.indices.front().reference_points.size();
	}

    std::ostream &operator<<(std::ostream &out, const roc_curve &q)
    {
		//out << "roc_curve:\n";

		const std::vector<std::pair<real_t, real_t>>& points = q.get_points();
		for (size_t ix = 0; ix < points.size(); ix++)
		{
			out <<"("<< points[ix].first << "," << points[ix].second << ")\n";
		}
		out << "AUC:" <<q._auc << "\n";
		return out;

    }

    // indexer :
	template <class Node>
	void
	build_dindex_recursive(const size_t curr_node, const size_t n_terminal,
						   const size_t ncomb, const size_t st,
						   const size_t end, std::vector<size_t> &node_indices, /* array with all terminal indices in 'tree' */
						   const std::vector<size_t> &node_mappings,			/* tree_index : terminal_index */
						   std::vector<real_t> &node_distances,					/* indexed by terminal_index */
						   std::vector<real_t> &node_depths,					/* indexed by terminal_index */
						   size_t curr_depth, const std::vector<Node> &tree)
	{
		if (end > st)
		{
			size_t i, j;
			for (size_t el1 = st; el1 < end; el1++)
			{
				for (size_t el2 = el1 + 1; el2 <= end; el2++)
				{
					i = node_mappings[node_indices[el1]];
					j = node_mappings[node_indices[el2]];
					node_distances[ix_comb(i, j, n_terminal, ncomb)]++;
				}
			}
		}

		if (!is_terminal_node(tree[curr_node]))
		{
			const size_t delim = get_idx_tree_right(tree[curr_node]);
			size_t frontier = st;
			size_t temp;
			for (size_t ix = st; ix <= end; ix++)
			{
				if (node_indices[ix] < delim)
				{
					temp = node_indices[frontier];
					node_indices[frontier] = node_indices[ix];
					node_indices[ix] = temp;
					frontier++;
				}
			}
			if (unlikely(frontier == st))
				unexpected_error();
			curr_depth++;

			build_dindex_recursive<Node>(get_idx_tree_left(tree[curr_node]),
										 n_terminal, ncomb, st, frontier - 1,
										 node_indices, node_mappings,
										 node_distances, node_depths, curr_depth,
										 tree);
			build_dindex_recursive<Node>(get_idx_tree_right(tree[curr_node]),
										 n_terminal, ncomb, frontier, end,
										 node_indices, node_mappings,
										 node_distances, node_depths, curr_depth,
										 tree);
		}
		else
		{
			node_depths[node_mappings[curr_node]] = curr_depth;
		}
	}

	void
	build_ref_node(SingleTreeIndex &node)
	{
		node.reference_mapping.resize(node.reference_points.size());
		node.reference_mapping.shrink_to_fit();
		std::iota(node.reference_mapping.begin(), node.reference_mapping.end(),
				  (size_t)0);
		std::sort(
			node.reference_mapping.begin(), node.reference_mapping.end(), [&node](const size_t a, const size_t b)
			{ return node.reference_points[a] < node.reference_points[b]; });

		size_t n_terminal = node.n_terminal;
		node.reference_indptr.assign(n_terminal + 1, (size_t)0);
		node.reference_indptr.shrink_to_fit();

		std::vector<size_t>::iterator curr_begin = node.reference_mapping.begin();
		std::vector<size_t>::iterator new_begin;
		size_t curr_node;
		while (curr_begin != node.reference_mapping.end())
		{
			curr_node = node.reference_points[*curr_begin];
			new_begin = std::upper_bound(
				curr_begin, node.reference_mapping.end(), curr_node, [&node](const size_t a, const size_t b)
				{ return a < node.reference_points[b]; });
			node.reference_indptr[curr_node + 1] = std::distance(curr_begin,
																 new_begin);
			curr_begin = new_begin;
		}

		for (size_t ix = 1; ix < n_terminal; ix++)
			node.reference_indptr[ix + 1] += node.reference_indptr[ix];
	}

	template <class Node>
	void
	build_dindex(std::vector<size_t> &node_indices,		   /* empty, but correctly sized */
				 const std::vector<size_t> &node_mappings, /* tree_index : terminal_index */
				 std::vector<real_t> &node_distances,	   /* indexed by terminal_index */
				 std::vector<real_t> &node_depths,		   /* indexed by terminal_index */
				 const size_t n_terminal, const std::vector<Node> &tree)
	{
		if (tree.size() <= 1)
			return;

		std::fill(node_distances.begin(), node_distances.end(), 0.);

		node_indices.clear();
		for (size_t node = 0; node < tree.size(); node++)
		{
			if (is_terminal_node(tree[node]))
				node_indices.push_back(node);
		}

		node_depths.resize(n_terminal);

		build_dindex_recursive<Node>((size_t)0, node_indices.size(),
									 calc_ncomb(node_indices.size()), 0,
									 node_indices.size() - 1, node_indices,
									 node_mappings, node_distances, node_depths,
									 (size_t)0, tree);
	}

	void
	build_dindex(std::vector<size_t> &node_indices,		   /* empty, but correctly sized */
				 const std::vector<size_t> &node_mappings, /* tree_index : terminal_index */
				 std::vector<real_t> &node_distances,	   /* indexed by terminal_index */
				 std::vector<real_t> &node_depths,		   /* indexed by terminal_index */
				 const size_t n_terminal, const std::vector<IsoTree> &tree)
	{
		build_dindex<IsoTree>(node_indices, node_mappings, node_distances,
							  node_depths, n_terminal, tree);
	}

	template <class Model>
	void
	build_terminal_node_mappings(TreesIndexer &indexer, const Model &model)
	{
		indexer.indices.resize(model.ntrees());

		indexer.indices.shrink_to_fit();

		if (!indexer.indices.empty() && !indexer.indices.front().reference_points.empty())
		{
			for (auto &ind : indexer.indices)
			{
				ind.reference_points.clear();
				ind.reference_indptr.clear();
				ind.reference_mapping.clear();
			}
		}

		for (size_t tree = 0; tree < indexer.indices.size(); tree++)
		{
			build_terminal_node_mappings_single_tree(
				indexer.indices[tree].terminal_node_mappings,
				indexer.indices[tree].n_terminal, get_tree(model, tree));
		}
	}

	template <class Tree>
	void
	build_terminal_node_mappings_single_tree(std::vector<size_t> &mappings,
											 size_t &n_terminal,
											 const std::vector<Tree> &tree)
	{
		mappings.resize(tree.size());
		mappings.shrink_to_fit();
		std::fill(mappings.begin(), mappings.end(), (size_t)0);

		n_terminal = 0;
		for (size_t node = 0; node < tree.size(); node++)
		{
			if (is_terminal_node(tree[node]))
			{
				mappings[node] = n_terminal;
				n_terminal++;
			}
		}
	}

	void
	build_terminal_node_mappings_single_tree(std::vector<size_t> &mappings,
											 size_t &n_terminal,
											 const std::vector<IsoTree> &tree)
	{
		build_terminal_node_mappings_single_tree<IsoTree>(mappings, n_terminal,
														  tree);
	}

	static inline const std::vector<IsoTree> &
	get_tree(const IsoForest &model, size_t tree)
	{
		return model.trees[tree];
	}

	static inline const std::vector<IsoHPlane> &
	get_tree(const ExtIsoForest &model, size_t tree)
	{
		return model.hplanes[tree];
	}

	static inline size_t
	get_idx_tree_left(const IsoTree &node)
	{
		return node.tree_left;
	}

	static inline size_t
	get_idx_tree_left(const IsoHPlane &node)
	{
		return node.hplane_left;
	}

	static inline size_t
	get_idx_tree_right(const IsoTree &node)
	{
		return node.tree_right;
	}

	static inline size_t
	get_idx_tree_right(const IsoHPlane &node)
	{
		return node.hplane_right;
	}

	void
	build_tree_indices(TreesIndexer *indexer, const IsoForest *model_outputs,
					   const ExtIsoForest *model_outputs_ext, int nthreads,
					   const bool with_distances)
	{
		if (model_outputs != NULL)
			build_tree_indices(*indexer, *model_outputs, nthreads, with_distances);
		else
			build_tree_indices(*indexer, *model_outputs_ext, nthreads,
							   with_distances);
	}

	void
	tmat_to_dense(real_t *tmat, real_t *dmat, size_t n, real_t fill_diag)
	{
		size_t ncomb = calc_ncomb(n);
		for (size_t i = 0; i < (n - 1); i++)
		{
			for (size_t j = i + 1; j < n; j++)
			{
				// dmat[i + j * n] = dmat[j + i * n] = tmat[i * (n - (i+1)/2) + j - i - 1];
				dmat[i + j * n] = dmat[j + i * n] = tmat[ix_comb(i, j, n, ncomb)];
			}
		}
		for (size_t i = 0; i < n; i++)
			dmat[i + i * n] = fill_diag;
	}
	void
	batched_csc_predict(PredictionData &data, int nthreads,
						IsoForest *model_outputs,
						ExtIsoForest *model_outputs_ext, real_t *output_depths,
						sparse_ix *tree_num,
						real_t *per_tree_depths)
	{
#ifdef _OPENMP
		size_t ntrees = (model_outputs != NULL) ? model_outputs->trees.size() : model_outputs_ext->hplanes.size();
		if ((size_t)nthreads > ntrees)
			nthreads = ntrees;
#else
		nthreads = 1;
#endif
		std::vector<WorkerForPredictCSC> worker_memory(nthreads);

		bool threw_exception = false;
		std::exception_ptr ex = NULL;

		if (model_outputs != NULL)
		{

			for (size_t tree = 0;
				 tree < (decltype(tree))model_outputs->trees.size(); tree++)
			{
				if (threw_exception)
					continue;
				try
				{
					WorkerForPredictCSC *ptr_worker =
						&worker_memory[omp_get_thread_num()];
					if (!ptr_worker->depths.size())
					{
						ptr_worker->depths.resize(data.nrows);
						ptr_worker->ix_arr.resize(data.nrows);
						std::iota(ptr_worker->ix_arr.begin(),
								  ptr_worker->ix_arr.end(), (size_t)0);

						if (model_outputs->missing_action == Divide || (model_outputs->new_cat_action == Weighted && model_outputs->cat_split_type == SubSet && data.categ_data != NULL))
						{
							ptr_worker->weights_arr.resize(data.nrows);
						}
					}
					ptr_worker->st = 0;
					ptr_worker->end = data.nrows - 1;
					if (model_outputs->missing_action == Divide)
						std::fill(ptr_worker->weights_arr.begin(),
								  ptr_worker->weights_arr.end(), (real_t)1);
					traverse_itree_csc(
						*ptr_worker,
						model_outputs->trees[tree],
						*model_outputs,
						data,
						(tree_num == NULL) ? ((sparse_ix *)NULL) : (tree_num + tree * data.nrows),
						per_tree_depths, (size_t)0,
						model_outputs->has_range_penalty);
				}
				catch (...)
				{
					{
						if (!threw_exception)
						{
							threw_exception = true;
							ex = std::current_exception();
						}
					}
				}
			}
		}
		else
		{

			for (size_t tree = 0;
				 tree < (decltype(tree))model_outputs_ext->hplanes.size(); tree++)
			{
				if (threw_exception)
					continue;
				try
				{
					WorkerForPredictCSC *ptr_worker =
						&worker_memory[omp_get_thread_num()];
					if (!ptr_worker->depths.size())
					{
						ptr_worker->depths.resize(data.nrows);
						ptr_worker->comb_val.resize(data.nrows);
						ptr_worker->ix_arr.resize(data.nrows);
						std::iota(ptr_worker->ix_arr.begin(),
								  ptr_worker->ix_arr.end(), (size_t)0);
					}

					ptr_worker->st = 0;
					ptr_worker->end = data.nrows - 1;
					traverse_hplane_csc(
						*ptr_worker,
						model_outputs_ext->hplanes[tree],
						*model_outputs_ext,
						data,
						(tree_num == NULL) ? ((sparse_ix *)NULL) : (tree_num + tree * data.nrows),
						per_tree_depths, (size_t)0,
						model_outputs_ext->has_range_penalty);
				}
				catch (...)
				{
					{
						if (!threw_exception)
						{
							threw_exception = true;
							ex = std::current_exception();
						}
					}
				}
				if (threw_exception)

					std::rethrow_exception(ex);
			}
#ifdef _OPENMP
			if (nthreads <= 1)
#endif
			{
				std::copy(worker_memory.front().depths.begin(),
						  worker_memory.front().depths.end(), output_depths);
			}

#ifdef _OPENMP
			else
			{
				std::fill(output_depths, output_depths + prediction_data.nrows, (real_t)0);
				for (auto &workspace : worker_memory)
					if (workspace.depths.size())
#if !defined(_MSC_VER) && !defined(_WIN32)
#pragma omp simd
#endif
						for (size_t row = 0; row < prediction_data.nrows; row++)
							output_depths[row] += workspace.depths[row];
			}
#endif
		}
	}
	template <class lreal_t_safe>
	void
	impute_missing_values_internal(
		real_t numeric_data[],
		int categ_data[], bool is_col_major,
		real_t Xr[],
		sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
		size_t nrows, int nthreads,
		iso_forest *model_outputs,
		ext_iso_forest *model_outputs_ext,
		Imputer &imputer)
	{
		prediction_data data =
			{numeric_data, categ_data, nrows, is_col_major, imputer.ncols_numeric,
			 imputer.ncols_categ,
			 NULL, NULL, NULL, Xr, Xr_ind, Xr_indptr};

		std::vector<size_t> ix_arr(nrows);
		std::iota(ix_arr.begin(), ix_arr.end(), (size_t)0);

		size_t end = check_for_missing(data, imputer, ix_arr.data(), nthreads);
		if (end == 0)
			return;

		if ((size_t)nthreads > end)
			nthreads = (int)end;
#ifdef _OPENMP
		std::vector<ImputedData> imp_memory(nthreads);
#else
		std::vector<ImputedData> imp_memory(1);
#endif

		bool threw_exception = false;
		std::exception_ptr ex = NULL;

		if (model_outputs != NULL)
		{
			for (size_t row = 0; row < (decltype(row))end; row++)
			{
				if (threw_exception)
					continue;
				try
				{
					initialize_impute_calc(imp_memory[omp_get_thread_num()],
										   data, imputer, ix_arr[row]);

					for (std::vector<IsoTree> &tree : model_outputs->trees)
					{
						traverse_itree(
							tree,
							*model_outputs,
							data,
							&imputer.imputer_tree[&tree - &(model_outputs->trees[0])],
							&imp_memory[omp_get_thread_num()], (real_t)1,
							ix_arr[row], (sparse_ix *)NULL, (real_t *)NULL,
							(size_t)0);
					}

					apply_imputation_results(data,
											 imp_memory[omp_get_thread_num()],
											 imputer, (size_t)ix_arr[row]);
				}

				catch (...)
				{
					{
						if (!threw_exception)
						{
							threw_exception = true;
							ex = std::current_exception();
						}
					}
				}
			}
		}
		else
		{
			real_t temp;

			for (size_t row = 0; row < (decltype(row))end; row++)
			{
				if (threw_exception)
					continue;
				try
				{
					initialize_impute_calc(imp_memory[omp_get_thread_num()],
										   data, imputer, ix_arr[row]);

					for (std::vector<IsoHPlane> &hplane : model_outputs_ext->hplanes)
					{
						traverse_hplane(
							hplane,
							*model_outputs_ext,
							data,
							temp,
							&imputer.imputer_tree[&hplane - &(model_outputs_ext->hplanes[0])],
							&imp_memory[omp_get_thread_num()], (sparse_ix *)NULL,
							(real_t *)NULL, ix_arr[row]);
					}

					apply_imputation_results(data,
											 imp_memory[omp_get_thread_num()],
											 imputer, (size_t)ix_arr[row]);
				}

				catch (...)
				{
					{
						if (!threw_exception)
						{
							threw_exception = true;
							ex = std::current_exception();
						}
					}
				}
			}
		}

		if (threw_exception)
			std::rethrow_exception(ex);
	}

	void
	impute_missing_values(real_t numeric_data[], int categ_data[],
						  bool is_col_major,
						  real_t Xr[],
						  sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
						  size_t nrows, bool use_long_real_t, int nthreads,
						  IsoForest *model_outputs,
						  ExtIsoForest *model_outputs_ext, Imputer &imputer)
	{


		if( use_long_real_t)
		{
		impute_missing_values_internal<long real_t>(numeric_data, categ_data,
													is_col_major, Xr, Xr_ind,
													Xr_indptr, nrows, nthreads,
													model_outputs,
													model_outputs_ext, imputer);

		}
		else
		{
		impute_missing_values_internal<real_t>(
				numeric_data, categ_data, is_col_major,
				Xr, Xr_ind, Xr_indptr,
				nrows, nthreads,
				model_outputs, model_outputs_ext,
				imputer);
		}
		
	}

	signal_switcher::signal_switcher() : is_active(false)
	{

		if (!handle_is_locked)
		{
			handle_is_locked = true;
			interrupt_switch = false;
			this->old_sig = ::signal(SIGINT, set_interrup_global_variable);
			this->is_active = true;
		}
	}

	signal_switcher::~signal_switcher()
	{
#ifndef _FOR_PYTHON
		// #pragma omp critical
		{
			if (this->is_active && handle_is_locked)
				interrupt_switch = false;
		}
#endif
		this->restore_handle();
	}

	void
	signal_switcher::restore_handle()
	{
#ifdef OPENMP_
#pragma omp critical
#endif // OPENMP
		{
			if (this->is_active && handle_is_locked)
			{
				signal(SIGINT, this->old_sig);
				this->is_active = false;
				handle_is_locked = false;
			}
		}
	}
	void
	signal_switcher::set_interrup_global_variable(int s)
	{
#ifdef OPENMP_

#pragma omp critical
#endif
		{
			interrupt_switch = true;
		}
		UNDEF_REFERENCE(s);
		UNDEF_REFERENCE2(s);

	}
	bool signal_switcher::interrupt_switch = false;
	bool signal_switcher::handle_is_locked = false;

	/* redef

	 void isolation_forest::predict_distance(real_t Xc[], int Xc_ind[], int Xc_indptr[], int categ_data[],
	 size_t nrows,
	 bool as_kernel,
	 bool assume_full_distr, bool standardize,
	 bool triangular,
	 real_t dist_matrix[])
	 {
	 this->check_is_fitted();
	 this->check_nthreads();
	 std::vector<real_t> tmat(triangular? 0 : calc_ncomb(nrows));

	 calc_similarity((real_t*)nullptr, (int*)nullptr,
	 Xc, Xc_ind, Xc_indptr,
	 nrows, false, this->nthreads, assume_full_distr, standardize, as_kernel,
	 (!this->model.trees.empty())? &this->model : nullptr,
	 (!this->model_ext.hplanes.empty())? &this->model_ext : nullptr,
	 triangular? dist_matrix : tmat.data(),
	 (real_t*)nullptr, (size_t)0, false,
	 (!this->indexer.indices.empty())? &this->indexer : nullptr,
	 true, (size_t)0, (size_t)0);
	 if (!triangular) {
	 real_t diag_filler;
	 if (as_kernel) {
	 if (standardize)
	 diag_filler = 1.;
	 else
	 diag_filler = std::max(this->model.trees.size(), this->model_ext.hplanes.size());
	 }
	 else {
	 if (standardize)
	 diag_filler = 0;
	 else
	 diag_filler = std::numeric_limits<real_t>::infinity();
	 }
	 tmat_to_dense(tmat.data(), dist_matrix, nrows, diag_filler);
	 }
	 }




	 void isolation_forest::predict_distance(real_t numeric_data[], int categ_data[],
	 size_t nrows,
	 bool as_kernel,
	 bool assume_full_distr, bool standardize,
	 bool triangular,
	 real_t dist_matrix[])
	 {
	 this->check_is_fitted();
	 this->check_nthreads();
	 std::vector<real_t> tmat(triangular? 0 : calc_ncomb(nrows));

	 calc_similarity(numeric_data, categ_data,
	 (real_t*)nullptr, (int*)nullptr, (int*)nullptr,
	 nrows, false, this->nthreads, assume_full_distr, standardize, as_kernel,
	 (!this->model.trees.empty())? &this->model : nullptr,
	 (!this->model_ext.hplanes.empty())? &this->model_ext : nullptr,
	 triangular? dist_matrix : tmat.data(),
	 (real_t*)nullptr, (size_t)0, false,
	 (!this->indexer.indices.empty())? &this->indexer : nullptr,
	 true, (size_t)0, (size_t)0);
	 if (!triangular) {
	 real_t diag_filler;
	 if (as_kernel) {
	 if (standardize)
	 diag_filler = 1.;
	 else
	 diag_filler = std::max(this->model.trees.size(), this->model_ext.hplanes.size());
	 }
	 else {
	 if (standardize)
	 diag_filler = 0;
	 else
	 diag_filler = std::numeric_limits<real_t>::infinity();
	 }
	 tmat_to_dense(tmat.data(), dist_matrix, nrows, diag_filler);
	 }
	 }
	 std::vector<real_t> isolation_forest::predict_distance(real_t X[], size_t nrows,
	 bool as_kernel,
	 bool assume_full_distr, bool standardize,
	 bool triangular)

	 {
	 this->check_is_fitted();
	 this->check_nthreads();
	 std::vector<real_t> tmat(calc_ncomb(nrows));
	 std::vector<real_t> dmat(triangular? square(nrows) : 0);

	 calc_similarity(X, (int*)nullptr,
	 (real_t*)nullptr, (int*)nullptr, (int*)nullptr,
	 nrows, false, this->nthreads, assume_full_distr, standardize, as_kernel,
	 (!this->model.trees.empty())? &this->model : nullptr,
	 (!this->model_ext.hplanes.empty())? &this->model_ext : nullptr,
	 tmat.data(), (real_t*)nullptr, (size_t)0, false,
	 (!this->indexer.indices.empty())? &this->indexer : nullptr,
	 true, (size_t)0, (size_t)0);
	 if (!triangular) {
	 real_t diag_filler;
	 if (as_kernel) {
	 if (standardize)
	 diag_filler = 1.;
	 else
	 diag_filler = std::max(this->model.trees.size(), this->model_ext.hplanes.size());
	 }
	 else {
	 if (standardize)
	 diag_filler = 0;
	 else
	 diag_filler = std::numeric_limits<real_t>::infinity();
	 }
	 tmat_to_dense(tmat.data(), dmat.data(), nrows, diag_filler);
	 }
	 return (triangular? tmat : dmat);

	 }
	 */

	void
	isolation_forest::remap_terminal_trees(IsoForest *model_outputs,
										   ExtIsoForest *model_outputs_ext,
										   prediction_data &data,
										   sparse_ix *tree_num,
										   int nthreads)
	{
		UNDEF_REFERENCE(nthreads)
		UNDEF_REFERENCE2(nthreads)

		size_t ntrees =
			(model_outputs != NULL) ? model_outputs->trees.size() : model_outputs_ext->hplanes.size();
		size_t max_tree, curr_term;
		std::vector<sparse_ix> tree_mapping;
		if (model_outputs != NULL)
		{
			max_tree = std::accumulate(
				model_outputs->trees.begin(), model_outputs->trees.end(),
				(size_t)0, [](const size_t curr_max, const std::vector<IsoTree> &tr)
				{ return std::max(curr_max, tr.size()); });
			tree_mapping.resize(max_tree);
			for (size_t tree = 0; tree < ntrees; tree++)
			{
				std::fill(tree_mapping.begin(), tree_mapping.end(), (size_t)0);
				curr_term = 0;
				for (size_t node = 0; node < model_outputs->trees[tree].size();
					 node++)
					if (model_outputs->trees[tree][node].tree_left == 0)
						tree_mapping[node] = curr_term++;

#ifdef OPENMP_
#pragma omp parallel for schedule(static) num_threads(nthreads) shared(tree_num, tree_mapping, tree, data)
				for (size_t_for row = 0; row < (decltype(row))data.nrows; row++)
#else
				for (size_t row = 0; row < (decltype(row))data.nrows; row++)
#endif
					tree_num[row + tree * data.nrows] = tree_mapping[tree_num[row + tree * data.nrows]];
			}
		}

		else
		{
			max_tree = std::accumulate(
				model_outputs_ext->hplanes.begin(),
				model_outputs_ext->hplanes.end(), (size_t)0, [](const size_t curr_max, const std::vector<IsoHPlane> &tr)
				{ return std::max(curr_max, tr.size()); });
			tree_mapping.resize(max_tree);
			for (size_t tree = 0; tree < ntrees; tree++)
			{
				std::fill(tree_mapping.begin(), tree_mapping.end(), (size_t)0);
				curr_term = 0;
				for (size_t node = 0;
					 node < model_outputs_ext->hplanes[tree].size(); node++)
					tree_mapping[node] =
						(model_outputs_ext->hplanes[tree][node].hplane_left == 0) ? curr_term++ : tree_mapping[node];

#ifdef OPENMP_
#pragma omp parallel for schedule(static) num_threads(nthreads) shared(tree_num, tree_mapping, tree, data)
				for (size_t_for row = 0; row < (decltype(row))data.nrows; row++)
#else
				for (size_t row = 0; row < (decltype(row))data.nrows; row++)
#endif
					tree_num[row + tree * data.nrows] = tree_mapping[tree_num[row + tree * data.nrows]];
			}
		}
	}

	template <class lreal_t_safe>
	real_t
	calc_kurtosis(real_t x[], size_t n, MissingAction missing_action)
	{
		lreal_t_safe m = 0;
		lreal_t_safe M2 = 0, M3 = 0, M4 = 0;
		lreal_t_safe delta, delta_s, delta_div;
		lreal_t_safe diff, n_;
		lreal_t_safe out;

		if (missing_action == Fail)
		{
			for (size_t row = 0; row < n; row++)
			{
				n_ = (lreal_t_safe)(row + 1);

				delta = x[row] - m;
				delta_div = delta / n_;
				delta_s = delta_div * delta_div;
				diff = delta * (delta_div * (lreal_t_safe)row);

				m += delta_div;
				M4 += diff * delta_s * (n_ * n_ - 3 * n_ + 3) + 6 * delta_s * M2 - 4 * delta_div * M3;
				M3 += diff * delta_div * (n_ - 2) - 3 * delta_div * M2;
				M2 += diff;
			}

			out = (M4 / M2) * ((lreal_t_safe)n / M2);
			return (!is_na_or_inf(out)) ? std::fmax((real_t)out, 0.) : (-HUGE_VAL);
		}

		else
		{
			size_t cnt = 0;
			for (size_t row = 0; row < n; row++)
			{
				if (likely(!is_na_or_inf(x[row])))
				{
					cnt++;
					n_ = (lreal_t_safe)cnt;

					delta = x[row] - m;
					delta_div = delta / n_;
					delta_s = delta_div * delta_div;
					diff = delta * (delta_div * (lreal_t_safe)(cnt - 1));

					m += delta_div;
					M4 += diff * delta_s * (n_ * n_ - 3 * n_ + 3) + 6 * delta_s * M2 - 4 * delta_div * M3;
					M3 += diff * delta_div * (n_ - 2) - 3 * delta_div * M2;
					M2 += diff;
				}
			}

			if (unlikely(cnt == 0))
				return -HUGE_VAL;

			out = (M4 / M2) * ((lreal_t_safe)cnt / M2);
			return (!is_na_or_inf(out)) ? std::fmax((real_t)out, 0.) : (-HUGE_VAL);
		}
	}

	/* TODO: is this algorithm correct? */
	template <class mapping, class lreal_t_safe = long real_t>
	real_t
	calc_kurtosis_weighted(size_t ix_arr[], size_t st, size_t end, real_t x[],
						   MissingAction missing_action, mapping &w)
	{
		lreal_t_safe m = 0;
		lreal_t_safe M2 = 0, M3 = 0, M4 = 0;
		lreal_t_safe delta, delta_s, delta_div;
		lreal_t_safe diff;
		lreal_t_safe n = 0;
		lreal_t_safe out;
		lreal_t_safe n_prev = 0.;
		lreal_t_safe w_this;

		for (size_t row = st; row <= end; row++)
		{
			if (likely(!is_na_or_inf(x[ix_arr[row]])))
			{
				w_this = w[ix_arr[row]];
				n += w_this;

				delta = x[ix_arr[row]] - m;
				delta_div = delta / n;
				delta_s = delta_div * delta_div;
				diff = delta * (delta_div * n_prev);
				n_prev = n;

				m += w_this * (delta_div);
				M4 += w_this * (diff * delta_s * (n * n - 3 * n + 3) + 6 * delta_s * M2 - 4 * delta_div * M3);
				M3 += w_this * (diff * delta_div * (n - 2) - 3 * delta_div * M2);
				M2 += w_this * (diff);
			}
		}

		if (unlikely(n <= 0))
			return -HUGE_VAL;
		if (unlikely(
				!is_na_or_inf(M2) && M2 <= std::numeric_limits<real_t>::epsilon()))
		{
			if (!check_more_than_two_unique_values(ix_arr, st, end, x,
												   missing_action))
				return -HUGE_VAL;
		}

		out = (M4 / M2) * (n / M2);
		return (!is_na_or_inf(out)) ? std::fmax((real_t)out, 0.) : (-HUGE_VAL);
	}
#if 0 // redef
		template < class lreal_t_safe>
		real_t calc_kurtosis_weighted(real_t *  x, size_t n_, MissingAction missing_action, real_t *  w)
		{
		    lreal_t_safe m = 0;
		    lreal_t_safe M2 = 0, M3 = 0, M4 = 0;
		    lreal_t_safe delta, delta_s, delta_div;
		    lreal_t_safe diff;
		    lreal_t_safe n = 0;
		    lreal_t_safe out;
		    lreal_t_safe n_prev = 0.;
		    lreal_t_safe w_this;

		    for (size_t row = 0; row < n_; row++)
		    {
		        if (likely(!is_na_or_inf(x[row])))
		        {
		            w_this = w[row];
		            n += w_this;

		            delta      =  x[row] - m;
		            delta_div  =  delta / n;
		            delta_s    =  delta_div * delta_div;
		            diff       =  delta * (delta_div * n_prev);
		            n_prev     =  n;

		            m   +=  w_this * (delta_div);
		            M4  +=  w_this * (diff * delta_s * (n * n - 3 * n + 3) + 6 * delta_s * M2 - 4 * delta_div * M3);
		            M3  +=  w_this * (diff * delta_div * (n - 2) - 3 * delta_div * M2);
		            M2  +=  w_this * (diff);
		        }
		    }

		    if (unlikely(n <= 0)) return -HUGE_VAL;

		    out = ( M4 / M2 ) * ( n / M2 );
		    return (!is_na_or_inf(out))? std::fmax((real_t)out, 0.) : (-HUGE_VAL);
		}

#endif
	/* TODO: make these compensated sums */
	/* TODO: can this use the same algorithm as above but with a correction at the end,
	 like it was done for the variance? */
	template <class lreal_t_safe>
	real_t
	calc_kurtosis(size_t *ix_arr, size_t st, size_t end, size_t col_num,
				  real_t Xc[],
				  sparse_ix *Xc_ind, sparse_ix *Xc_indptr,
				  MissingAction missing_action)
	{
		/* ix_arr must be already sorted beforehand */
		if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
			return -HUGE_VAL;

		lreal_t_safe s1 = 0;
		lreal_t_safe s2 = 0;
		lreal_t_safe s3 = 0;
		lreal_t_safe s4 = 0;
		lreal_t_safe x_sq;
		size_t cnt = end - st + 1;

		if (unlikely(cnt <= 1))
			return -HUGE_VAL;

		size_t st_col = Xc_indptr[col_num];
		size_t end_col = Xc_indptr[col_num + 1] - 1;
		size_t curr_pos = st_col;
		size_t ind_end_col = Xc_ind[end_col];
		size_t *ptr_st = std::lower_bound(ix_arr + st, ix_arr + end + 1,
										  Xc_ind[st_col]);

		lreal_t_safe xval;

		if (missing_action != Fail)
		{
			for (size_t *row = ptr_st;
				 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;)
			{
				if (Xc_ind[curr_pos] == (sparse_ix)(*row))
				{
					xval = Xc[curr_pos];
					if (unlikely(is_na_or_inf(xval)))
					{
						cnt--;
					}

					else
					{
						/* TODO: is it safe to use FMA here? some calculations rely on assuming that
						 some of these 's' are larger than the others. Would this procedure be guaranteed
						 to preserve such differences if done with a mixture of sums and FMAs? */
						x_sq = square(xval);
						s1 += xval;
						s2 = std::fma(xval, xval, s2);
						s3 = std::fma(x_sq, xval, s3);
						s4 = std::fma(x_sq, x_sq, s4);
						// s1 += pw1(xval);
						// s2 += pw2(xval);
						// s3 += pw3(xval);
						// s4 += pw4(xval);
					}

					if (row == ix_arr + end || curr_pos == end_col)
						break;
					curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
												Xc_ind + end_col + 1, *(++row)) -
							   Xc_ind;
				}

				else
				{
					if (Xc_ind[curr_pos] > (sparse_ix)(*row))
						row = std::lower_bound(row + 1, ix_arr + end + 1,
											   Xc_ind[curr_pos]);
					else
						curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
													Xc_ind + end_col + 1, *row) -
								   Xc_ind;
				}
			}

			if (unlikely(
					cnt <= (end - st + 1) - (Xc_indptr[col_num + 1] - Xc_indptr[col_num])))
				return -HUGE_VAL;
		}

		else
		{
			for (size_t *row = ptr_st;
				 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;)
			{
				if (Xc_ind[curr_pos] == (sparse_ix)(*row))
				{
					xval = Xc[curr_pos];
					x_sq = square(xval);
					s1 += xval;
					s2 = std::fma(xval, xval, s2);
					s3 = std::fma(x_sq, xval, s3);
					s4 = std::fma(x_sq, x_sq, s4);
					// s1 += pw1(xval);
					// s2 += pw2(xval);
					// s3 += pw3(xval);
					// s4 += pw4(xval);

					if (row == ix_arr + end || curr_pos == end_col)
						break;
					curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
												Xc_ind + end_col + 1, *(++row)) -
							   Xc_ind;
				}

				else
				{
					if (Xc_ind[curr_pos] > (sparse_ix)(*row))
						row = std::lower_bound(row + 1, ix_arr + end + 1,
											   Xc_ind[curr_pos]);
					else
						curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
													Xc_ind + end_col + 1, *row) -
								   Xc_ind;
				}
			}
		}

		if (unlikely(cnt <= 1 || s2 == 0 || s2 == pw2(s1)))
			return -HUGE_VAL;
		lreal_t_safe cnt_l = (lreal_t_safe)cnt;
		lreal_t_safe sn = s1 / cnt_l;
		lreal_t_safe v = s2 / cnt_l - pw2(sn);
		if (unlikely(std::isnan(v)))
			return -HUGE_VAL;
		if (v <= std::numeric_limits<real_t>::epsilon() && !check_more_than_two_unique_values(ix_arr, st, end, col_num,
																							  Xc_indptr, Xc_ind, Xc,
																							  missing_action))
			return -HUGE_VAL;
		if (unlikely(v <= 0))
			return 0.;
		lreal_t_safe out = (s4 - 4 * s3 * sn + 6 * s2 * pw2(sn) - 4 * s1 * pw3(sn) + cnt_l * pw4(sn)) / (cnt_l * pw2(v));
		return (!is_na_or_inf(out)) ? std::fmax((real_t)out, 0.) : (-HUGE_VAL);
	}

	template <class lreal_t_safe>
	real_t
	calc_kurtosis(size_t col_num, size_t nrows,
				  real_t Xc[],
				  sparse_ix *Xc_ind, sparse_ix *Xc_indptr,
				  MissingAction missing_action)
	{
		if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
			return -HUGE_VAL;

		lreal_t_safe s1 = 0;
		lreal_t_safe s2 = 0;
		lreal_t_safe s3 = 0;
		lreal_t_safe s4 = 0;
		lreal_t_safe x_sq;
		size_t cnt = nrows;

		if (unlikely(cnt <= 1))
			return -HUGE_VAL;

		lreal_t_safe xval;

		if (missing_action != Fail)
		{
			for (auto ix = Xc_indptr[col_num]; ix < Xc_indptr[col_num + 1]; ix++)
			{
				xval = Xc[ix];
				if (unlikely(is_na_or_inf(xval)))
				{
					cnt--;
				}

				else
				{
					x_sq = square(xval);
					s1 += xval;
					s2 = std::fma(xval, xval, s2);
					s3 = std::fma(x_sq, xval, s3);
					s4 = std::fma(x_sq, x_sq, s4);
					// s1 += pw1(xval);
					// s2 += pw2(xval);
					// s3 += pw3(xval);
					// s4 += pw4(xval);
				}
			}

			if (cnt <= (nrows) - (Xc_indptr[col_num + 1] - Xc_indptr[col_num]))
				return -HUGE_VAL;
		}

		else
		{
			for (auto ix = Xc_indptr[col_num]; ix < Xc_indptr[col_num + 1]; ix++)
			{
				xval = Xc[ix];
				x_sq = square(xval);
				s1 += xval;
				s2 = std::fma(xval, xval, s2);
				s3 = std::fma(x_sq, xval, s3);
				s4 = std::fma(x_sq, x_sq, s4);
				// s1 += pw1(xval);
				// s2 += pw2(xval);
				// s3 += pw3(xval);
				// s4 += pw4(xval);
			}
		}

		if (unlikely(cnt <= 1 || s2 == 0 || s2 == pw2(s1)))
			return -HUGE_VAL;
		lreal_t_safe cnt_l = (lreal_t_safe)cnt;
		lreal_t_safe sn = s1 / cnt_l;
		lreal_t_safe v = s2 / cnt_l - pw2(sn);
		if (unlikely(std::isnan(v)))
			return -HUGE_VAL;
		if (v <= std::numeric_limits<real_t>::epsilon() && !check_more_than_two_unique_values(nrows, col_num, Xc_indptr,
																							  Xc_ind, Xc, missing_action))
			return -HUGE_VAL;
		if (unlikely(v <= 0))
			return 0.;
		lreal_t_safe out = (s4 - 4 * s3 * sn + 6 * s2 * pw2(sn) - 4 * s1 * pw3(sn) + cnt_l * pw4(sn)) / (cnt_l * pw2(v));
		return (!is_na_or_inf(out)) ? std::fmax((real_t)out, 0.) : (-HUGE_VAL);
	}

	template <class mapping, class lreal_t_safe>
	real_t
	calc_kurtosis_weighted(size_t *ix_arr, size_t st, size_t end,
						   size_t col_num,
						   real_t Xc[],
						   sparse_ix *Xc_ind, sparse_ix *Xc_indptr,
						   MissingAction missing_action, mapping &w)
	{
		/* ix_arr must be already sorted beforehand */
		if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
			return -HUGE_VAL;

		lreal_t_safe s1 = 0;
		lreal_t_safe s2 = 0;
		lreal_t_safe s3 = 0;
		lreal_t_safe s4 = 0;
		lreal_t_safe x_sq;
		lreal_t_safe w_this;
		lreal_t_safe cnt = 0;
		for (size_t row = st; row <= end; row++)
			cnt += w[ix_arr[row]];

		if (unlikely(cnt <= 0))
			return -HUGE_VAL;

		size_t st_col = Xc_indptr[col_num];
		size_t end_col = Xc_indptr[col_num + 1] - 1;
		size_t curr_pos = st_col;
		size_t ind_end_col = Xc_ind[end_col];
		size_t *ptr_st = std::lower_bound(ix_arr + st, ix_arr + end + 1,
										  Xc_ind[st_col]);

		lreal_t_safe xval;

		if (missing_action != Fail)
		{
			for (size_t *row = ptr_st;
				 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;)
			{
				if (Xc_ind[curr_pos] == (sparse_ix)(*row))
				{
					w_this = w[*row];
					xval = Xc[curr_pos];

					if (unlikely(is_na_or_inf(xval)))
					{
						cnt -= w_this;
					}

					else
					{
						x_sq = xval * xval;
						s1 = std::fma(w_this, xval, s1);
						s2 = std::fma(w_this, x_sq, s2);
						s3 = std::fma(w_this, x_sq * xval, s3);
						s4 = std::fma(w_this, x_sq * x_sq, s4);
						// s1 += w_this * pw1(xval);
						// s2 += w_this * pw2(xval);
						// s3 += w_this * pw3(xval);
						// s4 += w_this * pw4(xval);
					}

					if (row == ix_arr + end || curr_pos == end_col)
						break;
					curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
												Xc_ind + end_col + 1, *(++row)) -
							   Xc_ind;
				}

				else
				{
					if (Xc_ind[curr_pos] > (sparse_ix)(*row))
						row = std::lower_bound(row + 1, ix_arr + end + 1,
											   Xc_ind[curr_pos]);
					else
						curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
													Xc_ind + end_col + 1, *row) -
								   Xc_ind;
				}
			}

			if (unlikely(cnt <= 0))
				return -HUGE_VAL;
		}

		else
		{
			for (size_t *row = ptr_st;
				 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;)
			{
				if (Xc_ind[curr_pos] == (sparse_ix)(*row))
				{
					w_this = w[*row];
					xval = Xc[curr_pos];

					x_sq = xval * xval;
					s1 = std::fma(w_this, xval, s1);
					s2 = std::fma(w_this, x_sq, s2);
					s3 = std::fma(w_this, x_sq * xval, s3);
					s4 = std::fma(w_this, x_sq * x_sq, s4);
					// s1 += w_this * pw1(xval);
					// s2 += w_this * pw2(xval);
					// s3 += w_this * pw3(xval);
					// s4 += w_this * pw4(xval);

					if (row == ix_arr + end || curr_pos == end_col)
						break;
					curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
												Xc_ind + end_col + 1, *(++row)) -
							   Xc_ind;
				}

				else
				{
					if (Xc_ind[curr_pos] > (sparse_ix)(*row))
						row = std::lower_bound(row + 1, ix_arr + end + 1,
											   Xc_ind[curr_pos]);
					else
						curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
													Xc_ind + end_col + 1, *row) -
								   Xc_ind;
				}
			}
		}

		if (unlikely(cnt <= 1 || s2 == 0 || s2 == pw2(s1)))
			return -HUGE_VAL;
		lreal_t_safe sn = s1 / cnt;
		lreal_t_safe v = s2 / cnt - pw2(sn);
		if (unlikely(std::isnan(v)))
			return -HUGE_VAL;
		if (v <= std::numeric_limits<real_t>::epsilon() && !check_more_than_two_unique_values(ix_arr, st, end, col_num,
																							  Xc_indptr, Xc_ind, Xc,
																							  missing_action))
			return -HUGE_VAL;
		if (v <= 0)
			return 0.;
		lreal_t_safe out = (s4 - 4 * s3 * sn + 6 * s2 * pw2(sn) - 4 * s1 * pw3(sn) + cnt * pw4(sn)) / (cnt * pw2(v));
		return (!is_na_or_inf(out)) ? std::fmax((real_t)out, 0.) : (-HUGE_VAL);
	}

	/* TODO: make this a compensated sum */
	template <class real_t_ = real_t>
	real_t
	find_split_rel_gain_t(real_t_ *x, size_t n, real_t split_point)
	{
		real_t this_gain;
		real_t best_gain = -HUGE_VAL;
		real_t x1 = 0, x2 = 0;
		real_t sum_left = 0, sum_right = 0, sum_tot = 0;
		for (size_t row = 0; row < n; row++)
			sum_tot += x[row];
		for (size_t row = 0; row < n - 1; row++)
		{
			sum_left += x[row];
			if (x[row] == x[row + 1])
				continue;

			sum_right = sum_tot - sum_left;
			this_gain = sum_left * (sum_left / (real_t)(row + 1)) + sum_right * (sum_right / (real_t)(n - row - 1));
			if (this_gain > best_gain)
			{
				best_gain = this_gain;
				x1 = x[row];
				x2 = x[row + 1];
			}
		}

		if (best_gain <= -HUGE_VAL)
			return best_gain;
		split_point = midpoint(x1, x2);
		return std::fmax((real_t)best_gain,
						 std::numeric_limits<real_t>::epsilon());
	}

	/* Note: there is no 'weighted' version of 'find_split_rel_gain' with unindexed 'x', because calling it would
	 imply having to argsort the 'x' values in order to sort the weights, which is less efficient. */

	template <class real_x, class real_y, class mapping>
	real_t
	find_split_rel_gain_weighted_t(real_x *x, real_x xmean, size_t *ix_arr,
								   size_t st, size_t end, real_t &split_point,
								   size_t &split_ix, mapping &w)
	{
		real_y this_gain;
		real_y best_gain = -HUGE_VAL;
		split_ix = 0; /* <- avoid out-of-bounds at the end */
		real_t sum_left = 0, sum_right = 0, sum_tot = 0, sumw = 0, sumw_tot = 0;

		for (size_t row = st; row <= end; row++)
			sumw_tot += w[ix_arr[row]];
		for (size_t row = st; row <= end; row++)
			sum_tot += x[ix_arr[row]] - xmean;
		for (size_t row = st; row < end; row++)
		{
			sumw += w[ix_arr[row]];
			sum_left += x[ix_arr[row]] - xmean;
			if (x[ix_arr[row]] == x[ix_arr[row + 1]])
				continue;

			sum_right = sum_tot - sum_left;
			this_gain = sum_left * (sum_left / sumw) + sum_right * (sum_right / (sumw_tot - sumw));
			if (this_gain > best_gain)
			{
				best_gain = this_gain;
				split_ix = row;
			}
		}

		if (best_gain <= -HUGE_VAL)
			return best_gain;
		split_point = midpoint(x[ix_arr[split_ix]], x[ix_arr[split_ix + 1]]);
		return std::fmax((real_t)best_gain,
						 std::numeric_limits<real_t>::epsilon());
	}

	template <class real_t_, class mapping, class lreal_t_safe>
	real_t
	find_split_rel_gain_weighted(real_t_ *x, real_t_ xmean, size_t *ix_arr,
								 size_t st, size_t end, real_t &split_point,
								 size_t &split_ix, mapping &w)
	{
		if ((end - st + 1) < THRESHOLD_LONG_DOUBLE)
			return find_split_rel_gain_weighted_t<real_t, real_t_, mapping>(
				x, xmean, ix_arr, st, end, split_point, split_ix, w);
		else
			return find_split_rel_gain_weighted_t<lreal_t_safe, real_t_, mapping>(
				(lreal_t_safe *)x, xmean, ix_arr, st, end, split_point, split_ix, w);
	}

	real_t
	calc_sd_right_to_left(real_t *x, size_t n, real_t *sd_arr)
	{
		real_t running_mean = 0;
		real_t running_ssq = 0;
		real_t mean_prev = x[n - 1];
		for (size_t row = 0; row < n - 1; row++)
		{
			running_mean += (x[n - row - 1] - running_mean) / (real_t)(row + 1);
			running_ssq += (x[n - row - 1] - running_mean) * (x[n - row - 1] - mean_prev);
			mean_prev = running_mean;
			sd_arr[n - row - 1] =
				(row == 0) ? 0. : std::sqrt(running_ssq / (real_t)(row + 1));
		}
		running_mean += (x[0] - running_mean) / (real_t)n;
		running_ssq += (x[0] - running_mean) * (x[0] - mean_prev);
		return std::sqrt(running_ssq / (real_t)n);
	}

	template <class lreal_t_safe>
	lreal_t_safe
	calc_sd_right_to_left_weighted(real_t *x, size_t n, real_t *sd_arr,
								   real_t *w, lreal_t_safe &cumw,
								   size_t *sorted_ix)
	{
		lreal_t_safe running_mean = 0;
		lreal_t_safe running_ssq = 0;
		lreal_t_safe mean_prev = x[sorted_ix[n - 1]];
		lreal_t_safe cnt = 0;
		real_t w_this;
		for (size_t row = 0; row < n - 1; row++)
		{
			w_this = w[sorted_ix[n - row - 1]];
			cnt += w_this;
			running_mean += w_this * (x[sorted_ix[n - row - 1]] - running_mean) / cnt;
			running_ssq += w_this * ((x[sorted_ix[n - row - 1]] - running_mean) * (x[sorted_ix[n - row - 1]] - mean_prev));
			mean_prev = running_mean;
			sd_arr[n - row - 1] = (row == 0) ? 0. : std::sqrt(running_ssq / cnt);
		}
		w_this = w[sorted_ix[0]];
		cnt += w_this;
		running_mean += (x[sorted_ix[0]] - running_mean) / cnt;
		running_ssq += w_this * ((x[sorted_ix[0]] - running_mean) * (x[sorted_ix[0]] - mean_prev));
		cumw = cnt;
		return std::sqrt(running_ssq / cnt);
	}

	real_t
	calc_sd_right_to_left(real_t *x, real_t xmean, size_t ix_arr[], size_t st,
						  size_t end, real_t *sd_arr)
	{
		real_t running_mean = 0;
		real_t running_ssq = 0;
		real_t mean_prev = x[ix_arr[end]] - xmean;
		size_t n = end - st + 1;
		for (size_t row = 0; row < n - 1; row++)
		{
			running_mean += ((x[ix_arr[end - row]] - xmean) - running_mean) / (real_t)(row + 1);
			running_ssq += ((x[ix_arr[end - row]] - xmean) - running_mean) * ((x[ix_arr[end - row]] - xmean) - mean_prev);
			mean_prev = running_mean;
			sd_arr[n - row - 1] =
				(row == 0) ? 0. : std::sqrt(running_ssq / (real_t)(row + 1));
		}
		running_mean += ((x[ix_arr[st]] - xmean) - running_mean) / (real_t)n;
		running_ssq += ((x[ix_arr[st]] - xmean) - running_mean) * ((x[ix_arr[st]] - xmean) - mean_prev);
		return std::sqrt(running_ssq / (real_t)n);
	}

#if 0

		template <class xreal = real_t, class yreal = real_t>
		real_t find_split_rel_gain_t(xreal * x, yreal xmean, size_t *ix_arr, size_t st, size_t end, real_t &split_point, size_t &split_ix)
		{
			yreal this_gain;
			yreal best_gain = -HUGE_VAL;
		    split_ix = 0; /* <- avoid out-of-bounds at the end */
		    real_t sum_left = 0, sum_right = 0, sum_tot = 0;
		    for (size_t row = st; row <= end; row++)
		        sum_tot += x[ix_arr[row]] - xmean;
		    for (size_t row = st; row < end; row++)
		    {
		        sum_left += x[ix_arr[row]] - xmean;
		        if (x[ix_arr[row]] == x[ix_arr[row+1]])
		            continue;

		        sum_right = sum_tot - sum_left;
		        this_gain =   sum_left  * (sum_left  / (yreal)(row - st + 1))
		                    + sum_right * (sum_right / (yreal)(end - row));
		        if (this_gain > best_gain)
		        {
		            best_gain = this_gain;
		            split_ix = row;
		        }
		    }

		    if (best_gain <= -HUGE_VAL)
		        return best_gain;
		    split_point = midpoint(x[ix_arr[split_ix]], x[ix_arr[split_ix+1]]);
		    return std::fmax((real_t)best_gain, std::numeric_limits<real_t>::epsilon());
		}
		real_t find_split_rel_gain(real_t * x, size_t n, real_t &split_point)
		{
		    if (n < THRESHOLD_LONG_DOUBLE)
		        return find_split_rel_gain_t<real_t>(x, n, split_point);
		    else
		        return find_split_rel_gain_t<long real_t>(x, n, split_point);
		}

		template <class real_t_ = real_t, class lreal_t_safe>
		real_t find_split_rel_gain(real_t_ *x, real_t_ xmean, size_t *ix_arr, size_t st, size_t end, real_t &split_point, size_t &split_ix)
		{
		    if ((end-st+1) < THRESHOLD_LONG_DOUBLE)
		        return find_split_rel_gain_t<real_t, real_t_>(x, xmean, ix_arr, st, end, split_point, split_ix);
		    else
		        return find_split_rel_gain_t<lreal_t_safe, real_t_>(x, xmean, ix_arr, st, end, split_point, split_ix);
		}

		--> comment
		template <class real_t_, class mapping, class lreal_t_safe>
		lreal_t_safe calc_sd_right_to_left_weighted(real_t_ * x, real_t_ xmean, size_t ix_arr[], size_t st, size_t end,
		                                           real_t * sd_arr, mapping & w, lreal_t_safe &cumw)
		{
		    lreal_t_safe running_mean = 0;
		    lreal_t_safe running_ssq = 0;
		    real_t_ mean_prev = x[ix_arr[end]] - xmean;
		    size_t n = end - st + 1;
		    lreal_t_safe cnt = 0;
		    real_t w_this;
		    for (size_t row = 0; row < n-1; row++)
		    {
		        w_this = w[ix_arr[end-row]];
		        cnt += w_this;
		        running_mean   += w_this * ((x[ix_arr[end-row]] - xmean) - running_mean) / cnt;
		        running_ssq    += w_this * (((x[ix_arr[end-row]] - xmean) - running_mean) * ((x[ix_arr[end-row]] - xmean) - mean_prev));
		        mean_prev       =  running_mean;
		        sd_arr[n-row-1] = (row == 0)? 0. : std::sqrt(running_ssq / cnt);
		    }
		    w_this = w[ix_arr[st]];
		    cnt += w_this;
		    running_mean   += ((x[ix_arr[st]] - xmean) - running_mean) / cnt;
		    running_ssq    += w_this * (((x[ix_arr[st]] - xmean) - running_mean) * ((x[ix_arr[st]] - xmean) - mean_prev));
		    cumw = cnt;
		    return std::sqrt(running_ssq / cnt);
		}

		template <class real_t, class real_t_>
		real_t find_split_std_gain_t(real_t_ * x, size_t n, real_t * sd_arr,
		                             GainCriterion criterion, real_t min_gain, real_t & split_point)
		{
		    real_t full_sd = calc_sd_right_to_left<real_t>(x, n, sd_arr);
		    real_t running_mean = 0;
		    real_t running_ssq = 0;
		    real_t mean_prev = x[0];
		    real_t best_gain = -HUGE_VAL;
		    real_t this_sd, this_gain;
		    real_t n_ = (real_t)n;
		    size_t best_ix = 0;
		    for (size_t row = 0; row < n-1; row++)
		    {
		        running_mean   += (x[row] - running_mean) / (real_t)(row+1);
		        running_ssq    += (x[row] - running_mean) * (x[row] - mean_prev);
		        mean_prev       =  running_mean;
		        if (x[row] == x[row+1])
		            continue;

		        this_sd = (row == 0)? 0. : std::sqrt(running_ssq / (real_t)(row+1));
		        this_gain = (criterion == Pooled)?
		                    pooled_gain(full_sd, n_, this_sd, sd_arr[row+1], row+1, n-row-1)
		                        :
		                    sd_gain(full_sd, this_sd, sd_arr[row+1]);
		        if (this_gain > best_gain && this_gain > min_gain)
		        {
		            best_gain = this_gain;
		            best_ix = row;
		        }
		    }

		    if (best_gain > -HUGE_VAL)
		        split_point = midpoint(x[best_ix], x[best_ix+1]);

		    return best_gain;
		}



		template <class real_t, class mapping, class lreal_t_safe>
		real_t find_split_std_gain_weighted(real_t * x, real_t xmean, size_t ix_arr[], size_t st, size_t end, real_t * sd_arr,
		                                    GainCriterion criterion, real_t min_gain, real_t & split_point, size_t & split_ix, mapping & w)
		{
		    lreal_t_safe cumw;
		    real_t full_sd = calc_sd_right_to_left_weighted(x, xmean, ix_arr, st, end, sd_arr, w, cumw);
		    lreal_t_safe running_mean = 0;
		    lreal_t_safe running_ssq = 0;
		    lreal_t_safe mean_prev = x[ix_arr[st]] - xmean;
		    real_t best_gain = -HUGE_VAL;
		    lreal_t_safe currw = 0;
		    real_t this_sd, this_gain;
		    real_t w_this;
		    split_ix = st;

		    for (size_t row = st; row < end; row++)
		    {
		        w_this = w[ix_arr[row]];
		        currw += w_this;
		        running_mean   += w_this * ((x[ix_arr[row]] - xmean) - running_mean) / currw;
		        running_ssq    += w_this * (((x[ix_arr[row]] - xmean) - running_mean) * ((x[ix_arr[row]] - xmean) - mean_prev));
		        mean_prev       =  running_mean;
		        if (x[ix_arr[row]] == x[ix_arr[row+1]])
		            continue;

		        this_sd = (row == st)? 0. : std::sqrt(running_ssq / currw);
		        this_gain = (criterion == Pooled)?
		                    pooled_gain(full_sd, cumw, this_sd, sd_arr[row-st+1], currw, cumw-currw)
		                        :
		                    sd_gain(full_sd, this_sd, sd_arr[row-st+1]);
		        if (this_gain > best_gain && this_gain > min_gain)
		        {
		            best_gain = this_gain;
		            split_ix = row;
		        }
		    }

		    if (best_gain > -HUGE_VAL)
		        split_point = midpoint(x[ix_arr[split_ix]], x[ix_arr[split_ix+1]]);

		    return best_gain;
		}

#ifndef _FOR_R
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#endif
#endif

#ifndef _FOR_R
		[[gnu::optimize("Ofast")]]
#endif
		static inline void xpy1(real_t * x, real_t * y, size_t n)
		{
		    for (size_t ix = 0; ix < n; ix++) y[ix] += x[ix];
		}

#ifndef _FOR_R
		[[gnu::optimize("Ofast")]]
#endif
		static inline void axpy1(const real_t a, real_t * x, real_t * y, size_t n)
		{
		    for (size_t ix = 0; ix < n; ix++) y[ix] = std::fma(a, x[ix], y[ix]);
		}

#ifndef _FOR_R
		[[gnu::optimize("Ofast")]]
#endif
		static inline void xpy1(real_t * xval, size_t ind[], size_t nnz, real_t * y)
		{
		    for (size_t ix = 0; ix < nnz; ix++) y[ind[ix]] += xval[ix];
		}

#ifndef _FOR_R
		[[gnu::optimize("Ofast")]]
#endif
		static inline void axpy1(const real_t a, real_t * xval, size_t ind[], size_t nnz, real_t * y)
		{
		    for (size_t ix = 0; ix < nnz; ix++) y[ind[ix]] = std::fma(a, xval[ix], y[ind[ix]]);
		}

#ifndef _FOR_R
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif

		template <class real_t, class lreal_t_safe>
		real_t find_split_full_gain(real_t * x, size_t st, size_t end, size_t * ix_arr,
		                            size_t * cols_use, size_t ncols_use, bool force_cols_use,
		                            real_t * X_row_major, size_t ncols,
		                            real_t * Xr, size_t * Xr_ind, size_t * Xr_indptr,
		                            real_t * buffer_sum_left, real_t * buffer_sum_tot,
		                            size_t & split_ix, real_t & split_point,
		                            bool x_uses_ix_arr)
		{
		    if (end <= st) return -HUGE_VAL;
		    if (cols_use != NULL && ncols_use && (real_t)ncols_use / (real_t)ncols < 0.1)
		        force_cols_use = true;

		    memset(buffer_sum_tot, 0, (force_cols_use? ncols_use : ncols)*sizeof(real_t));
		    if (Xr_indptr == NULL)
		    {
		        if (force_cols_use)
		        {
		            real_t * ptr_row;
		            for (size_t row = st; row <= end; row++)
		            {
		                ptr_row = X_row_major + ix_arr[row]*ncols;
		                for (size_t col = 0; col < ncols_use; col++)
		                    buffer_sum_tot[col] += ptr_row[cols_use[col]];
		            }
		        }

		        else
		        {
		            for (size_t row = st; row <= end; row++)
		                xpy1(X_row_major + ix_arr[row]*ncols, buffer_sum_tot, ncols);
		        }
		    }

		    else
		    {
		        if (force_cols_use)
		        {
		            size_t *curr_begin;
		            size_t *row_end;
		            size_t *curr_col;
		            real_t * Xr_this;
		            size_t *cols_end = cols_use + ncols_use;
		            for (size_t row = st; row <= end; row++)
		            {
		                curr_begin = Xr_ind + Xr_indptr[ix_arr[row]];
		                row_end = Xr_ind + Xr_indptr[ix_arr[row] + 1];
		                if (curr_begin == row_end) continue;
		                curr_col = cols_use;
		                Xr_this = Xr + Xr_indptr[ix_arr[row]];

		                while (curr_col < cols_end && curr_begin < row_end)
		                {
		                    if (*curr_begin == *curr_col)
		                    {
		                        buffer_sum_tot[std::distance(cols_use, curr_col)] += Xr_this[std::distance(curr_begin, row_end)];
		                        curr_col++;
		                        curr_begin++;
		                    }

		                    else
		                    {
		                        if (*curr_begin > *curr_col)
		                            curr_col = std::lower_bound(curr_col, cols_end, *curr_begin);
		                        else
		                            curr_begin = std::lower_bound(curr_begin, row_end, *curr_col);
		                    }
		                }
		            }
		        }

		        else
		        {
		            size_t ptr_this;
		            for (size_t row = st; row <= end; row++)
		            {
		                ptr_this = Xr_indptr[ix_arr[row]];
		                xpy1(Xr + ptr_this, Xr_ind + ptr_this, Xr_indptr[ix_arr[row]+1] - ptr_this, buffer_sum_tot);
		            }
		        }
		    }

		    real_t best_gain = -HUGE_VAL;
		    real_t this_gain;
		    real_t sl, sr;
		    real_t dl, dr;
		    real_t vleft, vright;
		    memset(buffer_sum_left, 0, (force_cols_use? ncols_use : ncols)*sizeof(real_t));
		    if (Xr_indptr == NULL)
		    {
		        if (!force_cols_use)
		        {
		            for (size_t row = st; row < end; row++)
		            {
		                xpy1(X_row_major + ix_arr[row]*ncols, buffer_sum_left, ncols);
		                if (x_uses_ix_arr) {
		                    if (unlikely(x[ix_arr[row]] == x[ix_arr[row+1]])) continue;
		                }
		                else {
		                    if (unlikely(x[row] == x[row+1])) continue;
		                }

		                vleft = 0;
		                vright = 0;
		                dl = (real_t)(row-st+1);
		                dr = (real_t)(end-row);
		                for (size_t col = 0; col < ncols; col++)
		                {
		                    sl = buffer_sum_left[col];
		                    vleft += sl * (sl / dl);
		                    sr = buffer_sum_tot[col] - sl;
		                    vright += sr * (sr / dr);
		                }

		                this_gain = vleft + vright;
		                if (this_gain > best_gain)
		                {
		                    best_gain = this_gain;
		                    split_ix = row;
		                }
		            }
		        }

		        else
		        {
		            real_t * ptr_row;
		            for (size_t row = st; row < end; row++)
		            {
		                ptr_row = X_row_major + ix_arr[row]*ncols;
		                for (size_t col = 0; col < ncols_use; col++)
		                    buffer_sum_left[col] += ptr_row[cols_use[col]];
		                if (x_uses_ix_arr) {
		                    if (unlikely(x[ix_arr[row]] == x[ix_arr[row+1]])) continue;
		                }
		                else {
		                    if (unlikely(x[row] == x[row+1])) continue;
		                }

		                vleft = 0;
		                vright = 0;
		                dl = (real_t)(row-st+1);
		                dr = (real_t)(end-row);
		                for (size_t col = 0; col < ncols_use; col++)
		                {
		                    sl = buffer_sum_left[col];
		                    vleft += sl * (sl / dl);
		                    sr = buffer_sum_tot[col] - sl;
		                    vright += sr * (sr / dr);
		                }

		                this_gain = vleft + vright;
		                if (this_gain > best_gain)
		                {
		                    best_gain = this_gain;
		                    split_ix = row;
		                }
		            }
		        }
		    }

		    else
		    {
		        if (!force_cols_use)
		        {
		            size_t ptr_this;
		            for (size_t row = st; row < end; row++)
		            {
		                ptr_this = Xr_indptr[ix_arr[row]];
		                xpy1(Xr + ptr_this, Xr_ind + ptr_this, Xr_indptr[ix_arr[row]+1] - ptr_this, buffer_sum_left);
		                if (x_uses_ix_arr) {
		                    if (unlikely(x[ix_arr[row]] == x[ix_arr[row+1]])) continue;
		                }
		                else {
		                    if (unlikely(x[row] == x[row+1])) continue;
		                }

		                vleft = 0;
		                vright = 0;
		                dl = (real_t)(row-st+1);
		                dr = (real_t)(end-row);
		                for (size_t col = 0; col < ncols; col++)
		                {
		                    sl = buffer_sum_left[col];
		                    vleft += sl * (sl / dl);
		                    sr = buffer_sum_tot[col] - sl;
		                    vright += sr * (sr / dr);
		                }

		                this_gain = vleft + vright;
		                if (this_gain > best_gain)
		                {
		                    best_gain = this_gain;
		                    split_ix = row;
		                }
		            }
		        }

		        else
		        {
		            size_t *curr_begin;
		            size_t *row_end;
		            size_t *curr_col;
		            real_t * Xr_this;
		            size_t *cols_end = cols_use + ncols_use;
		            for (size_t row = st; row < end; row++)
		            {
		                curr_begin = Xr_ind + Xr_indptr[ix_arr[row]];
		                row_end = Xr_ind + Xr_indptr[ix_arr[row] + 1];
		                if (curr_begin == row_end) goto skip_sum;
		                curr_col = cols_use;
		                Xr_this = Xr + Xr_indptr[ix_arr[row]];
		                while (curr_col < cols_end && curr_begin < row_end)
		                {
		                    if (*curr_begin == *curr_col)
		                    {
		                        buffer_sum_left[std::distance(cols_use, curr_col)] += Xr_this[std::distance(curr_begin, row_end)];
		                        curr_col++;
		                        curr_begin++;
		                    }

		                    else
		                    {
		                        if (*curr_begin > *curr_col)
		                            curr_col = std::lower_bound(curr_col, cols_end, *curr_begin);
		                        else
		                            curr_begin = std::lower_bound(curr_begin, row_end, *curr_col);
		                    }
		                }

		                skip_sum:
		                if (x_uses_ix_arr) {
		                    if (unlikely(x[ix_arr[row]] == x[ix_arr[row+1]])) continue;
		                }
		                else {
		                    if (unlikely(x[row] == x[row+1])) continue;
		                }

		                vleft = 0;
		                vright = 0;
		                dl = (real_t)(row-st+1);
		                dr = (real_t)(end-row);
		                for (size_t col = 0; col < ncols_use; col++)
		                {
		                    sl = buffer_sum_left[col];
		                    vleft += sl * (sl / dl);
		                    sr = buffer_sum_tot[col] - sl;
		                    vright += sr * (sr / dr);
		                }

		                this_gain = vleft + vright;
		                if (this_gain > best_gain)
		                {
		                    best_gain = this_gain;
		                    split_ix = row;
		                }
		            }
		        }
		    }

		    if (best_gain <= -HUGE_VAL) return best_gain;

		    if (x_uses_ix_arr)
		        split_point = midpoint(x[ix_arr[split_ix]], x[ix_arr[split_ix+1]]);
		    else
		        split_point = midpoint(x[split_ix], x[split_ix+1]);
		    return best_gain / (lreal_t_safe)(end - st + 1);
		}

		template <class real_t, class mapping, class lreal_t_safe>
		real_t find_split_full_gain_weighted(real_t * x, size_t st, size_t end, size_t * ix_arr,
		                                     size_t * cols_use, size_t ncols_use, bool force_cols_use,
		                                     real_t * X_row_major, size_t ncols,
		                                     real_t * Xr, size_t * Xr_ind, size_t * Xr_indptr,
		                                     real_t * buffer_sum_left, real_t * buffer_sum_tot,
		                                     size_t & split_ix, real_t & split_point,
		                                     bool x_uses_ix_arr,
		                                     mapping & w)
		{
		    if (end <= st) return -HUGE_VAL;
		    if (cols_use != NULL && ncols_use && (real_t)ncols_use / (real_t)ncols < 0.1)
		        force_cols_use = true;

		    real_t wtot = 0;
		    if (x_uses_ix_arr)
		    {
		        for (size_t row = st; row <= end; row++)
		            wtot += w[ix_arr[row]];
		    }

		    else
		    {
		        for (size_t row = st; row <= end; row++)
		            wtot += w[row];
		    }

		    memset(buffer_sum_tot, 0, (force_cols_use? ncols_use : ncols)*sizeof(real_t));
		    if (Xr_indptr == NULL)
		    {
		        if (!force_cols_use)
		        {
		            if (x_uses_ix_arr)
		            {
		                for (size_t row = st; row <= end; row++)
		                    axpy1(w[ix_arr[row]], X_row_major + ix_arr[row]*ncols, buffer_sum_tot, ncols);
		            }

		            else
		            {
		                for (size_t row = st; row <= end; row++)
		                    axpy1(w[row], X_row_major + ix_arr[row]*ncols, buffer_sum_tot, ncols);
		            }
		        }

		        else
		        {
		            real_t * ptr_row;
		            real_t w_row;

		            if (x_uses_ix_arr)
		            {
		                for (size_t row = st; row <= end; row++)
		                {
		                    ptr_row = X_row_major + ix_arr[row]*ncols;
		                    w_row = w[ix_arr[row]];
		                    for (size_t col = 0; col < ncols_use; col++)
		                        buffer_sum_tot[col] = std::fma(w_row, ptr_row[cols_use[col]], buffer_sum_tot[col]);
		                }
		            }

		            else
		            {
		                for (size_t row = st; row <= end; row++)
		                {
		                    ptr_row = X_row_major + ix_arr[row]*ncols;
		                    w_row = w[row];
		                    for (size_t col = 0; col < ncols_use; col++)
		                        buffer_sum_tot[col] = std::fma(w_row, ptr_row[cols_use[col]], buffer_sum_tot[col]);
		                }
		            }
		        }
		    }

		    else
		    {
		        if (!force_cols_use)
		        {
		            size_t ptr_this;
		            if (x_uses_ix_arr)
		            {
		                for (size_t row = st; row <= end; row++)
		                {
		                    ptr_this = Xr_indptr[ix_arr[row]];
		                    axpy1(w[ix_arr[row]], Xr + ptr_this, Xr_ind + ptr_this, Xr_indptr[ix_arr[row]+1] - ptr_this, buffer_sum_tot);
		                }
		            }

		            else
		            {
		                for (size_t row = st; row <= end; row++)
		                {
		                    ptr_this = Xr_indptr[ix_arr[row]];
		                    axpy1(w[row], Xr + ptr_this, Xr_ind + ptr_this, Xr_indptr[ix_arr[row]+1] - ptr_this, buffer_sum_tot);
		                }
		            }
		        }

		        else
		        {
		            size_t *curr_begin;
		            size_t *row_end;
		            size_t *curr_col;
		            real_t * Xr_this;
		            size_t *cols_end = cols_use + ncols_use;
		            real_t w_row;
		            for (size_t row = st; row <= end; row++)
		            {
		                curr_begin = Xr_ind + Xr_indptr[ix_arr[row]];
		                row_end = Xr_ind + Xr_indptr[ix_arr[row] + 1];
		                if (curr_begin == row_end) continue;
		                curr_col = cols_use;
		                Xr_this = Xr + Xr_indptr[ix_arr[row]];
		                w_row = w[x_uses_ix_arr? ix_arr[row] : row];
		                size_t dtemp;

		                while (curr_col < cols_end && curr_begin < row_end)
		                {
		                    if (*curr_begin == *curr_col)
		                    {
		                        dtemp = std::distance(cols_use, curr_col);
		                        buffer_sum_tot[dtemp]
		                            =
		                        std::fma(w_row, Xr_this[std::distance(curr_begin, row_end)], buffer_sum_tot[dtemp]);
		                        curr_col++;
		                        curr_begin++;
		                    }

		                    else
		                    {
		                        if (*curr_begin > *curr_col)
		                            curr_col = std::lower_bound(curr_col, cols_end, *curr_begin);
		                        else
		                            curr_begin = std::lower_bound(curr_begin, row_end, *curr_col);
		                    }
		                }
		            }
		        }
		    }

		    real_t best_gain = -HUGE_VAL;
		    real_t this_gain;
		    real_t sl, sr;
		    real_t vleft, vright;
		    real_t wleft = 0;
		    real_t w_row;
		    real_t wright;
		    memset(buffer_sum_left, 0, (force_cols_use? ncols_use : ncols)*sizeof(real_t));
		    if (Xr_indptr == NULL)
		    {
		        if (!force_cols_use)
		        {
		            for (size_t row = st; row < end; row++)
		            {
		                w_row = w[x_uses_ix_arr? ix_arr[row] : row];
		                wleft += w_row;
		                axpy1(w_row, X_row_major + ix_arr[row]*ncols, buffer_sum_left, ncols);
		                if (x_uses_ix_arr) {
		                    if (unlikely(x[ix_arr[row]] == x[ix_arr[row+1]])) continue;
		                }
		                else {
		                    if (unlikely(x[row] == x[row+1])) continue;
		                }

		                vleft = 0;
		                vright = 0;
		                wright = wtot - wleft;
		                for (size_t col = 0; col < ncols; col++)
		                {
		                    sl = buffer_sum_left[col];
		                    vleft += sl * (sl / wleft);
		                    sr = buffer_sum_tot[col] - sl;
		                    vright += sr * (sr / wright);
		                }

		                this_gain = vleft + vright;
		                if (this_gain > best_gain)
		                {
		                    best_gain = this_gain;
		                    split_ix = row;
		                }
		            }
		        }

		        else
		        {
		            real_t * ptr_row;
		            real_t w_row;
		            for (size_t row = st; row < end; row++)
		            {
		                w_row = w[x_uses_ix_arr? ix_arr[row] : row];
		                wleft += w_row;

		                ptr_row = X_row_major + ix_arr[row]*ncols;
		                for (size_t col = 0; col < ncols_use; col++)
		                    buffer_sum_left[col] = std::fma(w_row, ptr_row[cols_use[col]], buffer_sum_left[col]);
		                if (x_uses_ix_arr) {
		                    if (unlikely(x[ix_arr[row]] == x[ix_arr[row+1]])) continue;
		                }
		                else {
		                    if (unlikely(x[row] == x[row+1])) continue;
		                }

		                vleft = 0;
		                vright = 0;
		                wright = wtot - wleft;
		                for (size_t col = 0; col < ncols_use; col++)
		                {
		                    sl = buffer_sum_left[col];
		                    vleft += sl * (sl / wleft);
		                    sr = buffer_sum_tot[col] - sl;
		                    vright += sr * (sr / wright);
		                }

		                this_gain = vleft + vright;
		                if (this_gain > best_gain)
		                {
		                    best_gain = this_gain;
		                    split_ix = row;
		                }
		            }
		        }
		    }

		    else
		    {
		        if (!force_cols_use)
		        {
		            size_t ptr_this;
		            real_t w_row;
		            for (size_t row = st; row < end; row++)
		            {
		                w_row= w[x_uses_ix_arr? ix_arr[row] : row];
		                wleft += w_row;
		                ptr_this = Xr_indptr[ix_arr[row]];
		                axpy1(w_row, Xr + ptr_this, Xr_ind + ptr_this, Xr_indptr[ix_arr[row]+1] - ptr_this, buffer_sum_left);
		                if (x_uses_ix_arr) {
		                    if (unlikely(x[ix_arr[row]] == x[ix_arr[row+1]])) continue;
		                }
		                else {
		                    if (unlikely(x[row] == x[row+1])) continue;
		                }

		                vleft = 0;
		                vright = 0;
		                wright = wtot - wleft;
		                for (size_t col = 0; col < ncols; col++)
		                {
		                    sl = buffer_sum_left[col];
		                    vleft += sl * (sl / wleft);
		                    sr = buffer_sum_tot[col] - sl;
		                    vright += sr * (sr / wright);
		                }

		                this_gain = vleft + vright;
		                if (this_gain > best_gain)
		                {
		                    best_gain = this_gain;
		                    split_ix = row;
		                }
		            }
		        }

		        else
		        {
		            size_t *curr_begin;
		            size_t *row_end;
		            size_t *curr_col;
		            real_t * Xr_this;
		            size_t *cols_end = cols_use + ncols_use;
		            real_t w_row;
		            size_t dtemp;
		            for (size_t row = st; row < end; row++)
		            {
		                w_row = w[x_uses_ix_arr? ix_arr[row] : row];
		                wleft += w_row;

		                curr_begin = Xr_ind + Xr_indptr[ix_arr[row]];
		                row_end = Xr_ind + Xr_indptr[ix_arr[row] + 1];
		                if (curr_begin == row_end) goto skip_sum;
		                curr_col = cols_use;
		                Xr_this = Xr + Xr_indptr[ix_arr[row]];
		                while (curr_col < cols_end && curr_begin < row_end)
		                {
		                    if (*curr_begin == *curr_col)
		                    {
		                        dtemp = std::distance(cols_use, curr_col);
		                        buffer_sum_left[dtemp]
		                            =
		                        std::fma(w_row, Xr_this[std::distance(curr_begin, row_end)], buffer_sum_left[dtemp]);
		                        curr_col++;
		                        curr_begin++;
		                    }

		                    else
		                    {
		                        if (*curr_begin > *curr_col)
		                            curr_col = std::lower_bound(curr_col, cols_end, *curr_begin);
		                        else
		                            curr_begin = std::lower_bound(curr_begin, row_end, *curr_col);
		                    }
		                }

		                skip_sum:
		                if (x_uses_ix_arr) {
		                    if (unlikely(x[ix_arr[row]] == x[ix_arr[row+1]])) continue;
		                }
		                else {
		                    if (unlikely(x[row] == x[row+1])) continue;
		                }

		                vleft = 0;
		                vright = 0;
		                wright = wtot - wleft;
		                for (size_t col = 0; col < ncols_use; col++)
		                {
		                    sl = buffer_sum_left[col];
		                    vleft += sl * (sl / wleft);
		                    sr = buffer_sum_tot[col] - sl;
		                    vright += sr * (sr / wright);
		                }

		                this_gain = vleft + vright;
		                if (this_gain > best_gain)
		                {
		                    best_gain = this_gain;
		                    split_ix = row;
		                }
		            }
		        }
		    }

		    if (best_gain  <= -HUGE_VAL) return best_gain;

		    split_point = midpoint(x[ix_arr[split_ix]], x[ix_arr[split_ix+1]]);
		    return best_gain / wtot;
		}

		template <class real_t_, class real_t>
		real_t find_split_dens_shortform_t(real_t * x, size_t n, real_t & split_point)
		{
		    real_t best_gain = -HUGE_VAL;
		    size_t n_minus_one = n - 1;
		    real_t_ xmin = x[0];
		    real_t_ xmax = x[n-1];
		    real_t_ xleft, xright;
		    real_t_ xmid;
		    real_t this_gain;
		    size_t split_ix = 0;

		    for (size_t ix = 0; ix < n_minus_one; ix++)
		    {
		        if (x[ix] == x[ix+1]) continue;
		        xmid = (real_t_)x[ix] + ((real_t_)x[ix+1] - (real_t_)x[ix]) / (real_t_)2;
		        xleft = xmid - xmin;
		        xright = xmax - xmid;
		        if (unlikely(!xleft || !xright)) continue;
		        this_gain = (real_t_)square(ix+1) / xleft + (real_t_)square(n_minus_one - ix) / xright;
		        if (this_gain > best_gain)
		        {
		            best_gain = this_gain;
		            split_ix = ix;
		        }
		    }

		    if (best_gain <= -HUGE_VAL) return best_gain;

		    real_t_ xtot = (real_t_)xmax - (real_t_)xmin;
		    real_t_ nleft = (real_t_)(split_ix+1);
		    real_t_ nright = (real_t_)(n_minus_one - split_ix);
		    split_point = midpoint(x[split_ix], x[split_ix+1]);
		    real_t_ rpct_left = split_point / xtot;
		    rpct_left = std::fmax(rpct_left, std::numeric_limits<real_t>::min());
		    real_t_ rpct_right = (real_t_)1 - rpct_left;
		    rpct_right = std::fmax(rpct_right, std::numeric_limits<real_t>::min());

		    real_t_ nl_sq = nleft  / (real_t_)n; nl_sq = square(nl_sq);
		    real_t_ nr_sq = nright / (real_t_)n; nl_sq = square(nr_sq);

		    return nl_sq / rpct_left + nr_sq / rpct_right;
		}

		template <class real_t, class lreal_t_safe>
		real_t find_split_dens_shortform(real_t * x, size_t n, real_t & split_point)
		{
		    if (n < INT32_MAX)
		        return find_split_dens_shortform_t<real_t, real_t>(x, n, split_point);
		    else
		        return find_split_dens_shortform_t<lreal_t_safe, real_t>(x, n, split_point);
		}

		template <class real_t_, class real_t, class mapping>
		real_t find_split_dens_shortform_weighted_t(real_t * x, size_t n, real_t & split_point, mapping & w, size_t * buffer_indices)
		{
		    real_t best_gain = -HUGE_VAL;
		    size_t n_minus_one = n - 1;
		    real_t_ xmin = x[buffer_indices[0]];
		    real_t_ xmax = x[buffer_indices[n-1]];
		    real_t_ xleft, xright;
		    real_t_ xmid;
		    real_t this_gain;

		    real_t_ wtot = 0;
		    for (size_t ix = 0; ix < n; ix++)
		        wtot += w[buffer_indices[ix]];
		    real_t_ w_left = 0;
		    real_t_ w_right;
		    real_t_ best_w = 0;
		    size_t split_ix = 0;

		    for (size_t ix = 0; ix < n_minus_one; ix++)
		    {
		        w_left += w[buffer_indices[ix]];
		        if (x[buffer_indices[ix]] == x[buffer_indices[ix+1]]) continue;
		        xmid = (real_t_)x[buffer_indices[ix]] + ((real_t_)x[buffer_indices[ix+1]] - (real_t_)x[buffer_indices[ix]]) / (real_t_)2;
		        xleft = xmid - xmin;
		        xright = xmax - xmid;
		        if (unlikely(!xleft || !xright)) continue;

		        w_right = wtot - w_left;
		        this_gain = square(w_left) / xleft + square(w_right) / xright;
		        if (this_gain > best_gain)
		        {
		            best_gain = this_gain;
		            best_w = w_left;
		            split_ix = xmid;
		        }
		    }

		    if (best_gain <= -HUGE_VAL) return best_gain;

		    real_t_ xtot = xmax - xmin;
		    w_left = best_w;
		    w_right = wtot - w_left;
		    w_left = std::fmax(w_left, std::numeric_limits<real_t>::min());
		    w_right = std::fmax(w_right, std::numeric_limits<real_t>::min());
		    split_point = midpoint(x[buffer_indices[split_ix]], x[buffer_indices[split_ix+1]]);
		    real_t_ rpct_left = split_point / xtot;
		    rpct_left = std::fmax(rpct_left, std::numeric_limits<real_t>::min());
		    real_t_ rpct_right = (real_t_)1 - rpct_left;
		    rpct_right = std::fmax(rpct_right, std::numeric_limits<real_t>::min());

		    real_t_ wl_sq = w_left  / wtot; wl_sq = square(wl_sq);
		    real_t_ wr_sq = w_right / wtot; wl_sq = square(wr_sq);

		    return wl_sq / rpct_left + wr_sq / rpct_right;
		}

		template <class real_t, class mapping, class lreal_t_safe>
		real_t find_split_dens_shortform_weighted(real_t * x, size_t n, real_t & split_point, mapping & w, size_t * buffer_indices)
		{
		    if (n < INT32_MAX)
		        return find_split_dens_shortform_weighted_t<real_t, real_t, mapping>(x, n, split_point, w, buffer_indices);
		    else
		        return find_split_dens_shortform_weighted_t<lreal_t_safe, real_t, mapping>(x, n, split_point, w, buffer_indices);
		}

		template <class real_t>
		real_t find_split_dens_shortform(real_t * x, size_t * ix_arr, size_t st, size_t end,
		                                 real_t & split_point, size_t & split_ix)
		{
		    real_t best_gain = -HUGE_VAL;
		    real_t xmin = x[ix_arr[st]];
		    real_t xmax = x[ix_arr[end]];
		    real_t xleft, xright;
		    real_t xmid;
		    real_t this_gain;

		    for (size_t row = st; row < end; row++)
		    {
		        if (x[ix_arr[row]] == x[ix_arr[row+1]]) continue;
		        xmid = x[ix_arr[row]] + (x[ix_arr[row+1]] - x[ix_arr[row]]) / (real_t)2;
		        xleft = xmid - xmin;
		        xright = xmax - xmid;
		        if (unlikely(!xleft || !xright)) continue;
		        this_gain = square(row-st+1) / xleft + square(end-row) / xright;
		        if (this_gain > best_gain)
		        {
		            best_gain = this_gain;
		            split_ix = row;
		        }
		    }

		    if (best_gain <= -HUGE_VAL) return best_gain;

		    real_t xtot = (real_t)xmax - (real_t)xmin;
		    real_t nleft = (real_t)(split_ix-st+1);
		    real_t nright = (real_t)(end - split_ix);
		    split_point = midpoint(x[ix_arr[split_ix]], x[ix_arr[split_ix+1]]);
		    real_t rpct_left = split_point / xtot;
		    rpct_left = std::fmax(rpct_left, std::numeric_limits<real_t>::min());
		    real_t rpct_right = 1. - rpct_left;
		    rpct_right = std::fmax(rpct_right, std::numeric_limits<real_t>::min());
		    real_t ntot = (real_t)(end - st + 1);

		    real_t nl_sq = nleft  / ntot; nl_sq = square(nl_sq);
		    real_t nr_sq = nright / ntot; nl_sq = square(nr_sq);

		    return nl_sq / rpct_left + nr_sq / rpct_right;
		}

		template <class real_t, class mapping>
		real_t find_split_dens_shortform_weighted(real_t * x, size_t * ix_arr, size_t st, size_t end,
		                                          real_t & split_point, size_t & split_ix, mapping & w)
		{
		    real_t best_gain = -HUGE_VAL;
		    real_t xmin = x[ix_arr[st]];
		    real_t xmax = x[ix_arr[end]];
		    real_t xleft, xright;
		    real_t xmid;
		    real_t this_gain;

		    real_t wtot = 0;
		    for (size_t row = st; row <= end; row++)
		        wtot += w[ix_arr[row]];
		    real_t w_left = 0;
		    real_t w_right;
		    real_t best_w = 0;

		    for (size_t row = st; row < end; row++)
		    {
		        w_left += w[ix_arr[row]];
		        if (x[ix_arr[row]] == x[ix_arr[row+1]]) continue;
		        xmid = x[ix_arr[row]] + (x[ix_arr[row+1]] - x[ix_arr[row]]) / (real_t)2;
		        xleft = xmid - xmin;
		        xright = xmax - xmid;
		        if (unlikely(!xleft || !xright)) continue;

		        w_right = wtot - w_left;
		        this_gain = square(w_left) / xleft + square(w_right) / xright;
		        if (this_gain > best_gain)
		        {
		            best_gain = this_gain;
		            best_w = w_left;
		            split_ix = row;
		        }
		    }

		    if (best_gain <= -HUGE_VAL) return best_gain;

		    real_t xtot = (real_t)xmax - (real_t)xmin;
		    w_left = best_w;
		    w_right = wtot - w_left;
		    w_left = std::fmax(w_left, std::numeric_limits<real_t>::min());
		    w_right = std::fmax(w_right, std::numeric_limits<real_t>::min());
		    split_point = midpoint(x[split_ix], x[split_ix+1]);
		    real_t rpct_left = split_point / xtot;
		    rpct_left = std::fmax(rpct_left, std::numeric_limits<real_t>::min());
		    real_t rpct_right = 1. - rpct_left;
		    rpct_right = std::fmax(rpct_right, std::numeric_limits<real_t>::min());

		    real_t wl_sq = w_left  / wtot; wl_sq = square(wl_sq);
		    real_t wr_sq = w_right / wtot; wl_sq = square(wr_sq);

		    return wl_sq / rpct_left + wr_sq / rpct_right;
		}

		/* This is a slower but more numerically-robust form */
		template <class real_t_=real_t, class lreal_t_safe = long real_t>
		real_t find_split_dens_longform(real_t_ * x, size_t * ix_arr, size_t st, size_t end,
		                                real_t & split_point, size_t & split_ix)
		{
		    real_t best_gain = -HUGE_VAL;
		    real_t_ xmin = x[ix_arr[st]];
		    real_t_ xmax = x[ix_arr[end]];
		    real_t_ xleft, xright;
		    real_t_ xmid;
		    lreal_t_safe pct_left, pct_right;
		    lreal_t_safe rpct_left, rpct_right;
		    lreal_t_safe n_tot = end - st + 1;
		    lreal_t_safe xtot = (lreal_t_safe)xmax - (lreal_t_safe)xmin;
		    lreal_t_safe cnt_left;
		    real_t this_gain;

		    for (size_t row = st; row < end; row++)
		    {
		        if (x[ix_arr[row]] == x[ix_arr[row+1]]) continue;
		        xmid = midpoint(x[ix_arr[row]], x[ix_arr[row+1]]);
		        xleft = xmid - xmin;
		        xright = xmax - xmid;
		        if (unlikely(!xleft || !xright)) continue;

		        cnt_left = (lreal_t_safe)(row-st+1);

		        xleft = std::fmax(xleft, (real_t)std::numeric_limits<real_t>::min());
		        xright = std::fmax(xright, (real_t)std::numeric_limits<real_t>::min());
		        pct_left = cnt_left / n_tot;
		        pct_right = (lreal_t_safe)1 - pct_left;
		        rpct_left = (lreal_t_safe)xleft / xtot;
		        rpct_right = (lreal_t_safe)xright / xtot;

		        this_gain = square(pct_left) / rpct_left + square(pct_right) / rpct_right;
		        if (unlikely(is_na_or_inf(this_gain))) continue;
		        if (this_gain > best_gain)
		        {
		            best_gain = this_gain;
		            split_point = xmid;
		            split_ix = row;
		        }
		    }

		    return best_gain;
		}

		template <class real_t, class mapping, class lreal_t_safe>
		real_t find_split_dens_longform_weighted(real_t * x, size_t * ix_arr, size_t st, size_t end,
		                                         real_t & split_point, size_t & split_ix, mapping & w)
		{
		    real_t best_gain = -HUGE_VAL;
		    real_t xmin = x[ix_arr[st]];
		    real_t xmax = x[ix_arr[end]];
		    real_t xleft, xright;
		    real_t xmid;
		    lreal_t_safe pct_left, pct_right;
		    lreal_t_safe rpct_left, rpct_right;
		    lreal_t_safe xtot = (lreal_t_safe)xmax - (lreal_t_safe)xmin;
		    real_t this_gain;

		    lreal_t_safe wtot = 0;
		    for (size_t row = st; row <= end; row++)
		        wtot += w[ix_arr[row]];
		    lreal_t_safe w_left = 0;

		    for (size_t row = st; row < end; row++)
		    {
		        w_left += w[ix_arr[row]];
		        if (x[ix_arr[row]] == x[ix_arr[row+1]]) continue;
		        xmid = midpoint(x[ix_arr[row]], x[ix_arr[row+1]]);
		        xleft = xmid - xmin;
		        xright = xmax - xmid;
		        if (unlikely(!xleft || !xright)) continue;

		        xleft = std::fmax(xleft, (real_t)std::numeric_limits<real_t>::min());
		        xright = std::fmax(xright, (real_t)std::numeric_limits<real_t>::min());
		        pct_left = w_left / wtot;
		        pct_right = (lreal_t_safe)1 - pct_left;
		        rpct_left = (lreal_t_safe)xleft / xtot;
		        rpct_right = (lreal_t_safe)xright / xtot;

		        this_gain = square(pct_left) / rpct_left + square(pct_right) / rpct_right;
		        if (unlikely(is_na_or_inf(this_gain))) continue;
		        if (this_gain > best_gain)
		        {
		            best_gain = this_gain;
		            split_point = xmid;
		            split_ix = row;
		        }
		    }

		    return best_gain;
		}

		template <class real_t, class lreal_t_safe>
		real_t find_split_dens(real_t * x, size_t * ix_arr, size_t st, size_t end,
		                       real_t & split_point, size_t & split_ix)
		{
		    if (end - st + 1 < INT32_MAX && x[ix_arr[end]] - x[ix_arr[st]] >= 1)
		        return find_split_dens_shortform<real_t>(x, ix_arr, st, end, split_point, split_ix);
		    else
		        return find_split_dens_longform<real_t, lreal_t_safe>(x, ix_arr, st, end, split_point, split_ix);
		}

		template <class real_t, class mapping, class lreal_t_safe>
		real_t find_split_dens_weighted(real_t * x, size_t * ix_arr, size_t st, size_t end,
		                                real_t & split_point, size_t & split_ix, mapping & w)
		{
		    if (end - st + 1 < INT32_MAX && x[ix_arr[end]] - x[ix_arr[st]] >= 1)
		        return find_split_dens_shortform_weighted<real_t, mapping>(x, ix_arr, st, end, split_point, split_ix, w);
		    else
		        return find_split_dens_longform_weighted<real_t, mapping, lreal_t_safe>(x, ix_arr, st, end, split_point, split_ix, w);
		}

		template <class int_t, class lreal_t_safe>
		real_t find_split_dens_longform(int * x, int ncat, size_t * ix_arr, size_t st, size_t end,
		                                CategSplit cat_split_type, MissingAction missing_action,
		                                int & chosen_cat, signed char * split_categ, int * saved_cat_mode,
		                                size_t * buffer_cnt, int_t * buffer_indices)
		{
		    if (st >= end || ncat <= 1) return -HUGE_VAL;
		    size_t n_nas = 0;
		    int xval;

		    /* count categories */
		    memset(buffer_cnt, 0, sizeof(size_t) * ncat);
		    if (missing_action == Fail)
		    {
		        for (size_t row = st; row <= end; row++)
		            if (likely(x[ix_arr[row]] >= 0))
		                buffer_cnt[x[ix_arr[row]]]++;
		    }

		    else if (missing_action == Impute)
		    {
		        for (size_t row = st; row <= end; row++)
		        {
		            xval = x[ix_arr[row]];
		            if (unlikely(xval < 0))
		                n_nas++;
		            else
		                buffer_cnt[xval]++;
		        }

		        if (unlikely(n_nas >= end-st)) return -HUGE_VAL;

		        if (n_nas)
		        {
		            auto idxmax = std::max_element(buffer_cnt, buffer_cnt + ncat);
		            *idxmax += n_nas;
		            *saved_cat_mode = (int)std::distance(buffer_cnt, idxmax);
		        }
		    }

		    else
		    {
		        for (size_t row = st; row <= end; row++)
		        {
		            xval = x[ix_arr[row]];
		            if (likely(xval >= 0)) buffer_cnt[xval]++;
		        }
		    }

		    std::iota(buffer_indices, buffer_indices + ncat, (int_t)0);
		    std::sort(buffer_indices, buffer_indices + ncat,
		              [&buffer_cnt](const int_t a, const int_t b)
		              {return buffer_cnt[a] < buffer_cnt[b];});

		    int curr = 0;
		    if (split_categ != NULL)
		    {
		        while (buffer_cnt[buffer_indices[curr]] == 0)
		        {
		            split_categ[buffer_indices[curr]] = -1;
		            curr++;
		        }
		    }

		    else
		    {
		        while (buffer_cnt[buffer_indices[curr]] == 0) curr++;
		    }

		    int ncat_present = ncat - curr;
		    if (ncat_present <= 1) return -HUGE_VAL;
		    if (ncat_present == 2)
		    {
		        switch (cat_split_type)
		        {
		            case SingleCateg:
		            {
		                chosen_cat = buffer_indices[curr];
		                break;
		            }

		            case SubSet:
		            {
		                split_categ[buffer_indices[curr]] = 1;
		                split_categ[buffer_indices[curr+1]] = 0;
		                break;
		            }
		        }

		        lreal_t_safe pct_left
		            =
		        (lreal_t_safe)buffer_cnt[buffer_indices[curr]]
		            /
		        (lreal_t_safe)(
		            buffer_cnt[buffer_indices[curr]]
		                +
		            buffer_cnt[buffer_indices[curr+1]]
		        );

		        return  ((lreal_t_safe)buffer_cnt[buffer_indices[curr]] * (2. * pct_left)
		                     +
		                 (lreal_t_safe)buffer_cnt[buffer_indices[curr+1]] * (2. - 2.*pct_left))
		                    /
		                 (lreal_t_safe)(buffer_cnt[buffer_indices[curr]] + buffer_cnt[buffer_indices[curr+1]]);
		    }

		    size_t ntot;
		    if (missing_action == Impute)
		        ntot = end - st + 1;
		    else
		        ntot = std::accumulate(buffer_cnt, buffer_cnt + ncat, (size_t)0);
		    if (unlikely(ntot <= 1)) unexpected_error();
		    lreal_t_safe ntot_ = (lreal_t_safe)ntot;

		    switch (cat_split_type)
		    {
		        case SingleCateg:
		        {
		            real_t pct_one_cat = 1. / (real_t)ncat_present;
		            real_t pct_left_smallest = (lreal_t_safe)buffer_cnt[buffer_indices[curr]] / ntot_;
		            real_t gain_smallest
		                =
		            (lreal_t_safe)buffer_cnt[buffer_indices[curr]] * (pct_left_smallest / pct_one_cat)
		            +
		            (lreal_t_safe)(ntot - buffer_cnt[buffer_indices[curr]]) * ((1. - pct_left_smallest) / (1. - pct_one_cat))
		            ;

		            real_t pct_left_biggest = (lreal_t_safe)buffer_cnt[buffer_indices[ncat-1]] / ntot_;
		            real_t gain_biggest
		                =
		            (lreal_t_safe)buffer_cnt[buffer_indices[ncat-1]] * (pct_left_biggest / pct_one_cat)
		            +
		            (lreal_t_safe)(ntot - buffer_cnt[buffer_indices[ncat-1]]) * ((1. - pct_left_biggest) / (1. - pct_one_cat))
		            ;

		            if (gain_smallest >= gain_biggest)
		            {
		                chosen_cat = buffer_indices[curr];
		                return gain_smallest / ntot_;
		            }

		            else
		            {
		                chosen_cat = buffer_indices[ncat-1];
		                return gain_biggest / ntot_;
		            }
		            break;
		        }

		        case SubSet:
		        {
		            size_t cnt_left = 0;
		            size_t cnt_right;
		            int st_cat = curr - 1;
		            real_t this_gain;
		            real_t best_gain = -HUGE_VAL;
		            int best_cat = 0;
		            lreal_t_safe pct_left;
		            real_t pct_cat_left;
		            real_t ncat_present_ = (real_t)ncat_present;
		            for (; curr < ncat; curr++)
		            {
		                cnt_left += buffer_cnt[buffer_indices[curr]];
		                cnt_right = ntot - cnt_left;
		                pct_left = (lreal_t_safe)cnt_left / ntot_;
		                pct_cat_left = (real_t)(curr - st_cat) / ncat_present_;
		                this_gain
		                    =
		                (lreal_t_safe)cnt_left * (pct_left / pct_cat_left)
		                +
		                (lreal_t_safe)cnt_right * (((lreal_t_safe)1 - pct_left) / (1. - pct_cat_left))
		                ;
		                if (this_gain > best_gain)
		                {
		                    best_gain = this_gain;
		                    best_cat = curr;
		                }
		            }

		            if (best_gain <= -HUGE_VAL) return best_gain;
		            st_cat++;
		            for (; st_cat <= best_cat; st_cat++)
		                split_categ[buffer_indices[st_cat]] = 1;
		            for (; st_cat < ncat; st_cat++)
		                split_categ[buffer_indices[st_cat]] = 0;
		            return best_gain / ntot_;
		            break;
		        }
		    }

		    /* This will not be reached, but CRAN might complain otherwise */
		    return -HUGE_VAL;
		}

		template <class mapping, class int_t, class lreal_t_safe>
		real_t find_split_dens_longform_weighted(int * x, int ncat, size_t * ix_arr, size_t st, size_t end,
		                                         CategSplit cat_split_type, MissingAction missing_action,
		                                         int & chosen_cat, signed char * split_categ, int * saved_cat_mode,
		                                         int_t * buffer_indices, mapping & w)
		{
		    if (st >= end || ncat <= 1) return -HUGE_VAL;
		    lreal_t_safe w_missing = 0;
		    int xval;
		    size_t ix_;

		    /* count categories */
		    /* TODO: allocate this buffer externally */
		    std::vector<lreal_t_safe> buffer_cnt(ncat, (lreal_t_safe)0);
		    if (missing_action == Fail)
		    {
		        for (size_t row = st; row <= end; row++)
		        {
		            ix_ = ix_arr[row];
		            if (unlikely(x[ix_]) < 0) continue;
		            buffer_cnt[x[ix_]] += w[ix_];
		        }
		    }

		    else if (missing_action == Impute)
		    {
		        for (size_t row = st; row <= end; row++)
		        {
		            ix_ = ix_arr[row];
		            xval = x[ix_];
		            if (unlikely(xval < 0))
		                w_missing += w[ix_];
		            else
		                buffer_cnt[xval] += w[ix_];
		        }

		        if (w_missing)
		        {
		            auto idxmax = std::max_element(buffer_cnt.begin(), buffer_cnt.end());
		            *idxmax += w_missing;
		            *saved_cat_mode = (int)std::distance(buffer_cnt.begin(), idxmax);
		        }
		    }

		    else
		    {
		        for (size_t row = st; row <= end; row++)
		        {
		            ix_ = ix_arr[row];
		            xval = x[ix_];
		            if (likely(xval >= 0)) buffer_cnt[xval] += w[ix_];
		        }
		    }

		    std::iota(buffer_indices, buffer_indices + ncat, (int_t)0);
		    std::sort(buffer_indices, buffer_indices + ncat,
		              [&buffer_cnt](const int_t a, const int_t b)
		              {return buffer_cnt[a] < buffer_cnt[b];});

		    int curr = 0;
		    if (split_categ != NULL)
		    {
		        while (buffer_cnt[buffer_indices[curr]] == 0)
		        {
		            split_categ[buffer_indices[curr]] = -1;
		            curr++;
		        }
		    }

		    else
		    {
		        while (buffer_cnt[buffer_indices[curr]] == 0) curr++;
		    }

		    int ncat_present = ncat - curr;
		    if (ncat_present <= 1) return -HUGE_VAL;
		    if (ncat_present == 2)
		    {
		        switch (cat_split_type)
		        {
		            case SingleCateg:
		            {
		                chosen_cat = buffer_indices[curr];
		                break;
		            }

		            case SubSet:
		            {
		                split_categ[buffer_indices[curr]] = 1;
		                split_categ[buffer_indices[curr+1]] = 0;
		                break;
		            }
		        }

		        lreal_t_safe pct_left
		            =
		        buffer_cnt[buffer_indices[curr]]
		            /
		        (
		            buffer_cnt[buffer_indices[curr]]
		                +
		            buffer_cnt[buffer_indices[curr+1]]
		        );

		        return  (buffer_cnt[buffer_indices[curr]] * (pct_left * 2.)
		                     +
		                 buffer_cnt[buffer_indices[curr+1]] * (2. - 2.*pct_left))
		                    /
		                (buffer_cnt[buffer_indices[curr]] + buffer_cnt[buffer_indices[curr+1]]);
		    }

		    lreal_t_safe ntot = std::accumulate(buffer_cnt.begin(), buffer_cnt.end(), (lreal_t_safe)0);
		    if (unlikely(ntot <= 0)) unexpected_error();

		    switch (cat_split_type)
		    {
		        case SingleCateg:
		        {
		            real_t pct_one_cat = 1. / (real_t)ncat_present;
		            real_t pct_left_smallest = buffer_cnt[buffer_indices[curr]] / ntot;
		            real_t gain_smallest
		                =
		            buffer_cnt[buffer_indices[curr]] * (pct_left_smallest / pct_one_cat)
		            +
		            (ntot - buffer_cnt[buffer_indices[curr]]) * ((1. - pct_left_smallest) / (1. - pct_one_cat))
		            ;

		            real_t pct_left_biggest = buffer_cnt[buffer_indices[ncat-1]] / ntot;
		            real_t gain_biggest
		                =
		            buffer_cnt[buffer_indices[ncat-1]] * (pct_left_biggest / pct_one_cat)
		            +
		            (ntot - buffer_cnt[buffer_indices[ncat-1]]) * ((1. - pct_left_biggest) / (1. - pct_one_cat))
		            ;

		            if (gain_smallest >= gain_biggest)
		            {
		                chosen_cat = buffer_indices[curr];
		                return gain_smallest / ntot;
		            }

		            else
		            {
		                chosen_cat = buffer_indices[ncat-1];
		                return gain_biggest / ntot;
		            }
		            break;
		        }

		        case SubSet:
		        {
		            lreal_t_safe cnt_left = 0;
		            lreal_t_safe cnt_right;
		            int st_cat = curr - 1;
		            real_t this_gain;
		            real_t best_gain = -HUGE_VAL;
		            int best_cat = 0;
		            lreal_t_safe pct_left;
		            real_t pct_cat_left;
		            real_t ncat_present_ = (real_t)ncat_present;
		            for (; curr < ncat; curr++)
		            {
		                cnt_left += buffer_cnt[buffer_indices[curr]];
		                cnt_right = ntot - cnt_left;
		                pct_left = cnt_left / ntot;
		                pct_cat_left = (real_t)(curr - st_cat) / ncat_present_;
		                this_gain
		                    =
		                (lreal_t_safe)cnt_left * (pct_left / pct_cat_left)
		                +
		                (lreal_t_safe)cnt_right * (((lreal_t_safe)1 - pct_left) / (1. - pct_cat_left))
		                ;
		                if (this_gain > best_gain)
		                {
		                    best_gain = this_gain;
		                    best_cat = curr;
		                }
		            }

		            if (best_gain <= -HUGE_VAL) return best_gain;
		            st_cat++;
		            for (; st_cat <= best_cat; st_cat++)
		                split_categ[buffer_indices[st_cat]] = 1;
		            for (; st_cat < ncat; st_cat++)
		                split_categ[buffer_indices[st_cat]] = 0;
		            return best_gain / ntot;
		            break;
		        }
		    }

		    /* This will not be reached, but CRAN might complain otherwise */
		    return -HUGE_VAL;
		}
#endif
	/* for split-criterion in hyperplanes (see below for version aimed at single-variable splits) */
	template <class lreal_t_safe>
	real_t
	eval_guided_crit(real_t *x, size_t n, GainCriterion criterion,
					 real_t min_gain, bool as_relative_gain, real_t *buffer_sd,
					 real_t &split_point, real_t &xmin, real_t &xmax,
					 size_t *ix_arr_plus_st, size_t *cols_use,
					 size_t ncols_use, bool force_cols_use,
					 real_t *X_row_major, size_t ncols, real_t *Xr,
					 size_t *Xr_ind, size_t *Xr_indptr)
	{
		/* Note: the input 'x' is supposed to be a linear combination of standardized variables, so
		 all numbers are assumed to be small and in the same scale */
		real_t gain = 0;
		if (criterion == DensityCrit || criterion == FullGain)
			min_gain = 0;

		/* here it's assumed the 'x' vector matches exactly with 'ix_arr' + 'st' */
		if (unlikely(n == 2))
		{
			if (x[0] == x[1])
				return -HUGE_VAL;
			split_point = midpoint_with_reorder(x[0], x[1]);
			gain = 1.;
			if (gain > min_gain)
				return gain;
			else
				return 0.;
		}

		if (criterion == FullGain)
		{
			/* TODO: these buffers should be allocated externally */
			std::vector<size_t> argsorted(n);
			std::iota(argsorted.begin(), argsorted.end(), (size_t)0);
			std::sort(argsorted.begin(), argsorted.end(), [&x](const size_t a, const size_t b)
					  { return x[a] < x[b]; });
			if (x[argsorted[0]] == x[argsorted[n - 1]])
				return -HUGE_VAL;
			std::vector<real_t> temp_buffer(n + mult2(ncols));
			for (size_t ix = 0; ix < n; ix++)
				temp_buffer[ix] = x[argsorted[ix]];
			for (size_t ix = 0; ix < n; ix++)
				argsorted[ix] = ix_arr_plus_st[argsorted[ix]];
			size_t ignored;
			return find_split_full_gain<real_t, lreal_t_safe>(
				temp_buffer.data(), (size_t)0, n - 1, argsorted.data(),
				cols_use, ncols_use, force_cols_use, X_row_major, ncols, Xr,
				Xr_ind, Xr_indptr, temp_buffer.data() + n,
				temp_buffer.data() + n + ncols, ignored, split_point, false);
		}

		/* sort in ascending order */
		std::sort(x, x + n);
		xmin = x[0];
		xmax = x[n - 1];
		if (x[0] == x[n - 1])
			return -HUGE_VAL;

		if (criterion == Pooled && as_relative_gain && min_gain <= 0)
			gain = find_split_rel_gain<real_t, lreal_t_safe>(x, n, split_point);
		else if (criterion == Pooled || criterion == Averaged)
			gain = find_split_std_gain<real_t, lreal_t_safe>(x, n, buffer_sd,
															 criterion, min_gain,
															 split_point);
		else if (criterion == DensityCrit)
			gain = find_split_dens_shortform<real_t, lreal_t_safe>(x, n,
																   split_point);
		/* Note: a gain of -Inf signals that the data is unsplittable. Zero signals it's below the minimum. */
		return std::fmax(0., gain);
	}

	template <class lreal_t_safe>
	real_t
	eval_guided_crit_weighted(real_t *x, size_t n, GainCriterion criterion,
							  real_t min_gain, bool as_relative_gain,
							  real_t *buffer_sd, real_t &split_point,
							  real_t &xmin, real_t &xmax, real_t *w,
							  size_t *buffer_indices, size_t *ix_arr_plus_st,
							  size_t *cols_use, size_t ncols_use,
							  bool force_cols_use, real_t *X_row_major,
							  size_t ncols, real_t *Xr, size_t *Xr_ind,
							  size_t *Xr_indptr)
	{
		UNDEF_REFERENCE(as_relative_gain)
		UNDEF_REFERENCE2(as_relative_gain)

		/* Note: the input 'x' is supposed to be a linear combination of standardized variables, so
		 all numbers are assumed to be small and in the same scale */
		real_t gain = 0;
		if (criterion == DensityCrit || criterion == FullGain)
			min_gain = 0;

		/* here it's assumed the 'x' vector matches exactly with 'ix_arr' + 'st' */
		if (unlikely(n == 2))
		{
			if (x[0] == x[1])
				return -HUGE_VAL;
			split_point = midpoint_with_reorder(x[0], x[1]);
			gain = 1.;
			if (gain > min_gain)
				return gain;
			else
				return 0.;
		}

		/* sort in ascending order */
		std::iota(buffer_indices, buffer_indices + n, (size_t)0);
		std::sort(buffer_indices, buffer_indices + n, [&x](const size_t a, const size_t b)
				  { return x[a] < x[b]; });
		xmin = x[buffer_indices[0]];
		xmax = x[buffer_indices[n - 1]];
		if (xmin == xmax)
			return -HUGE_VAL;

		if (criterion == Pooled || criterion == Averaged)
			gain = find_split_std_gain_weighted<real_t, lreal_t_safe>(
				x, n, buffer_sd, criterion, min_gain, split_point, w,
				buffer_indices);
		else if (criterion == DensityCrit)
			gain =
				find_split_dens_shortform_weighted<real_t, real_t *, lreal_t_safe>(
					x, n, split_point, w, buffer_indices);
		else if (criterion == FullGain)
		{
			std::vector<size_t> argsorted(n);
			std::iota(argsorted.begin(), argsorted.end(), (size_t)0);
			std::sort(argsorted.begin(), argsorted.end(), [&x](const size_t a, const size_t b)
					  { return x[a] < x[b]; });
			if (x[argsorted[0]] == x[argsorted[n - 1]])
				return -HUGE_VAL;
			std::vector<real_t> temp_buffer(n + mult2(ncols));
			for (size_t ix = 0; ix < n; ix++)
				temp_buffer[ix] = x[argsorted[ix]];
			for (size_t ix = 0; ix < n; ix++)
				argsorted[ix] = ix_arr_plus_st[argsorted[ix]];
			size_t ignored;
			gain = find_split_full_gain_weighted<real_t, real_t *, lreal_t_safe>(
				temp_buffer.data(), (size_t)0, n - 1, argsorted.data(),
				cols_use, ncols_use, force_cols_use, X_row_major, ncols, Xr,
				Xr_ind, Xr_indptr, temp_buffer.data() + n,
				temp_buffer.data() + n + ncols, ignored, split_point, false, w);
		}
		/* Note: a gain of -Inf signals that the data is unsplittable. Zero signals it's below the minimum. */
		return std::fmax(0., gain);
	}

	/* for split-criterion in single-variable splits */
	template <class real_t_, class lreal_t_safe>
	real_t
	eval_guided_crit(size_t *ix_arr, size_t st, size_t end, real_t_ *x,
					 real_t *buffer_sd, bool as_relative_gain,
					 real_t *buffer_imputed_x, real_t *saved_xmedian,
					 size_t &split_ix, real_t &split_point, real_t &xmin,
					 real_t &xmax, GainCriterion criterion, real_t min_gain,
					 MissingAction missing_action, size_t *cols_use,
					 size_t ncols_use, bool force_cols_use,
					 real_t *X_row_major, size_t ncols, real_t *Xr,
					 size_t *Xr_ind, size_t *Xr_indptr)
	{
		size_t st_orig = st;
		real_t gain = 0;
		if (criterion == DensityCrit || criterion == FullGain)
			min_gain = 0;

		/* move NAs to the front if there's any, exclude them from calculations */
		if (missing_action != Fail)
			st = move_NAs_to_front(ix_arr, st, end, x);

		if (unlikely(st >= end))
			return -HUGE_VAL;
		else if (unlikely(st == (end - 1)))
		{
			if (x[ix_arr[st]] == x[ix_arr[end]])
				return -HUGE_VAL;
			split_point = midpoint_with_reorder(x[ix_arr[st]], x[ix_arr[end]]);
			split_ix = st;
			gain = 1.;
			if (gain > min_gain)
				return gain;
			else
				return 0.;
		}

		/* sort in ascending order */
		std::sort(ix_arr + st, ix_arr + end + 1, [&x](const size_t a, const size_t b)
				  { return x[a] < x[b]; });
		if (x[ix_arr[st]] == x[ix_arr[end]])
			return -HUGE_VAL;
		xmin = x[ix_arr[st]];
		xmax = x[ix_arr[end]];

		/* unlike the previous case for the extended model, the data here has not been centered,
		 which could make the standard deviations have poor precision. It's nevertheless not
		 necessary for this mean to have good precision, since it's only meant for centering,
		 so it can be calculated inexactly with simd instructions. */
		real_t_ xmean = 0;
		if (criterion == Pooled || criterion == Averaged)
		{
			for (size_t ix = st; ix <= end; ix++)
				xmean += x[ix_arr[ix]];
			xmean /= (real_t_)(end - st + 1);
		}

		if (missing_action == Impute && st > st_orig)
		{
			missing_action = Fail;
			fill_NAs_with_median(ix_arr, st_orig, st, end, x, buffer_imputed_x,
								 saved_xmedian);
			if (criterion == Pooled && as_relative_gain && min_gain <= 0)
				gain = find_split_rel_gain<real_t, lreal_t_safe>(buffer_imputed_x,
																 (real_t)xmean,
																 ix_arr, st_orig,
																 end, split_point,
																 split_ix);
			else if (criterion == Pooled || criterion == Averaged)
				gain = find_split_std_gain<real_t, lreal_t_safe>(buffer_imputed_x,
																 (real_t)xmean,
																 ix_arr, st_orig,
																 end, buffer_sd,
																 criterion,
																 min_gain,
																 split_point,
																 split_ix);
			else if (criterion == DensityCrit)
				gain = find_split_dens<real_t, lreal_t_safe>(buffer_imputed_x,
															 ix_arr, st_orig, end,
															 split_point,
															 split_ix);
			else if (criterion == FullGain)
			{
				/* TODO: this buffer should be allocated from outside */
				std::vector<real_t> temp_buffer(mult2(ncols));
				gain = find_split_full_gain<real_t, lreal_t_safe>(
					buffer_imputed_x, st_orig, end, ix_arr, cols_use, ncols_use,
					force_cols_use, X_row_major, ncols, Xr, Xr_ind, Xr_indptr,
					temp_buffer.data(), temp_buffer.data() + ncols, split_ix,
					split_point, true);
			}

			/* Note: in theory, it should be possible to use a faster version assuming a contiguous array for 'x',
			 but such an approach might give inexact split points. Better to avoid such inexactness at the
			 expense of more computations. */
		}

		else
		{
			if (criterion == Pooled && as_relative_gain && min_gain <= 0)
				gain = find_split_rel_gain<real_t_, lreal_t_safe>(x, xmean, ix_arr,
																  st, end,
																  split_point,
																  split_ix);
			else if (criterion == Pooled || criterion == Averaged)
				gain = find_split_std_gain<real_t_, lreal_t_safe>(x, xmean, ix_arr,
																  st, end,
																  buffer_sd,
																  criterion,
																  min_gain,
																  split_point,
																  split_ix);
			else if (criterion == DensityCrit)
				gain = find_split_dens<real_t_, lreal_t_safe>(x, ix_arr, st, end,
															  split_point,
															  split_ix);
			else if (criterion == FullGain)
			{
				/* TODO: this buffer should be allocated from outside */
				std::vector<real_t> temp_buffer(mult2(ncols));
				gain = find_split_full_gain<real_t_, lreal_t_safe>(
					x, st, end, ix_arr, cols_use, ncols_use, force_cols_use,
					X_row_major, ncols, Xr, Xr_ind, Xr_indptr,
					temp_buffer.data(), temp_buffer.data() + ncols, split_ix,
					split_point, true);
			}
		}

		/* Note: a gain of -Inf signals that the data is unsplittable. Zero signals it's below the minimum. */
		return std::fmax(0., gain);
	}
	template <class real_t_, class mapping, class lreal_t_safe>
	real_t
	eval_guided_crit_weighted(size_t *ix_arr, size_t st, size_t end,
							  real_t_ *x, real_t *buffer_sd,
							  bool as_relative_gain, real_t *buffer_imputed_x,
							  real_t *saved_xmedian, size_t &split_ix,
							  real_t &split_point, real_t &xmin, real_t &xmax,
							  GainCriterion criterion, real_t min_gain,
							  MissingAction missing_action, size_t *cols_use,
							  size_t ncols_use, bool force_cols_use,
							  real_t *X_row_major, size_t ncols, real_t *Xr,
							  size_t *Xr_ind, size_t *Xr_indptr, mapping &w)
	{
		size_t st_orig = st;
		real_t gain = 0;
		if (criterion == DensityCrit || criterion == FullGain)
			min_gain = 0;

		/* move NAs to the front if there's any, exclude them from calculations */
		if (missing_action != Fail)
			st = move_NAs_to_front(ix_arr, st, end, x);

		if (unlikely(st >= end))
			return -HUGE_VAL;
		else if (unlikely(st == (end - 1)))
		{
			if (x[ix_arr[st]] == x[ix_arr[end]])
				return -HUGE_VAL;
			split_point = midpoint_with_reorder(x[ix_arr[st]], x[ix_arr[end]]);
			split_ix = st;
			gain = 1.;
			if (gain > min_gain)
				return gain;
			else
				return 0.;
		}

		/* sort in ascending order */
		std::sort(ix_arr + st, ix_arr + end + 1, [&x](const size_t a, const size_t b)
				  { return x[a] < x[b]; });
		if (x[ix_arr[st]] == x[ix_arr[end]])
			return -HUGE_VAL;
		xmin = x[ix_arr[st]];
		xmax = x[ix_arr[end]];

		/* unlike the previous case for the extended model, the data here has not been centered,
		 which could make the standard deviations have poor precision. It's nevertheless not
		 necessary for this mean to have good precision, since it's only meant for centering,
		 so it can be calculated inexactly with simd instructions. */
		real_t_ xmean = 0;
		real_t_ cnt = 0;
		if (criterion == Pooled || criterion == Averaged)
		{
			for (size_t ix = st; ix <= end; ix++)
			{
				xmean += x[ix_arr[ix]];
				cnt += w[ix_arr[ix]];
			}
			xmean /= cnt;
		}

		if (missing_action == Impute && st > st_orig)
		{
			missing_action = Fail;
			fill_NAs_with_median(ix_arr, st_orig, st, end, x, buffer_imputed_x,
								 saved_xmedian);
			if (criterion == Pooled && as_relative_gain && min_gain <= 0)
				gain = find_split_rel_gain_weighted<real_t, mapping, lreal_t_safe>(
					buffer_imputed_x, (real_t)xmean, ix_arr, st_orig, end,
					split_point, split_ix, w);
			else if (criterion == Pooled || criterion == Averaged)
				gain = find_split_std_gain_weighted<real_t, mapping, lreal_t_safe>(
					buffer_imputed_x, (real_t)xmean, ix_arr, st_orig, end,
					buffer_sd, criterion, min_gain, split_point, split_ix, w);
			else if (criterion == DensityCrit)
				gain = find_split_dens_weighted<real_t, mapping, lreal_t_safe>(
					buffer_imputed_x, ix_arr, st_orig, end, split_point, split_ix,
					w);
			else if (criterion == FullGain)
			{
				std::vector<real_t> temp_buffer(mult2(ncols));
				gain =
					find_split_full_gain_weighted<real_t, mapping, lreal_t_safe>(
						buffer_imputed_x, st_orig, end, ix_arr, cols_use,
						ncols_use, force_cols_use, X_row_major, ncols, Xr, Xr_ind,
						Xr_indptr, temp_buffer.data(),
						temp_buffer.data() + ncols, split_ix, split_point, true,
						w);
			}
		}

		else
		{
			if (criterion == Pooled && as_relative_gain && min_gain <= 0)
				gain =
					find_split_rel_gain_weighted<real_t_, mapping, lreal_t_safe>(
						x, xmean, ix_arr, st, end, split_point, split_ix, w);
			else if (criterion == Pooled || criterion == Averaged)
				gain =
					find_split_std_gain_weighted<real_t_, mapping, lreal_t_safe>(
						x, xmean, ix_arr, st, end, buffer_sd, criterion, min_gain,
						split_point, split_ix, w);
			else if (criterion == DensityCrit)
				gain = find_split_dens_weighted<real_t_, mapping, lreal_t_safe>(
					x, ix_arr, st, end, split_point, split_ix, w);
			else if (criterion == FullGain)
			{
				std::vector<real_t> temp_buffer(mult2(ncols));
				gain = find_split_full_gain_weighted<real_t_, mapping,
													 lreal_t_safe>(x, st, end, ix_arr, cols_use, ncols_use,
																   force_cols_use, X_row_major, ncols, Xr, Xr_ind,
																   Xr_indptr, temp_buffer.data(),
																   temp_buffer.data() + ncols, split_ix,
																   split_point, true, w);
			}
		}

		/* Note: a gain of -Inf signals that the data is unsplittable. Zero signals it's below the minimum. */
		return std::fmax(0., gain);
	}
	/* TODO: here it should only need to look at the non-zero entries. It can then use the
	 same algorithm as above, but putting an extra check to see where do the zeros fit in
	 the sorted order of the non-zero entries while calculating gains and SDs, and then
	 call the 'divide_subset' function after-the-fact to reach the same end result.
	 It should be much faster than this if the non-zero entries are few. */
	template <class real_t_, class sparse_ix_, class lreal_t_safe>
	real_t
	eval_guided_crit(size_t ix_arr[], size_t st, size_t end, size_t col_num,
					 real_t_ Xc[], sparse_ix_ Xc_ind[], sparse_ix_ Xc_indptr[],
					 real_t buffer_arr[], size_t buffer_pos[],
					 bool as_relative_gain, real_t *saved_xmedian,
					 real_t &split_point, real_t &xmin, real_t &xmax,
					 GainCriterion criterion, real_t min_gain,
					 MissingAction missing_action, size_t *cols_use,
					 size_t ncols_use, bool force_cols_use,
					 real_t *X_row_major, size_t ncols, real_t *Xr,
					 size_t *Xr_ind, size_t *Xr_indptr)
	{
		size_t ignored;

		todense(ix_arr, st, end, col_num, Xc, Xc_ind, Xc_indptr, buffer_arr);
		size_t tot = end - st + 1;
		std::iota(buffer_pos, buffer_pos + tot, (size_t)0);

		if (missing_action == Impute)
		{
			missing_action = Fail;
			for (size_t ix = 0; ix < tot; ix++)
			{
				if (unlikely(is_na_or_inf(buffer_arr[ix])))
				{
					goto fill_missing;
				}
			}
			goto no_nas;

		fill_missing:
		{
			size_t idx_half = div2(tot);
			std::nth_element(buffer_pos, buffer_pos + idx_half,
							 buffer_pos + tot, [&buffer_arr](const size_t a, const size_t b)
							 { return buffer_arr[a] < buffer_arr[b]; });
			*saved_xmedian = buffer_arr[buffer_pos[idx_half]];

			if ((tot % 2) == 0)
			{
				real_t xlow = *std::max_element(buffer_pos,
												buffer_pos + idx_half);
				*saved_xmedian = xlow + ((*saved_xmedian) - xlow) / 2.;
			}

			for (size_t ix = 0; ix < tot; ix++)
				buffer_arr[ix] =
					is_na_or_inf(buffer_arr[ix]) ? (*saved_xmedian) : buffer_arr[ix];
			std::iota(buffer_pos, buffer_pos + tot, (size_t)0);
		}
		}

		no_nas:
		return eval_guided_crit<real_t, lreal_t_safe>(buffer_pos, 0, end - st,
													  buffer_arr,
													  buffer_arr + tot,
													  as_relative_gain,
													  saved_xmedian,
													  (real_t *)NULL, ignored,
													  split_point, xmin, xmax,
													  criterion, min_gain,
													  missing_action, cols_use,
													  ncols_use,
													  force_cols_use,
													  X_row_major, ncols, Xr,
													  Xr_ind, Xr_indptr);
	}

	template <class real_t_, class sparse_ix_, class mapping, class lreal_t_safe>
	real_t
	eval_guided_crit_weighted(size_t ix_arr[], size_t st, size_t end,
							  size_t col_num, real_t_ Xc[],
							  sparse_ix_ Xc_ind[], sparse_ix_ Xc_indptr[],
							  real_t buffer_arr[], size_t buffer_pos[],
							  bool as_relative_gain, real_t *saved_xmedian,
							  real_t &split_point, real_t &xmin, real_t &xmax,
							  GainCriterion criterion, real_t min_gain,
							  MissingAction missing_action, size_t *cols_use,
							  size_t ncols_use, bool force_cols_use,
							  real_t *X_row_major, size_t ncols, real_t *Xr,
							  size_t *Xr_ind, size_t *Xr_indptr, mapping &w)
	{
		size_t ignored;

		todense(ix_arr, st, end, col_num, Xc, Xc_ind, Xc_indptr, buffer_arr);
		size_t tot = end - st + 1;
		std::iota(buffer_pos, buffer_pos + tot, (size_t)0);

		if (missing_action == Impute)
		{
			missing_action = Fail;
			for (size_t ix = 0; ix < tot; ix++)
			{
				if (unlikely(is_na_or_inf(buffer_arr[ix])))
				{
					goto fill_missing;
				}
			}
			goto no_nas;

		fill_missing:
		{
			size_t idx_half = div2(tot);
			std::nth_element(buffer_pos, buffer_pos + idx_half,
							 buffer_pos + tot, [&buffer_arr](const size_t a, const size_t b)
							 { return buffer_arr[a] < buffer_arr[b]; });
			*saved_xmedian = buffer_arr[buffer_pos[idx_half]];

			if ((tot % 2) == 0)
			{
				real_t xlow = *std::max_element(buffer_pos,
												buffer_pos + idx_half);
				*saved_xmedian = xlow + ((*saved_xmedian) - xlow) / 2.;
			}

			for (size_t ix = 0; ix < tot; ix++)
				buffer_arr[ix] =
					is_na_or_inf(buffer_arr[ix]) ? (*saved_xmedian) : buffer_arr[ix];
			std::iota(buffer_pos, buffer_pos + tot, (size_t)0);
		}
		}

	no_nas:
		/* TODO: allocate this buffer externally */
		std::vector<real_t> buffer_w(tot);
		for (size_t row = st; row <= end; row++)
			buffer_w[row - st] = w[ix_arr[row]];
		/* TODO: in this case, as the weights match with the order of the indices, could use a faster version
		 with a weighted rel_gain function instead (not yet implemented). */
		return eval_guided_crit_weighted<real_t, std::vector<real_t>, lreal_t_safe>(
			buffer_pos, 0, end - st, buffer_arr, buffer_arr + tot,
			as_relative_gain, saved_xmedian, (real_t *)NULL, ignored, split_point,
			xmin, xmax, criterion, min_gain, missing_action, cols_use, ncols_use,
			force_cols_use, X_row_major, ncols, Xr, Xr_ind, Xr_indptr, buffer_w);
	}

	/* TODO: this kurtosis caps the minimum values to zero, but the minimum achievable value is 1,
	 see how are imprecise results used outside of the function in the different kind of calculations
	 that use kurtosis and maybe change the logic. */
	template <typename in>
	bool
	check_more_than_two_unique_values(size_t ix_arr[], size_t st, size_t end,
									  in x[], MissingAction missing_action)
	{
		if (end - st <= 1)
			return false;

		if (missing_action == Fail)
		{
			in x0 = x[ix_arr[st]];
			for (size_t ix = st + 1; ix <= end; ix++)
			{
				if (x[ix_arr[ix]] != x0)
					return true;
			}
		}

		else
		{
			in x0;
			size_t ix;
			for (ix = st; ix <= end; ix++)
			{
				if (x[ix_arr[ix]] >= 0)
				{
					x0 = x[ix_arr[ix]];
					ix++;
					break;
				}
			}

			for (; ix <= end; ix++)
			{
				if (x[ix_arr[ix]] >= 0 && x[ix_arr[ix]] != x0)
					return true;
			}
		}

		return false;
	}
	template <class InputData, class WorkerMemory, class lreal_t_safe>
	void
	fit_itree(std::vector<IsoTree> *tree_root,
			  std::vector<IsoHPlane> *hplane_root, WorkerMemory &workspace,
			  InputData &input_data, ModelParams &model_params,
			  std::vector<ImputeNode> *impute_nodes, size_t tree_num)
	{
		/* initialize array for depths if called for */
		if (workspace.ix_arr.empty() && model_params.calc_depth)
			workspace.row_depths.resize(input_data.nrows, 0);

		/* choose random sample of rows */
		if (workspace.ix_arr.empty())
			workspace.ix_arr.resize(model_params.sample_size);
		if (input_data.log2_n > 0)
			workspace.btree_weights.assign(input_data.btree_weights_init.begin(),
										   input_data.btree_weights_init.end());
		workspace.rnd_generator.seed(model_params.random_seed + tree_num);
		workspace.rbin = UniformUnitInterval(0, 1);
		sample_random_rows(
			workspace.ix_arr, input_data.nrows, model_params.with_replacement,
			workspace.rnd_generator, workspace.ix_all,
			(input_data.weight_as_sample) ? input_data.sample_weights : NULL,
			workspace.btree_weights, input_data.log2_n, input_data.btree_offset,
			workspace.is_repeated);
		workspace.st = 0;
		workspace.end = model_params.sample_size - 1;

		/* in some cases, it's not possible to use column weights even if they are given,
		 because every single column will always need to be checked or end up being used. */
		bool avoid_col_weights = (tree_root != NULL && model_params.ntry >= model_params.ncols_per_tree && model_params.prob_pick_by_gain_avg + model_params.prob_pick_by_gain_pl + model_params.prob_pick_by_full_gain + model_params.prob_pick_by_dens >= 1) || (tree_root == NULL && model_params.ndim >= model_params.ncols_per_tree) || (model_params.ncols_per_tree == 1);
		if (input_data.preinitialized_col_sampler == NULL)
		{
			if (input_data.col_weights != NULL && !avoid_col_weights && !model_params.weigh_by_kurt)
				workspace.col_sampler.initialize(input_data.col_weights,
												 input_data.ncols_tot);
		}

		/* set expected tree size and add root node */
		{
			size_t exp_nodes = mult2(model_params.sample_size);
			if (model_params.sample_size >= div2(SIZE_MAX))
				exp_nodes = SIZE_MAX;
			else if (model_params.max_depth <= (size_t)30)
				exp_nodes = std::min(exp_nodes, pow2(model_params.max_depth));
			if (tree_root != NULL)
			{
				tree_root->reserve(exp_nodes);
				tree_root->emplace_back();
			}
			else
			{
				hplane_root->reserve(exp_nodes);
				hplane_root->emplace_back();
			}
			if (impute_nodes != NULL)
			{
				impute_nodes->reserve(exp_nodes);
				impute_nodes->emplace_back((size_t)0);
			}
		}

		/* initialize array with candidate categories if not already done */
		if (workspace.categs.empty())
			workspace.categs.resize(input_data.max_categ);

		/* initialize array with per-node column weights if needed */
		if ((model_params.prob_pick_col_by_range || model_params.prob_pick_col_by_var || model_params.prob_pick_col_by_kurt) && workspace.node_col_weights.empty())
		{
			workspace.node_col_weights.resize(input_data.ncols_tot);
			if (tree_root != NULL || model_params.standardize_data || model_params.missing_action != Fail)
			{
				workspace.saved_stat1.resize(input_data.ncols_numeric);
				workspace.saved_stat2.resize(input_data.ncols_numeric);
			}
		}

		/* IMPORTANT!!!!!
		 The standard library implementation is likely going to use the Box-Muller method
		 for normal sampling, which has some state memory in the **distribution object itself**
		 in addition to the state memory from the RNG engine. DO NOT avoid re-generating this
		 object on each tree, despite being inefficient, because then it can cause seed
		 irreproducibility when the number of splitting dimensions is odd and the number
		 of threads is more than 1. This is a very hard issue to debug since everything
		 works fine depending on the order in which trees are assigned to threads.
		 DO NOT PUT THESE LINES BELOW THE NEXT IF. */
		if (hplane_root != NULL)
		{
			if (input_data.ncols_categ || model_params.coef_type == Normal)
				workspace.coef_norm = std::normal_distribution<real_t>(0, 1);
			if (model_params.coef_type == Uniform)
				workspace.coef_unif = std::uniform_real_distribution<real_t>(-1,
																			 1);
		}

		/* for the extended model, initialize extra vectors and objects */
		if (hplane_root != NULL && workspace.comb_val.empty())
		{
			workspace.comb_val.resize(model_params.sample_size);
			workspace.col_take.resize(model_params.ndim);
			workspace.col_take_type.resize(model_params.ndim);

			if (input_data.ncols_numeric)
			{
				workspace.ext_offset.resize(input_data.ncols_tot);
				workspace.ext_coef.resize(input_data.ncols_tot);
				workspace.ext_mean.resize(input_data.ncols_tot);
			}

			if (input_data.ncols_categ)
			{
				workspace.ext_fill_new.resize(input_data.max_categ);
				switch (model_params.cat_split_type)
				{
				case SingleCateg:
				{
					workspace.chosen_cat.resize(input_data.max_categ);
					break;
				}

				case SubSet:
				{
					workspace.ext_cat_coef.resize(input_data.ncols_tot);
					for (std::vector<real_t> &v : workspace.ext_cat_coef)
						v.resize(input_data.max_categ);
					break;
				}
				}
			}

			workspace.ext_fill_val.resize(input_data.ncols_tot);
		}

		/* If there are density weights, need to standardize them to sum up to
		 the sample size here. Note that weights for missing values with 'Divide'
		 are only initialized on-demand later on. */
		workspace.changed_weights = false;
		if (hplane_root == NULL)
			workspace.weights_map.clear();

		lreal_t_safe weight_scaling = 0;
		if (input_data.sample_weights != NULL && !input_data.weight_as_sample)
		{
			workspace.changed_weights = true;

			/* For the extended model, if there is no sub-sampling, these weights will remain
			 constant throughout and do not need to be re-generated. */
			if (!(hplane_root != NULL && (!workspace.weights_map.empty() || !workspace.weights_arr.empty()) && model_params.sample_size == input_data.nrows && !model_params.with_replacement))
			{
				workspace.weights_map.clear();

				/* if the sub-sample size is small relative to the full sample size, use a mapping */
				if (input_data.Xc_indptr != NULL && model_params.sample_size < input_data.nrows / 50)
				{
					for (const size_t ix : workspace.ix_arr)
						weight_scaling += input_data.sample_weights[ix];
					weight_scaling = (lreal_t_safe)model_params.sample_size / weight_scaling;
					workspace.weights_map.reserve(workspace.ix_arr.size());
					for (const size_t ix : workspace.ix_arr)
						workspace.weights_map[ix] = input_data.sample_weights[ix] * weight_scaling;
				}

				/* if the sub-sample size is large, fill a full array matching to the sample size */
				else
				{
					if (workspace.weights_arr.empty())
					{
						workspace.weights_arr.assign(
							input_data.sample_weights,
							input_data.sample_weights + input_data.nrows);
						weight_scaling =
							std::accumulate(
								workspace.ix_arr.begin(),
								workspace.ix_arr.end(),
								(lreal_t_safe)0,
								[&input_data](const lreal_t_safe a,
											  const size_t b)
								{ return a + (lreal_t_safe)input_data.sample_weights[b]; });
						weight_scaling = (lreal_t_safe)model_params.sample_size / weight_scaling;
						for (real_t &w : workspace.weights_arr)
							w *= weight_scaling;
					}

					else
					{
						for (const size_t ix : workspace.ix_arr)
						{
							weight_scaling += input_data.sample_weights[ix];
							workspace.weights_arr[ix] =
								input_data.sample_weights[ix];
						}
						weight_scaling = (lreal_t_safe)model_params.sample_size / weight_scaling;
						for (real_t &w : workspace.weights_arr)
							w *= weight_scaling;
					}
				}
			}
		}

		/* if producing distance/similarity, also need to initialize the triangular matrix */
		if (model_params.calc_dist && workspace.tmat_sep.empty())
			workspace.tmat_sep.resize(
				(input_data.nrows * (input_data.nrows - 1)) / 2, 0);

		/* make space for buffers if not already allocated */
		if ((model_params.prob_pick_by_gain_avg > 0 || model_params.prob_pick_by_gain_pl > 0 || model_params.prob_pick_by_full_gain > 0 || model_params.prob_pick_by_dens > 0 || model_params.prob_pick_col_by_range > 0 || model_params.prob_pick_col_by_var > 0 || model_params.prob_pick_col_by_kurt > 0 || model_params.weigh_by_kurt || hplane_root != NULL) && (workspace.buffer_dbl.empty() && workspace.buffer_szt.empty() && workspace.buffer_chr.empty()))
		{
			size_t min_size_dbl = 0;
			size_t min_size_szt = 0;
			size_t min_size_chr = 0;

			bool gain = model_params.prob_pick_by_gain_avg > 0 || model_params.prob_pick_by_gain_pl > 0 || model_params.prob_pick_by_full_gain > 0 || model_params.prob_pick_by_dens > 0;

			if (input_data.ncols_categ)
			{
				min_size_szt = (size_t)2 * (size_t)input_data.max_categ;
				min_size_dbl = input_data.max_categ + 1;
				if (gain && model_params.cat_split_type == SubSet)
					min_size_chr = input_data.max_categ;
			}

			if (input_data.Xc_indptr != NULL && gain)
			{
				min_size_szt = std::max(min_size_szt, model_params.sample_size);
				min_size_dbl = std::max(min_size_dbl, model_params.sample_size);
			}

			/* TODO: revisit if this covers all the cases */
			if (model_params.ntry > 1 || gain)
			{
				min_size_dbl = std::max(min_size_dbl, model_params.sample_size);
				if (model_params.ndim < 2 && input_data.Xc_indptr != NULL)
					min_size_dbl = std::max(min_size_dbl,
											(size_t)2 * model_params.sample_size);
			}

			/* for sampled column choices */
			if (model_params.prob_pick_col_by_var)
			{
				if (input_data.ncols_categ)
				{
					min_size_szt = std::max(min_size_szt,
											(size_t)input_data.max_categ + 1);
					min_size_dbl = std::max(min_size_dbl,
											(size_t)input_data.max_categ + 1);
				}
			}

			if (model_params.prob_pick_col_by_kurt)
			{
				if (input_data.ncols_categ)
				{
					min_size_szt = std::max(min_size_szt,
											(size_t)input_data.max_categ + 1);
					min_size_dbl = std::max(min_size_dbl,
											(size_t)input_data.max_categ);
				}
			}

			/* for the extended model */
			if (hplane_root != NULL)
			{
				min_size_dbl = std::max(
					min_size_dbl, pow2(log2ceil(input_data.ncols_tot) + 1));
				if (model_params.missing_action != Fail)
				{
					min_size_szt = std::max(min_size_szt,
											model_params.sample_size);
					min_size_dbl = std::max(min_size_dbl,
											model_params.sample_size);
				}

				if (input_data.ncols_categ && model_params.cat_split_type == SubSet)
				{
					min_size_szt = std::max(
						min_size_szt,
						(size_t)2 * (size_t)input_data.max_categ + (size_t)1);
					min_size_dbl = std::max(min_size_dbl,
											(size_t)input_data.max_categ);
				}

				if (model_params.weigh_by_kurt)
					min_size_szt = std::max(min_size_szt, input_data.ncols_tot);

				if (gain && (!workspace.weights_arr.empty() || !workspace.weights_map.empty()))
				{
					workspace.sample_weights.resize(model_params.sample_size);
					min_size_szt = std::max(min_size_szt,
											model_params.sample_size);
				}
			}

			/* now resize */
			if (workspace.buffer_dbl.size() < min_size_dbl)
				workspace.buffer_dbl.resize(min_size_dbl);

			if (workspace.buffer_szt.size() < min_size_szt)
				workspace.buffer_szt.resize(min_size_szt);

			if (workspace.buffer_chr.size() < min_size_chr)
				workspace.buffer_chr.resize(min_size_chr);

			/* for guided column choice, need to also remember the best split so far */
			if (model_params.cat_split_type == SubSet && (model_params.prob_pick_by_gain_avg || model_params.prob_pick_by_gain_pl || model_params.prob_pick_by_full_gain || model_params.prob_pick_by_dens))
			{
				workspace.this_split_categ.resize(input_data.max_categ);
			}
		}

		/* Other potentially necessary buffers */
		if (tree_root != NULL && model_params.missing_action == Impute && (model_params.prob_pick_by_gain_avg || model_params.prob_pick_by_gain_pl || model_params.prob_pick_by_full_gain || model_params.prob_pick_by_dens) && input_data.Xc_indptr == NULL && input_data.ncols_numeric && workspace.imputed_x_buffer.empty())
		{
			workspace.imputed_x_buffer.resize(input_data.nrows);
		}

		if (model_params.prob_pick_by_full_gain && workspace.col_indices.empty())
			workspace.col_indices.resize(model_params.ncols_per_tree);

		if ((model_params.prob_pick_col_by_range || model_params.prob_pick_col_by_var) && model_params.weigh_by_kurt && model_params.sample_size == input_data.nrows && !model_params.with_replacement && (model_params.ncols_per_tree == input_data.ncols_tot) && !input_data.all_kurtoses.empty())
		{
			workspace.tree_kurtoses = input_data.all_kurtoses.data();
		}
		else
		{
			workspace.tree_kurtoses = NULL;
		}

		/* weigh columns by kurtosis in the sample if required */
		/* TODO: this one could probably be refactored to use the function in the helpers */
		std::vector<real_t> kurt_weights;
		bool avoid_leave_m_cols = false;
		if (model_params.weigh_by_kurt && !avoid_col_weights && (input_data.preinitialized_col_sampler == NULL || ((model_params.prob_pick_col_by_range || model_params.prob_pick_col_by_var) && workspace.tree_kurtoses == NULL)))
		{
			kurt_weights.resize(
				input_data.ncols_numeric + input_data.ncols_categ, 0.);

			if (model_params.ncols_per_tree >= input_data.ncols_tot)
			{

				if (input_data.Xc_indptr == NULL)
				{

					for (size_t col = 0; col < input_data.ncols_numeric; col++)
					{
						if (workspace.weights_arr.empty() && workspace.weights_map.empty())
							kurt_weights[col] = calc_kurtosis<
								typename std::remove_pointer<
									decltype(input_data.numeric_data)>::type,
								lreal_t_safe>(
								workspace.ix_arr.data(), workspace.st,
								workspace.end,
								input_data.numeric_data + col * input_data.nrows,
								model_params.missing_action);
						else if (!workspace.weights_arr.empty())
							kurt_weights[col] = calc_kurtosis_weighted<
								typename std::remove_pointer<
									decltype(input_data.numeric_data)>::type,
								decltype(workspace.weights_arr), lreal_t_safe>(
								workspace.ix_arr.data(), workspace.st,
								workspace.end,
								input_data.numeric_data + col * input_data.nrows,
								model_params.missing_action, workspace.weights_arr);
						else
							kurt_weights[col] = calc_kurtosis_weighted<
								typename std::remove_pointer<
									decltype(input_data.numeric_data)>::type,
								decltype(workspace.weights_map), lreal_t_safe>(
								workspace.ix_arr.data(), workspace.st,
								workspace.end,
								input_data.numeric_data + col * input_data.nrows,
								model_params.missing_action, workspace.weights_map);
					}
				}

				else
				{
					std::sort(workspace.ix_arr.begin(),
							  workspace.ix_arr.end());
					for (size_t col = 0; col < input_data.ncols_numeric; col++)
					{
						if (workspace.weights_arr.empty() && workspace.weights_map.empty())
							kurt_weights[col] =
								calc_kurtosis<
									typename std::remove_pointer<
										decltype(input_data.Xc)>::type,
									typename std::remove_pointer<
										decltype(input_data.Xc_indptr)>::type,
									lreal_t_safe>(workspace.ix_arr.data(),
												  workspace.st, workspace.end, col,
												  input_data.Xc, input_data.Xc_ind,
												  input_data.Xc_indptr,
												  model_params.missing_action);
						else if (!workspace.weights_arr.empty())
							kurt_weights[col] =
								calc_kurtosis_weighted<
									typename std::remove_pointer<
										decltype(input_data.Xc)>::type,
									typename std::remove_pointer<
										decltype(input_data.Xc_indptr)>::type,
									decltype(workspace.weights_arr), lreal_t_safe>(
									workspace.ix_arr.data(), workspace.st,
									workspace.end, col, input_data.Xc,
									input_data.Xc_ind, input_data.Xc_indptr,
									model_params.missing_action,
									workspace.weights_arr);
						else
							kurt_weights[col] =
								calc_kurtosis_weighted<
									typename std::remove_pointer<
										decltype(input_data.Xc)>::type,
									typename std::remove_pointer<
										decltype(input_data.Xc_indptr)>::type,
									decltype(workspace.weights_map), lreal_t_safe>(
									workspace.ix_arr.data(), workspace.st,
									workspace.end, col, input_data.Xc,
									input_data.Xc_ind, input_data.Xc_indptr,
									model_params.missing_action,
									workspace.weights_map);
					}
				}

				for (size_t col = 0; col < input_data.ncols_categ; col++)
				{
					if (workspace.weights_arr.empty() && workspace.weights_map.empty())
						kurt_weights[col + input_data.ncols_numeric] =
							calc_kurtosis<lreal_t_safe>(
								workspace.ix_arr.data(), workspace.st,
								workspace.end,
								input_data.categ_data + col * input_data.nrows,
								input_data.ncat[col], workspace.buffer_szt.data(),
								workspace.buffer_dbl.data(),
								model_params.missing_action,
								model_params.cat_split_type,
								workspace.rnd_generator);
					else if (!workspace.weights_arr.empty())
						kurt_weights[col + input_data.ncols_numeric] =
							calc_kurtosis_weighted<decltype(workspace.weights_arr),
												   lreal_t_safe>(
								workspace.ix_arr.data(), workspace.st,
								workspace.end,
								input_data.categ_data + col * input_data.nrows,
								input_data.ncat[col], workspace.buffer_dbl.data(),
								model_params.missing_action,
								model_params.cat_split_type,
								workspace.rnd_generator, workspace.weights_arr);
					else
						kurt_weights[col + input_data.ncols_numeric] =
							calc_kurtosis_weighted<decltype(workspace.weights_map),
												   lreal_t_safe>(
								workspace.ix_arr.data(), workspace.st,
								workspace.end,
								input_data.categ_data + col * input_data.nrows,
								input_data.ncat[col], workspace.buffer_dbl.data(),
								model_params.missing_action,
								model_params.cat_split_type,
								workspace.rnd_generator, workspace.weights_map);
				}

				for (auto &w : kurt_weights)
					w = (w == -HUGE_VAL) ? 0. : std::fmax(1e-8, -1. + w);
				if (input_data.col_weights != NULL)
				{
					for (size_t col = 0; col < input_data.ncols_tot; col++)
					{
						if (kurt_weights[col] <= 0)
							continue;
						kurt_weights[col] *= input_data.col_weights[col];
						kurt_weights[col] = std::fmax(kurt_weights[col], 1e-100);
					}
				}
				workspace.col_sampler.initialize(kurt_weights.data(),
												 kurt_weights.size());
			}

			else
			{
				std::vector<size_t> cols_take(model_params.ncols_per_tree);
				std::vector<size_t> buffer1;
				std::vector<bool> buffer2;
				sample_random_rows(cols_take, input_data.ncols_tot, false,
								   workspace.rnd_generator, buffer1,
								   (real_t *)NULL, kurt_weights, /* <- will not get used */
								   (size_t)0, (size_t)0, buffer2);

				if (model_params.sample_size == input_data.nrows && !model_params.with_replacement && !input_data.all_kurtoses.empty())
				{
					for (size_t col : cols_take)
						kurt_weights[col] = input_data.all_kurtoses[col];
					goto skip_kurt_calculations;
				}

				if (input_data.Xc_indptr != NULL)
					std::sort(workspace.ix_arr.begin(), workspace.ix_arr.end());

				for (size_t col : cols_take)
				{
					if (col < input_data.ncols_numeric)
					{
						if (input_data.Xc_indptr == NULL)
						{
							if (workspace.weights_arr.empty() && workspace.weights_map.empty())
								kurt_weights[col] = calc_kurtosis<
									typename std::remove_pointer<
										decltype(input_data.numeric_data)>::type,
									lreal_t_safe>(
									workspace.ix_arr.data(),
									workspace.st,
									workspace.end,
									input_data.numeric_data + col * input_data.nrows,
									model_params.missing_action);
							else if (!workspace.weights_arr.empty())
								kurt_weights[col] = calc_kurtosis_weighted<
									typename std::remove_pointer<
										decltype(input_data.numeric_data)>::type,
									decltype(workspace.weights_arr), lreal_t_safe>(
									workspace.ix_arr.data(),
									workspace.st,
									workspace.end,
									input_data.numeric_data + col * input_data.nrows,
									model_params.missing_action,
									workspace.weights_arr);
							else
								kurt_weights[col] = calc_kurtosis_weighted<
									typename std::remove_pointer<
										decltype(input_data.numeric_data)>::type,
									decltype(workspace.weights_map), lreal_t_safe>(
									workspace.ix_arr.data(),
									workspace.st,
									workspace.end,
									input_data.numeric_data + col * input_data.nrows,
									model_params.missing_action,
									workspace.weights_map);
						}

						else
						{
							if (workspace.weights_arr.empty() && workspace.weights_map.empty())
								kurt_weights[col] = calc_kurtosis<
									typename std::remove_pointer<
										decltype(input_data.Xc)>::type,
									typename std::remove_pointer<
										decltype(input_data.Xc_indptr)>::type,
									lreal_t_safe>(workspace.ix_arr.data(),
												  workspace.st, workspace.end, col,
												  input_data.Xc, input_data.Xc_ind,
												  input_data.Xc_indptr,
												  model_params.missing_action);
							else if (!workspace.weights_arr.empty())
								kurt_weights[col] = calc_kurtosis_weighted<
									typename std::remove_pointer<
										decltype(input_data.Xc)>::type,
									typename std::remove_pointer<
										decltype(input_data.Xc_indptr)>::type,
									decltype(workspace.weights_arr), lreal_t_safe>(
									workspace.ix_arr.data(), workspace.st,
									workspace.end, col, input_data.Xc,
									input_data.Xc_ind, input_data.Xc_indptr,
									model_params.missing_action,
									workspace.weights_arr);
							else
								kurt_weights[col] = calc_kurtosis_weighted<
									typename std::remove_pointer<
										decltype(input_data.Xc)>::type,
									typename std::remove_pointer<
										decltype(input_data.Xc_indptr)>::type,
									decltype(workspace.weights_map), lreal_t_safe>(
									workspace.ix_arr.data(), workspace.st,
									workspace.end, col, input_data.Xc,
									input_data.Xc_ind, input_data.Xc_indptr,
									model_params.missing_action,
									workspace.weights_map);
						}
					}

					else
					{
						if (workspace.weights_arr.empty() && workspace.weights_map.empty())
							kurt_weights[col] = calc_kurtosis<lreal_t_safe>(
								workspace.ix_arr.data(),
								workspace.st,
								workspace.end,
								input_data.categ_data + (col - input_data.ncols_numeric) * input_data.nrows,
								input_data.ncat[col - input_data.ncols_numeric],
								workspace.buffer_szt.data(),
								workspace.buffer_dbl.data(),
								model_params.missing_action,
								model_params.cat_split_type,
								workspace.rnd_generator);
						else if (!workspace.weights_arr.empty())
							kurt_weights[col] = calc_kurtosis_weighted<
								decltype(workspace.weights_arr), lreal_t_safe>(
								workspace.ix_arr.data(),
								workspace.st,
								workspace.end,
								input_data.categ_data + (col - input_data.ncols_numeric) * input_data.nrows,
								input_data.ncat[col - input_data.ncols_numeric],
								workspace.buffer_dbl.data(),
								model_params.missing_action,
								model_params.cat_split_type,
								workspace.rnd_generator, workspace.weights_arr);
						else
							kurt_weights[col] = calc_kurtosis_weighted<
								decltype(workspace.weights_map), lreal_t_safe>(
								workspace.ix_arr.data(),
								workspace.st,
								workspace.end,
								input_data.categ_data + (col - input_data.ncols_numeric) * input_data.nrows,
								input_data.ncat[col - input_data.ncols_numeric],
								workspace.buffer_dbl.data(),
								model_params.missing_action,
								model_params.cat_split_type,
								workspace.rnd_generator, workspace.weights_map);
					}

					/* Note to self: don't move this  to outside of the braces, as it needs to assign a weight
					 of zero to the columns that were not selected, thus it should only do this clipping
					 for columns that are chosen. */
					if (kurt_weights[col] == -HUGE_VAL)
					{
						kurt_weights[col] = 0;
					}

					else
					{
						kurt_weights[col] = std::fmax(1e-8,
													  -1. + kurt_weights[col]);
						if (input_data.col_weights != NULL)
						{
							kurt_weights[col] *= input_data.col_weights[col];
							kurt_weights[col] = std::fmax(kurt_weights[col],
														  1e-100);
						}
					}
				}

			skip_kurt_calculations:
				workspace.col_sampler.initialize(kurt_weights.data(),
												 kurt_weights.size());
				avoid_leave_m_cols = true;
			}

			if (model_params.prob_pick_col_by_range || model_params.prob_pick_col_by_var)
			{
				workspace.tree_kurtoses = kurt_weights.data();
			}
		}

		bool col_sampler_is_fresh = true;
		if (input_data.preinitialized_col_sampler == NULL)
		{
			workspace.col_sampler.initialize(input_data.ncols_tot);
		}
		else
		{
			workspace.col_sampler =
				*((column_sampler<lreal_t_safe> *)input_data.preinitialized_col_sampler);
			col_sampler_is_fresh = false;
		}
		/* TODO: this can be done more efficiently when sub-sampling columns */
		if (!avoid_leave_m_cols)
			workspace.col_sampler.leave_m_cols(model_params.ncols_per_tree,
											   workspace.rnd_generator);
		if (model_params.ncols_per_tree < input_data.ncols_tot)
			col_sampler_is_fresh = false;
		workspace.try_all = false;
		if (hplane_root != NULL && model_params.ndim >= input_data.ncols_tot)
			workspace.try_all = true;

		if (model_params.scoring_metric != Depth && !is_boxed_metric(model_params.scoring_metric))
		{
			workspace.density_calculator.initialize(
				model_params.max_depth,
				input_data.ncols_categ ? input_data.max_categ : 0,
				tree_root != NULL && input_data.ncols_categ,
				model_params.scoring_metric);
		}

		else if (is_boxed_metric(model_params.scoring_metric))
		{
			if (tree_root != NULL)
				workspace.density_calculator.initialize_bdens(
					input_data, model_params, workspace.ix_arr,
					workspace.col_sampler);
			else
				workspace.density_calculator.initialize_bdens_ext(
					input_data, model_params, workspace.ix_arr,
					workspace.col_sampler, col_sampler_is_fresh);
		}

		if (tree_root != NULL)
		{
			split_itree_recursive<InputData, WorkerMemory, lreal_t_safe>(
				*tree_root, workspace, input_data, model_params, impute_nodes, 0);
		}
		else
		{

			//prevent crash on emplace_back
 
			split_hplane_recursive<InputData, WorkerMemory, lreal_t_safe>(
				*hplane_root, workspace, input_data, model_params, impute_nodes,
				0);
		}

		/* if producing imputation structs, only need to keep the ones for terminal nodes */
		if (impute_nodes != NULL)
			drop_nonterminal_imp_node(*impute_nodes, tree_root, hplane_root);
	}

	template <class InputData, typename lreal_t_safe>
	std::vector<real_t>
	calc_kurtosis_all_data(InputData &input_data, ModelParams &model_params,
						   RNG_engine &rnd_generator)
	{
		std::unique_ptr<real_t[]> buffer_real_t;
		std::unique_ptr<size_t[]> buffer_size_t;
		if (input_data.ncols_categ)
		{
			buffer_real_t = std::unique_ptr<real_t[]>(
				new real_t[input_data.max_categ]);
			if (!(input_data.sample_weights != NULL && !input_data.weight_as_sample))
				buffer_size_t = std::unique_ptr<size_t[]>(
					new size_t[input_data.max_categ + 1]);
		}
		std::vector<real_t> kurt_weights(
			input_data.ncols_numeric + input_data.ncols_categ);
		for (size_t col = 0; col < input_data.ncols_tot; col++)
		{

			if (col < input_data.ncols_numeric)
			{
				if (input_data.Xc_indptr == NULL)
				{
					if (!(input_data.sample_weights != NULL && !input_data.weight_as_sample))
					{
						kurt_weights[col] = calc_kurtosis<lreal_t_safe>(
							input_data.numeric_data + col * input_data.nrows,
							input_data.nrows, model_params.missing_action);
					}

					else
					{
						kurt_weights[col] = calc_kurtosis_weighted<
							typename std::remove_pointer<
								decltype(input_data.numeric_data)>::type,
							lreal_t_safe>(
							input_data.numeric_data + col * input_data.nrows,
							input_data.nrows, model_params.missing_action,
							input_data.sample_weights);
					}
				}

				else
				{
					if (!(input_data.sample_weights != NULL && !input_data.weight_as_sample))
					{
						kurt_weights[col] =
							calc_kurtosis<
								typename std::remove_pointer<
									decltype(input_data.Xc)>::type,
								typename std::remove_pointer<
									decltype(input_data.Xc_indptr)>::type,
								lreal_t_safe>(col, input_data.nrows,
											  input_data.Xc, input_data.Xc_ind,
											  input_data.Xc_indptr,
											  model_params.missing_action);
					}

					else
					{
						kurt_weights[col] =
							calc_kurtosis_weighted<
								typename std::remove_pointer<
									decltype(input_data.Xc)>::type,
								typename std::remove_pointer<
									decltype(input_data.Xc_indptr)>::type,
								lreal_t_safe>(col, input_data.nrows,
											  input_data.Xc, input_data.Xc_ind,
											  input_data.Xc_indptr,
											  model_params.missing_action,
											  input_data.sample_weights);
					}
				}
			}
			else
			{
				if (!(input_data.sample_weights != NULL && !input_data.weight_as_sample))
				{
					kurt_weights[col] = calc_kurtosis<lreal_t_safe>(
						input_data.nrows,
						input_data.categ_data + (col - input_data.ncols_numeric) * input_data.nrows,
						input_data.ncat[col - input_data.ncols_numeric],
						buffer_size_t.get(), buffer_real_t.get(),
						model_params.missing_action, model_params.cat_split_type,
						rnd_generator);
				}

				else
				{
					kurt_weights[col] = calc_kurtosis_weighted<
						typename std::remove_pointer<
							decltype(input_data.sample_weights)>::type,
						lreal_t_safe>(
						input_data.nrows,
						input_data.categ_data + (col - input_data.ncols_numeric) * input_data.nrows,
						input_data.ncat[col - input_data.ncols_numeric],
						buffer_real_t.get(), model_params.missing_action,
						model_params.cat_split_type, rnd_generator,
						input_data.sample_weights);
				}
			}
		}

		for (auto &w : kurt_weights)
			w = (w == -HUGE_VAL) ? 0. : std::fmax(1e-8, -1. + w);
		if (input_data.col_weights != NULL)
		{
			for (size_t col = 0; col < input_data.ncols_tot; col++)
			{
				if (kurt_weights[col] <= 0)
					continue;
				kurt_weights[col] *= input_data.col_weights[col];
				kurt_weights[col] = std::fmax(kurt_weights[col], 1e-100);
			}
		}

		return kurt_weights;
	}

	tree_classifier::tree_classifier(const classifier *deserial) : classifier(deserial)
	{
		// Set tree
		//      const TreeNodeBase& deserial_tree = deserial->GetExtension(tree_classifier::child).tree();
		//     _tree.deserialize(&deserial_tree);
	}

	ensemble_classifier::ensemble_classifier(const classifier *deserial) : classifier(deserial)
	{
		// Check size
		/*const ensemble_classifier& ensemble_buffer = deserial->GetExtension(ensemble_classifier::child);
		 assert(ensemble_buffer.error_size() == ensemble_buffer.classifiers_size());
		 int size = ensemble_buffer.classifiers_size();
		 // Resize internal arrays
		 _error.resize(size);
		 _classifiers.resize(size);
		 // Get data from buffer
		 for(int i = 0 ; i < size ; ++i) {
		 _error[i] = ensemble_buffer.error(i);
		 const classifier& classifier_buffer = ensemble_buffer.classifiers(i);
		 _classifiers[i] = Classifier::buildClassifier(&classifier_buffer);

		 }
		 */
	}

	void
	ensemble_classifier::serialize(classifier *serial) const
	{

		if (serial->get_type() != this->get_type())
			throw std::runtime_error("Trying to serialize a classifier with the wrong type");
		if (serial == this)
			return;
#if 0
      // Get ensemble classifier buffer
      ensemble_classifier* ensemble_buffer(serial->MutableExtension(ensemble_classifier::child));
      // Write classifiers on the buffer
      for(size_t i = 0 ; i < _classifiers.size() ; ++i) {
          classifier* serial_class = ensemble_buffer->add_classifiers();
          _classifiers[i]->serialize(serial_class);
      }
      // Write errors
      for(size_t i = 0 ; i < _error.size() ; ++i)
          ensemble_buffer->add_error(_error[i]);

#endif
	}

	void
	ensemble_classifier::print(std::ostream &out) const
	{
		out << "Classifiers : " << std::endl;
		for (size_t i = 0; i < _classifiers.size(); ++i)
		{
			out << "weight = " << getWeight(i) << std::endl;
			out << *_classifiers[i] << std::endl;
			out << "----" << std::endl;
		}
	}

	ensemble_classifier::~ensemble_classifier()
	{
		// Delete classifiers
		for (size_t i = 0; i < _classifiers.size(); ++i)
			if ( _classifiers[i] ) 
				delete  _classifiers[i];
		_classifiers.clear();
	}

	// bayesian constructor :
	//

	// No parameters for now provallo::bayesian::bayesian(provallo::dataset&, const provallo::parameter_base&, std::random_device&, const provallo::split_method_factory&)

	bayesian::bayesian(const dataset &data, const parameter_base &parameters,
					   const std::random_device &random, const std::ostream &out, split_method_factory *factory) : bayesian(data, parameters, random, factory)
	{

		// check if ostream is good
		if (!out.good())
			throw std::runtime_error("Output stream is not good");
		init();
		// check split was initialized
	}
	void
	bayesian::init()
	{
		_name = "bayesian";

		// Split method for target tag
		const split_method *target_split = splitFactory().getTargetMethod();
		// Get some counts
		size_t class_number(target_split->size());
		size_t attrs_number(splitFactory().getSize());

		// Prior probability, number elements equal to the number of classes
		_prior.resize(class_number, 0.0);

		// Resize likelihood
		_likelihood.resize(attrs_number); // Number of rows
		for (size_t i = 0; i < _likelihood.size(); ++i)
			_likelihood[i].resize(class_number); // Number of columns

		// For each class + attribute, resize the probability array
		for (size_t i = 0; i < _likelihood.size(); ++i)		 // loop over attributes
			for (size_t j = 0; j < _likelihood[i].size(); ++j) // loop over classes
				{
					const split_method* method = splitFactory().getMethod(i);
					if(method!=nullptr)
						_likelihood[i][j].resize(splitFactory().getMethod(i)->size(), 0.0);
					else
						_likelihood[i][j].resize(1, 0.0);
				}

		// Iterate over each instance
		for (size_t i = 0; i < _data.size(); ++i)
		{
			// Get iterator to the attributes of this instance

			// Get target
			size_t target_branch = target_split->getBranch((_data.begin(i)));
			// Contribute to prior probability
			_prior[target_branch]++;

			// Loop over each attribute
			for (size_t j = 0; j < attrs_number; ++j)
			{
				// Get attribute value and branch
				//skip unspilttable or ignored attributes
				const split_method* method = splitFactory().getMethod(j);
				if(method==nullptr)
					continue;
				
				size_t attr_branch = method->getBranch(
					(_data.begin(i)));
				
				if ( attr_branch > method->size() )
				{
					//branch out of range, continue
					continue;
				}
				// Finally, contribute on likelihood
				if(_likelihood[j].size()<target_branch || _likelihood[j][target_branch].size() < method->size()) // resize the likelihood 
				{
					_likelihood[j].resize(target_branch+1);
					_likelihood[j][target_branch].resize(attr_branch+1, 0.0);
				}
				
				if(j<_likelihood.size() && target_branch<_likelihood[j].size() && attr_branch<_likelihood[j][target_branch].size() )
					_likelihood[j][target_branch][attr_branch]++;

			}
		}

		// Normalize all the probabilities
		float total_count = (float)_data.size();

		// Likelihood normalization
		for (size_t i = 0; i < attrs_number; ++i)
		{
			for (size_t j = 0; j < class_number; ++j)
			{
				// First check if we need to apply the laplacian correction for this attribute
				bool apply_laplacian(false);
				for (size_t k = 0; k < _likelihood[i][j].size(); ++k)
					if (_likelihood[i][j][k] == 0.0)
					{
						apply_laplacian = true;
						break;
					}

				if (apply_laplacian)
				{
					// Apply laplacian correction
					for (size_t k = 0; k < _likelihood[i][j].size(); ++k)
						_likelihood[i][j][k] = (_likelihood[i][j][k] + 1) / (_prior[j] + _likelihood[i][j].size());
				}
				else
				{
					// Don't apply laplacian correction
					for (size_t k = 0; k < _likelihood[i][j].size(); ++k)
						_likelihood[i][j][k] /= _prior[j];
				}
			}
		}

		// Finally, the prior probability normalization
		for (size_t j = 0; j < class_number; ++j)
			_prior[j] /= total_count;
	}
	// No parameters for now
	bayesian::bayesian(const dataset &data, const parameter_base &parameters,
					   const std::random_device &random, split_method_factory *factory)
		: classifier(data, parameters, random, factory)
	{
		init();
	}

	// Print classifier information
	void
	bayesian::print(std::ostream &out) const
	{
		out << " - Naive Bayes classifier " << std::endl;
	}

	// Internal function to serialize data into the buffer
	void
	bayesian::serialize(classifier *serial) const
	{
		// Get ensemble classifier buffer
		if (serial == this)
			return;

		if (serial->get_type() != this->get_type())
			throw std::runtime_error("Cannot serialize into a different classifier type");
	}


	bayesian::bayesian(const classifier *deserial) : classifier(deserial)
	{
		// Check size
		/*const NaiveBayes& nb_buffer = deserial->GetExtension(bayesian::child);

		 // Fill prior arrays
		 for(int i = 0 ; i < nb_buffer.prior_size() ; ++i)
		 _prior.push_back(nb_buffer.prior(i));

		 // Split method for target tag
		 const split_method* target_split = splitFactory().getTargetMethod();
		 // Get some counts
		 size_t class_number(target_split->size());
		 size_t attrs_number(splitFactory().getSize());
		 // Resize likelihood
		 _likelihood.resize(attrs_number); // Number of rows
		 for(size_t i = 0 ; i < _likelihood.size() ; ++i)
		 _likelihood[i].resize(class_number); // Number of columns

		 // Fill likelihood array
		 int cnt(0);
		 for(size_t i = 0 ; i < _likelihood.size() ; ++i) { // loop over attributes
		 const split_method* attr_split = splitFactory().getMethod(i);
		 for(size_t j = 0 ; j < _likelihood[i].size() ; ++j) { // loop over classes
		 // Loop over attributes values
		 for(size_t k = 0 ; k < attr_split->size() ; ++k) {
		 _likelihood[i][j].push_back(nb_buffer.likelihood(cnt));
		 ++cnt;
		 }
		 }
		 }
		 */
	}

	// Get type of classifier
	classifier_type
	bayesian::get_type() const
	{
		return BAYES;
	}

	bayesian::~bayesian()
	{
		// Nothing to do
		_likelihood.clear();

	}
	real_t &
    bayesian::getPrior(uint32_t class_value)
	{
		return _prior[class_value];
	}
    const real_t &
    bayesian::getPrior(uint32_t class_value) const
	{
		if (class_value >= _prior.size())
			throw std::runtime_error("Class value out of range");
		return _prior[class_value];
	}
 

	std::ostream &
	operator<<(std::ostream &out, const classifier &q)
	{
		q.print(out);
		return out;
	}

	TreeNodeBase::~TreeNodeBase()
	{
		// Delete children
		for (TreeNodeBase *child : _children)
			delete child;
		_children.clear();
		//don't delete split method. it may be reused.
		// Delete split method, since this is an unique instance
		//if (_split_method != NULL)
		//	delete _split_method;
	}
	void
	TreeLightLeaf::_print(std::ostream &out,
						  const attribute_information &information) const
	{
		// Print name and information of the distribution on the leaf
		out << "(";
		_split_method->printName(out, information);
		out << " = ";
		_split_method->printBranch(out, information, _target_tag);
		out << ")" << std::endl;
	}
	void
	TreeLightNode::_print(std::ostream &out,
						  const attribute_information &information) const
	{
		// Just print the name
		out << "(";
		_split_method->printName(out, information);
		out << ")" << std::endl;
	}
	void
	TreeNode::_print(std::ostream &out,
					 const attribute_information &information) const
	{
		// Just print the name
		out << "(";
		_split_method->printName(out, information);
		out << ") - " << (size_t)_distribution.sum() << std::endl;
	}
	void
	TreeLeaf::_print(std::ostream &out,
					 const attribute_information &information) const
	{
		// Print name and information of the distribution on the leaf
		out << "(";
		_split_method->printName(out, information);
		out << " = ";
		_split_method->printBranch(out, information,
								   _distribution.mode().discrete());
		out << ") ==> " << (size_t)_distribution.sum() << std::endl;
	}

	void
	TreeNodeBase::print(std::ostream &out,
						const attribute_information &information,
						int count) const
	{
		// Print internal data
		_print(out, information);
		// Increment level
		++count;
		// Children count
		discrete_value child(0);
		// Print children
		for (auto it = _children.begin(); it != _children.end(); ++it)
		{
			// Print tabulation
			for (int i = 0; i < count - 1; ++i)
				out << "    ";
			// Print information of this branch
			out << "|-- ";
			_split_method->printBranch(out, information, child);
			out << " ";
			// Recursively print the tree
			(*it)->print(out, information, count);
			// Advance branch
			++child;
		}
	}

	void
	TreeLightLeaf::serialize(TreeNodeBase *node) const
	{
		if (node == this){
			return;
		}
		if (node->get_type() != this->get_type())
			return;
	
	}
	void
	TreeLightLeaf::deserialize(const TreeNodeBase *node)
	{
		if (node == this)
			return;
		if (node->get_type() != this->get_type())
			return;
	
	}

	void
	TreeNode::serialize(TreeNodeBase *node) const
	{
		// Get tree node buffer
		if (node == this)
			return;
		if (node->get_type() != this->get_type())
			return;
	}

	void
	TreeNode::deserialize(const TreeNodeBase *node)
	{
		if (!node || node->get_type() != this->get_type())
			return;
	}

	void
	TreeNodeBase::serialize(TreeNodeBase *node) const
	{


		if (!node || node == this)
			return;
		if (node->get_type() != this->get_type())
			return;
		// Get tree node buffer
		// Serialize split methods
		//      _split_method->serialize(split_method_buffer);
		// Put type
		//  node->set_type(_getType());
		// Put internal base class data into buffer
		// Recursively apply the serialization process to the node's children
		//   for(std::vector<TreeNodeBase*>::const_iterator it = _children.begin() ; it != _children.end() ; ++it) {
		//     TreeNodeBase* child_node = node->add_children();
		//   (*it)->serialize(child_node);
		//}
	}

	void
	TreeNodeBase::deserialize(const TreeNodeBase *node)
	{
		// Create split method
		_split_method = split_method_factory::createMethod(
			node->get_split_method());
		// Get internal data
		//_dezeerialize(node);
		// Recursively create each child
		// for(int i = 0 ; i < node->children_size() ; ++i) {
		//  TreeNodeBase* child = TreeNodeBase::nodeBuilder(this, &node->children(i));
		//          _children.push_back(child);
		//    }
	}
	TreeNodeBase *
	TreeNodeBase::nodeBuilder(const TreeNodeBase *parent, const TreeNodeBase *node)
	{
		// build node from parent
		if (parent != node)
		{
			TreeNodeBase *newNode = new TreeNode(parent);
			newNode->deserialize(node);
			return newNode;
		}
		else
		{
			TreeNodeBase *newNode = new TreeLeaf(parent);
			newNode->deserialize(node);
			return newNode;
		}
	}

	size_t
	testClassifier(const classifier &_classifier, const dataset &data)
	{
		// Target tag
		attribute_tag target_tag = data.getattributes().get_target_tag();
		// Total error
		size_t error(0);

#ifdef HAVE_OPENMP
#pragma omp parallel
		{
			size_t local_error(0);

#pragma omp for
			for (size_t i = 0; i < data.size(); ++i)
			{
				// Get target value on the test data
				attribute test_attr(*(data.begin(i) + target_tag));
				// Classify the data
				attribute class_attr(_classifier.classify(data.begin(i), data.end(i)));
				// Check value
				if (test_attr.discrete() != class_attr.discrete())
					++local_error;
			}

#pragma omp critical
			error += local_error;
		}
#else
		// Test data
		for (size_t i = 0; i < data.size(); ++i)
		{
			// Get target value on the test data
			attribute test_attr (data.getattribute(i,target_tag));
			// Classify the data
			attribute class_attr(
				_classifier.classify(data.begin(i), data.end(i)));
			// Check value
			if (test_attr.discrete() != class_attr.discrete())
				++error;
		}
#endif
		// Return error
		return error;
	} // testClassifier

	// ROC curve
	roc_curve::roc_curve(const dataset &data, const classifier &classifier_)  
		// Target tag
		{
		// Target tag
		attribute_tag target_tag = data.getattributes().get_target_tag();
		// Get number of target attributes
		size_t target_count = data.getattributes().getTargetClassCount();
		real_t total_false_positive(0);
		real_t total_true_positive(0);
		real_t total_false_negative(0);
		real_t total_true_negative(0);

		// Allocate matrix
		_matrix.resize(target_count, target_count);
		// Test data
		for (size_t i = 0; i < data.size(); ++i)
		{
			// Get target value on the test data
			attribute test_attr(data.getattribute(i, target_tag));
			
			 
			// Classify the data

			attribute class_attr(
				classifier_.classify(data.begin(i), data.end(i)));
		 

			//update TPR/FPR/TNR/FNR
			if (test_attr.discrete() == class_attr.discrete())
			{
				if (test_attr.discrete() == 0)
					++total_true_negative;
				else
					++total_true_positive;
			}
			else
			{
				if (test_attr.discrete() == 0)
					++total_false_positive;
				else
					++total_false_negative;
			}	
				
				//update confusion matrix and update roc_curve
				++_matrix(test_attr.discrete(), class_attr.discrete());
			
			//update roc_curve
			_roc_points.push_back(std::make_pair(total_false_positive / (total_false_positive + total_true_negative),
												total_true_positive / (total_true_positive + total_false_negative)));

		}

	 
	} // roc_curve	

		
	
 
	// Confusion matrix

	confusion_matrix::confusion_matrix(const dataset &data,
									   const classifier &classifier_) : _matrix(), _class_values(),_fp(0),_fn(0),_tp(0),_tn(0), _total(0),_error(0), _accuracy(0.), _precision(0.), _recall(0.), _f1(0.) 
	{

		// Target tag
		attribute_tag target_tag = data.getattributes().get_target_tag();
		// Get number of target attributes
		size_t target_count = data.getattributes().getCount(target_tag);
		//fill class values 
		_class_values.resize(target_count);

		for (size_t i = 0; i < target_count; ++i)
			_class_values[i] = data.getattributes().getValue(target_tag,attribute( discrete_value(i) ));
		

		// Allocate matrix
		//_matrix = mmap_vector<mmap_vector<size_t>>(
		//			target_count, mmap_vector<size_t>(target_count, 0));
 		_matrix.resize(target_count,target_count);
		_matrix.clear();//set 0
		 
		 
		// Test data
		for (size_t i = 0; i < data.size(); ++i)
		{
			//Update total
			++_total;

			// Get target value on the test data
			attribute test_attr (data.getattribute(i ,target_tag));
			//translate the attribute : _class_values
		 	if(test_attr.discrete()>=_matrix.size1())
			{
			 
					test_attr  = discrete_value(0);

 			}

			attribute res(classifier_.classify(data.begin(i) , data.end(i)));
			if ( res.discrete() >=_matrix.size2())
				{
					attribute test (data.getattributes().getValue(target_tag,res));
					if(test.is_discrete()&&test.discrete()<_matrix.size2()) 
						res = test;
					else 
						//translate empty to 0.
						res  = discrete_value(0);
						
				}

	 		//update confusion matrix
			++_matrix(test_attr.discrete(), res.discrete());

			//update error count 
			if (test_attr.discrete() != res.discrete())
				++_error;
			

			//update tp,fp,tn,fn
			if (test_attr.discrete() == res.discrete())
			{
				if (test_attr.discrete() == 0)
					++_tn;
				else
					++_tp;
			}
			else
			{
				if (test_attr.discrete() == 0)
					++_fp;
				else
					++_fn;
			}	
			//update roc_curve
			
		}// for

		//update accuracy,precision,recall,f1
		_accuracy = (real_t)(_tp + _tn) / _total;
		_precision = (real_t)_tp / (_tp + _fp);
		_recall = (real_t)_tp / (_tp + _fn);
		_f1 = 2 * _precision * _recall / (_precision + _recall);
		
 	}// confusion_matrix

	//copy constructor
	confusion_matrix::confusion_matrix(const confusion_matrix &rhs) : _matrix(rhs._matrix), _class_values(rhs._class_values),_fp(rhs._fp),_fn(rhs._fn),_tp(rhs._tp),_tn(rhs._tn), _total(rhs._total),_error(rhs._error), _accuracy(rhs._accuracy), _precision(rhs._precision), _recall(rhs._recall), _f1(rhs._f1) 
	{
		_class_values.resize(rhs._class_values.size());
		for (size_t i = 0; i < rhs._class_values.size(); ++i)
			_class_values[i] = rhs._class_values[i];
	}
	//move constructor
	confusion_matrix::confusion_matrix(confusion_matrix &&rhs) : _matrix(std::move(rhs._matrix)), _class_values(std::move(rhs._class_values)),_fp(rhs._fp),_fn(rhs._fn),_tp(rhs._tp),_tn(rhs._tn), _total(rhs._total),_error(rhs._error), _accuracy(rhs._accuracy), _precision(rhs._precision), _recall(rhs._recall), _f1(rhs._f1) 
	{
		//update class_values 
		_class_values.resize(rhs._class_values.size());
		for (size_t i = 0; i < rhs._class_values.size(); ++i)
			_class_values[i] = rhs._class_values[i];

		//nothing to do
	}
    confusion_matrix &confusion_matrix::operator=(const confusion_matrix &other)
	{
		if(!(other==*this))
		{
			_matrix = other._matrix;
			_class_values = other._class_values;
			_fp = other._fp;
			_fn = other._fn;
			_tp = other._tp;
			_tn = other._tn;
			_total = other._total;
			_error = other._error;
			_accuracy = other._accuracy;
			_precision = other._precision;
			_recall = other._recall;
			_f1 = other._f1;
		}
		_class_values.resize(other._class_values.size());
		for (size_t i = 0; i < other._class_values.size(); ++i)
			_class_values[i] = other._class_values[i];

		return *this;
	}
    //move assignment
    confusion_matrix &confusion_matrix::operator=(confusion_matrix &&other)
	{
		if(!(other==*this))
		{
			_matrix = std::move(other._matrix);
			_class_values = std::move(other._class_values);
			_fp = other._fp;
			_fn = other._fn;
			_tp = other._tp;
			_tn = other._tn;
			_total = other._total;
			_error = other._error;
			_accuracy = other._accuracy;
			_precision = other._precision;
			_recall = other._recall;
			_f1 = other._f1;
		}
		_class_values.resize(other._class_values.size());
		for (size_t i = 0; i < other._class_values.size(); ++i)
			_class_values[i] = other._class_values[i];
	
		return *this;
	} 

	size_t
	confusion_matrix::getError() const
	{
		size_t error = 0;
		for (size_t i = 0; i < _matrix.size1(); ++i)
			for (size_t j = 0; j < _matrix.size2(); ++j)
				if (i != j)
					error += _matrix(i, j);
		return error;
	}
 
	std::ostream &
	operator<<(std::ostream &out, const confusion_matrix &q)
	{
 
		size_t target_count =  q.getMatrixDim();
		if(q._class_values.size()<target_count)
		{
			std::cerr << "class values size is different than matrix dimension" << std::endl;
		}
		// Print header
		out << std::setw(10) << "+";
		for (size_t i = 0; (i < target_count&& i<q._class_values.size()); ++i)
			out << std::setw(10)
				//<< "(" + q._attributes.getValue(target_tag, attribute(discrete_value(i))) + ")";
				<< "(" + q._class_values[i] + ")";
		out << std::endl;

		// Print matrix
		for (size_t i = 0; i < target_count; ++i)
		{
			out << std::setw(10)
				//<< "(" + q._attributes.getValue(target_tag, attribute(discrete_value(i))) + ")";
				<< "(" + q._class_values[i] + ")";

			for (size_t j = 0; j < target_count; ++j)
				out << std::setw(10) << q._matrix[i][j];
			out << std::endl;
		}
		//Print accuracy,precision,recall,f1
		out << std::setw(10) << "Accuracy:" << std::setw(10) << q._accuracy << std::endl;
		out << std::setw(10) << "Precision:" << std::setw(10) << q._precision << std::endl;
		out << std::setw(10) << "Recall:" << std::setw(10) << q._recall << std::endl;
		out << std::setw(10) << "F1:" << std::setw(10) << q._f1 << std::endl;
		
		return out;
	}

	// isoforest classifier implementation:
	//  Constructor

	//	Constructor

	iso_classifier::iso_classifier(const dataset &data) : classifier(data), _data(data) , 

      _params(	100, // trees	
	  			1000 ,10,
                    256,129,10,
                    0.,50, std::chrono::system_clock::now().time_since_epoch().count()  ),
	
      _isoforest(nullptr),
      _class_dist(data.get_values(data.getattributes().get_target_tag()).size() )
	{

		static uint64_t _seed = std::chrono::system_clock::now().time_since_epoch().count();

		// size_t target_tag= data.getattributes().get_target_tag();
		// std::cout << "target tag " << target_tag << std::endl;


		//_trees = 100;
		//_subsample_size = 256;
		//_max_depth = 10;
		validate_data();

		_seed += std::chrono::system_clock::now().time_since_epoch().count();

		// create isoforest and train it
		_isoforest = new isolation_forest(1,100,true,1);
		//_isoforest->set_max_depth(_max_depth);
		//_isoforest->set_seed(_seed>>32&0xFFFFFFFF|_seed&0xFFFFFFFF );

		matrix<real_t> training_data = transform_data();
		_isoforest->fit(training_data.begin(), training_data.rows(), training_data.cols()); // std::cout << "res size " << res.size() << std::endl;

		// std::cout << "res " << res[0] << std::endl;

	} // Constructor
	// Copy constructor
	iso_classifier::iso_classifier(const iso_classifier &rhs) : classifier(rhs), _data(rhs._data)
	 ,_params(rhs._params), _isoforest(rhs._isoforest),_class_dist(rhs._class_dist)		
	{
		validate_data();
		// create isoforest and train it
	}
	// Move constructor
	iso_classifier::iso_classifier(iso_classifier &&rhs) : classifier(std::move((classifier &&)rhs)), _data(std::move(rhs._data)),
														   _params(rhs._params),  _isoforest(std::move(rhs._isoforest)),_class_dist(std::move(rhs._class_dist))

	{

		validate_data();
		// create isoforest and train it
	}
	// transform dataset to matrix:
	provallo::matrix<real_t> iso_classifier::transform_data() const
	{
		provallo::matrix<real_t> samples(_data.size() / _data.getattributes().getSize(), _data.getattributes().getSize());
		for (size_t i = 0; i < _data.size(); ++i)
			for (size_t j = 0; j < _data.getattributes().getSize(); ++j)
				samples(i, j) = _data.getattribute(i / _data.getattributes().getSize(), j).continous();
		return samples;
	}

	// Train classifier
	// classify sample
	class_dist iso_classifier::classify(const std::vector<attribute> &data) const
	{

		// check data
		size_t cnt = _data.getattributes().getSize();

		if (data.size() < cnt)
			throw std::runtime_error("isoforest_classifier::classify(): Data size does not match number of attributes");
		// classify

		provallo::matrix<real_t> samples(data.size() / cnt, cnt);

		for (size_t i = 0; i < data.size(); ++i)
			for (size_t j = 0; j < classifier::_attributes_info.getSize(); j++)
				samples(i / cnt, j) = data[(i * cnt) + j].continous();

		size_t nclasses = classifier::_attributes_info.getCount(classifier::_attributes_info.get_target_tag());

		std::vector<real_t> result_probabilities = _isoforest->predict(samples);
		class_dist result(nclasses);
		class_dist return_result(result_probabilities.size());

		for (size_t i = 0; i < result_probabilities.size(); ++i)
		{
			return_result.set(i, result_probabilities[i]);
		}
		return result;
	}

	class_dist iso_classifier::posterior(const std::vector<attribute> &data) const
	{
		return classify(data);
	}	

	// Destructor

	iso_classifier::~iso_classifier()
	{
		if (_isoforest != nullptr)
			delete _isoforest;
		_isoforest = nullptr;
	}

	// Validate data
	void iso_classifier::validate_data()
	{

		// Get target tag
		attribute_tag target_tag = classifier::_attributes_info.get_target_tag();
		// Get number of target attributes
		size_t target_count = classifier::_attributes_info.getCount(target_tag);
		// Check number of target attributes
		if (target_count != 1)
			throw std::runtime_error("isoforest_classifier::isoforest_classifier(): Only one target attribute is allowed");

		// Get number of attributes
		size_t attributes_count = classifier::_attributes_info.getSize();
		// Check number of attributes
		if (attributes_count < 1)
			throw std::runtime_error("isoforest_classifier::isoforest_classifier(): At least one attribute is required");
		// Get number of samples
		size_t samples_count = _data.size();
		// Check number of samples
		if (samples_count < 1)
			throw std::runtime_error("isoforest_classifier::isoforest_classifier(): At least one sample is required");
		// Get number of samples
		size_t samples_size = _data.size() / attributes_count;
		// Check number of samples
		if (samples_size < 1)
			throw std::runtime_error("isoforest_classifier::isoforest_classifier(): At least one sample size is required");
		// Check number of samples
		if (samples_count < 2 * samples_size)
			throw std::runtime_error("isoforest_classifier::isoforest_classifier(): At least two samples are required");
		// Get number of trees
		size_t trees_count = (size_t)(std::log((real_t)samples_count) / std::log(2.0));
		// Check number of trees
		if (trees_count < 1)
			throw std::runtime_error("isoforest_classifier::isoforest_classifier(): At least one tree is required");
		// Get number of samples per tree
		size_t samples_per_tree = (size_t)(samples_count / trees_count);
		// Check number of samples per tree
		if (samples_per_tree < 1)
			throw std::runtime_error("isoforest_classifier::isoforest_classifier(): At least one sample per tree is required");
		// Get number of attributes per tree
		size_t _attributes_per_tree = (size_t)(attributes_count / 2);
		// Check number of attributes per tree
		if (_attributes_per_tree < 1)
			throw std::runtime_error("isoforest_classifier::isoforest_classifier(): At least one attribute per tree is required");
		// Get number of attributes per sample
		size_t attributes_per_sample = (size_t)(samples_size / 2);
		// Check number of attributes per sample
		if (attributes_per_sample < 1)
			throw std::runtime_error("isoforest_classifier::isoforest_classifier(): At least one attribute per sample is required");
	}
	// assignment operator
	iso_classifier &iso_classifier::operator=(const iso_classifier &rhs)
	{
		if (this != &rhs)
		{
			// _data( std::move(rhs._data));
			_isoforest = rhs._isoforest;
		}
		return *this;
	}

	// Print iso_classifier
	std::ostream &operator<<(std::ostream &os, const iso_classifier &rhs)
	{
		os << "isoforest_classifier: " << std::endl;
		rhs.print(os);
		// os << "  Number of trees = " << rhs.get_trees() << std::endl;
		// os << "  Number of samples per tree = " << rhs.get_samples() << std::endl;

		//	os << " Fitted: " << std::string(rhs.fitted()?"TRUE":"FALSE")<< std::endl;
		//
		return os;
	}
	void iso_classifier::print(std::ostream &os) const
	{
		os << *this;
	}

	// Print iso_classifier
	void iso_classifier::print() const
	{
		print(std::cout);
	}
	//
	void iso_classifier::print(std::ostream &out, const dataset &data, const std::vector<attribute> &predictions) const
	{
		print(out);
		data.print(out);
		out << std::string("Predictions: ") << std::endl;
		for (size_t i = 0; i < predictions.size(); ++i)
		{
			out <<std::string("  ") << predictions[i].continous() << std::endl;
		}	

	}
	void iso_classifier::print(std::ostream &out, const dataset &data) const
	{

		print(out);
		data.print(out);

	}

	const char* classifier_name[] =
  	{
    	"NONE","TREE","Decision Tree ","Random Tree ","Ensemble","Random Forest","AdaBoost","NaiveBayes","Metric","Nearest Neighbors","K-Means","Isolation Tree","Isolation Forest","Fixed KD Tree","LightGBM","XGBoost","CategoryBoost","SVM","LogRegression","kNN","MLP","LinearRegression","Perceptron","SGD","GA","QDA","LDA","GNB","GMM",
		"Hidden Markov Model","DBSCAN","OPTICS","K-Means++","K-Means Parallel","K-Means Lloyd","K-Means Lloyd Parallel" 
  	};


	void
	print_classifier_summary(const std::string &data_set_name,
							 const dataset &data, const classifier &_classifier)
	{
		// Check fitting over training test

		size_t classifier_index = _classifier.get_type()%classifier_type::CLASSIFIER_MAX;

		std::string cname =classifier_name[classifier_index ] ;	
		
		std::cout << "[#] ----- Check [ " << data_set_name << ","<<cname<< "]" << std::endl;
		confusion_matrix ConfusionMatrix(std::ref(data),std::ref( _classifier));
		// Print matrix
		std::cout << ConfusionMatrix;
		// Resume information
		std::cout << "[#] Number of [" << data_set_name << "] errors = "
				  << ConfusionMatrix.getError() << " - % " << std::fixed
				  << 100.0 * (float)ConfusionMatrix.getError() / (float)data.size()
				  << std::endl;
		std::cout << "[#] Number of " << data_set_name << " samples = "
				  << data.size() << std::endl
				  << std::endl;

		std::cout << "[#] ----- End Check [ " << data_set_name << "]" << std::endl;
		std::cout << "[#] dataset entropy = " << data.entropy() << std::endl;
		std::cout << "[#] dataset gini = " << data.gini() << std::endl;
		std::cout << "[#] dataset variance = " << data.variance() << std::endl;
 		
		/*	std::cout<<"[#] dataset median = "<<data.median()<<std::endl;
		std::cout<<"[#] dataset mode = "<<data.mode()<<std::endl;
		std::cout<<"[#] dataset range = "<<data.range()<<std::endl;
		std::cout<<"[#] dataset min = "<<data.min()<<std::endl;
		std::cout<<"[#] dataset max = "<<data.max()<<std::endl;
		std::cout<<"[#] dataset sum = "<<data.sum()<<std::endl;
		std::cout<<"[#] dataset sum of squares = "<<data.sum_of_squares()<<std::endl;
		std::cout<<"[#] dataset mean absolute deviation = "<<data.mean_absolute_deviation()<<std::endl;
		std::cout<<"[#] dataset median absolute deviation = "<<data.median_absolute_deviation()<<std::endl;
		std::cout<<"[#] dataset interquartile range = "<<data.interquartile_range()<<std::endl;
		std::cout<<"[#] dataset quartile deviation = "<<data.quartile_deviation()<<std::endl;
		std::cout<<"[#] dataset coefficient of variation = "<<data.coefficient_of_variation()<<std::endl;
		*/
	}

	void serialize_ExtIsoForest(	const ext_iso_forest& forest ,std::FILE* cstyle)
	{
		UNDEF_REFERENCE(forest)
		UNDEF_REFERENCE2(cstyle)

	}
	void serialize_ExtIsoForest(	const ext_iso_forest& forest ,std::ostream& cppstyle)
	{
		UNDEF_REFERENCE(forest)
		UNDEF_REFERENCE2(cppstyle)

	}
	void serialize_IsoForest(	const iso_forest& forest ,std::FILE* cstyle)
	{
		UNDEF_REFERENCE(forest)
		UNDEF_REFERENCE2(cstyle)

	}	
	void serialize_IsoForest(	const iso_forest& forest ,std::ostream& cppstyle)
	{
		UNDEF_REFERENCE(forest)
		UNDEF_REFERENCE2(cppstyle)

	}	
	void serialize_Indexer(	const TreesIndexer& indexer ,std::FILE* cstyle)
	{
		UNDEF_REFERENCE(indexer)
		UNDEF_REFERENCE2(cstyle)

	}
	//serialization / deserialization missing implementations: 
	//1)inspect_serialized_object
	//2)serialize_IsoForest(provallo::iso_forest const&, _IO_FILE*) - c/style
	//3)serialize_ExtIsoForest
	//4)serialize_Imputer
	//5)serialize_Indexer
	//6)serialize_IsoForest(provallo::iso_forest const&, std::ostream&)' cpp style
	//deserialize: 
	//1)deserialize_IsoForest
	//2)deserialize_ExtIsoForest
	//3)deserialize_Imputer
	//4)deserialize_Indexer
	

	//deserialize_IsoForest (cstyle) 
	void deserialize_IsoForest( iso_forest& forest, std::FILE* cstyle)
	{
		UNDEF_REFERENCE(cstyle)
		UNDEF_REFERENCE2(forest);
	}	
	//deserialize_Imputer (cstyle)
	void deserialize_Imputer(Imputer& imputer ,std::FILE* cstyle)
	{
		UNDEF_REFERENCE(cstyle)
		UNDEF_REFERENCE2(imputer)
 	}	
	//deserialize_Indexer (cstyle)
	void deserialize_Indexer(TreesIndexer& indexer ,std::FILE* cstyle)
	{
		UNDEF_REFERENCE(cstyle)
		UNDEF_REFERENCE2(indexer)

	}	

	void deserialize_Indexer(TreesIndexer& indexer ,const char* cstyle)
	{
		UNDEF_REFERENCE(cstyle)
		UNDEF_REFERENCE2(indexer)

	}	
	
	void deserialize_ExtIsoForest(ext_iso_forest& indexer ,std::FILE* cstyle)
	{
		UNDEF_REFERENCE(cstyle)
		UNDEF_REFERENCE2(indexer)

	}	

	//mini implementaion of lightgbm_classifier : 
 

	//lightgbm implementation : 
	//constructor : 
	lightgbm_classifier::lightgbm_classifier( const dataset &data, const parameter_base &parameters,
      const std::random_device &random, std::ostream &out/*default std::cout*/, split_method_factory *factory/*nullptr*/ ):ensemble_classifier(data,parameters,random,factory) 
	{
		
		UNDEF_REFERENCE(data)
		UNDEF_REFERENCE2(parameters)
		UNDEF_REFERENCE2(random)
		UNDEF_REFERENCE2(out)
		UNDEF_REFERENCE2(factory)
		//setup lightgbm parameters :
	    //cast parameters from parameter_base to lightgbm_parameter 
		
	}
	//classify 	:
	 attribute
    lightgbm_classifier::classify(dataset::attribute_iterator begin,
             dataset::attribute_iterator end) const 
			 {
						class_dist cd(1);
						UNDEF_REFERENCE(begin)
						UNDEF_REFERENCE2(end)
						//return class_dist();	
						return cd.mode();
			 }
     attribute
    lightgbm_classifier::classify(std::vector<attribute>::const_iterator begin,
             std::vector<attribute>::const_iterator end) const
			 {
						class_dist cd(1);
						UNDEF_REFERENCE(begin)
						UNDEF_REFERENCE2(end)
						//return class_dist();	
						return cd.mode();
			 }

    // Get distribution of outcomes
   class_dist
   lightgbm_classifier::posterior(dataset::attribute_iterator begin,dataset::attribute_iterator end) const
   {
	class_dist cd(1);
  	UNDEF_REFERENCE(begin)
   	UNDEF_REFERENCE2(end)
   	//return class_dist();	
	return cd;
   }
   class_dist
   lightgbm_classifier::posterior(std::vector<attribute>::const_iterator begin,
              std::vector<attribute>::const_iterator end) const

			  {

				class_dist cd(1);
				UNDEF_REFERENCE(begin)
				UNDEF_REFERENCE2(end)
				//return class_dist();	
				return cd;
			  }

 	//destructor :
	lightgbm_classifier::~lightgbm_classifier()
	{
	} 

	//print :	
	void lightgbm_classifier::print(std::ostream& out) const
	{
		UNDEF_REFERENCE(out)
		UNDEF_REFERENCE2(out)
	}
	class_dist  lightgbm_classifier::classify(const std::vector<attribute> &sample) const
	{
		UNDEF_REFERENCE(sample)
		UNDEF_REFERENCE2(sample)

		class_dist cd(1);
		return cd;
	}
      
    class_dist lightgbm_classifier::posterior (const std::vector<attribute> &sample) const 
	{
		UNDEF_REFERENCE(sample)
		UNDEF_REFERENCE2(sample)

		class_dist cd(1);
		return cd;

	}


  }
/* namespace provallo */
