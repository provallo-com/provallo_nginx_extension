/*
 * split_utils.hpp
 *
 *  Created on: Apr 18, 2023
 *      Author: kardon
 */

#ifndef DECISION_ENGINE_SPLIT_UTILS_HPP_
#define DECISION_ENGINE_SPLIT_UTILS_HPP_
#include <vector>
#include <atomic>

#include "utils.h"
#include "dataset.h"
#include "attribute.h"
#include "../glue/glueprocessinfo.h"	
namespace provallo
{

	//using dataset = provallo::dataset;
	using attribute_iterator = dataset::attribute_iterator;
	//using attribute = provallo::attribute;
	//using attribute_type = provallo::attribute_type;
	
	typedef struct iso_hplane
	{
		std::vector<size_t> col_num;
		std::vector<ColType> col_type;
		std::vector<double> coef;
		std::vector<double> mean;
		std::vector<std::vector<double>> cat_coef;
		std::vector<int> chosen_cat;
		std::vector<double> fill_val;
		std::vector<double> fill_new;

		double split_point;
		size_t hplane_left;
		size_t hplane_right;
		double score; /* will not be integer when there are weights or early stop */
		double range_low = -HUGE_VAL;
		double range_high = HUGE_VAL;
		double remainder; /* only used for distance/similarity */

	} IsoHPlane;

	size_t
	divide_subset_split(size_t ix_arr[], double x[], size_t st, size_t end,
						double split_point) noexcept;

	/* For categorical columns split by subset */
	void
	divide_subset_split(size_t *ix_arr, int x[], size_t st, size_t end,
						signed char split_categ[], MissingAction missing_action,
						size_t &st_NA, size_t &end_NA, size_t &split_ix) noexcept;

	/* For categorical columns split by subset, used at prediction time (with similarity) */
	void
	divide_subset_split(size_t *ix_arr, int x[], size_t st, size_t end,
						signed char split_categ[], int ncat,
						MissingAction missing_action,
						NewCategAction new_cat_action, bool move_new_to_left,
						size_t &st_NA, size_t &end_NA, size_t &split_ix) noexcept;
	/* For categoricals split on a single category */
	void
	divide_subset_split(size_t *ix_arr, int x[], size_t st, size_t end,
						int split_categ, MissingAction missing_action,
						size_t &st_NA, size_t &end_NA, size_t &split_ix) noexcept;

	/* For categoricals split on sub-set that turned out to have 2 categories only (prediction-time) */
	void
	divide_subset_split(size_t *ix_arr, int x[], size_t st, size_t end,
						MissingAction missing_action,
						NewCategAction new_cat_action, bool move_new_to_left,
						size_t &st_NA, size_t &end_NA, size_t &split_ix) noexcept;

	/* For numerical columns */
	template <class xreal>
	inline void
	divide_subset_split(size_t *ix_arr, xreal x[], size_t st, size_t end,
						double split_point, MissingAction missing_action,
						size_t &st_NA, size_t &end_NA,
						size_t &split_ix) noexcept;
	template <class xreal, class sparse_x>
	inline void
	divide_subset_split(size_t *ix_arr, size_t st, size_t end, size_t col_num,
						xreal Xc[], sparse_x *Xc_ind, sparse_x *Xc_indptr,
						double split_point, MissingAction missing_action,
						size_t &st_NA, size_t &end_NA,
						size_t &split_ix) noexcept;

	template <class InputData, class WorkerMemory, class ldouble_safe>
	inline void
	add_chosen_column(WorkerMemory &workspace, InputData &input_data,
					  ModelParams &model_params,
					  std::vector<bool> &col_is_taken,
					  hashed_set<size_t> &col_is_taken_s);

	void
	shrink_to_fit_hplane(iso_hplane &hplane, bool clear_vectors);

	template <class InputData, class WorkerMemory>
	void
	simplify_hplane(iso_hplane &hplane, WorkerMemory &workspace,
					InputData &input_data, ModelParams &model_params);

	double
	sample_random_uniform(double xmin, double xmax, RNG_engine &rng) noexcept;

	template <class ldouble_safe>
	double
	eval_guided_crit(double *x, size_t n, GainCriterion criterion,
					 double min_gain, bool as_relative_gain, double *buffer_sd,
					 double &split_point, double &xmin, double &xmax,
					 size_t *ix_arr_plus_st, size_t *cols_use,
					 size_t ncols_use, bool force_cols_use,
					 double *X_row_major, size_t ncols, double *Xr,
					 size_t *Xr_ind, size_t *Xr_indptr);

	/* For numerical columns */
	template <class xreal>
	void
	divide_subset_split(size_t *ix_arr, xreal x[], size_t st, size_t end,
						double split_point, MissingAction missing_action,
						size_t &st_NA, size_t &end_NA,
						size_t &split_ix) noexcept
	{
		size_t temp;

		/* if NAs are not to be bothered with, just need to do a single pass */
		if (missing_action == Fail)
		{
			/* move to the left if it's l.e. split point */
			for (size_t row = st; row <= end; row++)
			{
				if (x[ix_arr[row]] <= split_point)
				{
					temp = ix_arr[st];
					ix_arr[st] = ix_arr[row];
					ix_arr[row] = temp;
					st++;
				}
			}
			split_ix = st;
		}

		/* otherwise, first put to the left all l.e. and not NA, then all NAs to the end of the left */
		else
		{
			for (size_t row = st; row <= end; row++)
			{
				if (!std::isnan(x[ix_arr[row]]) && x[ix_arr[row]] <= split_point)
				{
					temp = ix_arr[st];
					ix_arr[st] = ix_arr[row];
					ix_arr[row] = temp;
					st++;
				}
			}
			st_NA = st;

			for (size_t row = st; row <= end; row++)
			{
				if (unlikely(std::isnan(x[ix_arr[row]])))
				{
					temp = ix_arr[st];
					ix_arr[st] = ix_arr[row];
					ix_arr[row] = temp;
					st++;
				}
			}
			end_NA = st;
		}
	}

	/* For sparse numeric columns */
	template <class xreal, class sparse_x>
	inline void
	divide_subset_split(size_t *ix_arr, size_t st, size_t end, size_t col_num,
						xreal Xc[], sparse_x *Xc_ind, sparse_x *Xc_indptr,
						double split_point, MissingAction missing_action,
						size_t &st_NA, size_t &end_NA,
						size_t &split_ix) noexcept
	{
		/* TODO: this is a mess, needs refactoring */
		/* TODO: when moving zeros, would be better to instead move by '>' (opposite as in here) */
		/* TODO: should create an extra version to go along with 'predict' that would
		 add the range penalty right here to spare operations. */
		if (Xc_indptr[col_num] == Xc_indptr[col_num + 1])
		{
			if (missing_action == Fail)
			{
				split_ix = (0 <= split_point) ? (end + 1) : st;
			}

			else
			{
				st_NA = (0 <= split_point) ? (end + 1) : st;
				end_NA = (0 <= split_point) ? (end + 1) : st;
			}
		}

		size_t st_col = Xc_indptr[col_num];
		size_t end_col = Xc_indptr[col_num + 1] - 1;
		size_t curr_pos = st_col;
		size_t ind_end_col = Xc_ind[end_col];
		size_t temp;
		bool move_zeros = 0 <= split_point;
		size_t *ptr_st = std::lower_bound(ix_arr + st, ix_arr + end + 1,
										  Xc_ind[st_col]);

		if (move_zeros && ptr_st > ix_arr + st)
			st = ptr_st - ix_arr;

		if (missing_action == Fail)
		{
			if (move_zeros)
			{
				for (size_t *row = ptr_st; row != ix_arr + end + 1;)
				{
					if (curr_pos >= end_col + 1)
					{
						for (size_t *r = row; r <= ix_arr + end; r++)
						{
							temp = ix_arr[st];
							ix_arr[st] = *r;
							*r = temp;
							st++;
						}
						break;
					}

					if (Xc_ind[curr_pos] == (sparse_x)(*row))
					{
						if (Xc[curr_pos] <= split_point)
						{
							temp = ix_arr[st];
							ix_arr[st] = *row;
							*row = temp;
							st++;
						}
						if (curr_pos == end_col && row < ix_arr + end)
						{
							for (size_t *r = row + 1; r <= ix_arr + end; r++)
							{
								temp = ix_arr[st];
								ix_arr[st] = *r;
								*r = temp;
								st++;
							}
						}
						if (row == ix_arr + end || curr_pos == end_col)
							break;
						curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
													Xc_ind + end_col + 1,
													*(++row)) -
								   Xc_ind;
					}

					else
					{
						if (Xc_ind[curr_pos] > (sparse_x)(*row))
						{
							while (row <= ix_arr + end && Xc_ind[curr_pos] > (sparse_x)(*row))
							{
								temp = ix_arr[st];
								ix_arr[st] = *row;
								*row = temp;
								st++;
								row++;
							}
						}

						else
							curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
														Xc_ind + end_col + 1, *row) -
									   Xc_ind;
					}
				}
			}

			else /* don't move zeros */
			{
				for (size_t *row = ptr_st;
					 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;)
				{
					if (Xc_ind[curr_pos] == (sparse_x)(*row))
					{
						if (Xc[curr_pos] <= split_point)
						{
							temp = ix_arr[st];
							ix_arr[st] = *row;
							*row = temp;
							st++;
						}
						if (row == ix_arr + end || curr_pos == end_col)
							break;
						curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
													Xc_ind + end_col + 1,
													*(++row)) -
								   Xc_ind;
					}

					else
					{
						if (Xc_ind[curr_pos] > (sparse_x)(*row))
							row = std::lower_bound(row + 1, ix_arr + end + 1,
												   Xc_ind[curr_pos]);
						else
							curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
														Xc_ind + end_col + 1, *row) -
									   Xc_ind;
					}
				}
			}

			split_ix = st;
		}

		else /* can have NAs */
		{

			bool has_NAs = false;
			if (move_zeros)
			{
				for (size_t *row = ptr_st; row != ix_arr + end + 1;)
				{
					if (curr_pos >= end_col + 1)
					{
						for (size_t *r = row; r <= ix_arr + end; r++)
						{
							temp = ix_arr[st];
							ix_arr[st] = *r;
							*r = temp;
							st++;
						}
						break;
					}

					if (Xc_ind[curr_pos] == (sparse_x)(*row))
					{
						if (unlikely(std::isnan(Xc[curr_pos])))
							has_NAs = true;
						else if (Xc[curr_pos] <= split_point)
						{
							temp = ix_arr[st];
							ix_arr[st] = *row;
							*row = temp;
							st++;
						}
						if (curr_pos == end_col && row < ix_arr + end)
							for (size_t *r = row + 1; r <= ix_arr + end; r++)
							{
								temp = ix_arr[st];
								ix_arr[st] = *r;
								*r = temp;
								st++;
							}
						if (row == ix_arr + end || curr_pos == end_col)
							break;
						curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
													Xc_ind + end_col + 1,
													*(++row)) -
								   Xc_ind;
					}

					else
					{
						if (Xc_ind[curr_pos] > (sparse_x)(*row))
						{
							while (row <= ix_arr + end && Xc_ind[curr_pos] > (sparse_x)(*row))
							{
								temp = ix_arr[st];
								ix_arr[st] = *row;
								*row = temp;
								st++;
								row++;
							}
						}

						else
						{
							curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
														Xc_ind + end_col + 1,
														*row) -
									   Xc_ind;
						}
					}
				}
			}

			else /* don't move zeros */
			{
				for (size_t *row = ptr_st;
					 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;)
				{
					if (Xc_ind[curr_pos] == (sparse_x)(*row))
					{
						if (unlikely(std::isnan(Xc[curr_pos])))
							has_NAs = true;
						if (!std::isnan(Xc[curr_pos]) && Xc[curr_pos] <= split_point)
						{
							temp = ix_arr[st];
							ix_arr[st] = *row;
							*row = temp;
							st++;
						}
						if (row == ix_arr + end || curr_pos == end_col)
							break;
						curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
													Xc_ind + end_col + 1,
													*(++row)) -
								   Xc_ind;
					}

					else
					{
						if (Xc_ind[curr_pos] > (sparse_x)(*row))
							row = std::lower_bound(row + 1, ix_arr + end + 1,
												   Xc_ind[curr_pos]);
						else
							curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
														Xc_ind + end_col + 1, *row) -
									   Xc_ind;
					}
				}
			}

			st_NA = st;
			if (has_NAs)
			{
				curr_pos = st_col;
				std::sort(ix_arr + st, ix_arr + end + 1);
				for (size_t *row = ix_arr + st;
					 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;)
				{
					if (Xc_ind[curr_pos] == (sparse_x)(*row))
					{
						if (unlikely(std::isnan(Xc[curr_pos])))
						{
							temp = ix_arr[st];
							ix_arr[st] = *row;
							*row = temp;
							st++;
						}
						if (row == ix_arr + end || curr_pos == end_col)
							break;
						curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
													Xc_ind + end_col + 1,
													*(++row)) -
								   Xc_ind;
					}

					else
					{
						if (Xc_ind[curr_pos] > (sparse_x)(*row))
							row = std::lower_bound(row + 1, ix_arr + end + 1,
												   Xc_ind[curr_pos]);
						else
							curr_pos = std::lower_bound(Xc_ind + curr_pos + 1,
														Xc_ind + end_col + 1, *row) -
									   Xc_ind;
					}
				}
			}
			end_NA = st;
		}
	}

	template <class InputData, class WorkerMemory>
	void
	simplify_hplane(IsoHPlane &hplane, WorkerMemory &workspace,
					InputData &input_data, ModelParams &model_params)
	{
		if (workspace.ntaken_best < model_params.ndim)
		{
			hplane.col_num.resize(workspace.ntaken_best);
			hplane.col_type.resize(workspace.ntaken_best);
			if (model_params.missing_action != Fail)
				hplane.fill_val.resize(workspace.ntaken_best);
		}

		size_t ncols_numeric = 0;
		size_t ncols_categ = 0;

		if (input_data.ncols_categ)
		{
			for (size_t col = 0; col < workspace.ntaken_best; col++)
			{
				switch (hplane.col_type[col])
				{
				case Numeric:
				{
					workspace.ext_coef[ncols_numeric] = hplane.coef[col];
					workspace.ext_mean[ncols_numeric] = hplane.mean[col];
					ncols_numeric++;
					break;
				}

				case Categorical:
				{
					workspace.ext_fill_new[ncols_categ] = hplane.fill_new[col];
					switch (model_params.cat_split_type)
					{
					case SingleCateg:
					{
						workspace.chosen_cat[ncols_categ] =
							hplane.chosen_cat[col];
						break;
					}

					case SubSet:
					{
						std::copy(
							hplane.cat_coef[col].begin(),
							hplane.cat_coef[col].begin() + input_data.ncat[hplane.col_num[col]],
							workspace.ext_cat_coef[ncols_categ].begin());
						break;
					}
					}
					ncols_categ++;
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

		else
		{
			ncols_numeric = workspace.ntaken_best;
		}

		hplane.coef.resize(ncols_numeric);
		hplane.mean.resize(ncols_numeric);
		if (input_data.ncols_numeric)
		{
			std::copy(workspace.ext_coef.begin(),
					  workspace.ext_coef.begin() + ncols_numeric,
					  hplane.coef.begin());
			std::copy(workspace.ext_mean.begin(),
					  workspace.ext_mean.begin() + ncols_numeric,
					  hplane.mean.begin());
		}

		/* If there are no categorical columns, all of them will be numerical and there is no need to reorder */
		if (ncols_categ)
		{
			hplane.fill_new.resize(ncols_categ);
			std::copy(workspace.ext_fill_new.begin(),
					  workspace.ext_fill_new.begin() + ncols_categ,
					  hplane.fill_new.begin());

			hplane.cat_coef.resize(ncols_categ);
			switch (model_params.cat_split_type)
			{
			case SingleCateg:
			{
				hplane.chosen_cat.resize(ncols_categ);
				std::copy(workspace.chosen_cat.begin(),
						  workspace.chosen_cat.begin() + ncols_categ,
						  hplane.chosen_cat.begin());
				hplane.cat_coef.clear();
				break;
			}

			case SubSet:
			{
				hplane.chosen_cat.clear();
				ncols_categ = 0;
				for (size_t col = 0; col < workspace.ntaken_best; col++)
				{
					if (hplane.col_type[col] == Categorical)
					{
						hplane.cat_coef[ncols_categ].resize(
							input_data.ncat[hplane.col_num[col]]);
						std::copy(
							workspace.ext_cat_coef[ncols_categ].begin(),
							workspace.ext_cat_coef[ncols_categ].begin() + input_data.ncat[hplane.col_num[col]],
							hplane.cat_coef[ncols_categ].begin());
						hplane.cat_coef[ncols_categ].shrink_to_fit();
						ncols_categ++;
					}
				}
				break;
			}
			}
		}

		else
		{
			hplane.cat_coef.clear();
			hplane.chosen_cat.clear();
			hplane.fill_new.clear();
		}
	}

	template <class InputData, class WorkerMemory, class ldouble_safe>
	inline void
	add_chosen_column(WorkerMemory &workspace, InputData &input_data,
					  ModelParams &model_params,
					  std::vector<bool> &col_is_taken,
					  hashed_set<size_t> &col_is_taken_s)
	{
		if (workspace.col_criterion == Uniformly)
		{
			set_col_as_taken(col_is_taken, col_is_taken_s, input_data,
							 workspace.col_chosen, workspace.col_type);
		}
		else
		{
			if (workspace.col_chosen < input_data.ncols_numeric)
			{
				workspace.col_type = Numeric;
			}
			else
			{
				workspace.col_chosen -= input_data.ncols_numeric;
				workspace.col_type = Categorical;
			}
		}
		workspace.col_take[workspace.ntaken] = workspace.col_chosen;
		workspace.col_take_type[workspace.ntaken] = workspace.col_type;

		switch (workspace.col_type)
		{
		case Numeric:
		{
			switch (model_params.coef_type)
			{
			case Uniform:
			{
				workspace.ext_coef[workspace.ntaken] = workspace.coef_unif(
					workspace.rnd_generator);
				break;
			}

			case Normal:
			{
				workspace.ext_coef[workspace.ntaken] = workspace.coef_norm(
					workspace.rnd_generator);
				break;
			}
			}

			if (input_data.Xc_indptr == NULL)
			{
				if (workspace.weights_arr.empty() && workspace.weights_map.empty())
				{
					if (model_params.missing_action == Fail && !model_params.standardize_data)
					{
						workspace.ext_mean[workspace.ntaken] = 0;
						workspace.ext_sd = 1;
					}

					else if (!model_params.standardize_data)
					{
						workspace.ext_sd = 1;
						if (workspace.col_criterion != Uniformly && workspace.has_saved_stats)
							workspace.ext_mean[workspace.ntaken] =
								workspace.saved_stat1[workspace.col_chosen];
						else
						{
							workspace.ext_mean[workspace.ntaken] =
								calc_mean_only(
									workspace.ix_arr.data(),
									workspace.st,
									workspace.end,
									input_data.numeric_data + workspace.col_chosen * input_data.nrows);
						}
					}

					else
					{
						if (workspace.col_criterion != Uniformly && workspace.has_saved_stats)
						{
							workspace.ext_mean[workspace.ntaken] =
								workspace.saved_stat1[workspace.col_chosen];
							workspace.ext_sd =
								workspace.saved_stat2[workspace.col_chosen];
						}

						else
						{
							calc_mean_and_sd<
								typename std::remove_pointer<
									decltype(input_data.numeric_data)>::type,
								ldouble_safe>(
								workspace.ix_arr.data(),
								workspace.st,
								workspace.end,
								input_data.numeric_data + workspace.col_chosen * input_data.nrows,
								model_params.missing_action, workspace.ext_sd,
								workspace.ext_mean[workspace.ntaken]);
						}
					}

					add_linear_comb(
						workspace.ix_arr.data(),
						workspace.st,
						workspace.end,
						workspace.comb_val.data(),
						input_data.numeric_data + workspace.col_chosen * input_data.nrows,
						workspace.ext_coef[workspace.ntaken], workspace.ext_sd,
						workspace.ext_mean[workspace.ntaken],
						workspace.ext_fill_val[workspace.ntaken],
						model_params.missing_action,
						workspace.buffer_dbl.data(),
						workspace.buffer_szt.data(), true);
				}
				else if (!workspace.weights_arr.empty())
				{
					if (model_params.missing_action == Fail && !model_params.standardize_data)
					{
						workspace.ext_mean[workspace.ntaken] = 0;
						workspace.ext_sd = 1;
					}

					else if (!model_params.standardize_data)
					{
						workspace.ext_sd = 1;
						if (workspace.col_criterion != Uniformly && workspace.has_saved_stats)
							workspace.ext_mean[workspace.ntaken] =
								workspace.saved_stat1[workspace.col_chosen];
						else
						{
							workspace.ext_mean[workspace.ntaken] =
								calc_mean_only_weighted(
									workspace.ix_arr.data(),
									workspace.st,
									workspace.end,
									input_data.numeric_data + workspace.col_chosen * input_data.nrows,
									workspace.weights_arr);
						}
					}

					else
					{
						if (workspace.col_criterion != Uniformly && workspace.has_saved_stats)
						{
							workspace.ext_mean[workspace.ntaken] =
								workspace.saved_stat1[workspace.col_chosen];
							workspace.ext_sd =
								workspace.saved_stat2[workspace.col_chosen];
						}

						else
						{
							calc_mean_and_sd_weighted<
								typename std::remove_pointer<
									decltype(input_data.numeric_data)>::type,
								decltype(workspace.weights_arr), ldouble_safe>(
								workspace.ix_arr.data(),
								workspace.st,
								workspace.end,
								input_data.numeric_data + workspace.col_chosen * input_data.nrows,
								workspace.weights_arr,
								model_params.missing_action, workspace.ext_sd,
								workspace.ext_mean[workspace.ntaken]);
						}
					}

					add_linear_comb_weighted<
						typename std::remove_pointer<
							decltype(input_data.numeric_data)>::type,
						decltype(workspace.weights_arr), ldouble_safe>(
						workspace.ix_arr.data(),
						workspace.st,
						workspace.end,
						workspace.comb_val.data(),
						input_data.numeric_data + workspace.col_chosen * input_data.nrows,
						workspace.ext_coef[workspace.ntaken], workspace.ext_sd,
						workspace.ext_mean[workspace.ntaken],
						workspace.ext_fill_val[workspace.ntaken],
						model_params.missing_action,
						workspace.buffer_dbl.data(),
						workspace.buffer_szt.data(), true,
						workspace.weights_arr);
				}

				else
				{
					if (model_params.missing_action == Fail && !model_params.standardize_data)
					{
						workspace.ext_mean[workspace.ntaken] = 0;
						workspace.ext_sd = 1;
					}

					else if (!model_params.standardize_data)
					{
						workspace.ext_sd = 1;
						if (workspace.col_criterion != Uniformly && workspace.has_saved_stats)
							workspace.ext_mean[workspace.ntaken] =
								workspace.saved_stat1[workspace.col_chosen];
						else
						{
							workspace.ext_mean[workspace.ntaken] =
								calc_mean_only_weighted(
									workspace.ix_arr.data(),
									workspace.st,
									workspace.end,
									input_data.numeric_data + workspace.col_chosen * input_data.nrows,
									workspace.weights_map);
						}
					}

					else
					{
						if (workspace.col_criterion != Uniformly && workspace.has_saved_stats)
						{
							workspace.ext_mean[workspace.ntaken] =
								workspace.saved_stat1[workspace.col_chosen];
							workspace.ext_sd =
								workspace.saved_stat2[workspace.col_chosen];
						}

						else
						{
							calc_mean_and_sd_weighted<
								typename std::remove_pointer<
									decltype(input_data.numeric_data)>::type,
								decltype(workspace.weights_map), ldouble_safe>(
								workspace.ix_arr.data(),
								workspace.st,
								workspace.end,
								input_data.numeric_data + workspace.col_chosen * input_data.nrows,
								workspace.weights_map,
								model_params.missing_action, workspace.ext_sd,
								workspace.ext_mean[workspace.ntaken]);
						}
					}

					add_linear_comb_weighted<
						typename std::remove_pointer<
							decltype(input_data.numeric_data)>::type,
						decltype(workspace.weights_map), ldouble_safe>(
						workspace.ix_arr.data(),
						workspace.st,
						workspace.end,
						workspace.comb_val.data(),
						input_data.numeric_data + workspace.col_chosen * input_data.nrows,
						workspace.ext_coef[workspace.ntaken], workspace.ext_sd,
						workspace.ext_mean[workspace.ntaken],
						workspace.ext_fill_val[workspace.ntaken],
						model_params.missing_action,
						workspace.buffer_dbl.data(),
						workspace.buffer_szt.data(), true,
						workspace.weights_map);
				}
			}

			else
			{
				if (workspace.weights_arr.empty() && workspace.weights_map.empty())
				{
					if (model_params.missing_action == Fail && !model_params.standardize_data)
					{
						workspace.ext_mean[workspace.ntaken] = 0;
						workspace.ext_sd = 1;
					}

					else if (!model_params.standardize_data)
					{
						workspace.ext_sd = 1;
						if (workspace.col_criterion != Uniformly && workspace.has_saved_stats)
							workspace.ext_mean[workspace.ntaken] =
								workspace.saved_stat1[workspace.col_chosen];
						else
						{
							workspace.ext_mean[workspace.ntaken] =
								calc_mean_only<
									typename std::remove_pointer<
										decltype(input_data.numeric_data)>::type,
									typename std::remove_pointer<
										decltype(input_data.Xc_indptr)>::type,
									ldouble_safe>(workspace.ix_arr.data(),
												  workspace.st, workspace.end,
												  workspace.col_chosen,
												  input_data.Xc,
												  input_data.Xc_ind,
												  input_data.Xc_indptr);
						}
					}

					else
					{
						if (workspace.col_criterion != Uniformly && workspace.has_saved_stats)
						{
							workspace.ext_mean[workspace.ntaken] =
								workspace.saved_stat1[workspace.col_chosen];
							workspace.ext_sd =
								workspace.saved_stat2[workspace.col_chosen];
						}

						else
						{
							calc_mean_and_sd<
								typename std::remove_pointer<
									decltype(input_data.Xc)>::type,
								typename std::remove_pointer<
									decltype(input_data.Xc_indptr)>::type,
								ldouble_safe>(
								workspace.ix_arr.data(), workspace.st,
								workspace.end, workspace.col_chosen,
								input_data.Xc, input_data.Xc_ind,
								input_data.Xc_indptr, workspace.ext_sd,
								workspace.ext_mean[workspace.ntaken]);
						}
					}

					add_linear_comb(workspace.ix_arr.data(), workspace.st,
									workspace.end, workspace.col_chosen,
									workspace.comb_val.data(), input_data.Xc,
									input_data.Xc_ind, input_data.Xc_indptr,
									workspace.ext_coef[workspace.ntaken],
									workspace.ext_sd,
									workspace.ext_mean[workspace.ntaken],
									workspace.ext_fill_val[workspace.ntaken],
									model_params.missing_action,
									workspace.buffer_dbl.data(),
									workspace.buffer_szt.data(), true);
				}

				else if (!workspace.weights_arr.empty())
				{
					if (model_params.missing_action == Fail && !model_params.standardize_data)
					{
						workspace.ext_mean[workspace.ntaken] = 0;
						workspace.ext_sd = 1;
					}

					else if (!model_params.standardize_data)
					{
						workspace.ext_sd = 1;
						if (workspace.col_criterion != Uniformly && workspace.has_saved_stats)
							workspace.ext_mean[workspace.ntaken] =
								workspace.saved_stat1[workspace.col_chosen];
						else
						{
							workspace.ext_mean[workspace.ntaken] =
								calc_mean_only_weighted<
									typename std::remove_pointer<
										decltype(input_data.numeric_data)>::type,
									typename std::remove_pointer<
										decltype(input_data.Xc_indptr)>::type,
									decltype(workspace.weights_arr),
									ldouble_safe>(workspace.ix_arr.data(),
												  workspace.st, workspace.end,
												  workspace.col_chosen,
												  input_data.Xc,
												  input_data.Xc_ind,
												  input_data.Xc_indptr,
												  workspace.weights_arr);
						}
					}

					else
					{
						if (workspace.col_criterion != Uniformly && workspace.has_saved_stats)
						{
							workspace.ext_mean[workspace.ntaken] =
								workspace.saved_stat1[workspace.col_chosen];
							workspace.ext_sd =
								workspace.saved_stat2[workspace.col_chosen];
						}

						else
						{
							calc_mean_and_sd_weighted<
								typename std::remove_pointer<
									decltype(input_data.numeric_data)>::type,
								typename std::remove_pointer<
									decltype(input_data.Xc_indptr)>::type,
								decltype(workspace.weights_arr), ldouble_safe>(
								workspace.ix_arr.data(), workspace.st,
								workspace.end, workspace.col_chosen,
								input_data.Xc, input_data.Xc_ind,
								input_data.Xc_indptr, workspace.ext_sd,
								workspace.ext_mean[workspace.ntaken],
								workspace.weights_arr);
						}
					}

					add_linear_comb_weighted<
						typename std::remove_pointer<
							decltype(input_data.numeric_data)>::type,
						typename std::remove_pointer<
							decltype(input_data.Xc_indptr)>::type,
						decltype(workspace.weights_arr), ldouble_safe>(
						workspace.ix_arr.data(), workspace.st, workspace.end,
						workspace.col_chosen, workspace.comb_val.data(),
						input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
						workspace.ext_coef[workspace.ntaken], workspace.ext_sd,
						workspace.ext_mean[workspace.ntaken],
						workspace.ext_fill_val[workspace.ntaken],
						model_params.missing_action,
						workspace.buffer_dbl.data(),
						workspace.buffer_szt.data(), true,
						workspace.weights_arr);
				}

				else
				{
					if (model_params.missing_action == Fail && !model_params.standardize_data)
					{
						workspace.ext_mean[workspace.ntaken] = 0;
						workspace.ext_sd = 1;
					}

					else if (!model_params.standardize_data)
					{
						workspace.ext_sd = 1;
						if (workspace.col_criterion != Uniformly && workspace.has_saved_stats)
							workspace.ext_mean[workspace.ntaken] =
								workspace.saved_stat1[workspace.col_chosen];
						else
						{
							workspace.ext_mean[workspace.ntaken] =
								calc_mean_only_weighted<
									typename std::remove_pointer<
										decltype(input_data.numeric_data)>::type,
									typename std::remove_pointer<
										decltype(input_data.Xc_indptr)>::type,
									decltype(workspace.weights_map),
									ldouble_safe>(workspace.ix_arr.data(),
												  workspace.st, workspace.end,
												  workspace.col_chosen,
												  input_data.Xc,
												  input_data.Xc_ind,
												  input_data.Xc_indptr,
												  workspace.weights_map);
						}
					}

					else
					{
						if (workspace.col_criterion != Uniformly && workspace.has_saved_stats)
						{
							workspace.ext_mean[workspace.ntaken] =
								workspace.saved_stat1[workspace.col_chosen];
							workspace.ext_sd =
								workspace.saved_stat2[workspace.col_chosen];
						}

						else
						{
							calc_mean_and_sd_weighted<
								typename std::remove_pointer<
									decltype(input_data.numeric_data)>::type,
								typename std::remove_pointer<
									decltype(input_data.Xc_indptr)>::type,
								decltype(workspace.weights_map), ldouble_safe>(
								workspace.ix_arr.data(), workspace.st,
								workspace.end, workspace.col_chosen,
								input_data.Xc, input_data.Xc_ind,
								input_data.Xc_indptr, workspace.ext_sd,
								workspace.ext_mean[workspace.ntaken],
								workspace.weights_map);
						}
					}

					add_linear_comb_weighted<
						typename std::remove_pointer<
							decltype(input_data.numeric_data)>::type,
						typename std::remove_pointer<
							decltype(input_data.Xc_indptr)>::type,
						decltype(workspace.weights_map), ldouble_safe>(
						workspace.ix_arr.data(), workspace.st, workspace.end,
						workspace.col_chosen, workspace.comb_val.data(),
						input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
						workspace.ext_coef[workspace.ntaken], workspace.ext_sd,
						workspace.ext_mean[workspace.ntaken],
						workspace.ext_fill_val[workspace.ntaken],
						model_params.missing_action,
						workspace.buffer_dbl.data(),
						workspace.buffer_szt.data(), true,
						workspace.weights_map);
				}
			}
			break;
		}

		case Categorical:
		{
			switch (model_params.cat_split_type)
			{
			case SingleCateg:
			{
				workspace.chosen_cat[workspace.ntaken] =
					choose_cat_from_present(workspace, input_data,
											workspace.col_chosen);
				workspace.ext_fill_new[workspace.ntaken] =
					workspace.coef_norm(workspace.rnd_generator);
				if (workspace.weights_arr.empty() && workspace.weights_map.empty())
				{
					add_linear_comb<ldouble_safe>(
						workspace.ix_arr.data(),
						workspace.st,
						workspace.end,
						workspace.comb_val.data(),
						input_data.categ_data + workspace.col_chosen * input_data.nrows,
						input_data.ncat[workspace.col_chosen],
						NULL,
						workspace.ext_fill_new[workspace.ntaken],
						workspace.chosen_cat[workspace.ntaken],
						workspace.ext_fill_val[workspace.ntaken],
						workspace.ext_fill_new[workspace.ntaken],
						NULL,
						NULL, model_params.new_cat_action,
						model_params.missing_action, SingleCateg, true);
				}

				else if (!workspace.weights_arr.empty())
				{
					add_linear_comb_weighted<decltype(workspace.weights_arr),
											 ldouble_safe>(
						workspace.ix_arr.data(),
						workspace.st,
						workspace.end,
						workspace.comb_val.data(),
						input_data.categ_data + workspace.col_chosen * input_data.nrows,
						input_data.ncat[workspace.col_chosen],
						NULL,
						workspace.ext_fill_new[workspace.ntaken],
						workspace.chosen_cat[workspace.ntaken],
						workspace.ext_fill_val[workspace.ntaken],
						workspace.ext_fill_new[workspace.ntaken],
						NULL,
						model_params.new_cat_action,
						model_params.missing_action, SingleCateg, true,
						workspace.weights_arr);
				}

				else
				{
					add_linear_comb_weighted<decltype(workspace.weights_map),
											 ldouble_safe>(
						workspace.ix_arr.data(),
						workspace.st,
						workspace.end,
						workspace.comb_val.data(),
						input_data.categ_data + workspace.col_chosen * input_data.nrows,
						input_data.ncat[workspace.col_chosen],
						NULL,
						workspace.ext_fill_new[workspace.ntaken],
						workspace.chosen_cat[workspace.ntaken],
						workspace.ext_fill_val[workspace.ntaken],
						workspace.ext_fill_new[workspace.ntaken],
						NULL,
						model_params.new_cat_action,
						model_params.missing_action, SingleCateg, true,
						workspace.weights_map);
				}

				break;
			}

			case SubSet:
			{
				for (int cat = 0; cat < input_data.ncat[workspace.col_chosen];
					 cat++)
					workspace.ext_cat_coef[workspace.ntaken][cat] =
						workspace.coef_norm(workspace.rnd_generator);

				if (model_params.coef_by_prop)
				{
					int ncat = input_data.ncat[workspace.col_chosen];
					size_t *counts = workspace.buffer_szt.data();
					size_t *sorted_ix = workspace.buffer_szt.data() + ncat;
					/* calculate counts and sort by them */
					std::fill(counts, counts + ncat, (size_t)0);
					for (size_t ix = workspace.st; ix <= workspace.end; ix++)
						if (input_data.categ_data[workspace.col_chosen * input_data.nrows + ix] >= 0)
							counts[input_data.categ_data[workspace.col_chosen * input_data.nrows + ix]]++;
					std::iota(sorted_ix, sorted_ix + ncat, (size_t)0);
					std::sort(sorted_ix, sorted_ix + ncat, [&counts](const size_t a, const size_t b)
							  { return counts[a] < counts[b]; });
					/* now re-order the coefficients accordingly */
					std::sort(
						workspace.ext_cat_coef[workspace.ntaken].begin(),
						workspace.ext_cat_coef[workspace.ntaken].begin() + ncat);
					std::copy(
						workspace.ext_cat_coef[workspace.ntaken].begin(),
						workspace.ext_cat_coef[workspace.ntaken].begin() + ncat,
						workspace.buffer_dbl.begin());
					for (int ix = 0; ix < ncat; ix++)
						workspace.ext_cat_coef[workspace.ntaken][ix] =
							workspace.buffer_dbl[sorted_ix[ix]];
				}

				if (workspace.weights_arr.empty() && workspace.weights_map.empty())
				{
					add_linear_comb<ldouble_safe>(
						workspace.ix_arr.data(),
						workspace.st,
						workspace.end,
						workspace.comb_val.data(),
						input_data.categ_data + workspace.col_chosen * input_data.nrows,
						input_data.ncat[workspace.col_chosen],
						workspace.ext_cat_coef[workspace.ntaken].data(),
						(double)0,
						(int)0,
						workspace.ext_fill_val[workspace.ntaken],
						workspace.ext_fill_new[workspace.ntaken],
						workspace.buffer_szt.data(),
						workspace.buffer_szt.data() + input_data.max_categ + 1,
						model_params.new_cat_action,
						model_params.missing_action, SubSet, true);
				}

				else if (!workspace.weights_arr.empty())
				{
					add_linear_comb_weighted<decltype(workspace.weights_arr),
											 ldouble_safe>(
						workspace.ix_arr.data(),
						workspace.st,
						workspace.end,
						workspace.comb_val.data(),
						input_data.categ_data + workspace.col_chosen * input_data.nrows,
						input_data.ncat[workspace.col_chosen],
						workspace.ext_cat_coef[workspace.ntaken].data(),
						(double)0, (int)0,
						workspace.ext_fill_val[workspace.ntaken],
						workspace.ext_fill_new[workspace.ntaken],
						workspace.buffer_szt.data(),
						model_params.new_cat_action,
						model_params.missing_action, SubSet, true,
						workspace.weights_arr);
				}

				else
				{
					add_linear_comb_weighted<decltype(workspace.weights_map),
											 ldouble_safe>(
						workspace.ix_arr.data(),
						workspace.st,
						workspace.end,
						workspace.comb_val.data(),
						input_data.categ_data + workspace.col_chosen * input_data.nrows,
						input_data.ncat[workspace.col_chosen],
						workspace.ext_cat_coef[workspace.ntaken].data(),
						(double)0, (int)0,
						workspace.ext_fill_val[workspace.ntaken],
						workspace.ext_fill_new[workspace.ntaken],
						workspace.buffer_szt.data(),
						model_params.new_cat_action,
						model_params.missing_action, SubSet, true,
						workspace.weights_map);
				}

				break;
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

	class split_method
	{
	protected:
		attribute_tag tag_factory;
		split_type type;
		std::vector<attribute_tag > attributes;

	public:
		// default split - random 
		split_method() : tag_factory(), type(split_type::CONE_RANDOM), attributes()
		{
			split_method::_instance_counter++;
		}	 
		split_method(const split_method &other) : tag_factory(other.tag_factory), type(other.type), attributes(other.attributes)
		{
			split_method::_instance_counter++;
		}
		split_method(split_method &&other) : tag_factory(std::move(other.tag_factory)), type(other.type), attributes(std::move(other.attributes))
		{
			split_method::_instance_counter++;
		}
		split_method &	operator=(const split_method &other)
		{
			tag_factory = other.tag_factory;
			type = other.type;
			attributes = other.attributes;
			return *this;
		}
		virtual size_t
		size() const = 0;

		attribute_tag
		get_tag(const size_t i) const
		{

			static attribute_tag empty;
			if (i < attributes.size())
				return attributes.at(i);
			else
				return empty;
		}
		split_method(const attribute_tag &factory,
					 const std::vector<attribute_tag> &attribs) : tag_factory(factory), type(split_type::DISC), attributes(attribs)
		{
			split_method::_instance_counter++;
		}


		attribute_tag
		getFactoryTag() const
		{
			return tag_factory;
		}
		size_t
		get_num_of_attributes() const
		{
			return attributes.size();
		}
		const std::vector<attribute_tag> &
		get_attributes() const
		{
			return attributes;
		}
		virtual size_t
		print(std::ostream &out,
			  const attribute_information &attribute_info) const = 0;
		virtual size_t
		print(std::ostream &out, const attribute_information &attribute_info,
			  const size_t branch) const = 0;
		/*virtual void
		 serialize (std::ostream &out)=0;
		 virtual split_method*
		 deserialize (std::istream &in)=0;
		 */

		bool operator!=(const split_method &other) const
		{
			return (_TYPEINFO(*this) != _TYPEINFO(other)																	 // different types
					|| tag_factory != other.tag_factory || attributes.size() != other.attributes.size() || _compare(other)); // different attributes
		}
		virtual bool
		operator==(const split_method &other) const
		{
			if (typeid(*this) != typeid(other))
				return false;
			if (tag_factory != other.tag_factory)
				return false;
			if (attributes.size() != other.attributes.size())
				return false;
			for (size_t i = 0; i < attributes.size(); ++i)
				if (attributes[i] != other.attributes[i])
					return false;
			return _compare(other);
		}

		virtual bool
		_compare(const split_method &other) const = 0;

		// Check if the array of attributes belongs to some branch
		virtual bool
		isInBranch(const attribute_iterator &begin, uint32_t nbranch) const = 0;
		virtual bool
		isInBranch(const std::vector<attribute>::const_iterator &begin,
				   uint32_t nbranch) const = 0;

		// Return the branch that this arrays belongs

		virtual uint32_t
		getBranch(const attribute_iterator &begin) const = 0;
		virtual uint32_t
		getBranch(const std::vector<attribute>::const_iterator &begin) const = 0;

		virtual split_method *
		clone() const = 0;
		virtual split_type
		get_type() const = 0;
		void
		printName(std::ostream &out,
				  const attribute_information &attributes_info) const
		{
			// Print name of this splitter the attribute
			out << attributes_info.getName(get_tag(0));
		}

		void
		printBranch(std::ostream &out,
					const attribute_information &attributes_info,
					uint32_t nbranch) const
		{
			// Print branch description
			attribute branch(nbranch);

			out << attributes_info.getValue(get_tag(0), branch);
		}

		virtual ~split_method()
		{

			split_method::_instance_counter--;
			attributes.clear();
		}

		static uint64_t _instance_counter;
	};

	class discrete_split : virtual public split_method
	{
		size_t _size;

	public:
		virtual size_t
		print(std::ostream &out, const attribute_information &attribute_info) const
		{
			printName(out, attribute_info);
			return 0;
		}
		virtual size_t
		print(std::ostream &out, const attribute_information &attribute_info,
			  const size_t branch) const
		{
			printBranch(out, attribute_info, branch);
			return 0;
		}

		bool
		_compare(const split_method &other) const
		{
			if (
				_size != other.size() ||
				get_type() != other.get_type() ||
				get_tag(0) != other.get_tag(0))
				return false;

			for (size_t i = 0; i < _size; ++i)
				if (get_tag(i) != other.get_tag(i))
					return false;

			return true;
		}

		discrete_split(const attribute_tag &factory,
					   const std::vector<attribute_tag> &attribs) : split_method(factory, attribs), _size(attribs.size())
		{
		}
		discrete_split(attribute_tag factory_tag, const attribute_tag &tag,
					   const dataset &data_set) : split_method(factory_tag, std::vector<attribute_tag>(1, tag)), _size(data_set.getattributes().getCount(get_tag(0)))
		{
		}

		discrete_split() : _size(0)
		{
		}
		virtual ~discrete_split()
		{
		}

		virtual split_type
		get_type() const
		{
			return DISC;
		}

		// Get number of branches (size of the split method)
		size_t
		size() const
		{
			return _size;
		}

		// Check if the array of attributes belongs to some branch

		/*
	   template<class InputIterator>
		 bool
		 isInBranch (const InputIterator& b, uint32_t nbranch) const
		 {
			size_t n = get_tag(0);

			return ((b+n)->discrete () == nbranch );
		  return false;

		 }

	   // Return the branch that this arrays belongs
	   template<class InputIterator>
		 uint32_t
		 getBranch (const InputIterator& b) const
		 {
		size_t n = get_tag(0);
		   return  ((b.operator + (n) )->discrete());
		 }
		 */
		template < class InputIterator >
		bool
		isInBranch(const InputIterator &b, uint32_t nbranch) const
		{
			size_t n = get_tag(0);

			return ((b + n)->discrete() == nbranch);
			//  return isInBranch<std::vector<attribute>::const_iterator> (b, nbranch);
		}
		template < class InputIterator >
		uint32_t
		getBranch(const InputIterator &b) const
		{
			size_t n = get_tag(0);
			return ((b + n)->discrete());
		}	

		virtual bool
		isInBranch(const attribute_iterator &b, uint32_t nbranch) const
		{
			return isInBranch<attribute_iterator>(b, nbranch);

 		}
		virtual bool
		isInBranch(const std::vector<attribute>::const_iterator &b,
				   uint32_t nbranch) const
		{
			return isInBranch<const std::vector<attribute>::const_iterator>(b, nbranch);
			//  return isInBranch<std::vector<attribute>::const_iterator> (b, nbranch);
		}

		virtual uint32_t
		getBranch(const attribute_iterator &b) const
		{
			return getBranch<attribute_iterator>(b);
		}

		virtual uint32_t
		getBranch(const std::vector<attribute>::const_iterator &b) const
		{
			return getBranch<const std::vector<attribute>::const_iterator>(b);
		}

		split_method *
		clone() const
		{
			return new discrete_split(*this);
		}
		virtual void
		serialize(std::ostream &out)
		{
			out	<<	"DISC"	<<	":"	<<	_size	<<	std::endl;
			
			for (size_t i = 0; i < _size; ++i)
				out <<std::to_string(get_tag(i))<< std::string((i==_size-1)? " ":"," ) ;

			out << std::endl;
		}
		virtual split_method *
		deserialize(std::istream &in)
		{
			in>>_size;	
			this->attributes.resize(_size);

 			for (size_t i = 0; i < _size; ++i)
				in >> attributes[i];

 			return  clone();

		}
	};

	struct continous_base
	{
		
	public:
		continous_base()
		{
		}
		continous_base(continous_base &&other) = default;
		continous_base(const continous_base &other) = default;
		virtual ~continous_base()
		{
		}
		// Binary cut of an interval (returns true if a split was done, and false
		// if all the data on the interval belongs to the same class)
		// Output -> cut_pair : first = index of the cut point, second = value of the cut point
		bool
		binarySplit(const dataset &data, uint32_t begin, uint32_t end,
					const attribute_tag &tag,
					std::pair<uint32_t, Float> &cut_pair) const;
		// Calculate the gain of a given continuous attribute
		void
		split(const dataset &data, const attribute_tag &tag,
			  std::vector<Float> &interval) const;


		void seed(const std::random_device& seed) { _seed = (std::random_device*)( ptrdiff_t( &seed)) ;}
	protected:
		// Accept or reject splitting
		virtual bool
		checkSplitting(const dataset &data, uint32_t begin, uint32_t end,
					   uint32_t cut_point, const attribute_tag &tag) const = 0;
		// Select cut point
		virtual uint32_t
		selectPoint(std::vector<std::pair<Float, uint32_t>>::iterator begin,
					std::vector<std::pair<Float, uint32_t>>::iterator end) const;
		// Recursively split an interval
		void
		splitInterval(const dataset &data, uint32_t begin, uint32_t end,
					  const attribute_tag &tag,
					  std::vector<Float> &interval) const;

		std::random_device * _seed;

	};

	struct binary_split : public continous_base
	{

		binary_split(const binary_split &other) = default;
		binary_split(binary_split &&other) = default;
		binary_split() = default;

		// Accept or reject splitting
		bool
		checkSplitting(const dataset &data, uint32_t begin, uint32_t end,
					   uint32_t cut_point, const attribute_tag &tag) const
		{

			// Never split halfways

				if	(cut_point==0 || begin==end)
					return false;
				if	(cut_point==end)
					return false;

				if (cut_point == end-1)
					return true;
				if(data.get_target_tag()==tag )
					return true;					

 			return false;
		}		
		split_type
		get_type() const
		{
			return CONE_BINARY;
		}
		//serialize and deserialize
	

	};

	// MultiInterval splitting continuous attribute
	struct multi_interval_split : public continous_base
	{

		multi_interval_split() = default;
		multi_interval_split(const multi_interval_split &other) = default;
		multi_interval_split(multi_interval_split &&mv) = default;

		// Accept or reject splitting
		bool
		checkSplitting(const dataset &data, uint32_t begin, uint32_t end,
					   uint32_t cut_point, const attribute_tag &tag) const
		{
			// Always return true
			if (   begin==end)
				return false;	
			if (cut_point == end-1)	
				return true;	
			if(data.get_target_tag()==tag )	
				return true;
			
			return true;
		}
		
		split_type
		get_type() const
		{
			return CONE_MULTI;
		}

	 
	};

	// Random MultiInterval splitting continuous attribute
	struct random_split : public continous_base
	{

		std::random_device *_ran;

		random_split(std::random_device &rand_) : _ran( &rand_)
		{
		}
		
		// Default constructor (called when constructing this split method after deserialization and we don't need
		// a random number engine in that case)
		random_split() : _ran(nullptr)
		{
		}

		random_split(random_split &&other) : _ran(std::move(other._ran))
		{
		}

		random_split(const random_split &other) : _ran(std::move(other._ran))
		{
		}
		uint32_t
		selectPoint(std::vector<std::pair<Float, uint32_t>>::iterator begin,
					std::vector<std::pair<Float, uint32_t>>::iterator end) const;

		// Accept or reject splitting
		bool
		checkSplitting(const dataset &data, uint32_t begin, uint32_t end,
					   uint32_t cut_point, const attribute_tag &tag) const
		{
			// Like binary split
			std::random_device dev;
			std::mt19937 gen(dev());
			std::uniform_int_distribution<> uniform(0, RAND_MAX);
			
			if(data.get_target_tag()==tag )	
				return true;	
			if (cut_point == end-1)
				return true;
			if (cut_point == 0)
				return false;
			if (begin == end)
				return false;
			
			return (uniform(gen) % 2 == 1);


		}
		bool binarySplit (const dataset &data, uint32_t begin,
			       uint32_t end, const attribute_tag &tag,
			       std::pair<uint32_t, Float> &cut_pair) const;
				   		//serialize and deserialize

		

		split_type
		get_type() const
		{
			return CONE_RANDOM;
		}

	 
	};

	// minimum description length
	struct mdlp_split : public continous_base
	{
		// Accept or reject splitting
		bool
		checkSplitting(const provallo::dataset &data, uint32_t begin, uint32_t end,
					   uint32_t cut_point, const attribute_tag &tag) const;

		split_type
		get_type() const
		{
			return CONE_MDLP;
		}

	 
	};

	/*
	 class continous_split : virtual public split_method
	 {
	 size_t _size;
	 virtual split_method*
	 deserialize (std::istream &in);
	 virtual void
	 serialize (std::ostream &out)=0;
	 public:
	 split_type get_type() const {
	 return CONTINUOUS;
	 }


	 continous_split (const attribute_tag &factory,
	 const std::vector<attribute_tag> &attribs) :
	 split_method (factory, attribs), _size (attribs.size ())
	 {
	 }
	 continous_split () :
	 _size (0)
	 {
	 }
	 virtual
	 ~continous_split ();
	 split_method* clone() const {
	 return new continous_split(*this);
	 }

	 };
	 */

	template <class SplittingPolicy>
	class cont1d : public split_method, public SplittingPolicy
	{
		// Set of points that defines the intervals
		std::vector<Float> _interval;

		// Internal method to compare a split method
		bool
		_compare(const split_method &other) const;

		// Private copy constructor
		cont1d(const cont1d &other) :  		  split_method(other.getFactoryTag(),std::vector<attribute_tag>(1, other.get_tag(0))),SplittingPolicy((const SplittingPolicy&)other), _interval( other._interval)
		{

		}
		
		protected:

	
	public:

		cont1d() : split_method(), _interval()
		{

		}
	 
		cont1d(attribute_tag factory_tag, const attribute_tag &tag,
			   const dataset &data_set) : split_method(factory_tag, std::vector<attribute_tag>(1, tag))
		{
			// Set interval
			SplittingPolicy::split(data_set, get_tag(0), _interval);
		}

		cont1d(attribute_tag factory_tag, const attribute_tag &tag,
			   const dataset &data_set,const std::random_device& random_) : split_method(factory_tag, std::vector<attribute_tag>(1, tag))
		{
			// Set interval
			//

			//feed seed to splitting policy. 
			//
			SplittingPolicy::seed(random_);
				

			SplittingPolicy::split(data_set, get_tag(0), _interval);
			//std::cout<<"interval size "<<_interval.size()<<std::endl;

			

		}

		cont1d(attribute_tag factory_tag, const attribute_tag &tag,
			   const dataset &data_set, std::vector<Float> interval) : split_method(factory_tag, std::vector<attribute_tag>(1, tag)), _interval(interval)
		{
			SplittingPolicy::set_interval(interval);	
			

		}
		// Extra constructor to use on splitting policy
		template <class SplittingArg>
		cont1d(attribute_tag factory_tag, const attribute_tag &tag,
			   const dataset &data_set, SplittingArg split_arg) : split_method(factory_tag, std::vector<attribute_tag>(1, tag)), SplittingPolicy(split_arg)
		{
			// Set interval
			SplittingPolicy::split(data_set, get_tag(0), _interval);
		}
		// Extra constructor to use on splitting policy
		template <class SplittingArg>
		cont1d(attribute_tag factory_tag, const attribute_tag &tag,
			   const dataset &data_set, SplittingArg split_arg, std::vector<Float> interval) : split_method(factory_tag, std::vector<attribute_tag>(1, tag)), SplittingPolicy(split_arg), _interval(interval)
		{
			SplittingPolicy::set_interval(interval);	
		}	
		// Extran constructor with random_device 
		cont1d(attribute_tag factory_tag, const attribute_tag &tag,
			   const dataset &data_set, std::random_device &dev) : split_method(factory_tag, std::vector<attribute_tag>(1, tag))
		{
			// Set interval
			auto x= dev();
			x/=++x;


			SplittingPolicy::split(data_set, get_tag(0), _interval/*, dev*/);
		}	
		// Extran constructor with random_device
		cont1d(attribute_tag factory_tag, const attribute_tag &tag,
			   const dataset &data_set, std::random_device &dev, std::vector<Float> interval) : split_method(factory_tag, std::vector<attribute_tag>(1, tag)), _interval(interval)
		{
			SplittingPolicy::set_interval(interval);	
		}	
		// Extran constructor with random_device and splitting policy
		template <class SplittingArg>
		cont1d(attribute_tag factory_tag, const attribute_tag &tag,
			   const dataset &data_set, SplittingArg split_arg, std::random_device &dev) : split_method(factory_tag, std::vector<attribute_tag>(1, tag)), SplittingPolicy(split_arg)
		{
			// Set interval
			auto x= dev();
			x/=++x;

			SplittingPolicy::split(data_set, get_tag(0), _interval);
		}	




		// Clone split method
		split_method *
		clone() const
		{
			return new cont1d<SplittingPolicy>(*this);
		}

		split_type
		get_type() const
		{
			return SplittingPolicy::get_type();
		}

		// Get number of branches (size of the split method)
		size_t
		size() const
		{
			return _interval.size() - 1;
		}

		// Check if the array of attributes belongs to some branch
		template <class InputIterator>
		bool
		isInBranch(const InputIterator &begin, uint32_t nbranch) const
		{
			// Get value
			if (_interval.size() > nbranch)
			{
				Float value = (*(begin + get_tag(0))).continous();
				return (value > _interval[nbranch] && value <= _interval[nbranch + 1]);
			}
			return false;
		}

		// Return the branch that this arrays belongs
		template <class InputIterator>
		uint32_t
		getBranch(const InputIterator &begin) const
		{
			if (_interval.size() == 0)
				return 0;
			// Value
			Float value = (*(begin + get_tag(0))).continous();
			// Binary search
			return std::lower_bound(_interval.begin(), _interval.end(), value) - _interval.begin() - 1;
		}

		bool
		isInBranch(const attribute_iterator &begin, uint32_t nbranch) const
		{
			return isInBranch<attribute_iterator>(begin, nbranch);
		}
		bool
		isInBranch(const std::vector<attribute>::const_iterator &begin,
				   uint32_t nbranch) const
		{
			return isInBranch<std::vector<attribute>::const_iterator>(begin,
																	  nbranch);
		}

		uint32_t
		getBranch(const attribute_iterator &begin) const
		{
			return getBranch<attribute_iterator>(begin);
		}
		uint32_t
		getBranch(const std::vector<attribute>::const_iterator &begin) const
		{
			return getBranch<std::vector<attribute>::const_iterator>(begin);
		}

		void
		printName(std::ostream &out,
				  const attribute_information &attributes_info) const
		{
			// Print name of this splitter the attribute
			out << attributes_info.getName(get_tag(0));
		}

		void
		printBranch(std::ostream &out,
					const attribute_information &attributes_info,
					uint32_t nbranch) const
		{
			// Print branch description

			out << "(" << std::fixed << _interval[nbranch] << ","
				<< _interval[nbranch + 1] << ") ";
			out<< "attribute info : [ " << attributes_info.getName(get_tag(0)) << " ]";

		}
		virtual size_t
		print(std::ostream &out,
			  const attribute_information &attribute_info) const
		{
			printName(out, attribute_info);
			return 0;
		}
		virtual size_t
		print(std::ostream &out, const attribute_information &attribute_info,
			  const size_t branch) const
		{
			printBranch(out, attribute_info, branch);
			return 0;
		}

		virtual ~cont1d()
		{
			_interval.clear();
		}

		// Compare two split methods
		bool
		operator==(const split_method &other) const
		{
			// Check type
			if (get_type() != other.get_type())
				return false;
			// Static cast is safe because we already check the type on the base method
			const cont1d<SplittingPolicy> &method = *static_cast<const cont1d<
				SplittingPolicy> *>(&other);
			return _compare(method);
		}	
		bool operator!=(const split_method &other) const
		{
			return !(*this == other);
		}		
		bool	operator==(const cont1d<SplittingPolicy> &other) const
		{
			return _compare(other);
		}		
		bool	operator!=(const cont1d<SplittingPolicy> &other) const
		{
			return !(*this == other);
		}		

		//serialize splits : 
		// 1. type of split
		// 2. tag of split
		// 3. number of intervals
		// 4. intervals

		void serialize(std::ostream &out) const
		{
			out << get_type() << ":" << get_tag(0) << ":" << _interval.size() << " ";
			for (uint32_t i = 0; i < _interval.size(); ++i)
				out <<std::to_string( _interval[i] ) << std::string( (i==_interval.size()-1)?" ":",");
			out<<std::endl;
			SplittingPolicy::serialize(out);

		}	

		split_method* deserialize(std::istream& in)
		{

			/**/
			std::string line;
			std::getline(in, line);
			std::vector<std::string> tokens;
			provallo::tokenize(line, tokens, std::string(":"));
			if (tokens.size() != 3)
				throw std::runtime_error("Error deserializing split method");
			
			//if ( split_type(std::strtod( tokens[0].c_str ))!= cont1d<SplittingPolicy>::get_type())
			//	throw std::runtime_error("Error deserializing split method");


			uint32_t tag = std::stoi(tokens[1].c_str());
			uint32_t nintervals = std::stoi(tokens[2].c_str());
			std::getline(in, line);	
			std::vector<std::string> tokens2;
			provallo::tokenize(line,tokens2, std::string(","));
			if (tokens2.size() != nintervals)
				throw std::runtime_error("Error deserializing split method");
			std::vector<Float> intervals;
			for (uint32_t i = 0; i < nintervals; ++i)
				intervals.push_back(std::stof(tokens2[i].c_str())); 

			this->_interval = intervals;
			SplittingPolicy::deserialize(in);
			return clone();

		}
		 
	};

	template <class SplittingPolicy>
	bool
	cont1d<SplittingPolicy>::_compare(const split_method &other) const
	{
		// Static cast is safe because we already check the type on the base method
		const cont1d<SplittingPolicy> &method = *static_cast<const cont1d<
			SplittingPolicy> *>(&other);
		if (get_tag(0) != method.get_tag(0))
			return false;
		for (uint32_t i = 0; i < _interval.size(); ++i)
			if (_interval[i] != method._interval[i])
				return false;

		return true;
	}

	// Split method factory. Given a data set, the factory will return a split method
	class split_method_factory
	{

		// Map of tags and split methods
		std::vector<split_method *> _split_methods;
		// Target split method, so far is just a Discrete Split (because target attributes are categorical)
		split_method *_target_method;
		provallo::dataset &r_dataset;

		bool override_split_method ;
		split_type  override_split_type;

		// Return a split method
		/// @brief 	
		/// @param  
		/// @param type	 
		/// @param data_set 	
		/// @param tag 
		/// @param factory_tag 	
		/// @return 	split_method*	

		static split_method *
		createMethod(const std::random_device &, split_type type,
					 const provallo::dataset &data_set, const attribute_tag &tag,
					 const attribute_tag &factory_tag ,split_method_factory &factory);

	public:
		split_method_factory(const provallo::dataset &ds,
							 const std::random_device &);

		split_method_factory(const split_method_factory &other) : _split_methods(other._split_methods),
																  _target_method(other._target_method ? other._target_method->clone() : nullptr),
																  r_dataset(other.r_dataset),override_split_method(false),override_split_type(CONE_RANDOM)
		{

		}
		split_method_factory(split_method_factory &&other) : _split_methods(std::move(other._split_methods)),
															 _target_method(other._target_method),
															 r_dataset(other.r_dataset),override_split_method(other.override_split_method),override_split_type(other.override_split_type)
		{
			other._target_method = nullptr;
		}
		split_method_factory &operator=(split_method_factory &&other)
		{
			override_split_method = other.override_split_method;
			if (this->_target_method != other._target_method)
			{
				if (this->_target_method)
					delete _target_method;
				this->_target_method = other._target_method;
				other._target_method = nullptr;
			}
			this->_split_methods = std::move(other._split_methods);
			this->r_dataset = *(&other.r_dataset);
			this->override_split_type = other.override_split_type;
			return *this;
		}

		// Return a method from buffer data
		static split_method *
		createMethod(const split_method &deserial);

		const split_method_factory &operator=(const split_method_factory &other)
		{
			if (this->_target_method != other._target_method)
			{
				if (this->_target_method)
					delete _target_method;

				this->_target_method = other._target_method ? other._target_method->clone() : nullptr;
				this->_split_methods.clear();
				for (auto method : other._split_methods)
					_split_methods.push_back(method->clone());
			}

			return *this;
		}

		// Comparison operator
		bool
		operator==(const split_method_factory &other) const
		{
			if (not(*_target_method == *other._target_method))
				return false;
			for (uint32_t i = 0; i < _split_methods.size(); ++i)
				if (not(*_split_methods[i] == *other._split_methods[i]))
					return false;
			return true;
		}

		// Get a split method
		const split_method *
		getMethod(const attribute_tag &tag) const;

		// Get total number of partition in the attribute space. This is the effective
		// number of attributes seen by a classifier (not counting the target attribute)
		// Target attribute is not taken into account on this value
		virtual size_t
		getSize() const
		{
			return _split_methods.size();
		}
		void set_override_split_method(split_type override)
		{
			this->override_split_type = override;
			this->override_split_method = true;
		}
		// Return a split method for the target attribute
		const split_method *
		getTargetMethod() const
		{
			return _target_method;
		}

		// Write data to output buffer
		void
		serialize(split_method_factory *serial) const;
		// Get data from buffer
		void
		deserialize(const split_method_factory *serial);

		virtual ~split_method_factory();
		private:
		std::map<std::pair<size_t, size_t>, split_method *> _split_cache;
	};

	// Split by entropy,gain ratio,chi-square  :

	struct EntropyGain
	{
		// Calculate the gain of a given attribute
		Float
		gain(const dataset &data, const split_method &selector);
	};

	// Splitting criteria for C4.5
	struct GainRatio
	{
		// Calculate the gain of a given attribute
		Float
		gain(const dataset &data, const split_method &selector);
	};

	// Splitting criteria using ChiSquare test
	struct ChiSquare
	{
		// Calculate the gain of a given attribute
		Float
		gain(const dataset &data, const split_method &selector);
	};

	// distance  metrics
	struct Euclidean
	{
		Float
		distance(const attribute &a, const attribute &b) const
		{
			// Return square distance
			if(a.is_continous()){
			return (a.continous() - b.continous()) * (a.continous() - b.continous());
			}
			else
			{
				return (a.discrete() - b.discrete()) * (a.discrete() - b.discrete());
			}	
		}
	};

	// Basic overlap metric
	struct Overlap
	{
		Float
		distance(const attribute &a, const attribute &b) const
		{
			//return overlap condition
			if(a.is_discrete()) {
			if (a.discrete() != b.discrete())
				return (a.discrete()-b.discrete())*(a.discrete()-b.discrete())	;
			}
			else
			{
				if (a.continous() != b.continous())
					return (a.continous()-b.continous())*(a.continous()-b.continous())	;
			}
			return 1.0;
		}
	};

	// Generic metric (with weights)
	template <class DiscreteDistancePolicy, class ContinuousDistancePolicy>
	struct metric : public DiscreteDistancePolicy,
					public ContinuousDistancePolicy
	{

		// Calculate the distance between two samples
		template <class DataLeftIterator, class DataRightIterator,
				  class TypeIterator, class WeightIterator>
		Float
		distance(DataLeftIterator a_begin, DataLeftIterator a_end,
				 DataRightIterator b_begin, DataRightIterator b_end,
				 TypeIterator t_begin, WeightIterator w_begin) const
		{


			Float dist(0.0);
 			// same sample, no distance ?
			if (a_begin == b_begin ||b_begin == b_end)
				return dist;
			// Iterate over all attributes
			while (a_begin != a_end && b_begin != b_end)
			{
				attribute a (*a_begin);
				attribute b (*b_begin);
				attribute_type t (*t_begin);
				Float w (*w_begin);

				// skip ignored attributes.

				if(t==ignored_attribute::_type())
				{
					a_begin++;
					b_begin++;
					t_begin++;
					w_begin++;
					continue;
				}

				if(t == continous_attribute::_type() || t == discrete_attribute::_type())
					{
							if (t == continous_attribute::_type())
								dist += w * ContinuousDistancePolicy::distance(a, b);
							else if (t == discrete_attribute::_type())
								dist += w * DiscreteDistancePolicy::distance(a, b);
					}
					else
					{
						return dist;
					}
				a_begin++;
				b_begin++;
				t_begin++;
				w_begin++;
			} //while
			assert(b_begin == b_end);
			// std::cout<<"[+] metric distance CPU time (s) "<< std::to_string( double((double(end-start)/CLOCKS_PER_SEC) ))<<std::endl;
			// Return distance
			return dist;
		}
	};

	// Calculate entropy of a data set
	Float
	entropy(const dataset &data);

	// Get best class on a data set
	attribute
	getBestClass(const dataset &data);

	// Calculate the gini index of a data set
	Float gini(const dataset& data);

}

#endif /* DECISION_ENGINE_SPLIT_UTILS_HPP_ */
