#ifndef DENSITY_ESTIMATOR_H_
#define DENSITY_ESTIMATOR_H_
#include "classifier.h"
#include "split_utils.hpp"
namespace provallo
{

	extern bool interrupt_switch;
	// FWD DECLR:
	template <class InputData, class WorkerMemory, class ldouble_safe>
	void
	calc_var_all_cols(InputData &data, WorkerMemory &workspace,
					  ModelParams &model_params, double *variances,
					  double *saved_xmin, double *saved_xmax,
					  double *saved_means, double *saved_sds);

	inline bool
	is_col_taken(std::vector<bool> &col_is_taken,
				 hashed_set<size_t> &col_is_taken_s,
				 size_t col_num);
	template <class ldouble_safe, class xreal_>
	inline void
	provallo::density_estimator<ldouble_safe, xreal_>::initialize(
		size_t max_depth, int max_categ, bool reserve_counts,
		ScoringMetric scoring_metric)
	{

		this->multipliers.reserve(max_depth + 3);
		this->multipliers.clear();
		if (scoring_metric != AdjDensity)
			this->multipliers.push_back(0);
		else
			this->multipliers.push_back(1);

		if (reserve_counts)
		{
			this->counts.resize(max_categ);
		}
	}

	template <class ldouble_safe, class xreal_>
	template <class InputData>
	inline void
	provallo::density_estimator<ldouble_safe, xreal_>::initialize_bdens(
		const InputData &data, const ModelParams &model_params,
		std::vector<size_t> &ix_arr,
		column_sampler<ldouble_safe> &col_sampler)
	{
		this->fast_bratio = model_params.fast_bratio;
		if (this->fast_bratio)
		{
			this->multipliers.reserve(model_params.max_depth + 3);
			this->multipliers.push_back(0);
		}

		if (data.range_low != NULL || data.ncat_ != NULL)
		{
			if (data.ncols_numeric)
			{
				this->queue_box.reserve(model_params.max_depth + 3);
				this->box_low.assign(data.range_low,
									 data.range_low + data.ncols_numeric);
				this->box_high.assign(data.range_high,
									  data.range_high + data.ncols_numeric);
			}

			if (data.ncols_categ)
			{
				this->queue_ncat.reserve(model_params.max_depth + 2);
				this->ncat.assign(data.ncat_, data.ncat_ + data.ncols_categ);
			}

			if (!this->fast_bratio)
			{
				if (data.ncols_numeric)
				{
					this->ranges.resize(data.ncols_numeric);
					for (size_t col = 0; col < data.ncols_numeric; col++)
						this->ranges[col] = this->box_high[col] - this->box_low[col];
				}

				if (data.ncols_categ)
				{
					this->ncat_orig = this->ncat;
				}
			}

			return;
		}

		if (data.ncols_numeric)
		{
			this->queue_box.reserve(model_params.max_depth + 3);
			this->box_low.resize(data.ncols_numeric);
			this->box_high.resize(data.ncols_numeric);
			if (!this->fast_bratio)
				this->ranges.resize(data.ncols_numeric);
		}
		if (data.ncols_categ)
		{
			this->queue_ncat.reserve(model_params.max_depth + 2);
		}
		bool unsplittable = false;

		size_t npresent = 0;
		std::vector<signed char> categ_present;
		if (data.ncols_categ)
		{
			categ_present.resize(data.max_categ);
		}

		col_sampler.prepare_full_pass();
		size_t col;
		while (col_sampler.sample_col(col))
		{
			if (col < data.ncols_numeric)
			{
				if (data.Xc_indptr != NULL)
				{
					get_range((size_t *)ix_arr.data(), (size_t)0,
							  ix_arr.size() - (size_t)1, col, data.Xc,
							  data.Xc_ind, data.Xc_indptr,
							  model_params.missing_action, this->box_low[col],
							  this->box_high[col], unsplittable);
				}

				else
				{
					get_range((size_t *)ix_arr.data(),
							  data.numeric_data + data.nrows * col, (size_t)0,
							  ix_arr.size() - (size_t)1,
							  model_params.missing_action, this->box_low[col],
							  this->box_high[col], unsplittable);
				}

				if (unsplittable)
				{
					this->box_low[col] = 0;
					this->box_high[col] = 0;
					if (!this->fast_bratio)
						this->ranges[col] = 0;
					col_sampler.drop_col(col);
				}

				if (!this->fast_bratio)
				{
					this->ranges[col] = (ldouble_safe)this->box_high[col] - (ldouble_safe)this->box_low[col];
					this->ranges[col] = std::fmax(this->ranges[col],
												  (ldouble_safe)0);
				}
			}

			else
			{
				get_categs(
					(size_t *)ix_arr.data(),
					data.categ_data + data.nrows * (col - data.ncols_numeric),
					(size_t)0, ix_arr.size() - (size_t)1, data.ncat[col],
					model_params.missing_action, categ_present.data(),
					npresent, unsplittable);

				if (unsplittable)
				{
					this->ncat[col - data.ncols_numeric] = 1;
					col_sampler.drop_col(col);
				}

				else
				{
					this->ncat[col - data.ncols_numeric] = npresent;
				}
			}
		}

		if (!this->fast_bratio)
			this->ncat_orig = this->ncat;
	}

	template <class ldouble_safe, class xreal>
	template <class InputData>
	inline void
	provallo::density_estimator<ldouble_safe, xreal>::initialize_bdens_ext(
		const InputData &data, const ModelParams &model_params,
		std::vector<size_t> &ix_arr,
		column_sampler<ldouble_safe> &col_sampler, bool col_sampler_is_fresh)
	{
		//test full pass:
		if (col_sampler_is_fresh)
			col_sampler.prepare_full_pass();	
		
		this->vals_ext_box.reserve(model_params.max_depth + 3);
		this->queue_ext_box.reserve(model_params.max_depth + 3);
		this->vals_ext_box.push_back(0);

		if (data.range_low != NULL)
		{
			this->box_low.assign(data.range_low,
								 data.range_low + data.ncols_numeric);
			this->box_high.assign(data.range_high,
								  data.range_high + data.ncols_numeric);
			return;
		}

		this->box_low.resize(data.ncols_numeric);
		this->box_high.resize(data.ncols_numeric);
		bool unsplittable = false;

		/* TODO: find out if there's an optimal point for choosing one or the other loop
		 when using 'leave_m_cols' and when using 'prob_pick_col_by_range', then fill in the
		 lines that are commented out. */
		// if (!data.ncols_categ || model_params.ncols_per_tree < data.ncols_numeric)
		if (data.ncols_numeric)
		{
			col_sampler.prepare_full_pass();
			size_t col;
			while (col_sampler.sample_col(col))
			{
				if (col >= data.ncols_numeric)
					continue;
				if (data.Xc_indptr != NULL)
				{
					get_range((size_t *)ix_arr.data(), (size_t)0,
							  ix_arr.size() - (size_t)1, col, data.Xc,
							  data.Xc_ind, data.Xc_indptr,
							  model_params.missing_action, this->box_low[col],
							  this->box_high[col], unsplittable);
				}

				else
				{
					get_range((size_t *)ix_arr.data(),
							  data.numeric_data + data.nrows * col, (size_t)0,
							  ix_arr.size() - (size_t)1,
							  model_params.missing_action, this->box_low[col],
							  this->box_high[col], unsplittable);
				}

				if (unsplittable)
				{
					this->box_low[col] = 0;
					this->box_high[col] = 0;
					col_sampler.drop_col(col);
				}
			}
		}
 		// else if (data.ncols_numeric)
		// {
		//     size_t n_unsplittable = 0;
		//     std::vector<size_t> unsplittable_cols;
		//     if (col_sampler_is_fresh && !col_sampler.has_weights())
		//         unsplittable_cols.reserve(data.ncols_numeric);

		//     /* TODO: this will do unnecessary calculations when using 'leave_m_cols' */
		//     for (size_t col = 0; col < data.ncols_numeric; col++)
		//     {
		//         if (data.Xc_indptr != NULL)
		//         {
		//             get_range((size_t*)ix_arr.data(), (size_t)0, ix_arr.size()-(size_t)1, col,
		//                       data.Xc, data.Xc_ind, data.Xc_indptr,
		//                       model_params.missing_action, this->box_low[col], this->box_high[col], unsplittable);
		//         }

		//         else
		//         {
		//             get_range((size_t*)ix_arr.data(), data.numeric_data + data.nrows * col, (size_t)0, ix_arr.size()-(size_t)1,
		//                       model_params.missing_action, this->box_low[col], this->box_high[col], unsplittable);
		//         }

		//         if (unsplittable)
		//         {
		//             this->box_low[col] = 0;
		//             this->box_high[col] = 0;
		//             n_unsplittable++;
		//             if (col_sampler.has_weights())
		//                 col_sampler.drop_col(col);
		//             else if (col_sampler_is_fresh)
		//                 unsplittable_cols.push_back(col);
		//         }
		//     }

		//     if (n_unsplittable && col_sampler_is_fresh && !col_sampler.has_weights())
		//     {
		//         #if (__cplusplus >= 202002L)
		//         for (auto col : unsplittable_cols | std::views::reverse)
		//             col_sampler.drop_from_tail(col);
		//         #else
		//         for (size_t inv_col = 0; inv_col < unsplittable_cols.size(); inv_col++)
		//         {
		//             size_t col = unsplittable_cols.size() - inv_col - 1;
		//             col_sampler.drop_from_tail(unsplittable_cols[col]);
		//         }
		//         #endif
		//     }

		//     else if (n_unsplittable > model_params.sample_size / 16 && !col_sampler_is_fresh && !col_sampler.has_weights())
		//     {
		//         /* TODO */
		//     }
		// }
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_density(
		double xmin, double xmax, double split_point)
	{
		if (std::isinf(xmax) || std::isinf(xmin) || std::isnan(xmin) || std::isnan(xmax) || std::isnan(split_point))
		{
			this->multipliers.push_back(0);
			return;
		}

		double range = std::fmax(xmax - xmin,
								 std::numeric_limits<double>::min());
		double dleft = std::fmax(split_point - xmin,
								 std::numeric_limits<double>::min());
		double dright = std::fmax(xmax - split_point,
								  std::numeric_limits<double>::min());
		double mult_left = std::log(dleft / range);
		double mult_right = std::log(dright / range);
		while (std::isinf(mult_left))
		{
			dleft = std::nextafter(dleft,
								   (mult_left < 0) ? HUGE_VAL : (-HUGE_VAL));
			mult_left = std::log(dleft / range);
		}
		while (std::isinf(mult_right))
		{
			dright = std::nextafter(dright,
									(mult_right < 0) ? HUGE_VAL : (-HUGE_VAL));
			mult_right = std::log(dright / range);
		}

		mult_left = std::isnan(mult_left) ? 0 : mult_left;
		mult_right = std::isnan(mult_right) ? 0 : mult_right;

		ldouble_safe curr = this->multipliers.back();
		this->multipliers.push_back(curr + mult_right);
		this->multipliers.push_back(curr + mult_left);
	}
	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_density(
		int n_left, int n_present)
	{
		this->push_density(0., (double)n_present, (double)n_left);
	}

	/* For single category splits */
	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_density(
		size_t counts[], int ncat)
	{
		/* this one assumes 'categ_present' has entries 0/1 for missing/present */
		int n_present = 0;
		for (int cat = 0; cat < ncat; cat++)
			n_present += counts[cat] > 0;
		this->push_density(0., (double)n_present, 1.);
	}

	/* For single category splits */
	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_density(
		int n_present)
	{
		this->push_density(0., (double)n_present, 1.);
	}

	/* For binary categorical splits */
	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_density()
	{
		this->multipliers.push_back(0);
		this->multipliers.push_back(0);
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_adj(
		double xmin, double xmax, double split_point, double pct_tree_left,
		ScoringMetric scoring_metric)
	{
		double range = std::fmax(xmax - xmin,
								 std::numeric_limits<double>::min());
		double dleft = std::fmax(split_point - xmin,
								 std::numeric_limits<double>::min());
		double dright = std::fmax(xmax - split_point,
								  std::numeric_limits<double>::min());
		double chunk_left = dleft / range;
		double chunk_right = dright / range;
		bool alignkr = true;
		if (std::isinf(xmax) || std::isinf(xmin) || std::isnan(xmin) || std::isnan(xmax) || std::isnan(split_point))
		{
			chunk_left = pct_tree_left;
			chunk_right = 1. - pct_tree_left;
			alignkr = false;
		}

		if (std::isnan(chunk_left) || std::isnan(chunk_right))
		{
			chunk_left = 0.5;
			chunk_right = 0.5;
		}
		if (alignkr)
		{
			chunk_left = pct_tree_left / chunk_left;
			chunk_right = (1. - pct_tree_left) / chunk_right;
		}
		chunk_left = 2. / (1. + .5 / chunk_left);
		chunk_right = 2. / (1. + .5 / chunk_right);
		// chunk_left = 2. / (1. + 1./chunk_left);
		// chunk_right = 2. / (1. + 1./chunk_right);
		// chunk_left = 2. - std::exp2(1. - chunk_left);
		// chunk_right = 2. - std::exp2(1. - chunk_right);

		ldouble_safe curr = this->multipliers.back();
		if (scoring_metric == AdjDepth)
		{
			this->multipliers.push_back(curr + chunk_right);
			this->multipliers.push_back(curr + chunk_left);
		}

		else
		{
			this->multipliers.push_back(
				std::fmax(
					curr * chunk_right,
					(ldouble_safe)std::numeric_limits<double>::epsilon()));
			this->multipliers.push_back(
				std::fmax(
					curr * chunk_left,
					(ldouble_safe)std::numeric_limits<double>::epsilon()));
		}
	}
	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_adj(
		signed char *categ_present, size_t *counts, int ncat,
		ScoringMetric scoring_metric)
	{
		/* this one assumes 'categ_present' has entries -1/0/1 for missing/right/left */
		int cnt_cat_left = 0;
		int cnt_cat = 0;
		size_t cnt = 0;
		size_t cnt_left = 0;
		for (int cat = 0; cat < ncat; cat++)
		{
			if (counts[cat] > 0)
			{
				cnt += counts[cat];
				cnt_cat_left += categ_present[cat];
				cnt_left += categ_present[cat] ? counts[cat] : 0;
				cnt_cat++;
			}
		}

		double pct_tree_left = (ldouble_safe)cnt_left / (ldouble_safe)cnt;
		this->push_adj(0., (double)cnt_cat, (double)cnt_cat_left,
					   pct_tree_left, scoring_metric);
	}

	/* For single category splits */
	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_adj(
		size_t *counts, int ncat, int chosen_cat, ScoringMetric scoring_metric)
	{
		/* this one assumes 'categ_present' has entries 0/1 for missing/present */
		int cnt_cat = 0;
		size_t cnt = 0;
		for (int cat = 0; cat < ncat; cat++)
		{
			cnt += counts[cat];
			cnt_cat += counts[cat] > 0;
		}

		double pct_tree_left = (ldouble_safe)counts[chosen_cat] / (ldouble_safe)cnt;
		this->push_adj(0., (double)cnt_cat, 1., pct_tree_left, scoring_metric);
	}

	/* For binary categorical splits */
	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_adj(
		double pct_tree_left, ScoringMetric scoring_metric)
	{
		this->push_adj(0., 1., 0.5, pct_tree_left, scoring_metric);
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_bdens(
		double split_point, size_t col)
	{
		if (this->fast_bratio)
			this->push_bdens_fast_route(split_point, col);
		else
			this->push_bdens_internal(split_point, col);
	}
	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_bdens_internal(
		double split_point, size_t col)
	{
		this->queue_box.push_back(this->box_high[col]);
		this->box_high[col] = split_point;
	}
	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_bdens_fast_route(
		double split_point, size_t col)
	{
		ldouble_safe curr_range = (ldouble_safe)this->box_high[col] - (ldouble_safe)this->box_low[col];
		ldouble_safe fraction_left = ((ldouble_safe)split_point - (ldouble_safe)this->box_low[col]) / curr_range;
		ldouble_safe fraction_right = ((ldouble_safe)this->box_high[col] - (ldouble_safe)split_point) / curr_range;
		fraction_left = std::fmax(
			fraction_left, (ldouble_safe)std::numeric_limits<double>::min());
		fraction_left = std::fmin(
			fraction_left,
			(ldouble_safe)(1. - std::numeric_limits<double>::epsilon()));
		fraction_left = std::log(fraction_left);
		fraction_left += this->multipliers.back();
		fraction_right = std::fmax(
			fraction_right, (ldouble_safe)std::numeric_limits<double>::min());
		fraction_right = std::fmin(
			fraction_right,
			(ldouble_safe)(1. - std::numeric_limits<double>::epsilon()));
		fraction_right = std::log(fraction_right);
		fraction_right += this->multipliers.back();
		this->multipliers.push_back(fraction_right);
		this->multipliers.push_back(fraction_left);

		this->push_bdens_internal(split_point, col);
	}
	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_bdens(
		int ncat_branch_left, size_t col)
	{
		if (this->fast_bratio)
			this->push_bdens_fast_route(ncat_branch_left, col);
		else
			this->push_bdens_internal(ncat_branch_left, col);
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_bdens_internal(
		int ncat_branch_left, size_t col)
	{
		this->queue_ncat.push_back(this->ncat[col]);
		this->ncat[col] = ncat_branch_left;
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_bdens_fast_route(
		int ncat_branch_left, size_t col)
	{
		double fraction_left = std::log(
			(double)ncat_branch_left / this->ncat[col]);
		double fraction_right = std::log(
			(double)(this->ncat[col] - ncat_branch_left) / this->ncat[col]);
		ldouble_safe curr = this->multipliers.back();
		this->multipliers.push_back(curr + fraction_right);
		this->multipliers.push_back(curr + fraction_left);

		this->push_bdens_internal(ncat_branch_left, col);
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_bdens(
		const std::vector<signed char> &cat_split, size_t col)
	{
		if (this->fast_bratio)
			this->push_bdens_fast_route(cat_split, col);
		else
			this->push_bdens_internal(cat_split, col);
	}
	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_bdens(
		const std::vector<char> &cat_split, size_t col)
	{
		this->push_bdens((const std::vector<char> &)cat_split, col);
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_bdens_internal(
		const std::vector<signed char> &cat_split, size_t col)
	{
		int ncat_branch_left = 0;
		for (auto el : cat_split)
			ncat_branch_left += el == 1;
		this->push_bdens_internal(ncat_branch_left, col);
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_bdens_fast_route(
		const std::vector<signed char> &cat_split, size_t col)
	{
		int ncat_branch_left = 0;
		for (auto el : cat_split)
			ncat_branch_left += el == 1;
		this->push_bdens_fast_route(ncat_branch_left, col);
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::push_bdens_ext(
		const IsoHPlane &hplane, const ModelParams &model_params)
	{
		double x1, x2;
		double xlow = 0, xhigh = 0;
		size_t col;
		size_t col_num = 0;
		size_t col_cat = 0;

		for (size_t col_outer = 0; col_outer < hplane.col_num.size();
			 col_outer++)
		{
			switch (hplane.col_type[col_outer])
			{
			case Numeric:
			{
				col = hplane.col_num[col_outer];
				x1 = hplane.coef[col_num] * (this->box_low[col] - hplane.mean[col_num]);
				x2 = hplane.coef[col_num] * (this->box_high[col] - hplane.mean[col_num]);
				xlow += std::fmin(x1, x2);
				xhigh += std::fmax(x1, x2);
				break;
			}

			case Categorical:
			{
				switch (model_params.cat_split_type)
				{
				case SingleCateg:
				{
					xlow += std::fmin(hplane.fill_new[col_cat], 0.);
					xhigh += std::fmax(hplane.fill_new[col_cat], 0.);
					break;
				}

				case SubSet:
				{
					xlow += *std::min_element(
						hplane.cat_coef[col_cat].begin(),
						hplane.cat_coef[col_cat].end());
					xhigh += *std::max_element(
						hplane.cat_coef[col_cat].begin(),
						hplane.cat_coef[col_cat].end());
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

		double chunk_left;
		double chunk_right;
		double xdiff = xhigh - xlow;

		if (model_params.scoring_metric != BoxedDensity)
		{
			chunk_left = (hplane.split_point - xlow) / xdiff;
			chunk_right = (xhigh - hplane.split_point) / xdiff;
			chunk_left = std::fmin(chunk_left,
								   std::numeric_limits<double>::min());
			chunk_left = std::fmax(chunk_left,
								   1. - std::numeric_limits<double>::epsilon());
			chunk_right = std::fmin(chunk_right,
									std::numeric_limits<double>::min());
			chunk_right = std::fmax(
				chunk_right, 1. - std::numeric_limits<double>::epsilon());
		}

		else
		{
			chunk_left = xdiff / (hplane.split_point - xlow);
			chunk_right = xdiff / (xhigh - hplane.split_point);
			chunk_left = std::fmin(chunk_left, 1.);
			chunk_right = std::fmin(chunk_right, 1.);
		}

		this->queue_ext_box.push_back(
			std::log(chunk_right) + this->vals_ext_box.back());
		this->vals_ext_box.push_back(
			std::log(chunk_left) + this->vals_ext_box.back());
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::pop()
	{
		this->multipliers.pop_back();
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::pop_right()
	{
		this->multipliers.pop_back();
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::pop_bdens(size_t col)
	{
		if (this->fast_bratio)
			this->pop_bdens_fast_route(col);
		else
			this->pop_bdens_internal(col);
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::pop_bdens_internal(
		size_t col)
	{
		double old_high = this->queue_box.back();
		this->queue_box.pop_back();
		this->queue_box.push_back(this->box_low[col]);
		this->box_low[col] = this->box_high[col];
		this->box_high[col] = old_high;
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::pop_bdens_fast_route(
		size_t col)
	{
		this->multipliers.pop_back();
		this->pop_bdens_internal(col);
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::pop_bdens_right(
		size_t col)
	{
		if (this->fast_bratio)
			this->pop_bdens_right_fast_route(col);
		else
			this->pop_bdens_right_internal(col);
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::pop_bdens_right_internal(
		size_t col)
	{
		double old_low = this->queue_box.back();
		this->queue_box.pop_back();
		this->box_low[col] = old_low;
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::pop_bdens_right_fast_route(
		size_t col)
	{
		this->multipliers.pop_back();
		this->pop_bdens_right_internal(col);
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::pop_bdens_cat(size_t col)
	{
		if (this->fast_bratio)
			this->pop_bdens_cat_fast_route(col);
		else
			this->pop_bdens_cat_internal(col);
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::pop_bdens_cat_internal(
		size_t col)
	{
		int old_ncat = this->queue_ncat.back();
		this->ncat[col] = old_ncat - this->ncat[col];
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::pop_bdens_cat_fast_route(
		size_t col)
	{
		this->multipliers.pop_back();
		this->pop_bdens_cat_internal(col);
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::pop_bdens_cat_right(
		size_t col)
	{
		if (this->fast_bratio)
			this->pop_bdens_cat_right_fast_route(col);
		else
			this->pop_bdens_cat_right_internal(col);
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::pop_bdens_cat_right_internal(
		size_t col)
	{
		int old_ncat = this->queue_ncat.back();
		this->queue_ncat.pop_back();
		this->ncat[col] = old_ncat;
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::pop_bdens_cat_right_fast_route(
		size_t col)
	{
		this->multipliers.pop_back();
		this->pop_bdens_cat_right_internal(col);
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::pop_bdens_ext()
	{
		this->vals_ext_box.pop_back();
		this->vals_ext_box.push_back(this->queue_ext_box.back());
		this->queue_ext_box.pop_back();
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::pop_bdens_ext_right()
	{
		this->vals_ext_box.pop_back();
	}

	/* this outputs the logarithm of the density */
	template <class ldouble_safe, class xreal>
	double
	provallo::density_estimator<ldouble_safe, xreal>::calc_density(
		ldouble_safe remainder, size_t sample_size)
	{
		return std::log(remainder) - std::log((ldouble_safe)sample_size) - this->multipliers.back();
	}

	template <class ldouble_safe, class xreal>
	ldouble_safe
	provallo::density_estimator<ldouble_safe, xreal>::calc_adj_depth()
	{
		ldouble_safe out = this->multipliers.back();
		return std::fmax(out, (ldouble_safe)std::numeric_limits<double>::min());
	}

	template <class ldouble_safe, class xreal>
	double
	provallo::density_estimator<ldouble_safe, xreal>::calc_adj_density()
	{
		return this->multipliers.back();
	}

	/* this outputs the logarithm of the density */
	template <class ldouble_safe, class xreal>
	ldouble_safe
	provallo::density_estimator<ldouble_safe, xreal>::calc_bratio_inv_log()
	{
		if (!this->multipliers.empty())
			return -this->multipliers.back();

		ldouble_safe sum_log_switdh = 0;
		ldouble_safe ratio_col;
		for (size_t col = 0; col < this->ranges.size(); col++)
		{
			if (!this->ranges[col])
				continue;
			ratio_col = this->ranges[col] / ((ldouble_safe)this->box_high[col] - (ldouble_safe)this->box_low[col]);
			ratio_col = std::fmax(ratio_col, (ldouble_safe)1);
			sum_log_switdh += std::log(ratio_col);
		}

		for (size_t col = 0; col < this->ncat.size(); col++)
		{
			if (this->ncat_orig[col] <= 1)
				continue;
			sum_log_switdh += std::log(
				(double)this->ncat_orig[col] / (double)this->ncat[col]);
		}

		return sum_log_switdh;
	}

	template <class ldouble_safe, class xreal>
	ldouble_safe
	provallo::density_estimator<ldouble_safe, xreal>::calc_bratio_log()
	{
		if (!this->multipliers.empty())
			return this->multipliers.back();

		ldouble_safe sum_log_switdh = 0;
		ldouble_safe ratio_col;
		for (size_t col = 0; col < this->ranges.size(); col++)
		{
			if (!this->ranges[col])
				continue;
			ratio_col = ((ldouble_safe)this->box_high[col] - (ldouble_safe)this->box_low[col]) / this->ranges[col];
			ratio_col = std::fmax(
				ratio_col, (ldouble_safe)std::numeric_limits<double>::min());
			ratio_col = std::fmin(
				ratio_col,
				(ldouble_safe)(1. - std::numeric_limits<double>::epsilon()));
			sum_log_switdh += std::log(ratio_col);
		}

		for (size_t col = 0; col < this->ncat.size(); col++)
		{
			if (this->ncat_orig[col] <= 1)
				continue;
			sum_log_switdh += std::log(
				(double)this->ncat[col] / (double)this->ncat_orig[col]);
		}

		return sum_log_switdh;
	}

	/* this does NOT output the logarithm of the density */
	template <class ldouble_safe, class xreal>
	double
	provallo::density_estimator<ldouble_safe, xreal>::calc_bratio()
	{
		return std::exp(this->calc_bratio_log());
	}

	const double MIN_DENS = std::log(std::numeric_limits<double>::min());

	/* this outputs the logarithm of the density */
	template <class ldouble_safe, class xreal>
	double
	provallo::density_estimator<ldouble_safe, xreal>::calc_bdens(
		ldouble_safe remainder, size_t sample_size)
	{
		double out = std::log(remainder) - std::log((ldouble_safe)sample_size) - this->calc_bratio_inv_log();
		return std::fmax(out, MIN_DENS);
	}

	/* this outputs the logarithm of the density */
	template <class ldouble_safe, class xreal>
	double
	provallo::density_estimator<ldouble_safe, xreal>::calc_bdens2(
		ldouble_safe remainder, size_t sample_size)
	{
		double out = std::log(remainder) - std::log((ldouble_safe)sample_size) - this->calc_bratio_log();
		return std::fmax(out, MIN_DENS);
	}

	/* this outputs the logarithm of the density */
	template <class ldouble_safe, class xreal>
	ldouble_safe
	provallo::density_estimator<ldouble_safe, xreal>::calc_bratio_log_ext()
	{
		return this->vals_ext_box.back();
	}

	template <class ldouble_safe, class xreal>
	double
	provallo::density_estimator<ldouble_safe, xreal>::calc_bratio_ext()
	{
		double out = std::exp(this->calc_bratio_log_ext());
		return std::fmax(out, std::numeric_limits<double>::min());
	}

	/* this outputs the logarithm of the density */
	template <class ldouble_safe, class xreal>
	double
	provallo::density_estimator<ldouble_safe, xreal>::calc_bdens_ext(
		ldouble_safe remainder, size_t sample_size)
	{
		double out = std::log(remainder) - std::log((ldouble_safe)sample_size) - this->calc_bratio_log_ext();
		return std::fmax(out, MIN_DENS);
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::save_range(double xmin,
																 double xmax)
	{
		this->xmin = xmin;
		this->xmax = xmax;
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::restore_range(
		double &xmin, double &xmax)
	{
		xmin = this->xmin;
		xmax = this->xmax;
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::save_counts(
		size_t *cat_counts, int ncat)
	{
		this->counts.assign(cat_counts, cat_counts + ncat);
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::save_n_present_and_left(
		signed char *split_left, int ncat)
	{
		this->n_present = 0;
		this->n_left = 0;
		for (int cat = 0; cat < ncat; cat++)
		{
			this->n_present += split_left[cat] >= 0;
			this->n_left += split_left[cat] == 1;
		}
	}

	template <class ldouble_safe, class xreal>
	void
	provallo::density_estimator<ldouble_safe, xreal>::save_n_present(
		size_t *cat_counts, int ncat)
	{
		this->n_present = 0;
		for (int cat = 0; cat < ncat; cat++)
			this->n_present += cat_counts[cat] > 0;
	}

	template <class InputData, class WorkerMemory, class ldouble_safe>
	void
	split_hplane_recursive(std::vector<IsoHPlane> &hplanes,
						   WorkerMemory &workspace, InputData &data,
						   ModelParams &model_params,
						   std::vector<ImputeNode>* impute_nodes,
						   size_t curr_depth)
	{
		if (interrupt_switch)
			return;
		ldouble_safe sum_weight = -HUGE_VAL;
		size_t hplane_from = hplanes.size() - 1;
		std::unique_ptr<RecursionState> recursion_state;
		std::vector<bool> col_is_taken;
		hashed_set<size_t> col_is_taken_s; 
		
		/* calculate imputation statistics if desired if impute nodes were allocated.*/
		if (impute_nodes != NULL&&impute_nodes->size()>0)
		{
			ImputeNode impute_node = impute_nodes->back(); 

			if (data.Xc_indptr != NULL)
				std::sort(workspace.ix_arr.begin() + workspace.st,
						  workspace.ix_arr.begin() + workspace.end + 1);
			build_impute_node<decltype(data), decltype(workspace), ldouble_safe>(
				std::ref(impute_node ), workspace, data, model_params,
				*impute_nodes, curr_depth, model_params.min_imp_obs); 

			/* check for potential isolated leafs or unique splits */ 
			if (workspace.end == workspace.st || (workspace.end - workspace.st) == 1 || curr_depth >= model_params.max_depth)
				goto terminal_statistics;
			
		}

		/* check for potential isolated leafs or unique splits */
		if (workspace.end == workspace.st || (workspace.end - workspace.st) == 1 || curr_depth >= model_params.max_depth)
			goto terminal_statistics;

		/* when using weights, the split should stop when the sum of weights is <= 1 */
		sum_weight = calculate_sum_weights<ldouble_safe>(workspace.ix_arr,
														 workspace.st,
														 workspace.end,
														 curr_depth,
														 workspace.weights_arr,
														 workspace.weights_map);

		if (curr_depth > 0 && (!workspace.weights_arr.empty() || !workspace.weights_map.empty()) && sum_weight <= 1)
			goto terminal_statistics;

		/* for sparse matrices, need to sort the indices */
		if (data.Xc_indptr != NULL && impute_nodes == NULL)
			std::sort(workspace.ix_arr.begin() + workspace.st,
					  workspace.ix_arr.begin() + workspace.end + 1);

		/* pick column to split according to criteria */
		workspace.prob_split_type = workspace.rbin(workspace.rnd_generator);

		if (workspace.prob_split_type < (model_params.prob_pick_by_gain_avg + model_params.prob_pick_by_gain_pl + model_params.prob_pick_by_full_gain + model_params.prob_pick_by_dens))
		{
			workspace.ntry = model_params.ntry;
			hplanes.back().score = -HUGE_VAL; /* this keeps track of the gain */
			if (workspace.prob_split_type < model_params.prob_pick_by_gain_avg)
				workspace.criterion = Averaged;
			else if (workspace.prob_split_type < model_params.prob_pick_by_gain_avg + model_params.prob_pick_by_gain_pl)
				workspace.criterion = Pooled;
			else if (workspace.prob_split_type < model_params.prob_pick_by_gain_avg + model_params.prob_pick_by_gain_pl + model_params.prob_pick_by_full_gain)
				workspace.criterion = FullGain;
			else
				workspace.criterion = DensityCrit;
		}

		else
		{
			workspace.criterion = NoCrit;
			workspace.ntry = 1;
		}

		/* pick column selection method also according to criteria */
		if ((workspace.criterion != NoCrit && std::max(workspace.ntry, (size_t)1) >= workspace.col_sampler.get_remaining_cols()) || (workspace.col_sampler.get_remaining_cols() <= model_params.ndim))
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
				for (size_t col = 0; col < data.ncols_numeric; col++)
					workspace.node_col_weights[col] = data.range_high[col] - data.range_low[col];
				goto add_col_weights_to_ranges;
			}

			else
			{
				calc_ranges_all_cols(data, workspace, model_params,
									 workspace.node_col_weights.data(),
									 NULL,
									 NULL);
			}

			workspace.has_saved_stats = false;
		}

		else if (workspace.prob_split_type < (model_params.prob_pick_col_by_range + model_params.prob_pick_col_by_var))
		{
			workspace.col_criterion = ByVar;
			workspace.has_saved_stats = model_params.standardize_data || model_params.missing_action != Fail;
			calc_var_all_cols<InputData, WorkerMemory, ldouble_safe>(
				data, workspace, model_params, workspace.node_col_weights.data(),
				NULL,
				NULL,
				workspace.has_saved_stats ? workspace.saved_stat1.data() : NULL,
				workspace.has_saved_stats ? workspace.saved_stat2.data() : NULL);
		}

		else if (workspace.prob_split_type < (model_params.prob_pick_col_by_range + model_params.prob_pick_col_by_var + model_params.prob_pick_col_by_kurt))
		{
			workspace.col_criterion = ByKurt;
			calc_kurt_all_cols<decltype(data), decltype(workspace), ldouble_safe>(
				data, workspace, model_params, workspace.node_col_weights.data(),
				NULL,
				NULL);
			workspace.has_saved_stats = false;
		}

		else
		{
			workspace.col_criterion = Uniformly;
			workspace.has_saved_stats = false;
		}

		if (workspace.col_criterion != Uniformly)
		{
			if (!workspace.node_col_sampler.initialize(
					workspace.node_col_weights.data(),
					&workspace.col_sampler.col_indices,
					workspace.col_sampler.curr_pos, model_params.ndim,
					model_params.ntry > 1))
			{
				goto terminal_statistics;
			}

			if (model_params.ntry > 1)
			{
				workspace.node_col_sampler.backup(
					workspace.node_col_sampler_backup, data.ncols_tot);
			}
		}

		if (workspace.criterion != NoCrit && (!workspace.weights_arr.empty() || !workspace.weights_map.empty()))
		{
			if (!workspace.weights_arr.empty())
			{
				for (size_t row = workspace.st; row <= workspace.end; row++)
					workspace.sample_weights[row - workspace.st] =
						workspace.weights_arr[workspace.ix_arr[row]];
			}

			else
			{
				for (size_t row = workspace.st; row <= workspace.end; row++)
					workspace.sample_weights[row - workspace.st] =
						workspace.weights_map[workspace.ix_arr[row]];
			}
		}

		if (workspace.criterion == FullGain)
		{
			workspace.col_sampler.get_array_remaining_cols(
				workspace.col_indices);
		}

		workspace.ntaken_best = 0;

		for (size_t attempt = 0; attempt < workspace.ntry; attempt++)
		{
			if (attempt > 0 && workspace.col_criterion != Uniformly)
			{
				workspace.node_col_sampler.restore(
					workspace.node_col_sampler_backup);
			}

			if (workspace.col_criterion == Uniformly)
			{
				if (data.ncols_tot < 1e5 || ((ldouble_safe)model_params.ndim / (ldouble_safe)workspace.col_sampler.get_remaining_cols()) > .25)
				{
					if (!col_is_taken.size())
						col_is_taken.resize(data.ncols_tot, false);
					else
						col_is_taken.assign(data.ncols_tot, false);
				}
				else
				{
					col_is_taken_s.clear();
					col_is_taken_s.reserve(model_params.ndim);
				}
			}

			workspace.ntaken = 0;
			workspace.ntried = 0;
			std::fill(
				workspace.comb_val.begin(),
				workspace.comb_val.begin() + (workspace.end - workspace.st + 1),
				(double)0);

			if (model_params.ndim >= data.ncols_tot)
				workspace.col_sampler.prepare_full_pass();
			else if (workspace.try_all && workspace.col_criterion == Uniformly)
				workspace.col_sampler.shuffle_remainder(workspace.rnd_generator);
			size_t threshold_shuffle =
				(workspace.col_sampler.get_remaining_cols() + 1) / 2;

			while (
				(workspace.col_criterion != Uniformly) ? workspace.node_col_sampler.sample_col(
															 workspace.col_chosen, workspace.rnd_generator)
													   : (workspace.try_all ? workspace.col_sampler.sample_col(workspace.col_chosen) : workspace.col_sampler.sample_col(workspace.col_chosen, workspace.rnd_generator)))
			{
				if (interrupt_switch)
					return;

				if (workspace.col_criterion != Uniformly)
					goto add_this_col;

				workspace.ntried++;
				if (!workspace.try_all && workspace.ntried >= threshold_shuffle)
				{
					workspace.try_all = true;
					workspace.col_sampler.shuffle_remainder(
						workspace.rnd_generator);
				}

				if (is_col_taken(col_is_taken, col_is_taken_s,
								 workspace.col_chosen))
					continue;

				get_split_range(workspace, data, model_params);
				if (workspace.unsplittable)
				{
					if (workspace.col_criterion != Uniformly) /* <- used 'node_col_sampler' */
						unexpected_error();
					workspace.col_sampler.drop_col(
						workspace.col_chosen + ((workspace.col_type == Numeric) ? (size_t)0 : data.ncols_numeric));
				}

				else
				{
				add_this_col:
					add_chosen_column<decltype(data), decltype(workspace),
									  ldouble_safe>(workspace, data, model_params,
													col_is_taken, col_is_taken_s);
					if (++workspace.ntaken >= model_params.ndim)
						break;
				}
			}

			if (!workspace.ntaken && !workspace.ntaken_best)
				goto terminal_statistics;
			else if (!workspace.ntaken)
				break;

			/* evaluate gain if necessary */
			if (workspace.criterion != NoCrit)
			{
				if (workspace.weights_arr.empty() && workspace.weights_map.empty())
					workspace.this_gain = eval_guided_crit<ldouble_safe>(
						workspace.comb_val.data(),
						workspace.end - workspace.st + 1, workspace.criterion,
						model_params.min_gain, workspace.ntry == 1,
						workspace.buffer_dbl.data(), workspace.this_split_point,
						workspace.xmin, workspace.xmax,
						workspace.ix_arr.data() + workspace.st,
						workspace.col_indices.data(),
						workspace.col_sampler.get_remaining_cols(),
						model_params.ncols_per_tree < data.ncols_numeric,
						data.X_row_major.data(), data.ncols_numeric,
						data.Xr.data(), data.Xr_ind.data(),
						data.Xr_indptr.data());
				else if (!workspace.weights_arr.empty())
					workspace.this_gain = eval_guided_crit_weighted<ldouble_safe>(
						workspace.comb_val.data(),
						workspace.end - workspace.st + 1, workspace.criterion,
						model_params.min_gain, workspace.ntry == 1,
						workspace.buffer_dbl.data(), workspace.this_split_point,
						workspace.xmin, workspace.xmax,
						workspace.sample_weights.data(),
						workspace.buffer_szt.data(),
						workspace.ix_arr.data() + workspace.st,
						workspace.col_indices.data(),
						workspace.col_sampler.get_remaining_cols(),
						model_params.ncols_per_tree < data.ncols_numeric,
						data.X_row_major.data(), data.ncols_numeric,
						data.Xr.data(), data.Xr_ind.data(),
						data.Xr_indptr.data());
				else
					workspace.this_gain = eval_guided_crit_weighted<ldouble_safe>(
						workspace.comb_val.data(),
						workspace.end - workspace.st + 1, workspace.criterion,
						model_params.min_gain, workspace.ntry == 1,
						workspace.buffer_dbl.data(), workspace.this_split_point,
						workspace.xmin, workspace.xmax,
						workspace.sample_weights.data(),
						workspace.buffer_szt.data(),
						workspace.ix_arr.data() + workspace.st,
						workspace.col_indices.data(),
						workspace.col_sampler.get_remaining_cols(),
						model_params.ncols_per_tree < data.ncols_numeric,
						data.X_row_major.data(), data.ncols_numeric,
						data.Xr.data(), data.Xr_ind.data(),
						data.Xr_indptr.data());
			}

			/* pass to the output object */
			if (workspace.ntry == 1 || workspace.this_gain > hplanes.back().score)
			{
				/* these should be shrunk later according to what ends up used */
				hplanes.back().score = workspace.this_gain;
				workspace.ntaken_best = workspace.ntaken;
				if (workspace.criterion != NoCrit)
				{
					hplanes.back().split_point = workspace.this_split_point;
					if (model_params.penalize_range)
					{
						hplanes.back().range_low = workspace.xmin - workspace.xmax + hplanes.back().split_point;
						hplanes.back().range_high = workspace.xmax - workspace.xmin + hplanes.back().split_point;
					}
				}
				hplanes.back().col_num.assign(
					workspace.col_take.begin(),
					workspace.col_take.begin() + workspace.ntaken);
				hplanes.back().col_type.assign(
					workspace.col_take_type.begin(),
					workspace.col_take_type.begin() + workspace.ntaken);
				if (data.ncols_numeric)
				{
					hplanes.back().coef.assign(
						workspace.ext_coef.begin(),
						workspace.ext_coef.begin() + workspace.ntaken);
					hplanes.back().mean.assign(
						workspace.ext_mean.begin(),
						workspace.ext_mean.begin() + workspace.ntaken);
				}

				if (model_params.missing_action != Fail)
					hplanes.back().fill_val.assign(
						workspace.ext_fill_val.begin(),
						workspace.ext_fill_val.begin() + workspace.ntaken);

				if (model_params.scoring_metric != Depth && !is_boxed_metric(model_params.scoring_metric))
				{
					workspace.density_calculator.save_range(workspace.xmin,
															workspace.xmax);
				}

				if (data.ncols_categ)
				{
					hplanes.back().fill_new.assign(
						workspace.ext_fill_new.begin(),
						workspace.ext_fill_new.begin() + workspace.ntaken);
					switch (model_params.cat_split_type)
					{
					case SingleCateg:
					{
						hplanes.back().chosen_cat.assign(
							workspace.chosen_cat.begin(),
							workspace.chosen_cat.begin() + workspace.ntaken);
						break;
					}

					case SubSet:
					{
						if (hplanes.back().cat_coef.size() < workspace.ntaken)
							hplanes.back().cat_coef.assign(
								workspace.ext_cat_coef.begin(),
								workspace.ext_cat_coef.begin() + workspace.ntaken);
						else
							for (size_t col = 0; col < workspace.ntaken_best;
								 col++)
								std::copy(workspace.ext_cat_coef[col].begin(),
										  workspace.ext_cat_coef[col].end(),
										  hplanes.back().cat_coef[col].begin());
						break;
					}
					}
				}
			}
		}

		col_is_taken.clear();
		col_is_taken.shrink_to_fit();
		col_is_taken_s.clear();

		/* if the best split is not good enough, don't split any further */
		if (workspace.criterion != NoCrit && hplanes.back().score <= 0)
			goto terminal_statistics;

		/* now need to reproduce the same split from before */
		if (workspace.criterion != NoCrit)
		{
			std::fill(
				workspace.comb_val.begin(),
				workspace.comb_val.begin() + (workspace.end - workspace.st + 1),
				(double)0);
			for (size_t col = 0; col < workspace.ntaken_best; col++)
			{
				switch (hplanes.back().col_type[col])
				{
				case Numeric:
				{
					if (data.Xc_indptr == NULL)
					{
						add_linear_comb(
							workspace.ix_arr.data(),
							workspace.st,
							workspace.end,
							workspace.comb_val.data(),
							data.numeric_data + hplanes.back().col_num[col] * data.nrows,
							hplanes.back().coef[col],
							(double)0,
							hplanes.back().mean[col],
							hplanes.back().fill_val.size() ? hplanes.back().fill_val[col] : workspace.this_split_point, /* second case is not used */
							model_params.missing_action, NULL, NULL, false);
					}

					else
					{
						add_linear_comb(
							workspace.ix_arr.data(),
							workspace.st,
							workspace.end,
							hplanes.back().col_num[col],
							workspace.comb_val.data(),
							data.Xc,
							data.Xc_ind,
							data.Xc_indptr,
							hplanes.back().coef[col],
							(double)0,
							hplanes.back().mean[col],
							hplanes.back().fill_val.size() ? hplanes.back().fill_val[col] : workspace.this_split_point, /* second case is not used */
							model_params.missing_action, NULL, NULL, false);
					}

					break;
				}

				case Categorical:
				{
					add_linear_comb<ldouble_safe>(
						workspace.ix_arr.data(),
						workspace.st,
						workspace.end,
						workspace.comb_val.data(),
						data.categ_data + hplanes.back().col_num[col] * data.nrows,
						data.ncat[hplanes.back().col_num[col]],
						(model_params.cat_split_type == SubSet) ? hplanes.back().cat_coef[col].data() : NULL,
						(model_params.cat_split_type == SingleCateg) ? hplanes.back().fill_new[col] : (double)0,
						(model_params.cat_split_type == SingleCateg) ? hplanes.back().chosen_cat[col] : 0,
						(hplanes.back().fill_val.size()) ? hplanes.back().fill_val[col] : workspace.this_split_point,		 /* second case is not used */
						(model_params.cat_split_type == SubSet) ? hplanes.back().fill_new[col] : workspace.this_split_point, /* second case is not used */
						NULL, NULL, model_params.new_cat_action,
						model_params.missing_action,
						model_params.cat_split_type, false);
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

		/* get the range */
		if (workspace.criterion == NoCrit)
		{
			workspace.xmin = HUGE_VAL;
			workspace.xmax = -HUGE_VAL;
			for (size_t row = 0; row < (workspace.end - workspace.st + 1); row++)
			{
				workspace.xmin =
					(workspace.xmin > workspace.comb_val[row]) ? workspace.comb_val[row] : workspace.xmin;
				workspace.xmax =
					(workspace.xmax < workspace.comb_val[row]) ? workspace.comb_val[row] : workspace.xmax;
			}
			if (workspace.xmin == workspace.xmax)
				goto terminal_statistics;
			/* in theory, could try again too, this could just be an unlucky case */

			hplanes.back().split_point = sample_random_uniform(
				workspace.xmin, workspace.xmax, workspace.rnd_generator);

			/* determine acceptable range */
			if (model_params.penalize_range)
			{
				hplanes.back().range_low = workspace.xmin - workspace.xmax + hplanes.back().split_point;
				hplanes.back().range_high = workspace.xmax - workspace.xmin + hplanes.back().split_point;
			}
		}

		if (model_params.missing_action == Fail && is_na_or_inf(hplanes.back().split_point))
			throw std::runtime_error(
				"Data has missing values. Try using a different value for 'missing_action'.\n");

		/* divide */
		workspace.split_ix = divide_subset_split(workspace.ix_arr.data(),
												 workspace.comb_val.data(),
												 workspace.st, workspace.end,
												 hplanes.back().split_point);

		/* set as non-terminal */
		hplanes.back().score = -1;

		/* add another round of separation depth for distance */
		if (model_params.calc_dist && curr_depth > 0)
			add_separation_step(workspace, data, (double)(-1));

		/* simplify vectors according to what ends up used */
		if (data.ncols_categ || workspace.ntaken_best < model_params.ndim)
			simplify_hplane(hplanes.back(), workspace, data, model_params);

		shrink_to_fit_hplane(hplanes.back(), false);

		/* if using a custom scoring metric, need to calculate it now */
		if (model_params.scoring_metric != Depth)
		{
			if (workspace.criterion != NoCrit)
				workspace.density_calculator.restore_range(workspace.xmin,
														   workspace.xmax);

			if (model_params.scoring_metric == Density)
			{
				workspace.density_calculator.push_density(
					workspace.xmin, workspace.xmax, hplanes.back().split_point);
			}

			else if (is_boxed_metric(model_params.scoring_metric))
			{
				workspace.density_calculator.push_bdens_ext(hplanes.back(),
															model_params);
			}

			else
			{
				double pct_tree_left;
				if (workspace.weights_arr.empty() && workspace.weights_map.empty())
				{
					pct_tree_left = (ldouble_safe)(workspace.split_ix - workspace.st) / (ldouble_safe)(workspace.end - workspace.st + 1);
				}

				else
				{
					ldouble_safe wtot = 0;
					ldouble_safe wleft = 0;
					if (!workspace.weights_arr.empty())
					{
						for (size_t ix = workspace.st; ix < workspace.split_ix;
							 ix++)
							wtot += workspace.weights_arr[workspace.ix_arr[ix]];
						wleft = wtot;
						for (size_t ix = workspace.split_ix; ix <= workspace.end;
							 ix++)
							wtot += workspace.weights_arr[workspace.ix_arr[ix]];
					}

					else
					{
						for (size_t ix = workspace.st; ix < workspace.split_ix;
							 ix++)
							wtot += workspace.weights_map[workspace.ix_arr[ix]];
						wleft = wtot;
						for (size_t ix = workspace.split_ix; ix <= workspace.end;
							 ix++)
							wtot += workspace.weights_map[workspace.ix_arr[ix]];
					}

					pct_tree_left = wleft / wtot;
				}

				workspace.density_calculator.push_adj(
					workspace.xmin, workspace.xmax, hplanes.back().split_point,
					pct_tree_left, model_params.scoring_metric);
			}
		}

		/* now split */

		/* back-up where it was */
		recursion_state = std::unique_ptr<RecursionState>(
			new RecursionState(workspace, true));

		/* follow left branch */
		hplanes[hplane_from].hplane_left = hplanes.size();
		hplanes.emplace_back();
		if (impute_nodes != NULL)
			impute_nodes->emplace_back(hplane_from);
		workspace.end = workspace.split_ix - 1;
		split_hplane_recursive<InputData, WorkerMemory, ldouble_safe>(
			hplanes, workspace, data, model_params, impute_nodes, curr_depth + 1);

		/* follow right branch */
		hplanes[hplane_from].hplane_right = hplanes.size();
		recursion_state->restore_state(workspace);
 		workspace.st = workspace.split_ix;	
		//this emplace_back is the problem 
		//
		//hplanes.emplace_back();
		// instead :
		//hplanes.emplace_back(hplanes[hplane_from]);

		if (impute_nodes != NULL)
			impute_nodes->emplace_back(hplane_from);
		if (is_boxed_metric(model_params.scoring_metric))
		{
			workspace.density_calculator.pop_bdens_ext();
		}
		else if (model_params.scoring_metric != Depth)
		{
			workspace.density_calculator.pop();
		}

		if (is_boxed_metric(model_params.scoring_metric))
		{
			workspace.density_calculator.pop_bdens_ext_right();
		}
		else if (model_params.scoring_metric != Depth)
		{
			workspace.density_calculator.pop_right();
		}

		return;

	terminal_statistics:
	{
		hplanes.back().hplane_left = 0;

		bool has_weights = !workspace.weights_arr.empty() || !workspace.weights_map.empty();
		if (has_weights)
		{
			if (sum_weight == -HUGE_VAL)
				sum_weight = calculate_sum_weights<ldouble_safe>(
					workspace.ix_arr, workspace.st, workspace.end, curr_depth,
					workspace.weights_arr, workspace.weights_map);
		}

		switch (model_params.scoring_metric)
		{
		case Depth:
		{
			if (!has_weights)
				hplanes.back().score = curr_depth + expected_avg_depth<ldouble_safe>(
														workspace.end - workspace.st + 1);
			else
				hplanes.back().score = curr_depth + expected_avg_depth<ldouble_safe>(sum_weight);
			break;
		}

		case AdjDepth:
		{
			if (!has_weights)
				hplanes.back().score =
					workspace.density_calculator.calc_adj_depth() + expected_avg_depth<ldouble_safe>(
																		workspace.end - workspace.st + 1);
			else
				hplanes.back().score =
					workspace.density_calculator.calc_adj_depth() + expected_avg_depth<ldouble_safe>(sum_weight);
			break;
		}

		case Density:
		{
			if (!has_weights)
				hplanes.back().score =
					workspace.density_calculator.calc_density(
						workspace.end - workspace.st + 1,
						model_params.sample_size);
			else
				hplanes.back().score =
					workspace.density_calculator.calc_density(
						sum_weight, model_params.sample_size);
			break;
		}

		case AdjDensity:
		{
			hplanes.back().score =
				workspace.density_calculator.calc_adj_density();
			break;
		}

		case BoxedRatio:
		{
			hplanes.back().score =
				workspace.density_calculator.calc_bratio_ext();
			break;
		}

		case BoxedDensity:
		{
			if (!has_weights)
				hplanes.back().score =
					workspace.density_calculator.calc_bdens_ext(
						workspace.end - workspace.st + 1,
						model_params.sample_size);
			else
				hplanes.back().score =
					workspace.density_calculator.calc_bdens_ext(
						sum_weight, model_params.sample_size);
			break;
		}

		case BoxedDensity2:
		{
			if (!has_weights)
				hplanes.back().score =
					workspace.density_calculator.calc_bdens_ext(
						workspace.end - workspace.st + 1,
						model_params.sample_size);
			else
				hplanes.back().score =
					workspace.density_calculator.calc_bdens_ext(
						sum_weight, model_params.sample_size);
			break;
		}
		}

		/* don't leave any vector initialized */
		shrink_to_fit_hplane(hplanes.back(), true);

		hplanes.back().remainder =
			(!workspace.weights_arr.empty()) ? sum_weight : ((!workspace.weights_map.empty()) ? sum_weight : ((double)(workspace.end - workspace.st + 1)));

		/* for distance, assume also the elements keep being split */
		if (model_params.calc_dist)
			add_remainder_separation_steps<InputData, WorkerMemory, ldouble_safe>(
				workspace, data, sum_weight);

		/* add this depth right away if requested */
		if (!workspace.row_depths.empty())
			for (size_t row = workspace.st; row <= workspace.end; row++)
				workspace.row_depths[workspace.ix_arr[row]] +=
					hplanes.back().score;

		/* add imputations from node if requested */
		if (model_params.impute_at_fit)
			add_from_impute_node(impute_nodes->back(), workspace, data);
	}
	}
} // namespace

#endif
