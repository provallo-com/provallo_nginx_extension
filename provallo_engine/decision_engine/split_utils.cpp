
#include "utils.h"
#include "dataset.h"
#include "split_utils.hpp"
#include <numeric>
#include <thread>
#include <mutex>
namespace provallo
{

	uint64_t split_method::_instance_counter = 0;
	typedef dataset *datasetptr;

	// Select cut point
	size_t
	continous_base::selectPoint(
		std::vector<std::pair<real_t, size_t>>::iterator begin,
		std::vector<std::pair<real_t, size_t>>::iterator end) const
	{
		return (*min_element(begin, end)).second;
	}

	size_t
	random_split::selectPoint(
		std::vector<std::pair<real_t, size_t>>::iterator begin,
		std::vector<std::pair<real_t, size_t>>::iterator end) const
	{
		int size = end - begin;
		std::random_device dev;
		std::mt19937 gen(dev());

		std::uniform_int_distribution<size_t> uniform(0, size);

		std::vector<std::pair<real_t, size_t>>::iterator it = begin + (uniform(gen) % size);
		return (*it).second;
	}
	static real_t
	entropy_diff(size_t ni, real_t l, real_t r, real_t li, real_t ri)
	{
		return (xlog<2>(r) - xlog<2>(l) + xlog<2>(li) + xlog<2>(l + ni) - xlog<2>(li + ni) - xlog<2>(ri) - xlog<2>(r + ni) + xlog<2>(ri + ni));
	}

	bool
	continous_base::binarySplit(const dataset &data, size_t begin,
								size_t end, const attribute_tag &tag,
								std::pair<size_t, real_t> &cut_pair) const
	{
		// Check for discrete attributes
		/*refactor:*/
		if (data.getattributes().getType(tag) == attribute_type::DISCRETE)
			return false;
		else
		{
			// Initialize auxiliary variables
			size_t tleft(0);			  // Total samples on the left
			size_t tright(end - begin); // Total samples on the right

			// Target tag
			attribute_tag target_tag = data.getattributes().get_target_tag();
			// Target count (i.e. number of classes)
			size_t target_count = data.getattributes().getCount(target_tag);

			// Container of sample of each class (right and left)
			std::vector<size_t> left(target_count, 0);
			std::vector<size_t> right(target_count, 0);

			// Initial position of the iterator on the target attribute
			dataset::sorted_iterator initial_target(data.begin_sorted(tag));

			// Number of consecutive sample of the same class
			size_t consecutive(1);
			// Current class
			size_t current_class((*(initial_target + begin)).discrete());
			if (current_class >= target_count)
			{
				if (discrete_value((*(initial_target + begin)).continous()) < target_count)
					current_class = discrete_value((*(initial_target + begin)).continous());
				else
					return false;
			}

			// Current position
			size_t current_position(begin);

			// Iterate over the samples
			for (dataset::sorted_iterator it(initial_target + begin);
				 it != initial_target + end; ++it)
			{
				// If the class is the same of the previous sample
				if ((*it).discrete() == current_class || discrete_value((*it).continous()) == current_class)
				{
					// Increment the number of consecutive samples
					++consecutive;
				}
				else
				{
					if (current_class >= left.size() || current_class >= right.size())
						return false;
					// Calculate the entropy difference
					real_t diff = entropy_diff(consecutive, tleft, tright,
											  left[current_class], right[current_class]);
					// If the difference is greater than the current one
					if (diff > cut_pair.second)
					{
						// Update the current entropy difference
						cut_pair.second = diff;
						// Update the cut point
						cut_pair.first = current_position;
					}
					// Update the current class
					current_class = (*it).discrete();
					if (current_class >= target_count)
					{
						if (discrete_value((*it).continous()) < target_count)
							current_class = discrete_value((*it).continous());
						else
							continue;
					}

					// Update the current position
					current_position = it.position();
					// Update the number of samples on the left
					tleft += consecutive;
					// Update the number of samples on the right
					tright -= consecutive;
					// Update the number of samples of the current class on the left
					left[current_class] += consecutive;
					// Update the number of samples of the current class on the right
					right[current_class] -= consecutive;
					// Reset the number of consecutive samples
					consecutive = 1;
				}
			} // for
			// Calculate the entropy difference
			if (current_class >= target_count)
				return false;

			real_t diff = entropy_diff(consecutive, tleft, tright,
									  left[current_class], right[current_class]);
			// If the difference is greater than the current one
			if (diff > cut_pair.second)
			{
				// Update the current entropy difference
				cut_pair.second = diff;
				// Update the cut point
				cut_pair.first = current_position;
			}
			// If the entropy difference is zero
			if (cut_pair.second == 0)
				return false;

			// Return true

			return true;
		} // else
		// Return false
		return false;
	}	  // binarySplit

	bool
	random_split::binarySplit(const dataset &data, size_t begin,
							  size_t end, const attribute_tag &tag,
							  std::pair<size_t, real_t> &cut_pair) const
	{

		// Check for discrete attributes
		if (data.getattributes().getType(tag) == attribute_type::DISCRETE)
			return false;
		else
		{
			// Initialize auxiliary variables
			size_t tleft(0);			  // Total samples on the left
			size_t tright(end - begin); // Total samples on the right

			// Target tag
			attribute_tag target_tag = data.getattributes().get_target_tag();
			// Target count (i.e. number of classes)
			size_t target_count = data.getattributes().getCount(target_tag);

			// Container of sample of each class (right and left)
			std::vector<size_t> left(target_count, 0);
			std::vector<size_t> right(target_count, 0);

			// Initial position of the iterator on the target attribute
			dataset::sorted_iterator initial_target(data.begin_sorted(tag));

			// Number of consecutive sample of the same class
			size_t consecutive(1);
			// Current class
			size_t current_class((*(initial_target + begin)).discrete());
			if (current_class > target_count)
				current_class = discrete_value((*(initial_target + begin)).continous());

			// Current position
			size_t current_position(begin);

			// Iterate over the samples
			for (dataset::sorted_iterator it(initial_target + begin);
				 it != initial_target + end; ++it)
			{
				// If the class is the same of the previous sample
				if (((*it).discrete() == current_class)   ) 
				{
					// Increment the number of consecutive samples
					++consecutive;
				}
				else
				{
					// Calculate the entropy difference
					real_t diff = entropy_diff(consecutive, tleft, tright,
											  left[current_class], right[current_class]);
					// If the difference is greater than the current one
					if (diff > cut_pair.second)
					{
						// Update the current entropy difference
						cut_pair.second = diff;
						// Update the cut point
						cut_pair.first = current_position;
					}
					// Update the current class
					current_class = (*it).discrete();
					if (current_class >= target_count)
					{
						if (discrete_value((*it).continous()) < target_count)
							current_class = discrete_value((*it).continous());
						else
							current_class = 0;
					}
					// Update the current position
					current_position = it.position();
					// Update the number of samples on the left
					tleft += consecutive;
					// Update the number of samples on the right
					tright -= consecutive;
					// Update the number of samples of the current class on the left
					left[current_class] += consecutive;
					// Update the number of samples of the current class on the right
					right[current_class] -= consecutive;
					// Reset the number of consecutive samples
					consecutive = 1;
				}
			} // for
			// Calculate the entropy difference
			real_t diff = entropy_diff(consecutive, tleft, tright,
									  left[current_class], right[current_class]);
			// If the difference is greater than the current one
			if (diff > cut_pair.second)
			{
				// Update the current entropy difference
				cut_pair.second = diff;
				// Update the cut point
				cut_pair.first = current_position;
			}
			return true;
		} // else
				// Split is not possible (all data is of the same class)
		return false;

	}	  // binarySplit

	/*refactor:
			// Calculate initial entropy
			real_t entropy (0.0);
			for (size_t i = 0; i < right.size (); ++i)
			{
				real_t prob = right[i] / (real_t) data.size ();
				if (prob != 0.0)
				entropy += -prob * log<2> (prob);
			}

			// Number of consecutive sample of the same class
			size_t consecutive (1);
			// Current class
			size_t current_class ((initial_target + begin)->discrete ());
			// Current position
			size_t current_position (begin);

			// Iterate over the samples
			for (dataset::sorted_iterator it (initial_target + begin);
			it != initial_target + end; ++it)
			{
				// If the class is the same of the previous sample
				if (it->discrete () == current_class)
				{
					// Increment the number of consecutive samples
					++consecutive;
				}
				else
				{
					// Calculate the entropy difference
					real_t diff = entropy_diff (consecutive, tleft, tright,
					left[current_class], right[current_class]);
					// If the difference is greater than the current one
					if (diff > cut_pair
		}
		#/

		// Initialize auxiliary variables
		size_t tleft (0);            // Total samples on the left
		size_t tright (end - begin); // Total samples on the right

		// Target tag
		attribute_tag target_tag = data.getattributes ().get_target_tag ();
		// Target count (i.e. number of classes)
		size_t target_count = data.getattributes ().getCount (target_tag);

		// Container of sample of each class (right and left)
		std::vector<size_t> left (target_count, 0);
		std::vector<size_t> right (target_count,0);

		// Initial position of the iterator on the target attribute
		dataset::sorted_iterator initial_target (data.begin_sorted (tag));

		// Initialize right array
		for (dataset::sorted_iterator it (initial_target + begin);
		it != initial_target + end; ++it)
		  {
				right[(it.begin () + target_tag)->discrete ()]++;
		  }

		// Calculate initial entropy
		real_t entropy (0.0);
		for (size_t i = 0; i < right.size (); ++i)
		  {
		real_t prob = right[i] / (real_t) data.size ();
		if (prob != 0.0)
		  entropy += -prob * log<2> (prob);
		  }

		// Number of consecutive sample of the same class
		size_t nssc (0);
		// Split entropy
		std::vector<std::pair<real_t, size_t> > split_entropy;

		// Loop over the sample and check potential cut points (i.e. boundary points)
		for (dataset::sorted_iterator it = initial_target + begin;
		it != initial_target + end - 1; ++it)
		  {
			attribute_iterator at = it.begin();
			// Get class of this and next sample class
			size_t this_class = (at + target_tag)->discrete ();
			size_t next_class = ((it + 1).begin () + target_tag)->discrete ();

			// Increment counter
			++nssc;

		// Position of the iterator
		size_t offset (it - initial_target);

		// Check if this is a boundary point (i.e. potential cut point)
		if (this_class != next_class)
		  {
			// We should check the values of the ordered attribute
			cont_value left_attr =
			((initial_target + offset).begin () + tag)->continous ();
			cont_value right_attr = ((initial_target + offset + 1).begin ()
			+ tag)->continous ();
			// Check if the attribute values are equal
			if (left_attr != right_attr)
			  {
					// Update counters
					tleft += nssc;
					tright -= nssc;
					left[this_class] += nssc;
					right[this_class] -= nssc;
					// Compute entropy of the split
					entropy += entropy_diff (nssc, tleft, tright, left[this_class],
								right[this_class])
						/ (real_t) data.size ();
					// Reset counter
					nssc = 0;
					// Push data of this boundary point
					split_entropy.push_back (std::make_pair (entropy, offset));
			  }
		  }
		  }

		if (split_entropy.size () > 0)
		  {
				// Get minimum entropy
				size_t cut_point = selectPoint (split_entropy.begin (),
								split_entropy.end ());
				// Get attribute value at the cut point
				real_t left_value =
					((initial_target + cut_point).begin () + tag)->continous ();
				real_t right_value =
					((initial_target + cut_point).begin () + tag)->continous ();
				real_t cut_value = (right_value + left_value) / 2;
				// Push cut point
				cut_pair = std::make_pair (cut_point, cut_value);
				// Can split interval
				return true;
		  }

		// Split is not possible (all data is of the same class)
		return false;
	  }
	*/
	void
	continous_base::splitInterval(const dataset &data, size_t begin,
								  size_t end, const attribute_tag &tag,
								  std::vector<real_t> &interval) const
	{
		// Check if we can split the interval

		if (begin == end)
			return;
		std::pair<size_t, real_t> cut_pair(0, 0.0);
		cut_pair.first = begin;
		cut_pair.second = (*(data.begin_sorted(tag) + begin)).continous();

		if (this->checkSplitting(data, begin, end, 0, tag))
		{

			//
			size_t cut_point = cut_pair.first;
			real_t cut_value = cut_pair.second;
			// Split left interval
			if (begin != end)
			{
				splitInterval(data, begin, end, tag, interval);
				// Push cut value
				interval.push_back(cut_value);
				// Split right interval
				splitInterval(data, cut_point + 1, end, tag, interval);
			}
			else
			{
				interval.push_back(cut_value);
			}
		}
	}
	void
	continous_base::split(const dataset &data, const attribute_tag &tag,
						  std::vector<real_t> &interval) const
	{
		// Set first value of the interval
		interval.push_back(-std::numeric_limits<float>::infinity());
		// Recursively split intervals
		splitInterval(data, 0, data.size(), tag, interval);
		// Set last value of the interval
		interval.push_back(std::numeric_limits<float>::infinity());
	}

	bool
	mdlp_split::checkSplitting(const dataset &data, size_t begin, size_t end,
							   size_t cut_point,
							   const attribute_tag &tag) const
	{
		// Target tag
		attribute_tag target_tag = data.getattributes().get_target_tag();
		// Target count (i.e. number of classes)
		size_t target_count = data.getattributes().getCount(target_tag);

		// Container of sample of each class (right and left)
		std::vector<size_t> left(target_count);
		std::vector<size_t> right(target_count);

		// Initial position of the iterator on the target attribute
		dataset::sorted_iterator initial_target = data.begin_sorted(tag);

		// Initialize right array
		for (dataset::sorted_iterator it(initial_target + begin);
			 it != initial_target + end; ++it)
		{
			size_t j(it - initial_target);
			size_t class_attr((it.begin() + target_tag)->discrete());
			// Accumulate class value
			if (j < cut_point + 1)
				left[class_attr]++;
			else
				right[class_attr]++;
		}
		// Size of each interval
		size_t size_left(cut_point + 1 - begin);
		size_t size_right(end - (cut_point + 1));

		// Calculate entropy
		real_t entropy_right(0.0);
		real_t entropy_left(0.0);
		real_t entropy(0.0);
		// Count of different classes on each interval
		real_t kl(target_count);
		real_t kr(target_count);
		for (size_t i = 0; i < right.size(); ++i)
		{
			// Calculate entropy of the sets defined by the partition
			real_t prob = right[i] / (real_t)size_right;
			if (prob != 0.0)
				entropy_right += -prob * log<2>(prob);
			else
				kr--;
			prob = left[i] / (real_t)size_left;
			if (prob != 0.0)
				entropy_left += -prob * log<2>(prob);
			else
				kl--;
			// Calculate total entropy
			prob = (right[i] + left[i]) / (real_t)data.size();
			if (prob != 0.0)
				entropy += -prob * log<2>(prob);
		}
		// Check criteria
		real_t delta = log<2>(pow(3, (real_t)target_count) - 2) - ((real_t)target_count * entropy - kl * entropy_left - kr * entropy_right);
		real_t gain = entropy - ((real_t)size_left / data.size()) * entropy_left - ((real_t)size_right / data.size()) * entropy_right;
		real_t test = log<2>(data.size() - 1) / (real_t)data.size() + delta / (real_t)data.size();

		// Accept the split
		if (gain > test)
			return true;
		// Reject the splitting
		return false;
	}

	typedef cont1d<binary_split> continous_binary_split;
	typedef cont1d<multi_interval_split> continous_multi_interval;
	typedef cont1d<mdlp_split> continous_mdlp;
	typedef cont1d<random_split> continous_random;

	split_method *
	split_method_factory::createMethod(const split_method &deserial)
	{
		// Method
		split_method *method(0);
		// Get type
		split_type type = deserial.get_type();
		// Check attribute
		if (type == DISC)
		{
			// Discrete attribute.
			// branch by value
			method = new discrete_split();
		}
		else if (type == CONE_BINARY)
		{
			// Return continuous split method
			method = new continous_binary_split;
		}
		else if (type == CONE_MULTI)
		{
			// Return continuous split method
			method = new continous_multi_interval;
		}
		else if (type == CONE_MDLP)
		{
			// Return continuous split method
			method = new continous_mdlp;
		}
		else if (type == CONE_RANDOM)
		{
			// Return continuous split method
			method = new continous_random;
		}
		else

			throw(std::runtime_error("Invalid method type: " + std::to_string(type)));

		// Get data from buffer
		// method->deserialize(&deserial);

		// Return method
		return method;
	}

	split_method *
	split_method_factory::createMethod(const std::random_device &random_,
									   split_type type, const dataset &data_set,
									   const attribute_tag &tag,
									   const attribute_tag &factory_tag,split_method_factory &factory)
	{

		static std::recursive_mutex _mute;
		
		std::lock_guard<std::recursive_mutex> guard(_mute);
		
		std::pair<size_t, size_t> lookup = std::make_pair(factory_tag, tag);
		
		split_method *ret = nullptr;
		//hardcode caching mechanism:
		bool use_cache = false;
		// Check attribute
		if (type == DISC)
		{
 			// do we really need to create? or just return from cache...
			if( use_cache) {

			if (factory._split_cache.size() && factory._split_cache.find(lookup) != factory._split_cache.end())
			{

				std::cout << "[+] returning cached [" << std::to_string(factory_tag) << "," << std::to_string(tag) << " ]" << std::endl;
				;

				ret =factory._split_cache[lookup];
			}
			else
			{

				std::cout << "[+] creating discrete split [" << std::to_string(factory_tag) << "," << std::to_string(tag) << " ]" << std::endl;
				;

				ret = new discrete_split(factory_tag, tag, std::ref(data_set));
				factory._split_cache.insert(std::make_pair(lookup, ret));
			}
			}
			else {
				ret = new discrete_split(factory_tag, tag, std::ref(data_set));
			}
		}
		else if (type == CONE_BINARY)
		{

			if( use_cache) {
			if (factory._split_cache.size() && factory._split_cache.find(lookup) != factory._split_cache.end())
			{
				std::cout << "[+] returning cached [" << std::to_string(factory_tag) << "," << std::to_string(tag) << " ]" << std::endl;			
				ret =factory._split_cache[lookup];
			}
			else
			{
				std::cout << "[+] creating binary continous split [" << std::to_string(factory_tag) << "," << std::to_string(tag) << " ]" << std::endl;

				// Return continuous split method
				ret = new continous_binary_split(factory_tag, tag, std::ref(data_set));
				factory._split_cache.insert(std::make_pair(lookup, ret));
			}
			}
			else {
				ret = new continous_binary_split(factory_tag, tag, std::ref(data_set));
			}
		}
		else if (type == CONE_MULTI)
		{
			if(use_cache)	{
			if (factory._split_cache.size() &&factory._split_cache.find(lookup) != factory._split_cache.end())
			{
				std::cout << "[+] returning cached [" << std::to_string(factory_tag) << "," << std::to_string(tag) << " ]" << std::endl;
				;

				ret =factory._split_cache[lookup];
			}
			else
			{

				std::cout << "[+] creating interval split [" << std::to_string(factory_tag) << "," << std::to_string(tag) << " ]" << std::endl;

				// Return continuous split method
				ret = new continous_multi_interval(factory_tag, tag, std::ref(data_set));
				factory._split_cache.insert(std::make_pair(lookup, ret));
			}
			}
			else {
				ret = new continous_multi_interval(factory_tag, tag, std::ref(data_set));
			}
		}
		else if (type == CONE_MDLP)
		{
			if(use_cache){
			if (factory._split_cache.size() && factory._split_cache.find(lookup) != factory._split_cache.end())
			{
				std::cout << "[+] returning cached [" << std::to_string(factory_tag) << "," << std::to_string(tag) << " ]" << std::endl;
				;

				ret =factory._split_cache[lookup];
			}
			else
			{

				std::cout << "[+] creating mdlp split [" << std::to_string(factory_tag) << "," << std::to_string(tag) << " ]" << std::endl;
				// Return continuous split method
				ret = new continous_mdlp(factory_tag, tag, std::ref(data_set));
				factory._split_cache.insert(std::make_pair(lookup, ret));
			}
			}
			else {
				ret = new continous_mdlp(factory_tag, tag, std::ref(data_set));
			}
		}
		else if (type == CONE_RANDOM)
		{
			// test random_dev
			
			if(use_cache) {
			if (factory._split_cache.size() && factory._split_cache.find(lookup) != factory._split_cache.end())
			{
				std::cout << "[+] returning cached [" << std::to_string(factory_tag) << "," << std::to_string(tag) << " ]" << std::endl;
				ret =factory._split_cache[lookup];
			}
			else
			{

				std::cout << "[+] creating random split [" << std::to_string(factory_tag) << "," << std::to_string(tag) << " ]" << std::endl;
				// Return continuous split method (random split need the random number engine)

				ret = new continous_random(factory_tag, tag, data_set, random_);
				factory._split_cache.insert(std::make_pair(lookup, ret));
			}
			}
			else {
				ret = new continous_random(factory_tag, tag, data_set, random_);
			}
		}
		else
		{
			// Get attribute information
			const attribute_information &info(data_set.getattributes());
			throw(std::runtime_error(
				std::string("Invalid type for attribute ") + info.getName(tag)));
		}
		return ret;
	}
	

	split_method_factory::split_method_factory(const dataset &data_set,
											   const std::random_device &r) : _split_methods(), _target_method(nullptr), r_dataset(*datasetptr(&data_set)),
																			  override_split_method(false), override_split_type(CONE_RANDOM)
	{
		// Get attribute information
 		// Target tag
		attribute_tag target_tag = data_set.getattributes().get_target_tag();
		// Create target method
		// get number of classes
		size_t num_classes = data_set.getattributes().getTargetClassCount();
		if ( num_classes == 0) throw std::runtime_error("No classes found in dataset");
		// Create context
		
		split_method_factory * context = this;

		// resize vector

		_split_methods.resize(data_set.getattributes().getSize(),nullptr);

		_target_method = split_method_factory::createMethod(r, DISC, data_set,
															target_tag, num_classes, *context);
		// Create split methods
		// Get group of attributes (the group define the effective number of attributes)
		size_t group_size(data_set.getattributes().getGroups().size());

		//set the target method just in case.

		_split_methods[target_tag] = _target_method;
 
		// Initialize the map with split methods
																	   
		for (size_t i = 0; i < group_size; ++i)
		{
			//make sure the groups are not empty
			if (data_set.getattributes().getGroups().getGroup(i).size() == 0) continue; 
			// Create split method

			split_method* ps = 	split_method_factory::createMethod(r, override_split_method ? override_split_type :data_set.getattributes().getGroups().getsplit_type(i),
												   data_set, data_set.getattributes().getGroups().getGroup(i)[0],
												   i,*context);
			// Add split method to vector
 				_split_methods[ i] = ps;
		}
		std::cout << "[+] created [[" << split_method::_instance_counter << "]] splits" << std::endl;
	}
	const split_method *
	split_method_factory::getMethod(const attribute_tag &tag) const
	{	

		if ( split_method::_instance_counter == 0 )
		{
			std::cout << "[+] no splits created" << std::endl;
			return nullptr;
		}
		if(_target_method&& tag == r_dataset.getattributes().get_target_tag() )
			return _target_method;
		else if (tag < _split_methods.size())
			return _split_methods[tag];
		else
		{
			throw(std::runtime_error(
				"Tag number " + std::to_string(tag) + " is not available on factory"));
		}
	}

	split_method_factory::~split_method_factory()
	{
		// Delete split methods
		for (size_t i=0;i<_split_methods.size();++i)
			if(_split_methods[i]!=nullptr)
				{delete _split_methods[i];_split_methods[i]=nullptr;}
		
		
	
		_split_methods.clear();
		_split_cache.clear();
		if(_target_method!=nullptr)
			delete _target_method;
		_target_method = nullptr;
	}

	void
	split_method_factory::serialize(split_method_factory *serial) const
	{
		if (serial == nullptr)
			throw(std::runtime_error("split_method_factory::serialize: null pointer"));

		serial->override_split_method = override_split_method;
		serial->override_split_type = override_split_type;
		serial->r_dataset = r_dataset;
		serial->_split_methods = _split_methods;
		serial->_target_method = _target_method;
	}

	// Get data from buffer
	void
	split_method_factory::deserialize(const split_method_factory *serial)
	{
		if (serial == nullptr)
			throw(std::runtime_error("split_method_factory::deserialize: null pointer"));
		// not implemented
	}

	real_t
	EntropyGain::gain(const dataset &data, const split_method &selector)
	{
		// Target tag
		attribute_tag target_tag = data.getattributes().get_target_tag();
		// Number of different values of the attribute
		size_t attr_count = selector.size();
		if (attr_count <= 2)
			return 0.0;
		// Number of different outcomes
		size_t target_count = data.getattributes().getCount(target_tag);

		std::vector<std::vector<size_t>> freqs(
			target_count+1,std::vector<size_t>(attr_count+1, 0));
		
		// Number of samples where the attribute takes some value
		std::vector<size_t> count(attr_count, 0);
		// Target attribute counter
		std::vector<size_t> target_probs(target_count, 0);

		// Loop over the data set
		for (size_t i = 0; i < data.size(); ++i)
		{
			// sanity check:
			if (data.begin(i) != data.end(i))
			{
				// Get attribute "branch"
				size_t attr_value = selector.getBranch(data.begin(i));
				// Get target value
				discrete_value target_value =
					data.getattribute(i, target_tag).discrete();
			

				freqs[target_value%freqs.size()][attr_value%attr_count]++;
				// Accumulate target occurrence
				target_probs[target_value%freqs.size()]++;
				// Accumulate attribute count
				count[attr_value%attr_count]++;
			}
		}

		// Entropy
		real_t entropy = 0.0;
		// Total count
		size_t total_count = std::accumulate(count.begin(), count.end(), 0);

		// Check if the data set contain at least one known value
		if (total_count == 0)
			return 0.0; // No gain

		// Subset entropies (for different values of the attribute)
		std::vector<real_t> attr_entropy(attr_count, 0.0);
		for (size_t i = 0; i < target_probs.size(); ++i)
		{
			real_t prob = (real_t)target_probs[i] / (real_t)total_count;
			if (prob != 0.0)
				entropy += -prob * log<2>(prob);
			for (size_t j = 0; j < attr_entropy.size(); ++j)
			{
				if (count[j] != 0.0)
				{
					real_t attr_prob = (real_t)freqs[i][j] / (real_t)count[j];
					attr_entropy[j] -= xlog<2>(attr_prob);
				}
			}
		}

		// Gain
		real_t gain = entropy;
		for (size_t i = 0; i < count.size(); ++i)
			gain -= ((real_t)count[i] / (real_t)data.size()) * attr_entropy[i];

		// Check for NAN
		assert(gain == gain);

		// Return entropy
		return gain;
	}

	real_t
	GainRatio::gain(const dataset &data, const split_method &selector)
	{
		// Target tag
		attribute_tag target_tag = data.getattributes().get_target_tag();
		// Number of different values of the attribute
				
		
		size_t attr_count = selector.size();
		if (attr_count == 1)
			return 0.0;
		// Number of different outcomes
		size_t target_count = data.getattributes().getCount(target_tag);
		// Proportion of instances (for each attribute value) in the data set that take
		// the a value of the target
		std::vector<std::vector<size_t>> freqs(
			target_count, std::vector<size_t>(attr_count, 0));
		// Number of samples where the attribute takes some value
		std::vector<size_t> count(attr_count, 0);
		// Target attribute counter
		std::vector<size_t> target_probs(target_count, 0);

		// Loop over the data set
		for (size_t i = 0; i < data.size(); ++i)
		{

			// sanity check:
			if (data.begin(i) != data.end(i))
			{
				// Get attribute "branch"

				// Get attribute "branch"
				size_t attr_value = selector.getBranch(data.begin(i));
				// Get target value
				discrete_value target_value =
					data.getattribute(i, target_tag).discrete();
				// Accumulate value for this attribute instance
				freqs[target_value%target_count][attr_value%attr_count]++;
				// Accumulate target occurrence
				target_probs[target_value%target_count]++;
				// Accumulate attribute count
				count[attr_value%attr_count]++;
			}
		}
		// Entropy
		real_t entropy = 0.0;
		// Total count
		size_t total_count = std::accumulate(count.begin(), count.end(), 0);

		// Check if the data set contain at least one known value
		if (total_count == 0)
			return 0.0; // No gain

		// Subset entropies (for different values of the attribute)
		std::vector<real_t> attr_entropy(attr_count, 0.0);
		for (size_t i = 0; i < target_probs.size(); ++i)
		{
			real_t prob = (real_t)target_probs[i] / (real_t)total_count;
			if (prob != 0.0)
				entropy += -prob * log<2>(prob);
			for (size_t j = 0; j < attr_entropy.size(); ++j)
			{
				if (count[j] != 0.0)
				{
					real_t attr_prob = (real_t)freqs[i][j] / (real_t)count[j];
					attr_entropy[j] -= xlog<2>(attr_prob);
				}
			}
		}

		// Gain
		real_t gain = entropy;
		// Split information
		real_t split = 0.0;
		for (size_t i = 0; i < count.size(); ++i)
		{
			// Probability
			real_t pi = (real_t)count[i] / (real_t)data.size();
			// Accumulate gain
			gain -= pi * attr_entropy[i];
			// Split information
			split -= xlog<2>(pi);
		}
		// Add unknown values, if any
		if (data.size() - total_count > 0)
			split -= (data.size() - total_count) * log<2>(data.size() - total_count);

		// Gain ratio
		real_t gain_ratio = gain / split;

		// If gain and split are zero, set gain_ratio to zero
		if (gain == 0 && split == 0)
			gain_ratio = 0.0;

		// Check for NAN
		assert(gain_ratio == gain_ratio);

		// Return gain
		return gain_ratio;
	}

	real_t
	ChiSquare::gain(const dataset &data, const split_method &selector)
	{
		// Target tag
		attribute_tag target_tag = data.getattributes().get_target_tag();
		// Number of different values of the attribute
		size_t attr_count = selector.size();
		if (attr_count == 1)
			return 0.0;
		// Number of different outcomes
		size_t target_count = data.getattributes().getCount(target_tag);

		// Proportion of instances (for each attribute value) in the data set that take
		// the a value of the target
		std::vector<std::vector<size_t>> freqs(
			target_count, std::vector<size_t>(attr_count, 0));
		// Total in rows
		std::vector<size_t> row_total(target_count);
		// Total in columns
		std::vector<size_t> column_total(attr_count);
		// Total samples
		size_t total(0);

		// Loop over the data set
		for (size_t i = 0; i < data.size(); ++i)
		{
			// Get attribute "branch"
			size_t attr_value = selector.getBranch(data.begin(i));
			// Get target value
			discrete_value target_value =
				data.getattribute(i, target_tag).discrete();
			// Accumulate value for this attribute instance
			if(attr_value < freqs[0].size() && target_value < freqs.size())	
				{
					freqs[target_value%freqs.size()][attr_value]++;
			// Compute totals
					row_total[target_value%freqs.size()]++;
					column_total[attr_value]++;
					total++;
				}
		}

		// Compute CHI square
		real_t chi(0.0);

		// Loop over target values
		for (size_t i = 0; i < target_count; ++i)
		{
			for (size_t j = 0; j < attr_count; ++j)
			{
				// Numerator
				real_t num = row_total[i] * column_total[j];
				if (num != 0.0 && total != 0.0)
				{
					// Compute expected value
					real_t expected = num / total;
					// Sum contribution to CHI square
					chi += (freqs[i][j] - expected) * (freqs[i][j] - expected) / expected;
				}
			}
		}

		// Check for NAN
		assert(chi == chi);

		// Return CHI square
		return chi;
	}

	size_t
	divide_subset_split(size_t ix_arr[], double x[], size_t st, size_t end,
						double split_point) noexcept
	{
		size_t temp;
		size_t st_orig = st;
		for (size_t row = st_orig; row <= end; row++)
		{
			if (x[row - st_orig] <= split_point)
			{
				temp = ix_arr[st];
				ix_arr[st] = ix_arr[row];
				ix_arr[row] = temp;
				st++;
			}
		}
		return st;
	}

	/* For categorical columns split by subset */
	void
	divide_subset_split(size_t *ix_arr, int x[], size_t st, size_t end,
						signed char split_categ[], MissingAction missing_action,
						size_t &st_NA, size_t &end_NA, size_t &split_ix) noexcept
	{
		size_t temp;

		/* if NAs are not to be bothered with, just need to do a single pass */
		if (missing_action == Fail)
		{
			/* move to the left if it's l.e. than the split point */
			for (size_t row = st; row <= end; row++)
			{
				if (split_categ[x[ix_arr[row]]] == 1)
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
				if (x[ix_arr[row]] >= 0 && split_categ[x[ix_arr[row]]] == 1)
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
				if (x[ix_arr[row]] < 0)
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

	/* For categorical columns split by subset, used at prediction time (with similarity) */
	void
	divide_subset_split(size_t *ix_arr, int x[], size_t st, size_t end,
						signed char split_categ[], int ncat,
						MissingAction missing_action,
						NewCategAction new_cat_action, bool move_new_to_left,
						size_t &st_NA, size_t &end_NA, size_t &split_ix) noexcept
	{
		size_t temp;
		int cval;

		/* if NAs are not to be bothered with, just need to do a single pass */
		if (missing_action == Fail && new_cat_action != Weighted)
		{
			/* in this case, will need to fill 'split_ix', otherwise need to fill 'st_NA' and 'end_NA' */
			if (new_cat_action == Smallest && move_new_to_left)
			{
				for (size_t row = st; row <= end; row++)
				{
					cval = x[ix_arr[row]];
					if (cval >= ncat || split_categ[cval] == 1 || split_categ[cval] == (-1))
					{
						temp = ix_arr[st];
						ix_arr[st] = ix_arr[row];
						ix_arr[row] = temp;
						st++;
					}
				}
			}

			else if (new_cat_action == Random)
			{
				for (size_t row = st; row <= end; row++)
				{
					cval = x[ix_arr[row]];
					cval = (cval >= ncat) ? (cval % ncat) : cval;
					if (split_categ[cval] == 1)
					{
						temp = ix_arr[st];
						ix_arr[st] = ix_arr[row];
						ix_arr[row] = temp;
						st++;
					}
				}
			}

			else
			{
				for (size_t row = st; row <= end; row++)
				{
					cval = x[ix_arr[row]];
					if (cval < ncat && split_categ[cval] == 1)
					{
						temp = ix_arr[st];
						ix_arr[st] = ix_arr[row];
						ix_arr[row] = temp;
						st++;
					}
				}
			}

			split_ix = st;
		}

		/* if there are new categories, and their direction was decided at random,
		 can just reuse what was randomly decided for previous columns by taking
		 a remainder w.r.t. the number of previous columns. Note however that this
		 will not be an unbiased decision if the model used a gain criterion. */
		else if (new_cat_action == Random)
		{
			if (missing_action == Impute && !move_new_to_left)
			{
				for (size_t row = st; row <= end; row++)
				{
					cval = x[ix_arr[row]];
					cval = (cval >= ncat) ? (cval % ncat) : cval;
					if (cval < 0 || split_categ[cval] == 1)
					{
						temp = ix_arr[st];
						ix_arr[st] = ix_arr[row];
						ix_arr[row] = temp;
						st++;
					}
				}
			}

			else
			{
				for (size_t row = st; row <= end; row++)
				{
					cval = x[ix_arr[row]];
					cval = (cval >= ncat) ? (cval % ncat) : cval;
					if (cval >= 0 && split_categ[cval] == 1)
					{
						temp = ix_arr[st];
						ix_arr[st] = ix_arr[row];
						ix_arr[row] = temp;
						st++;
					}
				}
			}
			st_NA = st;

			if (!(missing_action == Impute && !move_new_to_left))
			{
				for (size_t row = st; row <= end; row++)
				{
					if (unlikely(x[ix_arr[row]] < 0))
					{
						temp = ix_arr[st];
						ix_arr[st] = ix_arr[row];
						ix_arr[row] = temp;
						st++;
					}
				}
			}
			end_NA = st;
		}

		/* otherwise, first put to the left all l.e. and not NA, then all NAs to the end of the left */
		else
		{
			/* Note: if having 'new_cat_action'='Smallest' and 'missing_action'='Impute', missing values
			 and new categories will necessarily go into different branches, thus it's possible to do
			 all the movements in one pass if certain conditions match. */

			if (new_cat_action == Smallest && move_new_to_left)
			{
				for (size_t row = st; row <= end; row++)
				{
					cval = x[ix_arr[row]];
					if (cval >= 0 && (cval >= ncat || split_categ[cval] == 1 || split_categ[cval] == (-1)))
					{
						temp = ix_arr[st];
						ix_arr[st] = ix_arr[row];
						ix_arr[row] = temp;
						st++;
					}
				}
			}

			else if (missing_action == Impute && !move_new_to_left)
			{
				for (size_t row = st; row <= end; row++)
				{
					cval = x[ix_arr[row]];
					if (cval < ncat && (cval < 0 || split_categ[cval] == 1))
					{
						temp = ix_arr[st];
						ix_arr[st] = ix_arr[row];
						ix_arr[row] = temp;
						st++;
					}
				}
			}

			else
			{
				for (size_t row = st; row <= end; row++)
				{
					cval = x[ix_arr[row]];
					if (cval >= 0 && cval < ncat && split_categ[cval] == 1)
					{
						temp = ix_arr[st];
						ix_arr[st] = ix_arr[row];
						ix_arr[row] = temp;
						st++;
					}
				}
			}

			st_NA = st;

			if (new_cat_action == Weighted && missing_action == Divide)
			{
				for (size_t row = st; row <= end; row++)
				{
					cval = x[ix_arr[row]];
					if (cval < 0 || cval >= ncat || split_categ[cval] == (-1))
					{
						temp = ix_arr[st];
						ix_arr[st] = ix_arr[row];
						ix_arr[row] = temp;
						st++;
					}
				}
			}

			else if (new_cat_action == Weighted)
			{
				for (size_t row = st; row <= end; row++)
				{
					cval = x[ix_arr[row]];
					if (cval >= 0 && (cval >= ncat || split_categ[cval] == (-1)))
					{
						temp = ix_arr[st];
						ix_arr[st] = ix_arr[row];
						ix_arr[row] = temp;
						st++;
					}
				}
			}

			else if (missing_action == Divide)
			{
				for (size_t row = st; row <= end; row++)
				{
					if (unlikely(x[ix_arr[row]] < 0))
					{
						temp = ix_arr[st];
						ix_arr[st] = ix_arr[row];
						ix_arr[row] = temp;
						st++;
					}
				}
			}

			end_NA = st;
		}
	}

	/* For categoricals split on a single category */
	void
	divide_subset_split(size_t *ix_arr, int x[], size_t st, size_t end,
						int split_categ, MissingAction missing_action,
						size_t &st_NA, size_t &end_NA, size_t &split_ix) noexcept
	{
		size_t temp;

		/* if NAs are not to be bothered with, just need to do a single pass */
		if (missing_action == Fail)
		{
			/* move to the left if it's equal to the chosen category */
			for (size_t row = st; row <= end; row++)
			{
				if (x[ix_arr[row]] == split_categ)
				{
					temp = ix_arr[st];
					ix_arr[st] = ix_arr[row];
					ix_arr[row] = temp;
					st++;
				}
			}
			split_ix = st;
		}

		/* otherwise, first put to the left all equal to chosen and not NA, then all NAs to the end of the left */
		else
		{
			for (size_t row = st; row <= end; row++)
			{
				if (x[ix_arr[row]] == split_categ)
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
				if (unlikely(x[ix_arr[row]] < 0))
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

	/* For categoricals split on sub-set that turned out to have 2 categories only (prediction-time) */
	void
	divide_subset_split(size_t *ix_arr, int x[], size_t st, size_t end,
						MissingAction missing_action,
						NewCategAction new_cat_action, bool move_new_to_left,
						size_t &st_NA, size_t &end_NA, size_t &split_ix) noexcept
	{
		size_t temp;

		/* if NAs are not to be bothered with, just need to do a single pass */
		if (missing_action == Fail)
		{
			/* move to the left if it's l.e. than the split point */
			if (new_cat_action == Smallest && move_new_to_left)
			{
				for (size_t row = st; row <= end; row++)
				{
					if (x[ix_arr[row]] == 0 || x[ix_arr[row]] > 1)
					{
						temp = ix_arr[st];
						ix_arr[st] = ix_arr[row];
						ix_arr[row] = temp;
						st++;
					}
				}
			}

			else
			{
				for (size_t row = st; row <= end; row++)
				{
					if (x[ix_arr[row]] == 0)
					{
						temp = ix_arr[st];
						ix_arr[st] = ix_arr[row];
						ix_arr[row] = temp;
						st++;
					}
				}
			}
			split_ix = st;
		}

		/* otherwise, first put to the left all l.e. and not NA, then all NAs to the end of the left */
		else
		{
			if (new_cat_action == Smallest && move_new_to_left)
			{
				for (size_t row = st; row <= end; row++)
				{
					if (x[ix_arr[row]] == 0 || x[ix_arr[row]] > 1)
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
					if (unlikely(x[ix_arr[row]] < 0))
					{
						temp = ix_arr[st];
						ix_arr[st] = ix_arr[row];
						ix_arr[row] = temp;
						st++;
					}
				}
				end_NA = st;
			}

			else
			{
				for (size_t row = st; row <= end; row++)
				{
					if (x[ix_arr[row]] == 0)
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
					if (unlikely(x[ix_arr[row]] < 0))
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
	}

	void
	fill_NAs_with_median(size_t *ix_arr, size_t st_orig, size_t st, size_t end,
						 real_t *x,
						 double *buffer_imputed_x, double *xmedian)
	{
		size_t tot = end - st + 1;
		size_t idx_half = st + div2(tot);
		bool is_odd = (tot % 2) != 0;

		if (is_odd)
		{
			*xmedian = x[ix_arr[idx_half]];
			idx_half--;
		}

		else
		{
			idx_half--;
			double xlow = x[ix_arr[idx_half]];
			double xhigh = x[ix_arr[idx_half + (size_t)1]];
			*xmedian = xlow + (xhigh - xlow) / 2.;
		}

		for (size_t ix = st_orig; ix < st; ix++)
			buffer_imputed_x[ix_arr[ix]] = (*xmedian);
		for (size_t ix = st; ix <= end; ix++)
			buffer_imputed_x[ix_arr[ix]] = x[ix_arr[ix]];

		/* 'ix_arr' can be resorted in-place, but the logic is a bit complex */
		/* step 1: move all NAs to their place by swapping them with the lower-half
		 in ascending order (after this, the lower half will be unordered).
		 along the way, copy the indices that claim the places where earlier
		 there were missing values. these copied indices will be sorted in
		 descending order at the end, as they were inserted in reverse order. */
		size_t end_pointer = idx_half;
		size_t n_move = std::min(st - st_orig, idx_half - st + 1);
		size_t temp;
		for (size_t ix = st_orig; ix < st_orig + n_move; ix++)
		{
			temp = ix_arr[end_pointer];
			ix_arr[end_pointer] = ix_arr[ix];
			ix_arr[ix] = temp;
			end_pointer--;
		}

		/* step 2: reverse the indices that were moved to the beginning so
		 as to maintain the sorting order */
		std::reverse(ix_arr + st_orig, ix_arr + st_orig + n_move);
		/* step 3: rotate the total number of elements by the number of moved elements */
		size_t n_unmoved = (idx_half - st + 1) - n_move;
		std::rotate(ix_arr + st_orig, ix_arr + st_orig + n_move,
					ix_arr + st_orig + n_move + n_unmoved);
	}

	void
	todense(size_t *ix_arr, size_t st, size_t end, size_t col_num, real_t *Xc,
			sparse_ix *Xc_ind,
			sparse_ix *Xc_indptr, double *buffer_arr)
	{
		std::fill(buffer_arr, buffer_arr + (end - st + 1), (double)0);

		size_t st_col = Xc_indptr[col_num];
		size_t end_col = Xc_indptr[col_num + 1] - 1;
		size_t curr_pos = st_col;
		size_t ind_end_col = Xc_ind[end_col];
		size_t *ptr_st = std::lower_bound(ix_arr + st, ix_arr + end + 1,
										  Xc_ind[st_col]);

		for (size_t *row = ptr_st;
			 row != ix_arr + end + 1 && curr_pos != end_col + 1 && ind_end_col >= *row;)
		{
			if (Xc_ind[curr_pos] == (sparse_ix)(*row))
			{
				buffer_arr[row - (ix_arr + st)] = Xc[curr_pos];
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

	void
	shrink_to_fit_hplane(IsoHPlane &hplane, bool clear_vectors)
	{
		if (clear_vectors)
		{
			hplane.col_num.clear();
			hplane.col_type.clear();
			hplane.coef.clear();
			hplane.mean.clear();
			hplane.cat_coef.clear();
			hplane.chosen_cat.clear();
			hplane.fill_val.clear();
			hplane.fill_new.clear();
		}

		hplane.col_num.shrink_to_fit();
		hplane.col_type.shrink_to_fit();
		hplane.coef.shrink_to_fit();
		hplane.mean.shrink_to_fit();
		hplane.cat_coef.shrink_to_fit();
		hplane.chosen_cat.shrink_to_fit();
		hplane.fill_val.shrink_to_fit();
		hplane.fill_new.shrink_to_fit();
	}

	const std::string
	trim(const std::string &pString, const std::string &pWhitespace)
	{
		const std::string::size_type beginStr = pString.find_first_not_of(
			pWhitespace);
		if (beginStr == std::string::npos)
		{
			// No content
			return "";
		}
		const std::string::size_type endStr = pString.find_last_not_of(
			pWhitespace);
		const std::string::size_type range = endStr - beginStr + 1;
		return pString.substr(beginStr, range);
	}

	const std::string
	reduce(const std::string &pString, const std::string &pFill,
		   const std::string &pWhitespace)
	{
		// Trim first
		std::string result(trim(pString, pWhitespace));

		// Replace sub ranges
		std::string::size_type beginSpace = result.find_first_of(pWhitespace);
		while (beginSpace != std::string::npos)
		{
			const std::string::size_type endSpace = result.find_first_not_of(
				pWhitespace, beginSpace);
			const std::string::size_type range = endSpace - beginSpace;
			result.replace(beginSpace, range, pFill);
			const std::string::size_type newStart = beginSpace + pFill.length();
			beginSpace = result.find_first_of(pWhitespace, newStart);
		}
		return result;
	}

} // namespace
