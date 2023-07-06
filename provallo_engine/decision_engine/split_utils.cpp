#include "dataset.h"
#include "split_utils.hpp"
#include <numeric>
#include <thread>
#include <mutex>
namespace provallo
{

  uint64_t split_method::_instance_counter =0;
  // Select cut point
  uint32_t
  continous_base::selectPoint (
      std::vector<std::pair<Float, uint32_t> >::iterator begin,
      std::vector<std::pair<Float, uint32_t> >::iterator end) const
  {
    return (*min_element (begin, end)).second;
  }

  uint32_t
  random_split::selectPoint (
      std::vector<std::pair<Float, uint32_t> >::iterator begin,
      std::vector<std::pair<Float, uint32_t> >::iterator end) const
  {
    int size = end - begin;
    std::random_device dev;
    std::mt19937 gen (dev ());

    std::uniform_int_distribution<uint32_t> uniform (0, size);

    std::vector<std::pair<Float, uint32_t> >::iterator it = begin
	+ (uniform (gen) % size);
    return (*it).second;
  }
  static float
  entropy_diff (uint32_t ni, Float l, Float r, Float li, Float ri)
  {
    return (xlog<2> (r) - xlog<2> (l) + xlog<2> (li) + xlog<2> (l + ni)
	- xlog<2> (li + ni) - xlog<2> (ri) - xlog<2> (r + ni)
	+ xlog<2> (ri + ni));
  }

  bool
  continous_base::binarySplit (const dataset &data, uint32_t begin,
			       uint32_t end, const attribute_tag &tag,
			       std::pair<uint32_t, Float> &cut_pair) const
  {
    // Check for discrete attributes

	/*refactor:*/
	if (data.getattributes ().getType(tag) == attribute_type::DISCRETE)
	  return false;
	else	
	{	
		// Initialize auxiliary variables
		uint32_t tleft (0);            // Total samples on the left
		uint32_t tright (end - begin); // Total samples on the right

		// Target tag
		attribute_tag target_tag = data.getattributes ().get_target_tag ();
		// Target count (i.e. number of classes)
		uint32_t target_count = data.getattributes ().getCount (target_tag);

		// Container of sample of each class (right and left)
		std::vector<uint32_t> left (target_count, 0);
		std::vector<uint32_t> right (target_count,0);

		// Initial position of the iterator on the target attribute
		dataset::sorted_iterator initial_target (data.begin_sorted (tag));

		// Number of consecutive sample of the same class
		uint32_t consecutive (1);
		// Current class
		uint32_t current_class ((*(initial_target + begin)).discrete ());
		if( current_class >= target_count){
			if ( discrete_value((*(initial_target + begin)).continous()) < target_count)
				current_class = discrete_value((*(initial_target + begin)).continous()	);
			else
				return false;
		}

		// Current position
		uint32_t current_position (begin);

		// Iterate over the samples
		for (dataset::sorted_iterator it (initial_target + begin);
		it != initial_target + end; ++it)
		{
			// If the class is the same of the previous sample
			if ((*it).discrete () == current_class || discrete_value((*it).continous())==current_class)
			{
				// Increment the number of consecutive samples
				++consecutive;
			}
			else
			{
				if ( current_class>=left.size() || current_class>=right.size() )
					return false;
				// Calculate the entropy difference
				Float diff = entropy_diff (consecutive, tleft, tright,
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
				current_class = (*it).discrete (); 
				if(current_class >= target_count){
					if ( discrete_value((*it).continous()) < target_count)	
						current_class = discrete_value((*it).continous());	
					else
						continue;
				}
						
										// Update the current position
				current_position = it.position ();
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
		}//for
		// Calculate the entropy difference 
		if (current_class>=target_count)
			return false;

		Float diff = entropy_diff (consecutive, tleft, tright,
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
	}//else
  }//binarySplit
	
	  bool	
	  random_split::binarySplit (const dataset &data, uint32_t begin,
			       uint32_t end, const attribute_tag &tag,
			       std::pair<uint32_t, Float> &cut_pair) const
				     {

						// Check for discrete attributes
						if (data.getattributes ().getType(tag) == attribute_type::DISCRETE)
						  return false;
						else	
						{	
							// Initialize auxiliary variables
							uint32_t tleft (0);            // Total samples on the left
							uint32_t tright (end - begin); // Total samples on the right

							// Target tag
							attribute_tag target_tag = data.getattributes ().get_target_tag ();
							// Target count (i.e. number of classes)
							uint32_t target_count = data.getattributes ().getCount (target_tag);

							// Container of sample of each class (right and left)
							std::vector<uint32_t> left (target_count, 0);
							std::vector<uint32_t> right (target_count,0);

							// Initial position of the iterator on the target attribute
							dataset::sorted_iterator initial_target (data.begin_sorted (tag));

							// Number of consecutive sample of the same class
							uint32_t consecutive (1);
							// Current class
							uint32_t current_class ((*(initial_target + begin)).discrete ());
							if (current_class>target_count)
								current_class=discrete_value((*(initial_target + begin)).continous());

							// Current position
							uint32_t current_position (begin);

							// Iterate over the samples
							for (dataset::sorted_iterator it (initial_target + begin);
							it != initial_target + end; ++it)
							{
								// If the class is the same of the previous sample
								if ( ((*it).discrete () == current_class ) || discrete_value( (*it).continous())==current_class)
								{
									// Increment the number of consecutive samples
									++consecutive;
								}
								else
								{
									// Calculate the entropy difference
									Float diff = entropy_diff (consecutive, tleft, tright,
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
									current_class = (*it).discrete ();
									if (current_class>=target_count){
										if ( discrete_value((*it).continous()) < target_count)
											current_class=discrete_value((*it).continous());
										else
											current_class = 0	;
									}
									// Update the current position
									current_position = it.position ();
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
							}//for
							// Calculate the entropy difference
							Float diff = entropy_diff (consecutive, tleft, tright,
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
						}//else
					}//binarySplit



	
/*refactor:
		// Calculate initial entropy
		Float entropy (0.0);
		for (uint32_t i = 0; i < right.size (); ++i)
		{
			Float prob = right[i] / (Float) data.size ();
			if (prob != 0.0)
			entropy += -prob * log<2> (prob);
		}

		// Number of consecutive sample of the same class
		uint32_t consecutive (1);
		// Current class
		uint32_t current_class ((initial_target + begin)->discrete ());
		// Current position
		uint32_t current_position (begin);

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
				Float diff = entropy_diff (consecutive, tleft, tright,
				left[current_class], right[current_class]);
				// If the difference is greater than the current one
				if (diff > cut_pair
	}
	#/ 
	
    // Initialize auxiliary variables
    uint32_t tleft (0);            // Total samples on the left
    uint32_t tright (end - begin); // Total samples on the right

    // Target tag
    attribute_tag target_tag = data.getattributes ().get_target_tag ();
    // Target count (i.e. number of classes)
    uint32_t target_count = data.getattributes ().getCount (target_tag);

    // Container of sample of each class (right and left)
    std::vector<uint32_t> left (target_count, 0);
    std::vector<uint32_t> right (target_count,0);

    // Initial position of the iterator on the target attribute
    dataset::sorted_iterator initial_target (data.begin_sorted (tag));

    // Initialize right array
    for (dataset::sorted_iterator it (initial_target + begin);
	it != initial_target + end; ++it)
      {		
			right[(it.begin () + target_tag)->discrete ()]++;
      }

    // Calculate initial entropy
    Float entropy (0.0);
    for (uint32_t i = 0; i < right.size (); ++i)
      {
	Float prob = right[i] / (Float) data.size ();
	if (prob != 0.0)
	  entropy += -prob * log<2> (prob);
      }

    // Number of consecutive sample of the same class
    uint32_t nssc (0);
    // Split entropy
    std::vector<std::pair<Float, uint32_t> > split_entropy;

    // Loop over the sample and check potential cut points (i.e. boundary points)
    for (dataset::sorted_iterator it = initial_target + begin;
	it != initial_target + end - 1; ++it)
      {
		attribute_iterator at = it.begin();
		// Get class of this and next sample class
		uint32_t this_class = (at + target_tag)->discrete ();
		uint32_t next_class = ((it + 1).begin () + target_tag)->discrete ();

		// Increment counter
		++nssc;

	// Position of the iterator
	uint32_t offset (it - initial_target);

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
					/ (Float) data.size ();
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
			uint32_t cut_point = selectPoint (split_entropy.begin (),
							split_entropy.end ());
			// Get attribute value at the cut point
			Float left_value =
				((initial_target + cut_point).begin () + tag)->continous ();
			Float right_value =
				((initial_target + cut_point).begin () + tag)->continous ();
			Float cut_value = (right_value + left_value) / 2;
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
  continous_base::splitInterval (const dataset &data, uint32_t begin,
				 uint32_t end, const attribute_tag &tag,
				 std::vector<Float> &interval) const
  {
	// Check if we can split the interval

	if(begin==end)
		return;
	std::pair<uint32_t, Float> cut_pair(0, 0.0);
 	cut_pair.first = begin;	
	cut_pair.second = (*(data.begin_sorted (tag) + begin)).continous ();	

	if (this->checkSplitting (data, begin, end, 0, tag))
	  {		

			//  
			uint32_t cut_point = cut_pair.first;
			Float cut_value = cut_pair.second;
			// Split left interval
			if( begin!=end)
				{
					splitInterval (data, begin, end, tag, interval);
					// Push cut value
					interval.push_back (cut_value);
					// Split right interval
	   			splitInterval (data, cut_point + 1, end, tag, interval);
				}
				else
				{
					interval.push_back (cut_value);
				}	


	  }
  }
  void
  continous_base::split (const dataset &data, const attribute_tag &tag,
			 std::vector<Float> &interval) const
  {
    // Set first value of the interval
    interval.push_back (-std::numeric_limits<float>::infinity ());
    // Recursively split intervals
    splitInterval (data, 0, data.size (), tag, interval);
    // Set last value of the interval
    interval.push_back (std::numeric_limits<float>::infinity ());
  }

  bool
  mdlp_split::checkSplitting (const dataset &data, uint32_t begin, uint32_t end,
			      uint32_t cut_point,
			      const attribute_tag &tag) const
  {
    // Target tag
    attribute_tag target_tag = data.getattributes ().get_target_tag ();
    // Target count (i.e. number of classes)
    uint32_t target_count = data.getattributes ().getCount (target_tag);

    // Container of sample of each class (right and left)
    std::vector<uint32_t> left (target_count);
    std::vector<uint32_t> right (target_count);

    // Initial position of the iterator on the target attribute
    dataset::sorted_iterator initial_target = data.begin_sorted (tag);

    // Initialize right array
    for (dataset::sorted_iterator it (initial_target + begin);
			it != initial_target + end; ++it)
			{
			uint32_t j (it - initial_target);
			uint32_t class_attr ((it.begin () + target_tag)->discrete ());
			// Accumulate class value
			if (j < cut_point + 1)
			left[class_attr]++;
			else
			right[class_attr]++;
			}
			// Size of each interval
			uint32_t size_left (cut_point + 1 - begin);
			uint32_t size_right (end - (cut_point + 1));

			// Calculate entropy
			Float entropy_right (0.0);
			Float entropy_left (0.0);
			Float entropy (0.0);
			// Count of different classes on each interval
			Float kl (target_count);
			Float kr (target_count);
			for (uint32_t i = 0; i < right.size (); ++i)
			{
			// Calculate entropy of the sets defined by the partition
				Float prob = right[i] / (Float) size_right;
				if (prob != 0.0)
				entropy_right += -prob * log<2> (prob);
				else
				kr--;
				prob = left[i] / (Float) size_left;
				if (prob != 0.0)
					entropy_left += -prob * log<2> (prob);
					else
				kl--;
				// Calculate total entropy
				prob = (right[i] + left[i]) / (Float) data.size ();
				if (prob != 0.0)
				entropy += -prob * log<2> (prob);
			}
			// Check criteria
			Float delta = log<2> (pow (3, (Float) target_count) - 2)
			- ((Float) target_count * entropy - kl * entropy_left
				- kr * entropy_right);
			Float gain = entropy - ((Float) size_left / data.size ()) * entropy_left
			- ((Float) size_right / data.size ()) * entropy_right;
			Float test = log<2> (data.size () - 1) / (Float) data.size ()
			+ delta / (Float) data.size ();

    // Accept the split
    if (gain > test)
      return true;
    // Reject the splitting
    return false;
  }


  typedef cont1d<binary_split>  continous_binary_split;
  typedef cont1d<multi_interval_split> continous_multi_interval;
  typedef cont1d<mdlp_split> continous_mdlp;
  typedef cont1d<random_split>  continous_random;

  split_method*
  split_method_factory::createMethod (const split_method &deserial)
  {
    // Method
    split_method *method (0);
    // Get type
    split_type type = deserial.get_type ();
    // Check attribute
    if (type == DISC)
      {
		// Discrete attribute.
		// branch by value
		method = new discrete_split ();
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

      throw(std::runtime_error ("Invalid method type: " + std::to_string (type)));

    // Get data from buffer
    //method->deserialize(&deserial);

    // Return method
    return method;
  }

  split_method*
  split_method_factory::createMethod (const std::random_device &random_,
				      split_type type, const dataset &data_set,
				      const attribute_tag &tag,
				      const attribute_tag &factory_tag)
  {

    static std::recursive_mutex _mute;
    static std::map<std::pair<size_t,size_t>,split_method* > _split_cache;
    std::lock_guard<std::recursive_mutex > guard(_mute );
    std::pair<size_t,size_t> lookup =  std::make_pair(factory_tag,tag);
    split_method* ret = nullptr;

    // Check attribute
    if (type == DISC)
      {

			//do we really need to create? or just return from cache...
			if(_split_cache.size() && _split_cache.find(lookup)!=_split_cache.end() )
			{

				std::cout<<"[+] returning cached ["<<std::to_string(factory_tag)<< "," <<std::to_string(tag)<<" ]"<<std::endl;;

				ret = _split_cache[lookup];
			}
			else {

				std::cout<<"[+] creating discrete split ["<<std::to_string(factory_tag)<< "," <<std::to_string(tag)<<" ]"<<std::endl;;

				ret = new discrete_split (factory_tag, tag, std::ref(data_set));
				_split_cache.insert(std::make_pair(lookup,ret));

			}
      }
    else if (type == CONE_BINARY)
      {


	if(_split_cache.size() && _split_cache.find(lookup)!=_split_cache.end() )
	  {
	    std::cout<<"[+] returning cached ["<<std::to_string(factory_tag)<< "," <<std::to_string(tag)<<" ]"<<std::endl;;

	    ret = _split_cache[lookup];
	  }
	else {

	std::cout<<"[+] creating binary continous split ["<<std::to_string(factory_tag)<< "," <<std::to_string(tag)<<" ]"<<std::endl;

	// Return continuous split method
	ret=  new continous_binary_split(factory_tag, tag, std::ref(data_set));
	  _split_cache.insert(std::make_pair(lookup,ret));

	}
      }
    else if (type == CONE_MULTI)
      {
	if(_split_cache.size() && _split_cache.find(lookup)!=_split_cache.end() )
	  {
	    std::cout<<"[+] returning cached ["<<std::to_string(factory_tag)<< "," <<std::to_string(tag)<<" ]"<<std::endl;;

	    ret = _split_cache[lookup];
	  }
	else {

	std::cout<<"[+] creating interval split ["<<std::to_string(factory_tag)<< "," <<std::to_string(tag)<<" ]"<<std::endl;

	// Return continuous split method
	ret = new continous_multi_interval(factory_tag, tag, data_set);
	  _split_cache.insert(std::make_pair(lookup,ret));
	}
      }
    else if (type == CONE_MDLP)
      {
	if(_split_cache.size() && _split_cache.find(lookup)!=_split_cache.end() )
	  {
	    std::cout<<"[+] returning cached ["<<std::to_string(factory_tag)<< "," <<std::to_string(tag)<<" ]"<<std::endl;;

	    ret = _split_cache[lookup];
	  }
	else {

	std::cout<<"[+] creating mdlp split ["<<std::to_string(factory_tag)<< "," <<std::to_string(tag)<<" ]"<<std::endl;
 	// Return continuous split method
	ret = new continous_mdlp(factory_tag, tag, data_set);
	  _split_cache.insert(std::make_pair(lookup,ret));
	}
      }
    else if (type == CONE_RANDOM)
      {
			if(_split_cache.size() && _split_cache.find(lookup)!=_split_cache.end() )
			{
				std::cout<<"[+] returning cached ["<<std::to_string(factory_tag)<< "," <<std::to_string(tag)<<" ]"<<std::endl;;

				ret = _split_cache[lookup];
			}
			else {

					std::cout<<"[+] creating random split ["<<std::to_string(factory_tag)<< "," <<std::to_string(tag)<<" ]"<<std::endl;
					// Return continuous split method (random split need the random number engine)
					ret = new continous_random (factory_tag, tag, data_set,std::random_device ());
					_split_cache.insert(std::make_pair(lookup,ret));
			}
      }
			else{
					// Get attribute information
					const attribute_information &info (data_set.getattributes ());
					throw(std::runtime_error (
						std::string ("Invalid type for attribute ") + info.getName (tag)));
					
			}
			return ret ;
  }

typedef dataset* datasetptr;

  split_method_factory::split_method_factory (const dataset &data_set,
					      const std::random_device &r):_split_methods(),_target_method(nullptr), r_dataset(*datasetptr(&data_set)),
override_split_method(false),	override_split_type(CONE_RANDOM)	
  {
    // Get attribute information
    const attribute_information &info (data_set.getattributes ());
    // Target tag
    attribute_tag target_tag = info.get_target_tag ();
    // Create target method


    //resize vector
    
    _split_methods.resize(info.getSize()) ;

    _target_method = split_method_factory::createMethod (r, DISC, data_set,
							 target_tag,  6 );
	// Create split methods
    // Get group of attributes (the group define the effective number of attributes)
    const attribute_groups &groups (info.getGroups ());

    // Initialize the map with split methods



    for (size_t i = 0; i < groups.size (); ++i)
      _split_methods.push_back (
	  split_method_factory::createMethod (r, override_split_method?override_split_type: groups.getsplit_type (i),
					      data_set, groups.getGroup (i)[0],
					      i)); 
					
	
    std::cout<<"[+] created [[" <<split_method::_instance_counter<< "]] splits"<<std::endl;
  }

  const split_method*
  split_method_factory::getMethod (const attribute_tag &tag) const
  {
    // Find tag
    if (tag < _split_methods.size ())
      return _split_methods[tag];
    else
      {
	throw(std::runtime_error (
	    "Tag number " + std::to_string (tag)
		+ " is not available on factory"));
      }
  }

  split_method_factory::~split_method_factory ()
  {
    for(split_method* p:_split_methods)
	delete p;

    delete _target_method;
      _target_method=nullptr;
  }

  void
  split_method_factory::serialize (split_method_factory *serial) const
  {
    
	//not implemented
  }

  // Get data from buffer
  void
  split_method_factory::deserialize (const split_method_factory *serial)
  {
    //not implemented
  }

  Float
  EntropyGain::gain (const dataset &data, const split_method &selector)
  {
    // Target tag
    attribute_tag target_tag = data.getattributes ().get_target_tag ();
    // Number of different values of the attribute
    uint32_t attr_count = selector.size ();
    if (attr_count == 1)
      return 0.0;
    // Number of different outcomes
    uint32_t target_count = data.getattributes ().getCount (target_tag);

    std::vector<std::vector<uint32_t> > freqs (
	target_count, std::vector<uint32_t> (attr_count, 0));
    // Number of samples where the attribute takes some value
    std::vector<uint32_t> count (attr_count, 0);
    // Target attribute counter
    std::vector<uint32_t> target_probs (target_count, 0);

    // Loop over the data set
    for (uint32_t i = 0; i < data.size (); ++i)
      {
	//sanity check:
	if(data.begin (i)!=data.end(i)){
	// Get attribute "branch"
	uint32_t attr_value = selector.getBranch (data.begin (i));
	// Get target value
	discrete_value target_value =
	    data.getattribute (i, target_tag).discrete ();
	// Accumulate value for this attribute instance
	freqs[target_value][attr_value]++;
	// Accumulate target occurrence
	target_probs[target_value]++;
	// Accumulate attribute count
	count[attr_value]++;
	}
      }

    // Entropy
    Float entropy = 0.0;
    // Total count
    uint32_t total_count = std::accumulate (count.begin (), count.end (), 0);

    // Check if the data set contain at least one known value
    if (total_count == 0)
      return 0.0; // No gain

    // Subset entropies (for different values of the attribute)
    std::vector<Float> attr_entropy (attr_count, 0.0);
    for (uint32_t i = 0; i < target_probs.size (); ++i)
      {
	Float prob = (Float) target_probs[i] / (Float) total_count;
	if (prob != 0.0)
	  entropy += -prob * log<2> (prob);
	for (uint32_t j = 0; j < attr_entropy.size (); ++j)
	  {
	    if (count[j] != 0.0)
	      {
			Float attr_prob = (Float) freqs[i][j] / (Float) count[j];
			attr_entropy[j] -= xlog<2> (attr_prob);
	      }
	  }
      }

    // Gain
    Float gain = entropy;
    for (uint32_t i = 0; i < count.size (); ++i)
      gain -= ((Float) count[i] / (Float) data.size ()) * attr_entropy[i];

    // Check for NAN
    assert(gain == gain);

    // Return entropy
    return gain;
  }

  Float
  GainRatio::gain (const dataset &data, const split_method &selector)
  {
    // Target tag
    attribute_tag target_tag = data.getattributes ().get_target_tag ();
    // Number of different values of the attribute
    uint32_t attr_count = selector.size ();
    if (attr_count == 1)
      return 0.0;
    // Number of different outcomes
    uint32_t target_count = data.getattributes ().getCount (target_tag);
    // Proportion of instances (for each attribute value) in the data set that take
    // the a value of the target
    std::vector<std::vector<uint32_t> > freqs (
	target_count, std::vector<uint32_t> (attr_count, 0));
    // Number of samples where the attribute takes some value
    std::vector<uint32_t> count (attr_count, 0);
    // Target attribute counter
    std::vector<uint32_t> target_probs (target_count, 0);

    // Loop over the data set
    for (uint32_t i = 0; i < data.size (); ++i)
      {

	//sanity check:
	if(data.begin (i)!=data.end(i)){
	 // Get attribute "branch"

	// Get attribute "branch"
	uint32_t attr_value = selector.getBranch (data.begin (i));
	// Get target value
	discrete_value target_value =
	    data.getattribute (i, target_tag).discrete ();
	// Accumulate value for this attribute instance
	freqs[target_value][attr_value]++;
	// Accumulate target occurrence
	target_probs[target_value]++;
	// Accumulate attribute count
	count[attr_value]++;

	}
      }
    // Entropy
    Float entropy = 0.0;
    // Total count
    uint32_t total_count = std::accumulate (count.begin (), count.end (), 0);

    // Check if the data set contain at least one known value
    if (total_count == 0)
      return 0.0; // No gain

    // Subset entropies (for different values of the attribute)
    std::vector<Float> attr_entropy (attr_count, 0.0);
    for (uint32_t i = 0; i < target_probs.size (); ++i)
      {
	Float prob = (Float) target_probs[i] / (Float) total_count;
	if (prob != 0.0)
	  entropy += -prob * log<2> (prob);
	for (uint32_t j = 0; j < attr_entropy.size (); ++j)
	  {
	    if (count[j] != 0.0)
	      {
		Float attr_prob = (Float) freqs[i][j] / (Float) count[j];
		attr_entropy[j] -= xlog<2> (attr_prob);
	      }
	  }
      }

    // Gain
    Float gain = entropy;
    // Split information
    Float split = 0.0;
    for (uint32_t i = 0; i < count.size (); ++i)
      {
	// Probability
	Float pi = (Float) count[i] / (Float) data.size ();
	// Accumulate gain
	gain -= pi * attr_entropy[i];
	// Split information
	split -= xlog<2> (pi);
      }
    // Add unknown values, if any
    if (data.size () - total_count > 0)
      split -= (data.size () - total_count)
	  * log<2> (data.size () - total_count);

    // Gain ratio
    Float gain_ratio = gain / split;

    // If gain and split are zero, set gain_ratio to zero
    if (gain == 0 && split == 0)
      gain_ratio = 0.0;

    // Check for NAN
    assert(gain_ratio == gain_ratio);

    // Return gain
    return gain_ratio;
  }

  Float
  ChiSquare::gain (const dataset &data, const split_method &selector)
  {
    // Target tag
    attribute_tag target_tag = data.getattributes ().get_target_tag ();
    // Number of different values of the attribute
    uint32_t attr_count = selector.size ();
    if (attr_count == 1)
      return 0.0;
    // Number of different outcomes
    uint32_t target_count = data.getattributes ().getCount (target_tag);

    // Proportion of instances (for each attribute value) in the data set that take
    // the a value of the target
    std::vector<std::vector<uint32_t> > freqs (
	target_count, std::vector<uint32_t> (attr_count, 0));
    // Total in rows
    std::vector<uint32_t> row_total (target_count);
    // Total in columns
    std::vector<uint32_t> column_total (attr_count);
    // Total samples
    uint32_t total (0);

    // Loop over the data set
    for (uint32_t i = 0; i < data.size (); ++i)
      {
	// Get attribute "branch"
	uint32_t attr_value = selector.getBranch (data.begin (i));
	// Get target value
	discrete_value target_value =
	    data.getattribute (i, target_tag).discrete ();
	// Accumulate value for this attribute instance
	freqs[target_value][attr_value]++;
	// Compute totals
	row_total[target_value]++;
	column_total[attr_value]++;
	total++;
      }

    // Compute CHI square
    Float chi (0.0);

    // Loop over target values
    for (uint32_t i = 0; i < target_count; ++i)
      {
	for (uint32_t j = 0; j < attr_count; ++j)
	  {
	    // Numerator
	    Float num = row_total[i] * column_total[j];
	    if (num != 0.0 && total != 0.0)
	      {
		// Compute expected value
		Float expected = num / total;
		// Sum contribution to CHI square
		chi += (freqs[i][j] - expected) * (freqs[i][j] - expected)
		    / expected;
	      }
	  }
      }

    // Check for NAN
    assert(chi == chi);

    // Return CHI square
    return chi;
  }

  size_t
  divide_subset_split (size_t ix_arr[], double x[], size_t st, size_t end,
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
  divide_subset_split (size_t *ix_arr, int x[], size_t st, size_t end,
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
  divide_subset_split (size_t *ix_arr, int x[], size_t st, size_t end,
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
		if (cval >= ncat || split_categ[cval] == 1
		    || split_categ[cval] == (-1))
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
		if (cval >= 0
		    && (cval >= ncat || split_categ[cval] == 1
			|| split_categ[cval] == (-1)))
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
  divide_subset_split (size_t *ix_arr, int x[], size_t st, size_t end,
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
  divide_subset_split (size_t *ix_arr, int x[], size_t st, size_t end,
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
  fill_NAs_with_median (size_t *ix_arr, size_t st_orig, size_t st, size_t end,
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
	double xhigh = x[ix_arr[idx_half + (size_t) 1]];
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
    size_t n_move = std::min (st - st_orig, idx_half - st + 1);
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
    std::reverse (ix_arr + st_orig, ix_arr + st_orig + n_move);
    /* step 3: rotate the total number of elements by the number of moved elements */
    size_t n_unmoved = (idx_half - st + 1) - n_move;
    std::rotate (ix_arr + st_orig, ix_arr + st_orig + n_move,
		 ix_arr + st_orig + n_move + n_unmoved);
  }

  void
  todense (size_t *ix_arr, size_t st, size_t end, size_t col_num, real_t *Xc,
  sparse_ix *Xc_ind,
	   sparse_ix *Xc_indptr, double *buffer_arr)
  {
    std::fill (buffer_arr, buffer_arr + (end - st + 1), (double) 0);

    size_t st_col = Xc_indptr[col_num];
    size_t end_col = Xc_indptr[col_num + 1] - 1;
    size_t curr_pos = st_col;
    size_t ind_end_col = Xc_ind[end_col];
    size_t *ptr_st = std::lower_bound (ix_arr + st, ix_arr + end + 1,
				       Xc_ind[st_col]);

    for (size_t *row = ptr_st;
	row != ix_arr + end + 1 && curr_pos != end_col + 1
	    && ind_end_col >= *row;)
      {
	if (Xc_ind[curr_pos] == (sparse_ix) (*row))
	  {
	    buffer_arr[row - (ix_arr + st)] = Xc[curr_pos];
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
					   Xc_ind + end_col + 1, *row) - Xc_ind;
	  }
      }
  }

  void
  shrink_to_fit_hplane (IsoHPlane &hplane, bool clear_vectors)
  {
    if (clear_vectors)
      	{
			hplane.col_num.clear ();
			hplane.col_type.clear ();
			hplane.coef.clear ();
			hplane.mean.clear ();
			hplane.cat_coef.clear ();
			hplane.chosen_cat.clear ();
			hplane.fill_val.clear ();
			hplane.fill_new.clear ();
		
		}

		hplane.col_num.shrink_to_fit ();
		hplane.col_type.shrink_to_fit ();
		hplane.coef.shrink_to_fit ();
		hplane.mean.shrink_to_fit ();
		hplane.cat_coef.shrink_to_fit ();
		hplane.chosen_cat.shrink_to_fit ();
		hplane.fill_val.shrink_to_fit ();
		hplane.fill_new.shrink_to_fit ();
  }

  const std::string
  trim (const std::string &pString, const std::string &pWhitespace)
  {
    const std::string::size_type beginStr = pString.find_first_not_of (
	pWhitespace);
    if (beginStr == std::string::npos)
      {
		// No content
		return "";
      }
    const std::string::size_type endStr = pString.find_last_not_of (
	pWhitespace);
    const std::string::size_type range = endStr - beginStr + 1;
    return pString.substr (beginStr, range);
  }

  const std::string
  reduce (const std::string &pString, const std::string &pFill,
	  const std::string &pWhitespace)
  {
    // Trim first
    std::string result (trim (pString, pWhitespace));

    // Replace sub ranges
    std::string::size_type beginSpace = result.find_first_of (pWhitespace);
    while (beginSpace != std::string::npos)
      {
			const std::string::size_type endSpace = result.find_first_not_of (
				pWhitespace, beginSpace);
			const std::string::size_type range = endSpace - beginSpace;
			result.replace (beginSpace, range, pFill);
			const std::string::size_type newStart = beginSpace + pFill.length ();
			beginSpace = result.find_first_of (pWhitespace, newStart);
    
	  }
    return result;
  }



} //namespace
