/*
 * portanalyzer.cpp
 *
 *  Created on: Apr 3, 2021
 *      Author: kardon
 */

#include "portanalyzer.h"

namespace provallo
{

  provallo::port_distribution<uint16_t> g;
#ifdef STATIC_LOCK_
template <typename T>
std::recursive_mutex port_distribution<T>::locker_;
template <typename T>
std::map<std::pair<T,T>, T> port_distribution<T>::port_histogram_;
#endif //STATIC_LOCK_

//generate reference obj

} /* namespace provallo */

