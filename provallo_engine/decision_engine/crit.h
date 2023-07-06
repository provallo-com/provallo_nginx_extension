/*
 * crit.h
 *
 *  Created on: Apr 1, 2023
 *      Author: kardon
 */

#ifndef DECISION_ENGINE_CRIT_H_
#define DECISION_ENGINE_CRIT_H_

#define pw1(x) ((x))
#define pw2(x) ((x) * (x))
#define pw3(x) ((x) * (x) * (x))
#define pw4(x) ((x) * (x) * (x) * (x))
#define sd_gain(sd, sd_left, sd_right) (1. - ((sd_left) + (sd_right)) / (2. * (sd)))
#define pooled_gain(sd, cnt, sd_left, sd_right, cnt_left, cnt_right) \
	    (1. - (1./(sd))*(  ( ((real_t)(cnt_left))/(cnt) )*(sd_left) + ( ((real_t)(cnt_right)/(cnt)) )*(sd_right)  ))

#define extract_bit(number, bit) (((number) >> (bit)) & 1)

 
#endif /* DECISION_ENGINE_CRIT_H_ */
