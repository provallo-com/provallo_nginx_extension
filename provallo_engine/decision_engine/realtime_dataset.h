#ifndef _REALTIME_DATASET_H_
#define _REALTIME_DATASET_H_

#include <iostream>
#include <streambuf>
#include <string>
#include <memory.h>
#include <cstring>
#include <syncstream>
#include <iomanip>
#include "attribute.h"
#include "dataset.h"
#include "optimizers.h"
#include "matrix.h"
#include "../utils/csvfile.h"


namespace provallo
{
    class realtime_dataset : public dataset
    {
         
        enum mode
        {
            training,
            testing ,
            cross_validation
        }; 

    private:

        mode _mode; //dataset mode : training , testing , cross_validation
       
        matrix<attribute> _attributes;
        
        //filtered set is the set of attributes that are used for training 
        //with the exception of the target attribute
        
        matrix<Float> _filtered_set;
        std::vector<discrete_value> _filtered_set_targets;
        
        
        public:
        realtime_dataset(const attribute_info& info);
        
        realtime_datset(const provallo::csvfile &file_in  )

        //implement the pure virtual functions inherited from dataset :
        virtual dataset *
        getNew() ; //return a new dataset of the same type
    
        virtual const attribute *getattributeptr(uint32_t i, attribute_tag tag, bool *found) const;
        
        virtual  attribute *getattributeptr(uint32_t i, attribute_tag tag, bool *found) ;

        const attribute & getattribute(uint32_t i, attribute_tag tag) const override;
        attribute & getattribute(uint32_t i, attribute_tag tag) override;

        virtual const attribute *getattributeptr(uint32_t i, const std::string& name, bool *found) const;

        virtual attribute *getattributeptr(uint32_t i, const std::string& name, bool *found) ;
  
        virtual void
        addData(const std::string& data) ; //add data to the dataset

        virtual void
        addData(const std::string& data, const std::string& label) ; //add data to the dataset
};


}

















#endif // _REALTIME_DATASET_H_