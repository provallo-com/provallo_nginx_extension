#ifndef _FAST_MATRIX_FOREST_H_
#define _FAST_MATRIX_FOREST_H_

//fast_matrix_forest.h
//is a collection of matrices that can be used to replace trees,nodes,leafs and forests 
//in decision trees and random forests 

//super_tree is a matrix of indices that can be used to access the forest 
//super_tree_probabilities is a matrix of probabilities that can be used to access the forest
//super_tree_values is a matrix of values that can be used to access the forest

//super_tree_values_projection is a matrix of values that can be used to access the forest
//super_tree_hplane is a matrix of hyperplanes that can be used to access the forest


#include <string>
#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <set>
#include <cmath>
#include <numeric>
#include <functional>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <limits>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>

#include "matrix.h"
namespace provallo { 
    
    

//each matrix is a transition matrix from a tree structure
//the tree structure is a vector of vectors of discrete or continous values 
//the tree_matrix is a matrix of probabilities

struct  matrix_indices : public std::pair<size_t,size_t>
{
    matrix_indices(size_t i,size_t j):std::pair<size_t,size_t>(i,j){}
    size_t i() const {return first;}
    size_t j() const {return second;}
    size_t & i() {return first;}    
    size_t & j() {return second;}
    bool operator==(const matrix_indices & other) const
    {
        return (first==other.first && second==other.second);
    }
    bool operator!=(const matrix_indices & other) const
    {
        return !(*this==other);
    }
    bool operator<(const matrix_indices & other) const
    {
        return (first<other.first || (first==other.first && second<other.second));
    }
    bool operator>(const matrix_indices & other) const
    {
        return (first>other.first || (first==other.first && second>other.second));
    }
    bool operator<=(const matrix_indices & other) const
    {
        return (first<=other.first || (first==other.first && second<=other.second));
    }
    bool operator>=(const matrix_indices & other) const
    {
        return (first>=other.first || (first==other.first && second>=other.second));
    }
    matrix_indices operator+(const matrix_indices & other) const
    {
        return matrix_indices(first+other.first,second+other.second);
    }
    matrix_indices operator-(const matrix_indices & other) const
    {
        return matrix_indices(first-other.first,second-other.second);
    }
    matrix_indices operator*(const matrix_indices & other) const
    {
        return matrix_indices(first*other.first,second*other.second);
    }
    matrix_indices operator/(const matrix_indices & other) const
    {
        return matrix_indices(first/other.first,second/other.second);
    }
    matrix_indices operator%(const matrix_indices & other) const
    {
        return matrix_indices(first%other.first,second%other.second);
    }
    matrix_indices operator+(const size_t & other) const
    {
        return matrix_indices(first+other,second+other);
    }
    matrix_indices operator-(const size_t & other) const
    {
        return matrix_indices(first-other,second-other);
    }   
    matrix_indices operator*(const size_t & other) const
    {
        return matrix_indices(first*other,second*other);
    }   
    matrix_indices operator/(const size_t & other) const
    {
        return matrix_indices(first/other,second/other);
    }   
    //arithmetic operators
    matrix_indices & operator+=(const matrix_indices & other) 
    {
        first+=other.first;
        second+=other.second;
        return *this;
    }
    matrix_indices & operator-=(const matrix_indices & other) 
    {
        first-=other.first;
        second-=other.second;
        return *this;
    }   
    matrix_indices & operator*=(const matrix_indices & other) 
    {
        first*=other.first;
        second*=other.second;
        return *this;
    }
    matrix_indices & operator/=(const matrix_indices & other) 
    {
        first/=other.first;
        second/=other.second;
        return *this;
    }
    matrix_indices & operator%=(const matrix_indices & other) 
    {
        first%=other.first;
        second%=other.second;
        return *this;
    }
    matrix_indices & operator+=(const size_t & other) 
    {
        first+=other;
        second+=other;
        return *this;
    }
    matrix_indices & operator-=(const size_t & other) 
    {
        first-=other;
        second-=other;
        return *this;
    }
    matrix_indices & operator*=(const size_t & other) 
    {
        first*=other;
        second*=other;
        return *this;
    }
    matrix_indices & operator/=(const size_t & other) 
    {
        first/=other;
        second/=other;
        return *this;
    }
    //logical operators
    bool operator!() const
    {
        return (first==0 && second==0);
    }
    bool operator&&(const matrix_indices & other) const
    {
        return (first && other.first && second && other.second);
    }
    bool operator||(const matrix_indices & other) const
    {
        return (first || other.first || second || other.second);
    }
    bool operator&&(const size_t & other) const
    {
        return (first && other && second && other);
    }
    bool operator||(const size_t & other) const
    {
        return (first || other || second || other);
    }
    bool operator==(const size_t & other) const
    {
        return (first==other && second==other);
    }
    bool operator!=(const size_t & other) const
    {
        return (first!=other || second!=other);
    }
    friend std::ostream & operator<<(std::ostream & os,const matrix_indices & other)
    {
        os<<"("<<other.first<<","<<other.second<<")";
        return os;
    }   
    friend std::istream & operator>>(std::istream & is,matrix_indices & other)
    {
        is>>other.first>>other.second;
        return is;
    }
    static std::atomic_uint64_t matrix_indices_count;

};  
    
    typedef struct tag_hyperplane
	{
        uint64_t hplane_id;
        matrix_indices hplane_indices; //indices of the hplane in the super_tree 
        size_t hplane_depth;
        size_t hplane_level;
        size_t hplane_parent;
        size_t hplane_left;
        size_t hplane_right;

        size_t hplane_dim;
        size_t hplane_feature;
        size_t hplane_feature_index;
        size_t hplane_feature_index_left;
        size_t hplane_feature_index_right;
        real_t hplane_feature_value;
        real_t hplane_feature_value_left;
        real_t hplane_feature_value_right;
        real_t hplane_feature_value_min;
        real_t hplane_feature_value_max;
        real_t hplane_feature_value_range;
        real_t weight;
        real_t score;
        std::uniform_real_distribution<real_t> distribution;        
        //hplane constructor
        tag_hyperplane():hplane_id(++hplane_count),hplane_indices(matrix_indices(0,0)),hplane_depth(0),hplane_level(0),hplane_parent(0),hplane_left(0),hplane_right(0),hplane_dim(0),hplane_feature(0),hplane_feature_index(0),hplane_feature_index_left(0),hplane_feature_index_right(0),hplane_feature_value(0.0),hplane_feature_value_left(0.0),hplane_feature_value_right(0.0),hplane_feature_value_min(0.0),hplane_feature_value_max(0.0),hplane_feature_value_range(0.0),weight(0.0),score(0.0),distribution(0.0,1.0){}
        //copy constructor
        tag_hyperplane(const tag_hyperplane & other):hplane_id(other.hplane_id),hplane_indices(other.hplane_indices),hplane_depth(other.hplane_depth),hplane_level(other.hplane_level),hplane_parent(other.hplane_parent),hplane_left(other.hplane_left),hplane_right(other.hplane_right),hplane_dim(other.hplane_dim),hplane_feature(other.hplane_feature),hplane_feature_index(other.hplane_feature_index),hplane_feature_index_left(other.hplane_feature_index_left),hplane_feature_index_right(other.hplane_feature_index_right),hplane_feature_value(other.hplane_feature_value),hplane_feature_value_left(other.hplane_feature_value_left),hplane_feature_value_right(other.hplane_feature_value_right),hplane_feature_value_min(other.hplane_feature_value_min),hplane_feature_value_max(other.hplane_feature_value_max),hplane_feature_value_range(other.hplane_feature_value_range),weight(other.weight),score(other.score),distribution(other.distribution){}
        //move constructor
        tag_hyperplane(tag_hyperplane && other):hplane_id(other.hplane_id),hplane_indices(other.hplane_indices),hplane_depth(other.hplane_depth),hplane_level(other.hplane_level),hplane_parent(other.hplane_parent),hplane_left(other.hplane_left),hplane_right(other.hplane_right),hplane_dim(other.hplane_dim),hplane_feature(other.hplane_feature),hplane_feature_index(other.hplane_feature_index),hplane_feature_index_left(other.hplane_feature_index_left),hplane_feature_index_right(other.hplane_feature_index_right),hplane_feature_value(other.hplane_feature_value),hplane_feature_value_left(other.hplane_feature_value_left),hplane_feature_value_right(other.hplane_feature_value_right),hplane_feature_value_min(other.hplane_feature_value_min),hplane_feature_value_max(other.hplane_feature_value_max),hplane_feature_value_range(other.hplane_feature_value_range),weight(other.weight),score(other.score),distribution(other.distribution){}
        //copy assignment
        tag_hyperplane & operator=(const tag_hyperplane & other)
        {
            if(this!=&other)
            {
                //hplane_id=other.hplane_id;
                hplane_indices=other.hplane_indices;
                hplane_depth=other.hplane_depth;
                hplane_level=other.hplane_level;
                hplane_parent=other.hplane_parent;
                hplane_left=other.hplane_left;
                hplane_right=other.hplane_right;
                hplane_dim=other.hplane_dim;
                hplane_feature=other.hplane_feature;
                hplane_feature_index=other.hplane_feature_index;
                hplane_feature_index_left=other.hplane_feature_index_left;
                hplane_feature_index_right=other.hplane_feature_index_right;
                hplane_feature_value=other.hplane_feature_value;
                hplane_feature_value_left=other.hplane_feature_value_left;
                hplane_feature_value_right=other.hplane_feature_value_right;
                hplane_feature_value_min=other.hplane_feature_value_min;
                hplane_feature_value_max=other.hplane_feature_value_max;
                hplane_feature_value_range=other.hplane_feature_value_range;
                weight=other.weight;
                score=other.score;
                distribution=other.distribution;

            }
            return *this;
        }   
        //move assignment
        tag_hyperplane & operator=(tag_hyperplane && other)
        {
            if(this!=&other)
            {
                hplane_id=std::move(other.hplane_id);
                hplane_indices = std::move(other.hplane_indices);
                hplane_depth=std::move(other.hplane_depth);
                hplane_level=std::move(other.hplane_level);
                hplane_parent=std::move(other.hplane_parent);
                hplane_left=std::move(other.hplane_left);
                hplane_right=std::move(other.hplane_right);
                hplane_dim=std::move(other.hplane_dim);
                hplane_feature=std::move(other.hplane_feature);
                hplane_feature_index=std::move(other.hplane_feature_index);
                hplane_feature_index_left=std::move(other.hplane_feature_index_left);
                hplane_feature_index_right=std::move(other.hplane_feature_index_right); 
                hplane_feature_value=std::move(other.hplane_feature_value);
                hplane_feature_value_left=std::move(other.hplane_feature_value_left);
                hplane_feature_value_right=std::move(other.hplane_feature_value_right);
                hplane_feature_value_min=std::move(other.hplane_feature_value_min);

                hplane_feature_value_max=std::move(other.hplane_feature_value_max); 
                hplane_feature_value_range=std::move(other.hplane_feature_value_range);
                weight=std::move(other.weight);
                score=std::move(other.score);
                distribution=std::move(other.distribution);
                
                //other.hplane_id=0;
                //other.hplane_indices=matrix_indices(0,0);
                //other.hplane_depth=0;
                //other.hplane_level=0;
                //other.hplane_parent=0;
                
            }
            return *this;
        }   
        //destructor
        ~tag_hyperplane(){}
        //comparison operators
        bool operator==(const tag_hyperplane & other) const
        {
            return (hplane_id==other.hplane_id && hplane_indices==other.hplane_indices && hplane_depth==other.hplane_depth && hplane_level==other.hplane_level && hplane_parent==other.hplane_parent && hplane_left==other.hplane_left && hplane_right==other.hplane_right && hplane_dim==other.hplane_dim && hplane_feature==other.hplane_feature && hplane_feature_index==other.hplane_feature_index && hplane_feature_index_left==other.hplane_feature_index_left && hplane_feature_index_right==other.hplane_feature_index_right && hplane_feature_value==other.hplane_feature_value && hplane_feature_value_left==other.hplane_feature_value_left && hplane_feature_value_right==other.hplane_feature_value_right && hplane_feature_value_min==other.hplane_feature_value_min && hplane_feature_value_max==other.hplane_feature_value_max && hplane_feature_value_range==other.hplane_feature_value_range && weight==other.weight && score==other.score);
        }   
        bool operator!=(const tag_hyperplane & other) const
        {
            return !(*this==other);
        }
        //arithmetic operators
        tag_hyperplane& operator+(const tag_hyperplane & other)
        {
           this->hplane_indices=this->hplane_indices+other.hplane_indices;  

           return *this;
        }   
        tag_hyperplane& operator-(const tag_hyperplane& other)
        {
            this->hplane_indices-=other.hplane_indices;
            return *this;
        }
        tag_hyperplane& operator*(const tag_hyperplane& other)
        {
            this->hplane_indices*=other.hplane_indices;
            return *this;
        }
        tag_hyperplane& operator/(const tag_hyperplane& other)
        {
            this->hplane_indices/=other.hplane_indices;
            return *this;
        }
        //logical operators
        bool operator!() const
        {
            return (hplane_id==0 && hplane_indices==matrix_indices(0,0) && hplane_depth==0 && hplane_level==0 && hplane_parent==0 && hplane_left==0 && hplane_right==0 && hplane_dim==0 && hplane_feature==0 && hplane_feature_index==0 && hplane_feature_index_left==0 && hplane_feature_index_right==0 && hplane_feature_value==0.0 && hplane_feature_value_left==0.0 && hplane_feature_value_right==0.0 && hplane_feature_value_min==0.0 && hplane_feature_value_max==0.0 && hplane_feature_value_range==0.0 && weight==0.0 && score==0.0);
        }

        void print(std::ostream& os)const 
        {
            os<<"hplane_id="<<hplane_id<<std::endl;
            os<<"hplane_indices="<<hplane_indices<<std::endl;
            os<<"hplane_depth="<<hplane_depth<<std::endl;
            os<<"hplane_level="<<hplane_level<<std::endl;
            os<<"hplane_parent="<<hplane_parent<<std::endl;

            os<<"hplane_left="<<hplane_left<<std::endl;
            os<<"hplane_right="<<hplane_right<<std::endl;


            os<<"hplane_dim="<<hplane_dim<<std::endl;
            os<<"hplane_feature="<<hplane_feature<<std::endl;
            os<<"hplane_feature_index="<<hplane_feature_index<<std::endl;

            os<<"hplane_feature_index_left="<<hplane_feature_index_left<<std::endl;
            os<<"hplane_feature_index_right="<<hplane_feature_index_right<<std::endl;








            os<<"hplane_feature_value="<<hplane_feature_value<<std::endl;
            os<<"hplane_feature_value_left="<<hplane_feature_value_left<<std::endl;

            os<<"hplane_feature_value_right="<<hplane_feature_value_right<<std::endl;
            os<<"hplane_feature_value_min="<<hplane_feature_value_min<<std::endl;
            os<<"hplane_feature_value_max="<<hplane_feature_value_max<<std::endl;
            os<<"hplane_feature_value_range="<<hplane_feature_value_range<<std::endl;
            os<<"weight="<<weight<<std::endl;
            os<<"score="<<score<<std::endl;
            os<<"distribution="<<distribution<<std::endl;

            
        }
        //operator <<
        friend std::ostream & operator<<(std::ostream & os , const tag_hyperplane& other )
        {
            
            other.print(os);
            return os;
                        
       }


        static std::atomic_uint64_t hplane_count;

	} hplane; 
    //hyperplane  

template < typename T , typename U = std::vector<T> >   
class super_tree {
      private:
        std::vector<U> _forest; 
        provallo::matrix<matrix_indices> _super_tree;
        provallo::matrix<real_t> _super_tree_probabilities;
        provallo::matrix<T> _super_tree_values;
        provallo::matrix<T> _super_tree_values_projection;
        provallo::matrix<provallo::hplane> _super_tree_hplane;

        //metrics

        public:
        //constructors
                
        super_tree(const std::vector<U> & forest):
        _forest(forest),
        _super_tree(forest.size(),forest[0].size()),
        _super_tree_probabilities(forest.size(),
        forest[0].size()),
        _super_tree_values(forest.size(),forest[0].size()),
        _super_tree_values_projection(forest.size(),forest[0].size()),
        _super_tree_hplane(forest.size(),forest[0].size())  {
            for(size_t i=0;i<forest.size();i++)
            {
                for(size_t j=0;j<forest[i].size();j++)
                {
                    _super_tree(i,j)=matrix_indices(i,j);
                    _super_tree_probabilities(i,j)=0.0;
                    _super_tree_values(i,j)=forest[i][j];
                    _super_tree_values_projection(i,j)=forest[i][j];
                    _super_tree_hplane(i,j).hplane_indices=matrix_indices(i,j); 
                    _super_tree_hplane(i,j).hplane_id=0;
                    _super_tree_hplane(i,j).hplane_depth=0;
                    _super_tree_hplane(i,j).hplane_level=0;
                    _super_tree_hplane(i,j).hplane_parent=0;
                    _super_tree_hplane(i,j).hplane_left=0;
                    _super_tree_hplane(i,j).hplane_right=0;
                    _super_tree_hplane(i,j).hplane_dim=0;
                    _super_tree_hplane(i,j).hplane_feature=0;
                    _super_tree_hplane(i,j).hplane_feature_index=0;
                    _super_tree_hplane(i,j).hplane_feature_index_left=0;
                    _super_tree_hplane(i,j).hplane_feature_index_right=0;
                    _super_tree_hplane(i,j).hplane_feature_value=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_left=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_right=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_min=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_max=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_range=0.0;
                    _super_tree_hplane(i,j).weight=0.0;
                    _super_tree_hplane(i,j).score=0.0;
                    _super_tree_hplane(i,j).distribution=std::uniform_real_distribution<real_t>(0.0,1.0); 
                     
                }
            }
        }   
        super_tree(const std::vector<U> & forest,const provallo::matrix<matrix_indices> & super_tree):_forest(forest),_super_tree(super_tree)
        {
            _super_tree_probabilities.resize(forest.size(),forest[0].size());
            _super_tree_values.resize(forest.size(),forest[0].size());
            _super_tree_values_projection.resize(forest.size(),forest[0].size());
            _super_tree_hplane.resize(forest.size(),forest[0].size());
            for(size_t i=0;i<forest.size();i++)
            {
                for(size_t j=0;j<forest[i].size();j++)
                {
                    _super_tree_probabilities(i,j)=0.0;
                    _super_tree_values(i,j)=forest[i][j];
                    _super_tree_values_projection(i,j)=forest[i][j];
                    _super_tree_hplane(i,j).hplane_indices=matrix_indices(i,j); 
                    _super_tree_hplane(i,j).hplane_id=0;
                    _super_tree_hplane(i,j).hplane_depth=0;
                    _super_tree_hplane(i,j).hplane_level=0;
                    _super_tree_hplane(i,j).hplane_parent=0;
                    _super_tree_hplane(i,j).hplane_left=0;
                    _super_tree_hplane(i,j).hplane_right=0;
                    _super_tree_hplane(i,j).hplane_dim=0;
                    _super_tree_hplane(i,j).hplane_feature=0;
                    _super_tree_hplane(i,j).hplane_feature_index=0;
                    _super_tree_hplane(i,j).hplane_feature_index_left=0;
                    _super_tree_hplane(i,j).hplane_feature_index_right=0;
                    _super_tree_hplane(i,j).hplane_feature_value=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_left=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_right=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_min=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_max=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_range=0.0;
                    _super_tree_hplane(i,j).weight=0.0;
                    _super_tree_hplane(i,j).score=0.0;
                    _super_tree_hplane(i,j).distribution=std::uniform_real_distribution<real_t>(0.0,1.0);

                }

            }
        }   
        super_tree(const std::vector<U> & forest,const provallo::matrix<matrix_indices> & super_tree,const provallo::matrix<real_t> & super_tree_probabilities):_forest(forest),_super_tree(super_tree),_super_tree_probabilities(super_tree_probabilities)
        {
            _super_tree_values.resize(forest.size(),forest[0].size());
            _super_tree_values_projection.resize(forest.size(),forest[0].size());
            _super_tree_hplane.resize(forest.size(),forest[0].size());
            for(size_t i=0;i<forest.size();i++)
            {
                for(size_t j=0;j<forest[i].size();j++)
                {
                    _super_tree_values(i,j)=forest[i][j];
                    _super_tree_values_projection(i,j)=forest[i][j];
                    _super_tree_hplane(i,j).hplane_indices=matrix_indices(i,j); 
                    _super_tree_hplane(i,j).hplane_id=0;
                    _super_tree_hplane(i,j).hplane_depth=0;
                    _super_tree_hplane(i,j).hplane_level=0;
                    _super_tree_hplane(i,j).hplane_parent=0;
                    _super_tree_hplane(i,j).hplane_left=0;
                    _super_tree_hplane(i,j).hplane_right=0;
                    _super_tree_hplane(i,j).hplane_dim=0;
                    _super_tree_hplane(i,j).hplane_feature=0;
                    _super_tree_hplane(i,j).hplane_feature_index=0;
                    _super_tree_hplane(i,j).hplane_feature_index_left=0;
                    _super_tree_hplane(i,j).hplane_feature_index_right=0;
                    _super_tree_hplane(i,j).hplane_feature_value=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_left=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_right=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_min=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_max=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_range=0.0;
                    _super_tree_hplane(i,j).weight=0.0;
                    _super_tree_hplane(i,j).score=0.0;
                    _super_tree_hplane(i,j).distribution=std::uniform_real_distribution<real_t>(0.0,1.0);
                }
            }
        }   
        super_tree(const std::vector<U> & forest,const provallo::matrix<matrix_indices> & super_tree,const provallo::matrix<real_t> & super_tree_probabilities,const provallo::matrix<T> & super_tree_values):_forest(forest),_super_tree(super_tree),_super_tree_probabilities(super_tree_probabilities),_super_tree_values(super_tree_values)
        {
            _super_tree_values_projection.resize(forest.size(),forest[0].size());
            _super_tree_hplane.resize(forest.size(),forest[0].size());
            for(size_t i=0;i<forest.size();i++)
            {
                for(size_t j=0;j<forest[i].size();j++)
                {
                    _super_tree_values_projection(i,j)=forest[i][j];
                    _super_tree_hplane(i,j).hplane_indices=matrix_indices(i,j); 
                    _super_tree_hplane(i,j).hplane_id=0;
                    _super_tree_hplane(i,j).hplane_depth=0;
                    _super_tree_hplane(i,j).hplane_level=0;
                    _super_tree_hplane(i,j).hplane_parent=0;
                    _super_tree_hplane(i,j).hplane_left=0;
                    _super_tree_hplane(i,j).hplane_right=0;
                    _super_tree_hplane(i,j).hplane_dim=0;
                    _super_tree_hplane(i,j).hplane_feature=0;
                    _super_tree_hplane(i,j).hplane_feature_index=0;
                    _super_tree_hplane(i,j).hplane_feature_index_left=0;
                    _super_tree_hplane(i,j).hplane_feature_index_right=0;
                    _super_tree_hplane(i,j).hplane_feature_value=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_left=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_right=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_min=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_max=0.0;
                    _super_tree_hplane(i,j).hplane_feature_value_range=0.0;
                    _super_tree_hplane(i,j).weight=0.0;
                    _super_tree_hplane(i,j).score=0.0;
                    _super_tree_hplane(i,j).distribution=std::uniform_real_distribution<real_t>(0.0,1.0);
                }
            }
        }   
        super_tree(const std::vector<U> & forest,const provallo::matrix<matrix_indices> & super_tree,const provallo::matrix<real_t> & super_tree_probabilities,const provallo::matrix<T> & super_tree_values,const provallo::matrix<T> & super_tree_values_projection,const provallo::matrix<provallo::hplane> & super_tree_hplane):_forest(forest),_super_tree(super_tree),_super_tree_probabilities(super_tree_probabilities),_super_tree_values(super_tree_values),_super_tree_values_projection(super_tree_values_projection),_super_tree_hplane(super_tree_hplane){}
        //copy constructor
        super_tree(const super_tree & other):_forest(other._forest),_super_tree(other._super_tree),_super_tree_probabilities(other._super_tree_probabilities),_super_tree_values(other._super_tree_values),_super_tree_values_projection(other._super_tree_values_projection),_super_tree_hplane(other._super_tree_hplane){}
        //move constructor
        super_tree(super_tree && other):_forest(std::move(other._forest)),_super_tree(std::move(other._super_tree)),_super_tree_probabilities(std::move(other._super_tree_probabilities)),_super_tree_values(std::move(other._super_tree_values)),_super_tree_values_projection(std::move(other._super_tree_values_projection)),_super_tree_hplane(std::move(other._super_tree_hplane)){}
        //copy assignment

        super_tree & operator=(const super_tree & other)
        {
            if(this!=&other)
            {
                _forest=other._forest;
                _super_tree=other._super_tree;
                _super_tree_probabilities=other._super_tree_probabilities;
                _super_tree_values=other._super_tree_values;
                _super_tree_values_projection=other._super_tree_values_projection;
                _super_tree_hplane=other._super_tree_hplane;
            }
            return *this;
        }
        //move assignment
        super_tree & operator=(super_tree && other)
        {
            if(this!=&other)
            {
                _forest=std::move(other._forest);
                _super_tree=std::move(other._super_tree);
                _super_tree_probabilities=std::move(other._super_tree_probabilities);
                _super_tree_values=std::move(other._super_tree_values);
                _super_tree_values_projection=std::move(other._super_tree_values_projection);
                _super_tree_hplane=std::move(other._super_tree_hplane);
            }
            return *this;
        }

        //destructor
        virtual ~super_tree() = default;
        //getters
        const std::vector<U> & forest() const {return _forest;}
        std::vector<U> & forest() {return _forest;}
        const provallo::matrix<matrix_indices> & get_super_tree() const {return _super_tree;}
        provallo::matrix<matrix_indices> & get_super_tree() {return _super_tree;}
        const provallo::matrix<real_t> & super_tree_probabilities() const {return _super_tree_probabilities;}
        provallo::matrix<real_t> & super_tree_probabilities() {return _super_tree_probabilities;}
        const provallo::matrix<T> & super_tree_values() const {return _super_tree_values;}
        provallo::matrix<T> & super_tree_values() {return _super_tree_values;}
        const provallo::matrix<T> & super_tree_values_projection() const {return _super_tree_values_projection;}
        provallo::matrix<T> & super_tree_values_projection() {return _super_tree_values_projection;}
         
        const provallo::matrix<provallo::hplane> & super_tree_hplane() const {return _super_tree_hplane;}
        provallo::matrix<provallo::hplane> & super_tree_hplane() {return _super_tree_hplane;}
        //setters
        void set_forest(const std::vector<U> & forest) {_forest=forest;}
        void set_super_tree(const provallo::matrix<matrix_indices> & super_tree) {_super_tree=super_tree;}
        void set_super_tree_probabilities(const provallo::matrix<real_t> & super_tree_probabilities) {_super_tree_probabilities=super_tree_probabilities;}
        void set_super_tree_values(const provallo::matrix<T> & super_tree_values) {_super_tree_values=super_tree_values;}
        void set_super_tree_values_projection(const provallo::matrix<T> & super_tree_values_projection) {_super_tree_values_projection=super_tree_values_projection;}
        void set_super_tree_hplane(const provallo::hplane & super_tree_hplane) {_super_tree_hplane=super_tree_hplane;}
        //operators
        bool operator==(const super_tree & other) const
        {
            return (_forest==other._forest && _super_tree==other._super_tree && _super_tree_probabilities==other._super_tree_probabilities && _super_tree_values==other._super_tree_values && _super_tree_values_projection==other._super_tree_values_projection && _super_tree_hplane==other._super_tree_hplane);
        }       
        bool operator!=(const super_tree & other) const
        {
            return !(*this==other);
        }

        //methods
        void print(std::ostream & os=std::cout) const
        {
            os<<"forest:"<<std::endl;
            for(size_t i=0;i<_forest.size();i++)
            {
                os<<"tree "<<i<<":";
                for(size_t j=0;j<_forest[i].size();j++)
                {
                    os<<_forest[i][j]<<" ";
                }
                os<<std::endl;
            }
            os<<"super_tree:"<<std::endl;
            _super_tree.print(os);
            os<<"super_tree_probabilities:"<<std::endl;
            _super_tree_probabilities.print(os);
            os<<"super_tree_values:"<<std::endl;
            _super_tree_values.print(os);
            os<<"super_tree_values_projection:"<<std::endl;
            _super_tree_values_projection.print(os);
            //os<<"super_tree_hplane:"<<std::endl;
           // _super_tree_hplane.print(os);
        }

        //transform the super_tree into a matrix of probabilities
        //and set the hplane values according to random projections 
        //of the super_tree values

       inline void initialize_hplanes()
       {
            //initialize hplanes
            //set hplane values according to random projections of the super_tree values
            //set hplane values according to random projections of the super_tree values


            //initialize hplanes
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);
            std::uniform_int_distribution<> dis_rows(0, _super_tree_values.rows()-1);
            std::uniform_int_distribution<> dis_cols(0, _super_tree_values.cols()-1);   
            std::uniform_int_distribution<> dis_levels(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_depths(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_parents(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_lefts(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_rights(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_dims(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_features(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_indices(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_indices_left(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_indices_right(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values_left(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values_right(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values_min(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values_max(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values_range(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_weights(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_scores(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_probabilities(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_values(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_values_projection(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_hplanes(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_forest(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_super_tree(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_super_tree_probabilities(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_super_tree_values(0, _super_tree_values.cols()-1);


            //project the super_tree values into the hplane values 
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    _super_tree_values_projection(i,j)=_super_tree_values(i,j)*_super_tree_hplane(i,j).hplane_feature_value;
                }
            }
            //set hplane values according to random projections of the super_tree values
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    _super_tree_hplane(i,j).hplane_id=dis_hplanes(gen);
                    _super_tree_hplane(i,j).hplane_indices=matrix_indices(dis_rows(gen),dis_cols(gen));
                    _super_tree_hplane(i,j).hplane_depth=dis_depths(gen);
                    _super_tree_hplane(i,j).hplane_level=dis_levels(gen);
                    _super_tree_hplane(i,j).hplane_parent=dis_parents(gen);
                    _super_tree_hplane(i,j).hplane_left=dis_lefts(gen);
                    _super_tree_hplane(i,j).hplane_right=dis_rights(gen);
                    _super_tree_hplane(i,j).hplane_dim=dis_dims(gen);
                    _super_tree_hplane(i,j).hplane_feature=dis_features(gen);
                    _super_tree_hplane(i,j).hplane_feature_index=dis_feature_indices(gen);
                    _super_tree_hplane(i,j).hplane_feature_index_left=dis_feature_indices_left(gen);
                    _super_tree_hplane(i,j).hplane_feature_index_right=dis_feature_indices_right(gen);
                    _super_tree_hplane(i,j).hplane_feature_value=dis_feature_values(gen);
                    _super_tree_hplane(i,j).hplane_feature_value_left=dis_feature_values_left(gen);
                    _super_tree_hplane(i,j).hplane_feature_value_right=dis_feature_values_right(gen);
                    _super_tree_hplane(i,j).hplane_feature_value_min=dis_feature_values_min(gen);
                    _super_tree_hplane(i,j).hplane_feature_value_max=dis_feature_values_max(gen);
                    _super_tree_hplane(i,j).hplane_feature_value_range=dis_feature_values_range(gen);
                    _super_tree_hplane(i,j).weight=dis_weights(gen);
                    _super_tree_hplane(i,j).score=dis_scores(gen);
                }
            }
            
            
       }
        inline void initialize_probabilities()
        {
            //initialize probabilities
            //set probabilities according to random projections of the super_tree values
            //set probabilities according to random projections of the super_tree values    
            //initialize probabilities
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);
            std::uniform_real_distribution<> dis_scores(0.0, 1.0);
            std::uniform_real_distribution<> dis_weights(0.0, 1.0);
            std::uniform_int_distribution<> dis_rows(0, _super_tree_values.rows()-1);
            std::uniform_int_distribution<> dis_cols(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_levels(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_depths(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_parents(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_lefts(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_rights(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_dims(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_features(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_indices(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_indices_left(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_indices_right(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values_left(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values_right(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values_min(0, _super_tree_values.cols()-1);
            std::uniform_int_distribution<> dis_feature_values_max(0, _super_tree_values.cols()-1); 
            real_t sum = 0.0;

            //  //project the super_tree values into the probabilities
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {   
                    //get the hplane of the super_tree
                    hplane& hplane=_super_tree_hplane(i,j);
                    //update hplane values according to random projections of the super_tree values 
                    hplane.hplane_feature_value=_super_tree_values(i,j)*dis_feature_values(gen); 
                    hplane.hplane_depth=dis_depths(gen);
                    hplane.hplane_level=dis_levels(gen);
                    hplane.hplane_parent=dis_parents(gen);
                    hplane.hplane_left=dis_lefts(gen);
                    hplane.hplane_right=dis_rights(gen);
                    hplane.hplane_dim=dis_dims(gen);
                    hplane.hplane_feature=dis_features(gen);
                    hplane.hplane_feature_index=dis_feature_indices(gen);
                    hplane.hplane_feature_index_left=dis_feature_indices_left(gen);
                    hplane.hplane_feature_index_right=dis_feature_indices_right(gen);
                    hplane.hplane_feature_value_left=dis_feature_values_left(gen);
                    hplane.hplane_feature_value_right=dis_feature_values_right(gen);
                    hplane.hplane_feature_value_min=dis_feature_values_min(gen);
                    hplane.hplane_feature_value_max=dis_feature_values_max(gen);
                     //update probabilities according to random projections of the super_tree values
                    _super_tree_probabilities(i,j)=_super_tree_values(i,j)*hplane.hplane_feature_value;
                    hplane.weight=dis_weights(gen);

                    hplane.score=dis_scores(gen)/super_tree_values(i,j);
                    sum+=_super_tree_probabilities(i,j);

                 }
                 //normalize probabilities
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    _super_tree_probabilities(i,j)/=sum;
                }
            }       
             //normalize probabilities
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                real_t sum = 0.0;
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    sum+=_super_tree_probabilities(i,j);
                }
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    _super_tree_probabilities(i,j)/=sum;
                }

            }

            //for each tree in the forest update the probabilities of the leaves 
            //according to the probabilities of the super_tree

            for(size_t i=0;i<_forest.size();i++)
            {
                for(size_t j=0;j<_forest[i].size();j++)
                {
                    _forest[i][j]=_super_tree_probabilities(i,j);
                    sum+=_forest[i][j];

                }
            }   
            //for each tree in the forest normalize the probabilities of the leaves 
            //according to the probabilities of the super_tree  
            for(size_t i=0;i<_forest.size();i++)
            {
           
                for(size_t j=0;j<_forest[i].size();j++)
                {
                    _forest[i][j]/=sum;
                }
            }   
           //normalize probabilities
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    _super_tree_probabilities(i,j)/=sum;
                }

            }
            //for each node in the super_tree update the probabilities of the leaves
            //according to the probabilities of the super_tree
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    _super_tree(i,j)=_super_tree_probabilities(i,j);
                }
            }
            //for each node in the super_tree normalize the probabilities of the leaves
            //according to the probabilities of the super_tree
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                real_t sum = 0.0;
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    sum+=_super_tree_values(i,j);
                }
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    _super_tree_values(i,j)/=sum;
                }
            }             
        }
        real_t get_leaf_probability(const size_t & i,const size_t & j) const
        {
            return _super_tree_probabilities(i,j);
        }   
        real_t get_leaf_value(const size_t & i,const size_t & j) const
        {
            return _super_tree_values(i,j);
        }   
        real_t get_leaf_value_projection(const size_t & i,const size_t & j) const
        {
            return _super_tree_values_projection(i,j);
        }   
        real_t get_leaf_hplane(const size_t & i,const size_t & j) const
        {
            return _super_tree_hplane(i,j).hplane_feature_value;
        }   
        real_t get_leaf_forest(const size_t & i,const size_t & j) const
        {
            return _forest[i][j];
        }   
        real_t get_leaf_super_tree(const size_t & i,const size_t & j) const
        {
            return _super_tree_values(i,j);
        }
        //get trees,nodes,leaves
        std::vector<U> get_trees() const
        {
            return _forest;
        }   
        std::vector<T> get_nodes() const
        {
            std::vector<T> nodes;
            for(size_t i=0;i<_forest.size();i++)
            {
                for(size_t j=0;j<_forest[i].size();j++)
                {
                    nodes.push_back(_forest[i][j]);
                }
            }
            return nodes;
        }   
        std::vector<T> get_leaves() const
        {
            std::vector<T> leaves;
            for(size_t i=0;i<_forest.size();i++)
            {
                for(size_t j=0;j<_forest[i].size();j++)
                {
                    leaves.push_back(_forest[i][j]);
                }
            }
            return leaves;
        }   
        //get super_tree,nodes,leaves
        
        provallo::matrix<T> get_super_tree_nodes() const
        {
            return _super_tree_values;
        }
        provallo::matrix<T> get_super_tree_leaves() const
        {
            return _super_tree_values;
        }
        //get super_tree_probabilities,nodes,leaves
        //same probabilities for nodes and leaves
        provallo::matrix<real_t> get_super_tree_probabilities() const
        {
            return _super_tree_probabilities;
        }
        provallo::matrix<real_t> get_super_tree_nodes_probabilities() const
        {
            return _super_tree_probabilities;
        }   
        provallo::matrix<real_t> get_super_tree_leaves_probabilities() const
        {
            return _super_tree_probabilities;
        }   
        //get super_tree_values,nodes,leaves
        provallo::matrix<T> get_super_tree_values() const
        {
            return _super_tree_values;
        }   
        provallo::matrix<T> get_super_tree_nodes_values() const
        {
            return _super_tree_values;
        }
        provallo::matrix<T> get_super_tree_leaves_values() const
        {
            return _super_tree_values;
        }
        //get super_tree_values_projection,nodes,leaves
        provallo::matrix<T> get_super_tree_values_projection() const
        {
            return _super_tree_values_projection;
        }
        void process_hplanes()
        {
            //calculate hplane values for each node,leaf and tree:
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    _super_tree_hplane(i,j).hplane_feature_value=_super_tree_values(i,j);
                    _super_tree_hplane(i,j).hplane_dim=i;
                    _super_tree_hplane(i,j).hplane_feature=j;
                    _super_tree_hplane(i,j).hplane_feature_index=i;
                    _super_tree_hplane(i,j).hplane_feature_index_left=i;
                    _super_tree_hplane(i,j).hplane_feature_index_right=i;
                    _super_tree_hplane(i,j).hplane_feature_value_left=_super_tree_values(i,j);
                    _super_tree_hplane(i,j).hplane_feature_value_right=_super_tree_values(i,j);
                    _super_tree_hplane(i,j).hplane_feature_value_min=_super_tree_values(i,j);
                    _super_tree_hplane(i,j).hplane_feature_value_max=_super_tree_values(i,j);
                    _super_tree_hplane(i,j).hplane_feature_value_range=_super_tree_values(i,j);
                    _super_tree_hplane(i,j).weight=_super_tree_values(i,j);
                    _super_tree_hplane(i,j).score= _super_tree_values_projection(i,j)/_super_tree_values(i,j);
                    real_t projected_value = _super_tree_values_projection(i,j); 
                    real_t hyperplane_projected_value= _super_tree_hplane(i,j).hplane_feature_value;
 
                    real_t projection_intersection = _super_tree_hplane(i,j).distribution(projected_value);
                    real_t hyperplane_projected_intersection = _super_tree_hplane(i,j).distribution(hyperplane_projected_value); 

                     //update the probabilities 
                    _super_tree_probabilities(i,j)=projected_value/_super_tree_values(i,j); 
                    //update the hplane values according to the probabilities
                    _super_tree_hplane(i,j).hplane_feature_value=projected_value;
                    _super_tree_hplane(i,j).hplane_feature_value_left=projected_value;
                    _super_tree_hplane(i,j).hplane_feature_value_right=projected_value;
                    //update the super_tree projections according to the hplane values 
                    _super_tree_values_projection(i,j)=projected_value; 
                    _super_tree_hplane(i,j).hplane_feature_value=projected_value;
                    _super_tree_hplane(i,j).hplane_feature_value_left=projected_value;
                    _super_tree_hplane(i,j).hplane_feature_value_right=projected_value;
                    _super_tree_hplane(i,j).hplane_feature_value_min=projection_intersection; 
                    _super_tree_hplane(i,j).hplane_feature_value_max=hyperplane_projected_intersection;
                    _super_tree_hplane(i,j).hplane_feature_value_range=hyperplane_projected_intersection-projection_intersection; 
                    //update weight
                    matrix<matrix_indices> hplane_indices=_super_tree_hplane(i,j).hplane_indices; 
                    _super_tree_hplane(i,j).weight=_super_tree_values(i,j)*_super_tree_hplane(i,j).hplane_feature_value_range; 
                    _super_tree_hplane(i,j).score= _super_tree_values_projection(i,j)/_super_tree_values(i,j);
                    _super_tree_hplane(i,j).hplane_feature_value=projected_value;
                    _super_tree_hplane(i,j).hplane_feature_value_left=projected_value;


                }
            }   
         
        }//process_hplanes
        //get super_tree_hplane,nodes,leaves
        provallo::matrix<provallo::hplane> get_super_tree_hplane() const
        {
            return _super_tree_hplane;
        }
        provallo::matrix<provallo::hplane> get_super_tree_nodes_hplane() const
        {
            return _super_tree_hplane;
        }
        provallo::matrix<provallo::hplane> get_super_tree_leaves_hplane() const
        {
            return _super_tree_hplane;
        }
        //get super_tree_forest,nodes,leaves
        std::vector<std::vector<U>> get_super_tree_forest() const
        {
            std::vector<std::vector<U>> super_tree_forest;
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                std::vector<U> tree;
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    tree.push_back(_super_tree(i,j));
                }
                super_tree_forest.push_back(tree);
            }
            return super_tree_forest;
        }
        std::vector<U> get_super_tree_nodes_forest() const
        {
            std::vector<U> super_tree_nodes_forest;
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    super_tree_nodes_forest.push_back(_super_tree(i,j));
                }
            }
            return super_tree_nodes_forest;
        }
        std::vector<U> get_super_tree_leaves_forest() const
        {
            std::vector<U> super_tree_leaves_forest;
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    super_tree_leaves_forest.push_back(_super_tree(i,j));
                }
            }
            return super_tree_leaves_forest;
        }
        //get super_tree_probabilities_forest,nodes,leaves
        std::vector<std::vector<real_t>> get_super_tree_probabilities_forest() const
        {
            std::vector<std::vector<real_t>> super_tree_probabilities_forest;
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                std::vector<real_t> tree;
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    tree.push_back(_super_tree_probabilities(i,j));
                }
                super_tree_probabilities_forest.push_back(tree);
            }
            return super_tree_probabilities_forest;
        }
        //get super_tree_values_forest,nodes,leaves
        std::vector<std::vector<T>> get_super_tree_values_forest() const
        {
            std::vector<std::vector<T>> super_tree_values_forest;
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                std::vector<T> tree;
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    tree.push_back(_super_tree_values(i,j));
                }
                super_tree_values_forest.push_back(tree);
            }
            return super_tree_values_forest;
        }
        //get super_tree_values_projection_forest,nodes,leaves
        std::vector<std::vector<T>> get_super_tree_values_projection_forest() const
        {
            std::vector<std::vector<T>> super_tree_values_projection_forest;
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                std::vector<T> tree;
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    tree.push_back(_super_tree_values_projection(i,j));
                }
                super_tree_values_projection_forest.push_back(tree);
            }
            return super_tree_values_projection_forest;
        }
        //get super_tree_hplane_forest,nodes,leaves
        std::vector<std::vector<provallo::hplane>> get_super_tree_hplane_forest() const
        {
            std::vector<std::vector<provallo::hplane>> super_tree_hplane_forest;
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                std::vector<provallo::hplane> tree;
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    tree.push_back(_super_tree_hplane(i,j));
                }
                super_tree_hplane_forest.push_back(tree);
            }
            return super_tree_hplane_forest;
        }
        //get super_tree_forest,nodes,leaves
        std::vector<std::vector<U>> get_super_tree_forest(const size_t & i) const
        {
            std::vector<std::vector<U>> super_tree_forest;
            for(size_t j=0;j<_super_tree_values.cols();j++)
            {
                std::vector<U> tree;
                tree.push_back(_super_tree(i,j));
                super_tree_forest.push_back(tree);
            }
            return super_tree_forest;
        }
        std::vector<U> get_super_tree_nodes_forest(const size_t & i) const
        {
            std::vector<U> super_tree_nodes_forest;
            for(size_t j=0;j<_super_tree_values.cols();j++)
            {
                super_tree_nodes_forest.push_back(_super_tree(i,j));
            }
            return super_tree_nodes_forest;
        }   
        std::vector<U> get_super_tree_leaves_forest(const size_t & i) const
        {
            std::vector<U> super_tree_leaves_forest;
            for(size_t j=0;j<_super_tree_values.cols();j++)
            {
                super_tree_leaves_forest.push_back(_super_tree(i,j));
            }
            return super_tree_leaves_forest;
        }   
        //test hyperplane value projection quality
        //return the average error
        inline real_t test_projection_quality()
        {
            std::random_device rd;
            std::mt19937 gen(rd());

            if(this->_forest.size()>0)
            {   size_t tree=0;
                for(auto& tree : this->_forest)
                {
                    for(auto& node : tree)
                    {
                        //get the hplane of the super_tree
                        hplane& hplane=this->_super_tree_hplane(tree,node);


                        //get the projection of the super_tree value
                        real_t projected_value = this->_super_tree_values_projection(tree,node); 
                        //get the hyperplane projection of the super_tree value
                        real_t hyperplane_projected_value= hplane.hplane_feature_value;
                        //get the intersection of the projection and the hyperplane projection
                        real_t projection_intersection = hplane.distribution(projected_value);
                        real_t hyperplane_projected_intersection = hplane.distribution(hyperplane_projected_value);
                        //get the error
                        real_t error = hyperplane_projected_intersection-projection_intersection;
                        //update the error
                        this->_super_tree_hplane(tree,node).hplane_feature_value_min=projection_intersection;
                        this->_super_tree_hplane(tree,node).hplane_feature_value_max=hyperplane_projected_intersection;
                        this->_super_tree_hplane(tree,node).hplane_feature_value_range=hyperplane_projected_intersection-projection_intersection;
                        //update the probabilities
                        this->_super_tree_probabilities(tree,node)=projected_value/this->_super_tree_values(tree,node);
                        //update the hplane values according to the probabilities
                        this->_super_tree_hplane(tree,node).hplane_feature_value=projected_value;
                        this->_super_tree_hplane(tree,node).hplane_feature_value_left=projected_value;
                        this->_super_tree_hplane(tree,node).hplane_feature_value_right=projected_value;
                        //update the super_tree projections according to the hplane values
                        this->_super_tree_values_projection(tree,node)=projected_value;

                    }
                }
            }
            else
            {
                for(size_t tree=0;tree<this->_super_tree_values.rows();tree++)
                {
                    for(size_t node=0;node<this->_super_tree_values.cols();node++)
                    {
                        //get the hplane of the super_tree
                        hplane& hplane=this->_super_tree_hplane(tree,node);
                        //get the projection of the super_tree value
                        real_t projected_value = this->_super_tree_values_projection(tree,node);
                        //get the hyperplane projection of the super_tree value
                        real_t hyperplane_projected_value= hplane.hplane_feature_value;
                        //get the intersection of the projection and the hyperplane projection
                        real_t projection_intersection = hplane.distribution(projected_value);

                        real_t hyperplane_projected_intersection = hplane.distribution(hyperplane_projected_value);
                        //get the error
                        real_t error = hyperplane_projected_intersection-projection_intersection;
                        //update the error
                        this->_super_tree_hplane(tree,node).hplane_feature_value_min=projection_intersection;
                        this->_super_tree_hplane(tree,node).hplane_feature_value_max=hyperplane_projected_intersection;

                        this->_super_tree_hplane(tree,node).hplane_feature_value_range=hyperplane_projected_intersection-projection_intersection;
                        //update the probabilities
                        this->_super_tree_probabilities(tree,node)=projected_value/this->_super_tree_values(tree,node);
                        //update the hplane values according to the probabilities
                        this->_super_tree_hplane(tree,node).hplane_feature_value=projected_value;
                        this->_super_tree_hplane(tree,node).hplane_feature_value_left=projected_value;
                        this->_super_tree_hplane(tree,node).hplane_feature_value_right=projected_value;
                        //update the super_tree projections according to the hplane values
                        this->_super_tree_values_projection(tree,node)=projected_value;

                    }
                }

            }
            //calculate the average error
            real_t average_error=0.0;
            if(this->_forest.size()>0)
            {   size_t tree=0;

                for(auto& tree : this->_forest)
                {
                    for(auto& node : tree)
                    {
                        //get the hplane of the super_tree

                        hplane& hplane=this->_super_tree_hplane(tree,node);
                        //get the projection of the super_tree value
                        real_t projected_value = this->_super_tree_values_projection(tree,node);
                        //get the hyperplane projection of the super_tree value
                        real_t hyperplane_projected_value= hplane.hplane_feature_value;
                        //get the intersection of the projection and the hyperplane projection
                        real_t projection_intersection = hplane.distribution(projected_value);

                        real_t hyperplane_projected_intersection = hplane.distribution(hyperplane_projected_value);
                        //get the error
                        real_t error = hyperplane_projected_intersection-projection_intersection;
                        //update the error
                        this->_super_tree_hplane(tree,node).hplane_feature_value_min=projection_intersection;
                        this->_super_tree_hplane(tree,node).hplane_feature_value_max=hyperplane_projected_intersection;
                        //update the average error
                        average_error+=error;
                    }
                }
            }
            else
            {
                for(size_t tree=0;tree<this->_super_tree_values.rows();tree++)
                {
                    for(size_t node=0;node<this->_super_tree_values.cols();node++)
                    {
                        //get the hplane of the super_tree
                        hplane& hplane=this->_super_tree_hplane(tree,node);
                        //get the projection of the super_tree value
                        real_t projected_value = this->_super_tree_values_projection(tree,node);
                        //get the hyperplane projection of the super_tree value
                        real_t hyperplane_projected_value= hplane.hplane_feature_value;
                        //get the intersection of the projection and the hyperplane projection
                        real_t projection_intersection = hplane.distribution(projected_value);

                        real_t hyperplane_projected_intersection = hplane.distribution(hyperplane_projected_value);
                        //get the error
                        real_t error = hyperplane_projected_intersection-projection_intersection;
                        //update the error
                        this->_super_tree_hplane(tree,node).hplane_feature_value_min=projection_intersection;
                        this->_super_tree_hplane(tree,node).hplane_feature_value_max=hyperplane_projected_intersection;
                        //update the average error
                        average_error+=error;
                    }
                }
            }
            //calculate the average error
            average_error/=(this->_super_tree_values.rows()*this->_super_tree_values.cols());
            return average_error;
        }//test_projection_quality
        //test hyperplane value projection quality
        //return the average error
        inline real_t test_projection_quality(const size_t & tree,const size_t & node)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            //get the hplane of the super_tree
            hplane& hplane=this->_super_tree_hplane(tree,node);
            //get the projection of the super_tree value
            real_t projected_value = this->_super_tree_values_projection(tree,node);
            //get the hyperplane projection of the super_tree value
            real_t hyperplane_projected_value= hplane.hplane_feature_value;
            //get the intersection of the projection and the hyperplane projection
            real_t projection_intersection = hplane.distribution(projected_value);

            real_t hyperplane_projected_intersection = hplane.distribution(hyperplane_projected_value);
            //get the error
            real_t error = hyperplane_projected_intersection-projection_intersection;
            //update the error
            this->_super_tree_hplane(tree,node).hplane_feature_value_min=projection_intersection;
            this->_super_tree_hplane(tree,node).hplane_feature_value_max=hyperplane_projected_intersection;
            //update the probabilities
            this->_super_tree_probabilities(tree,node)=projected_value/this->_super_tree_values(tree,node);
            //update the hplane values according to the probabilities
            this->_super_tree_hplane(tree,node).hplane_feature_value=projected_value;
            this->_super_tree_hplane(tree,node).hplane_feature_value_left=projected_value;
            this->_super_tree_hplane(tree,node).hplane_feature_value_right=projected_value;
            //update the super_tree projections according to the hplane values
            this->_super_tree_values_projection(tree,node)=projected_value;
            //calculate the average error
            return error;
        }//test_projection_quality

      void print(std::ostream& out)
      {
            //print super tree:
            out<<"super_tree:"<<std::endl;
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    out<<std::to_string(i)<<","<<std::to_string(j)<<":"<<_super_tree(i,j).first << ","<<_super_tree(i,j).second  <<" ";  

                }
                out<<std::endl;
            }
            //print super tree values:
            out<<"super_tree_values:"<<std::endl;
            for(size_t i=0;i<_super_tree_values.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values.cols();j++)
                {
                    out<<_super_tree_values(i,j)<<" ";
                }
                out<<std::endl;
            }
            //print super tree values projection:
            out<<"super_tree_values_projection:"<<std::endl;
            for(size_t i=0;i<_super_tree_values_projection.rows();i++)
            {
                for(size_t j=0;j<_super_tree_values_projection.cols();j++)
                {
                    out<<_super_tree_values_projection(i,j)<<" ";
                }
                out<<std::endl;
            }   
            //print super tree probabilities:
            out<<"super_tree_probabilities:"<<std::endl;
            for(size_t i=0;i<_super_tree_probabilities.rows();i++)
            {
                for(size_t j=0;j<_super_tree_probabilities.cols();j++)
                {
                    out<<_super_tree_probabilities(i,j)<<" ";
                }
                out<<std::endl;
            }
            //print super tree hplane:
            out<<"super_tree_hplane:"<<std::endl;
            for(size_t i=0;i<_super_tree_hplane.rows();i++)
            {
                for(size_t j=0;j<_super_tree_hplane.cols();j++)
                {
                    out<<_super_tree_hplane(i,j)<<" ";
                }
                out<<std::endl;
            }   
            //print forest:
            out<<"forest:"<<std::endl;
            for(size_t i=0;i<_forest.size();i++)
            {
                for(size_t j=0;j<_forest[i].size();j++)
                {
                    out<<_forest[i][j]<<" ";
                }
                out<<std::endl;
            }   
            
      }
 };

}//namespace provallo


#endif // _FAST_MATRIX_FOREST_H_