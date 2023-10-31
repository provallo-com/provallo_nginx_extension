#ifndef __TREE_CLASSIFIERS_H__
#define __TREE_CLASSIFIERS_H__


#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include <cmath>
#include <numeric>
#include <functional>
#include <memory>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <limits>
#include <random>
#include <atomic>
#include "utils.h"
#include "matrix.h"
#include "optimizers.h"
#include "fast_matrix_forest.h"

 namespace provallo {

    namespace trees 
    {
 
   template <typename T> class leaf
   {

       public:
       typedef T value_type;
       typedef T & reference;
       typedef const T & const_reference;
       typedef T * pointer;
       typedef const T * const_pointer;

       leaf(const T &value) : _value(value),_id(++_id_gen){}
       leaf(T &&value) : _value(value),_id(++_id_gen) {}
       leaf(const leaf &other) : _value(other._value),_id(++_id_gen) {}
       leaf(leaf &&other) : _value(other._value),_id(other._id) {}
       leaf & operator=(const leaf &other) { _value = other._value; return *this; }
       leaf & operator=(leaf &&other) { _value = other._value; _id = other._id;return *this; }
       const value_type & value() const { return _value; }
       value_type & value() { return _value; }
       void print() const { std::cout << "Leaf : "<< std::to_string(_id)<<"\t"<<std::to_string( _value )<< std::endl; }

        private:
       value_type _value;
       uint64_t _id;
       static std::atomic_uint64_t _id_gen  ;

       
   };
       //node  has children  each has leaves  :
       template <typename T> 
       class node 
       {

              public:
              typedef T value_type;
              typedef T & reference;
              typedef const T & const_reference;
              typedef T * pointer;
              typedef const T * const_pointer;

              typedef std::vector<leaf<T>*> children_type;
              typedef typename children_type::iterator iterator;
              typedef typename children_type::const_iterator const_iterator;
              typedef typename children_type::reverse_iterator reverse_iterator;
              typedef typename children_type::const_reverse_iterator const_reverse_iterator;
              typedef typename children_type::size_type size_type;
              typedef typename children_type::difference_type difference_type;
              typedef typename children_type::value_type children_value_type;
              typedef typename children_type::allocator_type allocator_type;
              typedef typename children_type::pointer children_pointer;
              typedef typename children_type::const_pointer children_const_pointer;
              typedef typename children_type::reference children_reference;
              typedef typename children_type::const_reference children_const_reference;

              node(const T &value) : _values(1,value) {}
              node(T &&value) : _values(1,value) {}
              node(const node &other) : _values(other._values) {}
              node(node &&other) : _values(std::move(other._values)) {}
              
              node & operator=(const node &other) { _values = other._values; return *this; }
              node & operator=(node &&other) { _values = std::move(other._values); return *this; }
              //add value
              void add_value(const value_type &v) { _values.push_back(v); }
              void add_value(value_type &&v) { _values.push_back(std::move(v)); }
              void add_value(const value_type &v, const size_type &i) { _values.insert(_values.begin() + i, v); }
              void add_value(value_type &&v, const size_type &i) { _values.insert(_values.begin() + i, std::move(v)); }
              void add_value(const value_type &v, const_iterator &i) { _values.insert(i, v); }
              void add_value(value_type &&v, const_iterator &i) { _values.insert(i, std::move(v)); }
              void add_value(const value_type &v, const_reverse_iterator &i) { _values.insert(i, v); }
              void add_value(value_type &&v, const_reverse_iterator &i) { _values.insert(i, std::move(v)); }



              //remove value
              void remove_value(const size_type &i) { _values.erase(_values.begin() + i); }
              void remove_value(const_iterator &i) { _values.erase(i); }
              void remove_value(const_reverse_iterator &i) { _values.erase(i); }
              void remove_value(const const_iterator &i) { _values.erase(i); }
              void remove_value(const const_reverse_iterator &i) { _values.erase(i); }
              void remove_value(const const_pointer &i) { _values.erase(i); }
               

              //set value 
              void set_value(const value_type &v, const size_type &i) { _values[i] = v; }
              void set_value(value_type &&v, const size_type &i) { _values[i] = std::move(v); }
              void set_value(const value_type &v, const_iterator &i) { *i = v; }
              void set_value(value_type &&v, const_iterator &i) { *i = std::move(v); }
              void set_value(const value_type &v, const_reverse_iterator &i) { *i = v; }
              void set_value(value_type &&v, const_reverse_iterator &i) { *i = std::move(v); }
              void set_value(const value_type &v, const const_iterator &i) { *i = v; }

              //get value
              value_type & value(const size_type &i) { return _values[i]; }
              const value_type & value(const size_type &i) const { return _values[i].value(); }
              value_type & value(const_iterator &i) { return *i; }
              const value_type & value(const_iterator &i) const { return *i.value(); }
              value_type & value(const_reverse_iterator &i) { return *i.value(); }
              const value_type & value(const_reverse_iterator &i) const { return *i.value(); }
              value_type & value(const const_iterator &i) { return *i.value(); }
              const value_type & value(const const_iterator &i) const { return *i.value(); }
              value_type & value(const const_reverse_iterator &i) { return *i.value(); }
              const value_type & value(const const_reverse_iterator &i) const { return *i.value(); }
              value_type & value(const const_pointer &i) { return *i; }
              const value_type & value(const const_pointer &i) const { return *i; }
              
              //get value []       
              value_type & operator[](const size_type &i) { return _values[i]; }
              const value_type & operator[](const size_type &i) const { return _values[i]; }

              //get value at
              value_type & at(const size_type &i) { return _values.at(i); }
              const value_type & at(const size_type &i) const { return _values.at(i); }
              value_type & at(const_iterator &i) { return *i; }
              const value_type & at(const_iterator &i) const { return *i; }
              value_type & at(const_reverse_iterator &i) { return *i; }
              const value_type & at(const_reverse_iterator &i) const { return *i; }
              value_type & at(const const_iterator &i) { return *i; }
              const value_type & at(const const_iterator &i) const { return *i; }
              value_type & at(const const_reverse_iterator &i) { return *i; }
              const value_type & at(const const_reverse_iterator &i) const { return *i; }
              value_type & at(const const_pointer &i) { return *i; }
              const value_type & at(const const_pointer &i) const { return *i; }
              


              //add leaf
              void add_leaf(const leaf<T> &l) { _values.push_back(l); }
              void add_leaf(leaf<T> &&l) { _values.push_back(std::move(l)); }
              void add_leaf(const leaf<T> &l, const size_type &i) { _values.insert(_values.begin() + i, l); }
              void add_leaf(leaf<T> &&l, const size_type &i) { _values.insert(_values.begin() + i, std::move(l)); }
              void add_leaf(const leaf<T> &l, const_iterator &i) { _values.insert(i, l); }
              void add_leaf(leaf<T>&&l, const_iterator &i) { _values.insert(i, std::move(l)); }
              void add_leaf(const leaf<T> &l, const_reverse_iterator &i) { _values.insert(i, l); }
              void add_leaf(leaf<T> &&l, const_reverse_iterator &i) { _values.insert(i, std::move(l)); }

              std::vector<T> operator ()(){  
                     //transform to vector of values
                     std::vector<T> ret(_values.begin(),_values.end() , [](const leaf<T>&l) { return l.value(); } ); 
                     //return vector of values
                     return ret;

              }
              
              void print() const {  std::cout<<"\tNode : "<<std::to_string(_id);  
                                    std::for_each(_values.begin(), _values.end(), [ ](const leaf<T> &l) {   l.print(); }); }  
                     
              private:
              children_type _values;
              uint64_t _id;
              static std::atomic_uint64_t _id_gen  ;

       };
       //tree   contains nodes. each node has children. each child has leaves, a value initializes the first node with the first leaf with the first value.   :
       template <typename T> class tree 
       {
              private:
              typedef T value_type;
              typedef T & reference;
              typedef const T & const_reference;
              typedef T * pointer;
              typedef const T * const_pointer;
              typedef std::vector<node<T>*> children_type;
              typedef typename children_type::iterator iterator;
              typedef typename children_type::const_iterator const_iterator;

              typedef typename children_type::reverse_iterator reverse_iterator;

              typedef typename children_type::const_reverse_iterator const_reverse_iterator;
              typedef typename children_type::size_type size_type;
              typedef typename children_type::difference_type difference_type;
              typedef typename children_type::value_type children_value_type;
              typedef typename children_type::allocator_type allocator_type;
              typedef typename children_type::pointer children_pointer;
              typedef typename children_type::const_pointer children_const_pointer;
              typedef typename children_type::reference children_reference;
              typedef typename children_type::const_reference children_const_reference;


              public:
              tree(const T &value) : _values(1,value) {_id = ++_id_gen;}
              tree(T &&value) : _values(1,value) { _id = ++_id_gen;}
              tree(const tree &other) : _values(other._values) {_id = ++_id_gen;}
              tree(tree &&other) : _values(std::move(other._values)) {_id=++_id_gen;}

              tree & operator=(const tree &other) { _values = other._values; return *this; }
              tree & operator=(tree &&other) { _values = std::move(other._values); return *this; }

              private:
              children_type _values; //nodes

              //add node
              void add_node(const node<T>&n) { _values.push_back(n); }
              void add_node(node<T> &&n) { _values.push_back(std::move(n)); }
              void add_node(const node<T> &n, const size_type &i) { _values.insert(_values.begin() + i, n); }
 
              //remove node
              void remove_node(const size_type &i) { _values.erase(_values.begin() + i); }

              //count leaves
              size_type count_leaves() const { return std::accumulate(_values.begin(), _values.end(), 0, [](const size_type &a, const node<T> &b) { return a + b.size(); }); }    
              //count nodes
              size_type count_nodes() const { return _values.size(); }


              //add leaf to node   
              void add_leaf(const leaf<T> &l, const size_type &i, const size_type &j) { _values[i].add_leaf(l, j); }
              void add_leaf(leaf<T>&&l, const size_type &i, const size_type &j) { _values[i].add_leaf(std::move(l), j); }
              void add_leaf(const leaf<T> &l, const_iterator &i, const_iterator &j) { _values[i].add_leaf(l, j); }

              //remove leaf from node
              void remove_leaf(const size_type &i, const size_type &j) { _values[i].remove_leaf(j); }
              void remove_leaf(const_iterator &i, const_iterator &j) { _values[i].remove_leaf(j); }
              
              //add value create leaf and node if necessary 
              void add_value(const value_type &v, const size_type &i, const size_type &j, const size_type &k) { _values[i].add_value(v, j, k); }
              void add_value(value_type &&v, const size_type &i, const size_type &j, const size_type &k) { _values[i].add_value(std::move(v), j, k); }
              void add_value(const value_type &v, const_iterator &i, const_iterator &j, const_iterator &k) { _values[i].add_value(v, j, k); }
              void add_value(value_type &&v, const_iterator &i, const_iterator &j, const_iterator &k) { _values[i].add_value(std::move(v), j, k); }
              void add_value(const value_type &v, const_reverse_iterator &i, const_reverse_iterator &j, const_reverse_iterator &k) { _values[i].add_value(v, j, k); }


              //set value for an existing leaf
              void set_value(const value_type &v, const size_type &i, const size_type &j, const size_type &k) { _values[i].set_value(v, j, k); }
              void set_value(value_type &&v, const size_type &i, const size_type &j, const size_type &k) { _values[i].set_value(std::move(v), j, k); }
              void print() const { std::for_each(_values.begin(), _values.end(), [](const node<T> &n) { n.print(); }); } 

              private:
              uint64_t _id;
              static std::atomic_uint64_t _id_gen ;

           
       };
       //forest     :
       template <typename T> class forest 
       {
           
              public:
              typedef T value_type;
              typedef T & reference;
              typedef const T & const_reference;
              typedef T * pointer;
              typedef const T * const_pointer;
              typedef std::vector<tree<T>*> children_type;
              typedef typename children_type::iterator iterator;
              typedef typename children_type::const_iterator const_iterator;
              typedef typename children_type::reverse_iterator reverse_iterator;
              typedef typename children_type::const_reverse_iterator const_reverse_iterator;
              typedef typename children_type::size_type size_type;
              typedef typename children_type::difference_type difference_type;
              typedef typename children_type::value_type children_value_type;
              typedef typename children_type::allocator_type allocator_type;
              typedef typename children_type::pointer children_pointer;
              typedef typename children_type::const_pointer children_const_pointer;
              typedef typename children_type::reference children_reference;
              typedef typename children_type::const_reference children_const_reference;

              forest(const T &value) : _value(value) , _id(++_id_gen) {}
              forest(T &&value) : _value(value) , _id(++_id_gen) {}
              forest(const forest<T> &other) : _value(other._value), _children(other._children),_id(other._id){}
              forest( forest<T> &&other) : _value(other._value), _children(std::move(other._children)) ,_id(++_id_gen){}
              forest():_value(T(0)), _id(++_id_gen) {}
              forest & operator=(const  forest<T> &other) { _value = other._value; _children = other._children; return *this; }
              forest & operator=( forest<T> &&other) { _value = other._value; _children = std::move(other._children); return *this; }
               

              //add tree
              void add_tree(const tree<T> &t) { _children.push_back(t); }
              void add_tree(tree<T> &&t) { _children.push_back(std::move(t)); }

              //remove tree
              void remove_tree(const size_type &i) { _children.erase(_children.begin() + i); }
              void remove_tree(const_iterator &i) { _children.erase(i); }
              void remove_tree(const_reverse_iterator &i) { _children.erase(i); }
              void remove_tree(const const_iterator &i) { _children.erase(i); }

              //add node to tree
              void add_node(const node<T> &n, const size_type &i, const size_type &j) { _children[i].add_node(n, j); }
              void add_node(node<T> &&n, const size_type &i, const size_type &j) { _children[i].add_node(std::move(n), j); }
              void add_node(const node<T> &n, const_iterator &i, const_iterator &j) { _children[i].add_node(n, j); }
              void add_node(node<T> &&n, const_iterator &i, const_iterator &j) { _children[i].add_node(std::move(n), j); }
              void add_node(const node<T> &n, const_reverse_iterator &i, const_reverse_iterator &j) { _children[i].add_node(n, j); }

              //remove node from tree
              void remove_node(const size_type &i, const size_type &j) { _children[i].remove_node(j); }
              void remove_node(const_iterator &i, const_iterator &j) { _children[i].remove_node(j); }
              void remove_node(const_reverse_iterator &i, const_reverse_iterator &j) { _children[i].remove_node(j); }
              void remove_node(const const_iterator &i, const const_iterator &j) { _children[i].remove_node(j); }
              void remove_node(const const_reverse_iterator &i, const const_reverse_iterator &j) { _children[i].remove_node(j); }
              void remove_node(const const_pointer &i, const const_pointer &j) { _children[i].remove_node(j); }
 
              


              //add leaf to node
              void add_leaf(const leaf<T> &l, const size_type &i, const size_type &j, const size_type &k) { _children[i].add_leaf(l, j, k); }
              void add_leaf( leaf<T> &&l, const size_type &i, const size_type &j, const size_type &k) { _children[i].add_leaf(std::move(l), j, k); }
              void add_leaf(const  leaf<T> &l, const_iterator &i, const_iterator &j, const_iterator &k) { _children[i].add_leaf(l, j, k); }
              


              //remove leaf from node
              void remove_leaf(const size_type &i, const size_type &j, const size_type &k) { _children[i].remove_leaf(j, k); }
              void remove_leaf(const_iterator &i, const_iterator &j, const_iterator &k) { _children[i].remove_leaf(j, k); }
              void remove_leaf(const_reverse_iterator &i, const_reverse_iterator &j, const_reverse_iterator &k) { _children[i].remove_leaf(j, k); }
              void remove_leaf(const const_iterator &i, const const_iterator &j, const const_iterator &k) { _children[i].remove_leaf(j, k); }
              void remove_leaf(const const_reverse_iterator &i, const const_reverse_iterator &j, const const_reverse_iterator &k) { _children[i].remove_leaf(j, k); }
              void remove_leaf(const const_pointer &i, const const_pointer &j, const const_pointer &k) { _children[i].remove_leaf(j, k); }

              //add value to leaf
              void add_value(const value_type &v, const size_type &i, const size_type &j, const size_type &k, const size_type &l) { _children[i].add_value(v, j, k, l); }
              void add_value(value_type &&v, const size_type &i, const size_type &j, const size_type &k, const size_type &l) { _children[i].add_value(std::move(v), j, k, l); }
              void add_value(const value_type &v, const_iterator &i, const_iterator &j, const_iterator &k, const_iterator &l) { _children[i].add_value(v, j, k, l); }
              void add_value(value_type &&v, const_iterator &i, const_iterator &j, const_iterator &k, const_iterator &l) { _children[i].add_value(std::move(v), j, k, l); }

       

              //remove value from leaf

              void remove_value(const size_type &i, const size_type &j, const size_type &k, const size_type &l) { _children[i].remove_value(j, k, l); }
              void remove_value(const_iterator &i, const_iterator &j, const_iterator &k, const_iterator &l) { _children[i].remove_value(j, k, l); }
              void remove_value(const_reverse_iterator &i, const_reverse_iterator &j, const_reverse_iterator &k, const_reverse_iterator &l) { _children[i].remove_value(j, k, l); }
              void remove_value(const const_iterator &i, const const_iterator &j, const const_iterator &k, const const_iterator &l) { _children[i].remove_value(j, k, l); }
              void remove_value(const const_reverse_iterator &i, const const_reverse_iterator &j, const const_reverse_iterator &k, const const_reverse_iterator &l) { _children[i].remove_value(j, k, l); }
              void remove_value(const const_pointer &i, const const_pointer &j, const const_pointer &k, const const_pointer &l) { _children[i].remove_value(j, k, l); }

              //set value for an existing leaf
              void set_value(const value_type &v, const size_type &i, const size_type &j, const size_type &k, const size_type &l) { _children[i].set_value(v, j, k, l); }
              void set_value(value_type &&v, const size_type &i, const size_type &j, const size_type &k, const size_type &l) { _children[i].set_value(std::move(v), j, k, l); }
              void set_value(const value_type &v, const_iterator &i, const_iterator &j, const_iterator &k, const_iterator &l) { _children[i].set_value(v, j, k, l); }
              void set_value(value_type &&v, const_iterator &i, const_iterator &j, const_iterator &k, const_iterator &l) { _children[i].set_value(std::move(v), j, k, l); }
              void set_value(const value_type &v, const_reverse_iterator &i, const_reverse_iterator &j, const_reverse_iterator &k, const_reverse_iterator &l) { _children[i].set_value(v, j, k, l); }
              void set_value(value_type &&v, const_reverse_iterator &i, const_reverse_iterator &j, const_reverse_iterator &k, const_reverse_iterator &l) { _children[i].set_value(std::move(v), j, k, l); }
              void set_value(const value_type &v, const const_iterator &i, const const_iterator &j, const const_iterator &k, const const_iterator &l) { _children[i].set_value(v, j, k, l); }
              void set_value(value_type &&v, const const_iterator &i, const const_iterator &j, const const_iterator &k, const const_iterator &l) { _children[i].set_value(std::move(v), j, k, l); }
              void set_value(const value_type &v, const const_reverse_iterator &i, const const_reverse_iterator &j, const const_reverse_iterator &k, const const_reverse_iterator &l) { _children[i].set_value(v, j, k, l); }
              void set_value(value_type &&v, const const_reverse_iterator &i, const const_reverse_iterator &j, const const_reverse_iterator &k, const const_reverse_iterator &l) { _children[i].set_value(std::move(v), j, k, l); }
              void set_value(const value_type &v, const const_pointer &i, const const_pointer &j, const const_pointer &k, const const_pointer &l) { _children[i].set_value(v, j, k, l); }



              //get value from leaf
              value_type & value(const size_type &i, const size_type &j, const size_type &k, const size_type &l) { return _children[i].value(j, k, l); }

              //get value from leaf
              virtual void print() const { std::for_each(_children.begin(), _children.end(), [](const tree<T> &t) { t.print(); }); } 

               
              private:
              value_type _value;
              children_type _children;
              uint64_t _id;
              static std::atomic_uint64_t _id_gen ;

                     
       };

       //leaf     implementation:
       //node     implementation:
       //tree     implementation:
       //forest   implementation:

       template <typename T>
       class uniform_random_forest : public forest<T> 
       {
              std::random_device rd;
              std::mt19937 gen;
              std::uniform_int_distribution<> dis;
              std::uint64_t _id; 
              static std::atomic_uint64_t _id_gen ;


              public:
              uniform_random_forest(const T &value, const size_t &min_depth, const size_t  &max_depth, const size_t  &min_trees, const size_t  &max_trees) : forest<T>(value), gen(rd()), dis(min_depth, max_depth), _id(0) 
              {

                     std::uniform_int_distribution<> dis_trees(min_trees, max_trees);
                     
                     _id=++_id_gen;
                  
                     //generate random values around value for each tree
                     for (size_t i = 0; i < dis_trees(gen); ++i) 
                     {
                            tree<T> t(value);
                            for (size_t j = 0; j < dis(gen); ++j) 
                            {
                                   node<T> n(value);
                                   for (size_t k = 0; k < dis(gen); ++k) 
                                   {
                                          leaf<T> l(value);
                                          n.add_leaf(l);
                                   }
                                   t.add_node(n);
                            }
                            this->add_tree(t);
                     }             
                     //generate random values around value for each tree


 

               }   
               //
               size_t size()const { return this->_children.size(); }
              //
               size_t total_size()const { return std::accumulate(this->_children.begin(), this->_children.end(), 0, [](const size_t &a, const tree<T> &b) { return a + b.size(); }); } 
              
              
              //reset
              void reset(const T &value, const size_t &min_depth, const size_t &max_depth, const size_t&min_trees, const size_t&max_trees)  
              {
                     //reset
                     this->_value = value;
                     this->_children.clear();
                     dis = std::uniform_int_distribution<>(min_depth, max_depth);
                     std::uniform_int_distribution<> dis_trees(min_trees, max_trees);
                     //reset
                     //generate random values around value for each tree
                     for (size_t i = 0; i < dis_trees(gen); ++i) 
                     {
                            tree<T> t(value);
                            for (size_t j = 0; j < dis(gen); ++j) 
                            {
                                   node<T> n(value);
                                   for (size_t k = 0; k < dis(gen); ++k) 
                                   {
                                          leaf<T> l(value);
                                          n.add_leaf(l);
                                   }
                                   t.add_node(n);
                            }
                            this->add_tree(t);
                     }
                     //generate random values around value for each tree


              }

              virtual void print() const { 
                     
                     std::for_each(this->_children.begin(), this->_children.end(), [](const tree<T> &t) { t.print(); }); } 

                     //get value from leaf
              virtual T & value(const size_t &i, const size_t &j, const size_t &k, const size_t &l) { return this->_children[i].value(j, k, l); }
              virtual const T & value(const size_t &i, const size_t &j, const size_t &k, const size_t &l) const { return this->_children[i].value(j, k, l); }
              virtual T & value(const size_t &i, const size_t &j, const size_t &k) { return this->_children[i].value(j, k); }
              virtual const T & value(const size_t &i, const size_t &j, const size_t &k) const { return this->_children[i].value(j, k); }
              virtual T & value(const size_t &i, const size_t &j) { return this->_children[i].value(j); }
              virtual const T & value(const size_t &i, const size_t &j) const { return this->_children[i].value(j); }
              virtual T & value(const size_t &i) { return this->_children[i].value(); }
              virtual const T & value(const size_t &i) const { return this->_children[i].value(); }
              virtual T & value(const typename forest<T>::const_iterator &i, const typename tree<T>::const_iterator &j, const typename node<T>::const_iterator &k, const typename leaf<T>::const_iterator &l) { return this->_children[i].value(j, k, l); }
              virtual const T & value(const typename forest<T>::const_iterator &i, const typename tree<T>::const_iterator &j, const typename node<T>::const_iterator &k, const typename leaf<T>::const_iterator &l) const { return this->_children[i].value(j, k, l); }
              virtual T & value(const typename forest<T>::const_iterator &i, const typename tree<T>::const_iterator &j, const typename node<T>::const_iterator &k) { return this->_children[i].value(j, k); }

              



               ~uniform_random_forest() {}
              

       };


       //initialize static template 

       template <typename T> std::atomic_uint64_t leaf<T>::_id_gen = 0ull;
       template <typename T> std::atomic_uint64_t node<T>::_id_gen = 0ull;
       template <typename T> std::atomic_uint64_t tree<T>::_id_gen = 0ull;
       template <typename T> std::atomic_uint64_t forest<T>::_id_gen = 0ull;

       template <typename T> std::atomic_uint64_t uniform_random_forest<T>::_id_gen = 0ull;

       //Leo Breiman's Random Forests for Classification and Regression
       //https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
       //https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
       //https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr
       //https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#varimp
       //https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#rfcode
       //https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#remarks
       //https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#bibliography

       template <typename T> class random_forest : public forest<T> 
       {
              std::random_device rd;
              std::mt19937 gen;
              std::uniform_int_distribution<> dis;
              std::uint64_t _id; 
              static std::atomic_uint64_t _id_gen ;     
              //
              provallo::matrix<real_t> _oob_sample;
              //out of bag prediction
              provallo::matrix<real_t> _oob_prediction;
              //out of bag error
              provallo::matrix<real_t> _oob_error;
              //out of bag variance
              provallo::matrix<real_t> _oob_variance;
              //out of bag confidence
              provallo::matrix<real_t> _oob_confidence;
              //out of bag importance
              provallo::matrix<real_t> _oob_importance;
              
              //out of bag proximity
              provallo::matrix<real_t> _oob_proximity;



              std::vector<std::string> _labels;
              std::vector<std::string> _features;
              std::vector<std::string> _classes;
              

              //forest hyperparameters:
              //number of trees
              size_t _n_trees =1;
              //number of features
              size_t _n_features =1;
              //number of samples
              size_t _n_samples =1;
              //number of classes
              size_t _n_classes =1;
              
              //number of dimensions
              size_t _n_dimensions =1;
              //noutliers
              size_t _n_outliers =1;
              //max categorical
              size_t _max_categorical=1;
              size_t _max_depth =1;
              
              size_t _nthreads = 1;

              public:

              //initialize a random forest with a value, a minimum depth, a maximum depth, a minimum number of trees and a maximum number of trees 
              //the random forest is initialized with random values around the value

 
              //construct a random forest from training data   :      
              //
              //    If the number of cases in the training set is N, sample N cases at random - but with replacement, from the original data. This sample will be the training set for growing the tree.
              //    If there are M input variables, a number m<<M is specified such that at each node, m variables are selected at random out of the M and the best split on these m is used to split the node. The value of m is held constant during the forest growing.
              //    Each tree is grown to the largest extent possible. There is no pruning.

              random_forest(const std::vector<std::string>& labels,const std::vector<std::string> & features , const std::vector<std::string>& classes, provallo::matrix<T>& training, provallo::matrix<T>& testing, const size_t &n_trees, const size_t &n_features, const size_t &n_samples, const size_t &n_classes, const size_t &n_dimensions, const size_t &n_outliers, const size_t &max_categorical, const size_t &max_depth, const size_t &nthreads):
              forest<T>(T(0)), gen(rd()), dis(1, max_depth), _id(0), _oob_sample(training), _oob_prediction(training.rows(), classes.size()), _oob_error(training.rows(), classes.size()), _oob_variance(training.rows(), classes.size()), _oob_confidence(training.rows(), classes.size()), _oob_importance(training.rows(), classes.size()), _oob_proximity(training.rows(), classes.size()), _labels(labels), _features(features), _classes(classes), _n_trees(n_trees), _n_features(n_features), _n_samples(n_samples), _n_classes(n_classes), _n_dimensions(n_dimensions), _n_outliers(n_outliers), _max_categorical(max_categorical), _max_depth(max_depth), _nthreads(nthreads)
              {
                     //initialize random forest with random leaves and nodes around value
                     _id=++_id_gen;


                     //update samples,features,classes, dimensions, outliers, categorical, depth 
                     _n_samples = training.rows();
                     _n_features = training.cols();
                     _n_classes = classes.size();
                     _n_dimensions = training.cols();
                     _n_outliers = n_outliers;
                     _max_categorical = max_categorical;
                     _max_depth = max_depth;
                     _nthreads = nthreads;
                     

                     // initialize random forest using training data:
                     for(size_t i=0;i<_n_trees;++i)
                     {
                            //generate random values around value for each tree
                            tree<T> t(T(0));
                            for (size_t j = 0; j < dis(gen); ++j) 
                            {
                                   node<T> n(T(0));
                                   for (size_t k = 0; k < dis(gen); ++k) 
                                   {
                                          size_t row = std::uniform_int_distribution<>(0, _n_samples - 1)(gen); 
                                          size_t col = std::uniform_int_distribution<>(0, _n_dimensions - 1)(gen); 
                                          leaf<T> l(training(row,col));
                                          n.add_leaf(l);
                                   }
                                   t.add_node(n);
                            }
                            this->add_tree(t);
                            //generate random values around value for each tree
                     }
                     
                     //update samples,features,classes, dimensions, outliers, categorical, depth
                     //initialize random forest with random leaves and nodes around value

                     //train random forest
                     train(training);



              }      
                    
              //reset the random forest with a value, a minimum depth, a maximum depth, a minimum number of trees and a maximum number of trees
              //the random forest is reset with random values around the value

              //reset:
              void reset(const T &value, const size_t &min_depth, const size_t &max_depth, const size_t&min_trees, const size_t&max_trees)  
              {
                     //reset
                     this->_value = value;
                     this->_children.clear();
                     dis = std::uniform_int_distribution<>(min_depth, max_depth);
                     std::uniform_int_distribution<> dis_trees(min_trees, max_trees);
                     //reset
                     //generate random values around value for each tree
                     for (size_t i = 0; i < dis_trees(gen); ++i) 
                     {
                            tree<T> t(value);
                            for (size_t j = 0; j < dis(gen); ++j) 
                            {
                                   node<T> n(value);
                                   for (size_t k = 0; k < dis(gen); ++k) 
                                   {
                                          leaf<T> l(value);
                                          n.add_leaf(l);
                                   }
                                   t.add_node(n);
                            }
                            this->add_tree(t);
                     }
                     //generate random values around value for each tree

              }
              //ooob prediction
              provallo::matrix<real_t> & oob_prediction() { return _oob_prediction; }
              //oob error
              provallo::matrix<real_t> & oob_error() { return _oob_error; }       
              //oob variance
              provallo::matrix<real_t> & oob_variance() { return _oob_variance; }
              //oob confidence
              provallo::matrix<real_t> & oob_confidence() { return _oob_confidence; }
              //oob importance
              provallo::matrix<real_t> & oob_importance() { return _oob_importance; }
              //oob proximity
              provallo::matrix<real_t> & oob_proximity() { return _oob_proximity; }
              //oob sample
              provallo::matrix<real_t> & oob_sample() { return _oob_sample; }
              
              //size
              size_t size()const { return this->_children.size(); }
              //total size
              size_t total_size()const { return std::accumulate(this->_children.begin(), this->_children.end(), 0, [](const size_t &a, const tree<T> &b) { return a + b.size(); }); }
              //print
              virtual void print() const { 
                     
                     std::for_each(this->_children.begin(), this->_children.end(), [](const tree<T> &t) { t.print(); }); 
              } 

              //set training set:
              void set_training_set(const provallo::matrix<real_t> &training_set) 
              {
                     //set training set
                     _oob_sample = training_set;
                     //set training set
              }
              void set_training_set(provallo::matrix<real_t> &&training_set) 
              {
                     //set training set
                     _oob_sample = std::move(training_set);
                     //set training set
              }
              void set_labels(const std::vector<std::string> &labels) 
              {
                     //set labels
                     _labels = labels;
                     //set labels
              } 

              void set_labels(std::vector<std::string> &&labels) 
              {
                     //set labels
                     _labels = std::move(labels);
                     //set labels
              }
              void set_features(const std::vector<std::string> &features) 
              {
                     //set features
                     _features = features;
                     //set features
              }
              void set_features(std::vector<std::string> &&features) 
              {
                     //set features
                     _features = std::move(features);
                     //set features
              }
              void set_classes(const std::vector<std::string> &classes) 
              {
                     //set classes
                     _classes = classes;
                     //set classes
              }
              void set_classes(std::vector<std::string> &&classes) 
              {
                     //set classes
                     _classes = std::move(classes);
                     //set classes
              }
              void set_n_trees(const size_t &n_trees) 
              {
                     //set n_trees
                     _n_trees = n_trees;
                     //set n_trees
              }
              void set_n_features(const size_t &n_features) 
              {
                     //set n_features
                     _n_features = n_features;
                     //set n_features
              }
              void set_n_samples(const size_t &n_samples) 
              {
                     //set n_samples
                     _n_samples = n_samples;
                     //set n_samples
              }
              void set_n_classes(const size_t &n_classes) 
              {
                     //set n_classes
                     _n_classes = n_classes;
                     //set n_classes
              }
              //train
              void train(provallo::matrix<T>& training) 
              {

                     for(auto& t : this->_children)
                     {
                            for(size_t i=0;i<_labels.size();++i)
                            {
                                   //train for each label indice
                                   train(training,t, i); 
                            }
                            
                     } 

              } 
              //train
              void train(provallo::matrix<T>& training,tree<T>& t,const size_t &label) 
              {
                     
                     //calculate oob prediction for the training set 
                     for(size_t i=0;i<_oob_sample.rows();++i)
                     {
                            //calculate oob prediction for the training set 
                            _oob_prediction(i,label) = predict(_oob_sample.row(i),t); 
                            //calculate oob prediction for the training set 
                     }
                     //calculate oob prediction for the training set


                     

              }  
       };            



       }//namespace trees   
       }//namespace provallo
       

#endif  // __TREE_CLASSIFIERS_H__
