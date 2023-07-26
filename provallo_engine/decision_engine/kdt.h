#ifndef __KDT_H_
#define __KDT_H_
#include "utils.h"

namespace provallo
{
  template <std::size_t N>
  class point
  {
  public:
    // Types representing iterators that can traverse and optionally modify the elements of the point.
    typedef double *iterator;
    typedef const double *const_iterator;

    // Returns N, the dimension of the point.
    std::size_t
    size() const;

    // Queries or retrieves the value of the point at a particular point. The index is assumed to be in-range.
    double &
    operator[](std::size_t index);
    double
    operator[](std::size_t index) const;

    // Returns iterators delineating the full range of elements in the point.
    iterator
    begin();
    iterator
    end();
    const_iterator
    begin() const;
    const_iterator
    end() const;

  private:
    double coords[N];
  };
  template <std::size_t N>
  double
  point_distance(const point<N> &one, const point<N> &two);

  // Returns whether two points are equal / not equal
  template <std::size_t N>
  bool
  operator==(const point<N> &one, const point<N> &two);

  template <std::size_t N>
  bool
  operator!=(const point<N> &one, const point<N> &two);

  template <std::size_t N>
  std::size_t
  point<N>::size() const
  {
    return N;
  }

  template <std::size_t N>
  double &
  point<N>::operator[](std::size_t index)
  {
    return coords[index];
  }

  template <std::size_t N>
  double
  point<N>::operator[](std::size_t index) const
  {
    return coords[index];
  }

  template <std::size_t N>
  typename point<N>::iterator
  point<N>::begin()
  {
    return coords;
  }

  template <std::size_t N>
  typename point<N>::const_iterator
  point<N>::begin() const
  {
    return coords;
  }

  template <std::size_t N>
  typename point<N>::iterator
  point<N>::end()
  {
    return begin() + size();
  }

  template <std::size_t N>
  typename point<N>::const_iterator
  point<N>::end() const
  {
    return begin() + size();
  }

  template <std::size_t N>
  double
  point_distance(const point<N> &one, const point<N> &two)
  {
    double result = 0.0;
    for (std::size_t i = 0; i < N; ++i)
      result += (one[i] - two[i]) * (one[i] - two[i]);
    return result;
  }

  template <std::size_t N>
  bool
  operator==(const point<N> &one, const point<N> &two)
  {
    return std::equal(one.begin(), one.end(), two.begin());
  }

  template <std::size_t N>
  bool
  operator!=(const point<N> &one, const point<N> &two)
  {
    return !(one == two);
  }

  template <typename T>
  class bpqueue
  {
  public:
    explicit bpqueue(std::size_t maxSize);
    void
    enqueue(const T &value, double priority);

    T dequeueMin();

    std::size_t
    size() const;

    bool
    empty() const;

    std::size_t
    maxSize() const;

    double
    best() const;

    double
    worst() const;

  private:
    // This class is layered on top of a multimap mapping from priorities
    // to elements with those priorities.
    std::multimap<double, T> elems;
    std::size_t maximumSize;
  };
  template <typename T>
  bpqueue<T>::bpqueue(std::size_t maxSize)
  {
    maximumSize = maxSize;
  }

  // enqueue adds the element to the map, then deletes the last element of the
  // map if there size exceeds the maximum size.
  template <typename T>
  void
  bpqueue<T>::enqueue(const T &value, double priority)
  {
    // Add the element to the collection.
    elems.insert(std::make_pair(priority, value));

    // If there are too many elements in the queue, drop off the last one.
    if (size() > maxSize())
    {
      typename std::multimap<double, T>::iterator last = elems.end();
      --last; // Now points to highest-priority element
      elems.erase(last);
    }
  }

  // dequeueMin copies the lowest element of the map (the one pointed at by
  // begin()) and then removes it.
  template <typename T>
  T bpqueue<T>::dequeueMin()
  {
    // Copy the best value.
    T result = elems.begin()->second;

    // Remove it from the map.
    elems.erase(elems.begin());

    return result;
  }

  // size() and empty() call directly down to the underlying map.
  template <typename T>
  std::size_t
  bpqueue<T>::size() const
  {
    return elems.size();
  }

  template <typename T>
  bool
  bpqueue<T>::empty() const
  {
    return elems.empty();
  }

  // maxSize just returns the appropriate data member.
  template <typename T>
  std::size_t
  bpqueue<T>::maxSize() const
  {
    return maximumSize;
  }

  // The best() and worst() functions check if the queue is empty,
  // and if so return infinity.
  template <typename T>
  double
  bpqueue<T>::best() const
  {
    return empty() ? std::numeric_limits<double>::infinity() : elems.begin()->first;
  }

  template <typename T>
  double
  bpqueue<T>::worst() const
  {
    return empty() ? std::numeric_limits<double>::infinity() : elems.rbegin()->first;
  }

  template <std::size_t N, typename elem_type>
  class kd_tree
  {
  public:
    // Constructs an empty KDTree.
    kd_tree();

    // Efficiently build a balanced KD-tree from a large set of points
    kd_tree(std::vector<std::pair<point<N>, elem_type>> &points);

    // Frees up all the dynamically allocated resources
    ~kd_tree();

    // Deep-copies the contents of another KDTree into this one.
    kd_tree(const kd_tree &rhs);
    kd_tree &
    operator=(const kd_tree &rhs);

    // Returns the dimension of the points stored in this KDTree.
    std::size_t
    dimension() const;

    // Returns the number of elements in the kd-tree and whether the tree is empty
    std::size_t
    size() const;
    bool
    empty() const;

    // Returns whether the specified point is contained in the KDTree.
    bool
    contains(const point<N> &pt) const;

    /*
     * Inserts the point pt into the KDTree, associating it with the specified value.
     * If the element already existed in the tree, the new value will overwrite the existing one.
     */
    void
    insert(const point<N> &pt, const elem_type &value = elem_type());

    /*
     * Returns a reference to the value associated with point pt in the KDTree.
     * If the point does not exist, then it is added to the KDTree using the
     * default value of elem_type as its key.
     */
    elem_type &
    operator[](const point<N> &pt);

    /*
     * Returns a reference to the key associated with the point pt. If the point
     * is not in the tree, this function throws an out_of_range exception.
     */
    elem_type &
    at(const point<N> &pt);
    const elem_type &
    at(const point<N> &pt) const;

    /*
     * Given a point v and an integer k, finds the k points in the KDTree
     * nearest to v and returns the most common value associated with those
     * points. In the event of a tie, one of the most frequent value will be chosen.
     */
    elem_type
    kNNValue(const point<N> &key, std::size_t k) const;

  private:
    struct node
    {
      provallo::point<N> point;
      node *left;
      node *right;
      int level; // level of the node in the tree, starts at 0 for the root
      elem_type value;

      node(const provallo::point<N> &_pt, int _level,
           const elem_type &_value = elem_type()) : point(_pt), left(NULL), right(nullptr), level(_level), value(_value)
      {
      }
    };

    // Root node of the KD-Tree
    node *root_;

    // Number of points in the KD-Tree
    std::size_t size_;

    /*
     * Recursively build a subtree that satisfies the KD-Tree invariant using points in [start, end)
     * At each level, we split points into two halves using the median of the points as pivot
     * The root of the subtree is at level 'currLevel'
     * O(n) time partitioning algorithm is used to locate the median element
     */
    node *
    buildTree(
        typename std::vector<std::pair<point<N>, elem_type>>::iterator start,
        typename std::vector<std::pair<point<N>, elem_type>>::iterator end,
        int currLevel);

    /*
     * Returns the node that contains point pt if it is present in subtree 'currNode'
     * Returns the node below which pt should be inserted if pt is not in the subtree
     */
    node *
    findNode(node *currNode, const point<N> &pt) const;

    // Recursive helper method for kNNValue(pt, k)
    void
    nearestNeighborRecurse(const node *currNode, const point<N> &key,
                           bpqueue<elem_type> &pQueue) const;

    /*
     * Recursive helper method for copy constructor and assignment operator
     * Deep copies tree 'root' and returns the root of the copied tree
     */
    node *
    deepcopyTree(node *root);

    // Recursively free up all resources of subtree rooted at 'currNode'
    void
    freeResource(node *currNode);
    // save trees recursively on file.
    // load trees
  };

  template <std::size_t N, typename elem_type>
  kd_tree<N, elem_type>::kd_tree() : root_(NULL), size_(0)
  {
  }

  template <std::size_t N, typename elem_type>
  typename kd_tree<N, elem_type>::node *
  kd_tree<N, elem_type>::deepcopyTree(
      typename kd_tree<N, elem_type>::node *root)
  {
    if (root == NULL)
      return NULL;
    node *newRoot = new node(*root);
    newRoot->left = deepcopyTree(root->left);
    newRoot->right = deepcopyTree(root->right);
    return newRoot;
  }

  template <std::size_t N, typename elem_type>
  typename kd_tree<N, elem_type>::node *
  kd_tree<N, elem_type>::buildTree(
      typename std::vector<std::pair<point<N>, elem_type>>::iterator start,
      typename std::vector<std::pair<point<N>, elem_type>>::iterator end,
      int currLevel)
  {
    if (start >= end)
      return NULL; // empty tree

    int axis = currLevel % N; // the axis to split on
    auto cmp = [axis](const std::pair<point<N>, elem_type> &p1,
                      const std::pair<point<N>, elem_type> &p2)
    {
      return p1.first[axis] < p2.first[axis];
    };
    std::size_t len = end - start;
    auto mid = start + len / 2;
    std::nth_element(start, mid, end, cmp); // linear time partition

    // move left (if needed) so that all the equal points are to the right
    // The tree will still be balanced as long as there aren't many points that are equal along each axis
    while (mid > start && (mid - 1)->first[axis] == mid->first[axis])
    {
      --mid;
    }

    node *newNode = new node(mid->first, currLevel, mid->second);
    newNode->left = buildTree(start, mid, currLevel + 1);
    newNode->right = buildTree(mid + 1, end, currLevel + 1);
    return newNode;
  }

  template <std::size_t N, typename elem_type>
  kd_tree<N, elem_type>::kd_tree(
      std::vector<std::pair<point<N>, elem_type>> &points)
  {
    root_ = buildTree(points.begin(), points.end(), 0);
    size_ = points.size();
  }

  template <std::size_t N, typename elem_type>
  kd_tree<N, elem_type>::kd_tree(const kd_tree &rhs)
  {
    root_ = deepcopyTree(rhs.root_);
    size_ = rhs.size_;
  }

  template <std::size_t N, typename elem_type>
  kd_tree<N, elem_type> &
  kd_tree<N, elem_type>::operator=(const kd_tree &rhs)
  {
    if (this != &rhs)
    { // make sure we don't self-assign
      freeResource(root_);
      root_ = deepcopyTree(rhs.root_);
      size_ = rhs.size_;
    }
    return *this;
  }

  template <std::size_t N, typename elem_type>
  void
  kd_tree<N, elem_type>::freeResource(
      typename kd_tree<N, elem_type>::node *currNode)
  {
    if (currNode == NULL)
      return;
    freeResource(currNode->left);
    freeResource(currNode->right);
    delete currNode;
  }

  template <std::size_t N, typename elem_type>
  kd_tree<N, elem_type>::~kd_tree()
  {
    freeResource(root_);
  }

  template <std::size_t N, typename elem_type>
  std::size_t
  kd_tree<N, elem_type>::dimension() const
  {
    return N;
  }

  template <std::size_t N, typename elem_type>
  std::size_t
  kd_tree<N, elem_type>::size() const
  {
    return size_;
  }

  template <std::size_t N, typename elem_type>
  bool
  kd_tree<N, elem_type>::empty() const
  {
    return size_ == 0;
  }

  template <std::size_t N, typename elem_type>
  typename kd_tree<N, elem_type>::node *
  kd_tree<N, elem_type>::findNode(
      typename kd_tree<N, elem_type>::node *currNode,
      const point<N> &pt) const
  {
    if (currNode == NULL || currNode->point == pt)
      return currNode;

    const point<N> &currpoint = currNode->point;
    int currLevel = currNode->level;
    if (pt[currLevel % N] < currpoint[currLevel % N])
    { // recurse to the left side
      return currNode->left == NULL ? currNode : findNode(currNode->left, pt);
    }
    else
    { // recurse to the right side
      return currNode->right == NULL ? currNode : findNode(currNode->right, pt);
    }
  }

  template <std::size_t N, typename elem_type>
  bool
  kd_tree<N, elem_type>::contains(const point<N> &pt) const
  {
    auto node = findNode(root_, pt);
    return node != NULL && node->point == pt;
  }

  template <std::size_t N, typename elem_type>
  void
  kd_tree<N, elem_type>::insert(const point<N> &pt, const elem_type &value)
  {
    auto targetNode = findNode(root_, pt);
    if (targetNode == NULL)
    { // this means the tree is empty
      root_ = new node(pt, 0, value);
      size_ = 1;
    }
    else
    {
      if (targetNode->point == pt)
      { // pt is already in the tree, simply update its value
        targetNode->value = value;
      }
      else
      { // construct a new node and insert it to the right place (child of targetNode)
        int currLevel = targetNode->level;
        node *newNode = new node(pt, currLevel + 1, value);
        if (pt[currLevel % N] < targetNode->point[currLevel % N])
        {
          targetNode->left = newNode;
        }
        else
        {
          targetNode->right = newNode;
        }
        ++size_;
      }
    }
  }

  template <std::size_t N, typename elem_type>
  const elem_type &
  kd_tree<N, elem_type>::at(const point<N> &pt) const
  {
    auto node = findNode(root_, pt);
    if (node == NULL || node->point != pt)
    {
      throw std::out_of_range("point not found in the KD-Tree");
    }
    else
    {
      return node->value;
    }
  }

  template <std::size_t N, typename elem_type>
  elem_type &
  kd_tree<N, elem_type>::at(const point<N> &pt)
  {
    const kd_tree<N, elem_type> &constThis = *this;
    return const_cast<elem_type &>(constThis.at(pt));
  }

  template <std::size_t N, typename elem_type>
  elem_type &
  kd_tree<N, elem_type>::operator[](const point<N> &pt)
  {
    auto node = findNode(root_, pt);
    if (node != NULL && node->point == pt)
    { // pt is already in the tree
      return node->value;
    }
    else
    { // insert pt with default elem_type value, and return reference to the new elem_type
      insert(pt);
      if (node == NULL)
        return root_->value; // the new node is the root
      else
        return (node->left != NULL && node->left->point == pt) ? node->left->value : node->right->value;
    }
  }

  template <std::size_t N, typename elem_type>
  void
  kd_tree<N, elem_type>::nearestNeighborRecurse(
      const typename kd_tree<N, elem_type>::node *currNode,
      const point<N> &key, bpqueue<elem_type> &pQueue) const
  {
    if (currNode == NULL)
      return;
    const point<N> &currpoint = currNode->point;

    // Add the current point to the BPQ if it is closer to 'key' that some point in the BPQ
    pQueue.enqueue(currNode->value, point_distance(currpoint, key));

    // Recursively search the half of the tree that contains point 'key'
    int currLevel = currNode->level;
    bool isLeftTree;
    if (key[currLevel % N] < currpoint[currLevel % N])
    {
      nearestNeighborRecurse(currNode->left, key, pQueue);
      isLeftTree = true;
    }
    else
    {
      nearestNeighborRecurse(currNode->right, key, pQueue);
      isLeftTree = false;
    }

    if (pQueue.size() < pQueue.maxSize() || fabs(key[currLevel % N] - currpoint[currLevel % N]) < pQueue.worst())
    {
      // Recursively search the other half of the tree if necessary
      if (isLeftTree)
        nearestNeighborRecurse(currNode->right, key, pQueue);
      else
        nearestNeighborRecurse(currNode->left, key, pQueue);
    }
  }

  template <std::size_t N, typename elem_type>
  elem_type
  kd_tree<N, elem_type>::kNNValue(const point<N> &key, std::size_t k) const
  {
    bpqueue<elem_type> pQueue(k); // BPQ with maximum size k
    if (empty())
      return elem_type(); // default return value if KD-tree is empty

    // Recursively search the KD-tree with pruning
    nearestNeighborRecurse(root_, key, pQueue);

    // Count occurrences of all elem_type in the kNN set
    std::unordered_map<elem_type, int> counter;
    while (!pQueue.empty())
    {
      ++counter[pQueue.dequeueMin()];
    }

    // Return the most frequent element in the kNN set
    elem_type result;
    int cnt = -1;
    for (const auto &p : counter)
    {
      if (p.second > cnt)
      {
        result = p.first;
        cnt = p.second;
      }
    }
    return result;
  }

}

#endif
