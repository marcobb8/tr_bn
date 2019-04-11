from libcpp.vector cimport vector
from libcpp.algorithm cimport copy
from libcpp.memory cimport shared_ptr, weak_ptr
from data_type cimport data as datat
from cython.operator cimport dereference as deref, preincrement as inc
cimport numpy as np
import numpy as np
from libcpp cimport bool
ctypedef int UINT
cdef extern from "variable_elimination.h":
    cdef cppclass Factor:
        Factor() except +
        Factor(vector[int] variables, vector[int] num_categories) except +
        void learn_params(const int* data, int nrows, int ncols, double alpha)
        double learn_params_se(vector[double] se, double alpha)
        double log_lik_se(vector[double] se, double alpha)
        vector[int] variables 
        vector[int] num_categories
        vector[double] prob # Array with the probabilities 
        vector[vector[UINT]] mpe # Most-probable instances 
    
    cdef cppclass Factor_Node:
        Factor_Node() except +
        Factor_Node(vector[int] variables, vector[int] num_categories, int id) except +
        Factor_Node(const Factor_Node& fnode) except +
        Factor_Node(shared_ptr[Factor] factor, int id) except +
        Factor_Node(int id) except +
    
        shared_ptr[Factor] factor
        shared_ptr[Factor] joint_factor
        vector[weak_ptr[Factor_Node]] children
        weak_ptr[Factor_Node] parent 
        int id
      
    cdef cppclass Factor_Tree nogil:
        Factor_Tree() except +
        Factor_Tree(const Factor_Tree& ft) except +
        Factor_Tree(vector[int] parents_inner, vector[int] parents_leave, vector[vector[int]] vec_variables_leave, vector[int] num_categories) except +
        void learn_parameters(const int* data, int nrows, int ncols, double alpha) nogil
        void max_compute(int xi) nogil
        void sum_compute(int xi) nogil
        void distribute_evidence() nogil
        void distribute_leaves(vector[int] variables, vector[int] values, int ncols) nogil
        void prod_compute(int xi) nogil
        void max_compute_tree() nogil
        void sum_compute_tree() nogil
        void sum_compute_query(const vector[bool] variables) nogil
        void set_evidence(const vector[int] variables, const vector[int] values) nogil
        void retract_evidence() nogil
        void mpe_data(int* data_complete, const int* data_missing, int ncols, int nrows, vector[int] rows, vector[double] &prob_mpe) nogil
        void mpe_data_oneproc(int* data_complete, const int* data_missing, int ncols, int nrows, vector[int] rows, vector[double] &prob_mpe) nogil
        vector[double] se_data(int* data_missing, int ncols, int nrows, vector[int] rows, vector[int] weights, vector[int] cluster) nogil
        vector[double] se_data_parallel(int* data_missing, int ncols, int nrows, vector[int] rows, vector[int] weights, vector[int] cluster)
        vector[vector[double]] se_data_distribute(int* data_missing, int ncols, int nrows, vector[int] rows, vector[int] weights) nogil
        vector[vector[double]] pred_data(int* data, int ncols, int nrows, vector[int] qvalues, vector[int] query, vector[int] evidence) nogil
        double cll(int* data, int ncols, int nrows, vector[int] query, vector[int] weights) nogil
        double cll_parallel(int* data, int ncols, int nrows, vector[int] query, vector[int] weights) nogil
        double ll_parallel(int* data, int ncols, int nrows) nogil
        vector[vector[vector[double]]] pred_data_distribute(int* data, int ncols, int nrows, vector[int] query, vector[int] evidence) nogil
        double pseudo_likelihood(int* data, int ncols, int nrows, int xi, vector[int] ch_xi) nogil
        vector[double] pseudo_cll_score(int change_type,int xout, int xin, int* data, int ncols, int nrows, vector[vector[int]] &children, vector[int] &class_vars, vector[double] &pseudocll_old) nogil

        
        vector[shared_ptr[Factor_Node]] inner_nodes, leave_nodes
        shared_ptr[Factor_Node] root
    
    shared_ptr[Factor] sum_factor(const Factor &f1, int xi)
    shared_ptr[Factor] prod_factor(const vector[shared_ptr[Factor]] factors)
    shared_ptr[Factor] max_factor(const Factor &f1, int xi)
    double log_lik_se(vector[double] se, double alpha, vector[int] cluster, vector[int] num_categories)
    vector[double] pseudo_cll_score(const Factor_Tree &ft, int change_type,int xout, int xin, int* data, int ncols, int nrows, vector[vector[int]] &children, vector[int] &class_vars, vector[double] &pseudocll_old) nogil

cdef class PyFactor:
    cdef Factor c_factor
    
cdef class PyFactorTree:
    cdef Factor_Tree c_factor_tree 
    