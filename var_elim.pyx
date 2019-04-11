"""
Inference algorithms in c++

@author: Sergio Luengo-Sanchez and Marco Benjumeda
"""
from libcpp.vector cimport vector
from libcpp.algorithm cimport copy
from libcpp.memory cimport shared_ptr, weak_ptr
from data_type cimport data as datat
from cython.operator cimport dereference as deref, preincrement as inc
import numpy 
cimport numpy as np
from libcpp cimport bool

import multiprocessing
from data_type cimport data
from scoring_functions import score_function
from cython.parallel import prange, threadid
from libcpp.vector cimport vector
import data_type
from copy import deepcopy
from libcpp.memory cimport make_shared
from cython.operator cimport dereference as deref

cdef extern from "math.h":
    double log(double x) nogil


cdef extern from "variable_elimination.h":
    cdef cppclass Factor:
        Factor() except +
        Factor(vector[int] &variables, vector[int] &num_categories) except +
        void learn_params(const int* data, int nrows, int ncols, double alpha)
        double learn_params_se(vector[double] se, double alpha)
        double log_lik_se(vector[double] se, double alpha)
        vector[int] variables 
        vector[int] num_categories
        vector[double] prob # Array with the probabilities 
        vector[UINT] mpe # Most-probable instances 
    
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



cdef class PyFactorTree:
#    cdef Factor_Tree c_factor_tree      
    
    def __cinit__(self,list parents_inner, list parents_leave, list vec_variables_leave, list num_categories):
        cdef vector[int] parents_inner_c
        cdef vector[int] parents_leave_c
        cdef vector[vector[int]] vec_variables_leave_c
        cdef vector[int] num_categories_c
        cdef int p, i, v, c
        cdef list v_l
        
        for p in parents_inner:
            parents_inner_c.push_back(p)
        for p in parents_leave:
            parents_leave_c.push_back(p)
        vec_variables_leave_c.resize(len(vec_variables_leave))
        for i, v_l in enumerate(vec_variables_leave):
            for v in v_l:
                vec_variables_leave_c[i].push_back(v)
        for c in num_categories:
            num_categories_c.push_back(c)
        self.c_factor_tree = Factor_Tree(parents_inner_c, parents_leave, vec_variables_leave_c, num_categories_c)
    
    def num_nodes(self):
        return self.c_factor_tree.inner_nodes.size()
    
    def copy(self):
        cdef PyFactorTree fiti_new = PyFactorTree([-1],[0],[[0]],[2])
        fiti_new.c_factor_tree = Factor_Tree(self.c_factor_tree)
        return fiti_new
#    def __cinit__(self):
#        return

    
    def get_parent(self, int id):
        if id == -1:
            return -1
        elif id < self.c_factor_tree.inner_nodes.size() and id >= 0:
            return deref(deref(self.c_factor_tree.inner_nodes[id]).parent.lock()).id
        elif id < 2*self.c_factor_tree.inner_nodes.size() and id >= 0:
            return deref(deref(self.c_factor_tree.leave_nodes[id-self.c_factor_tree.inner_nodes.size()]).parent.lock()).id
        else:
            raise IndexError('id out of bounds')
        
    def get_children(self, int id):
        if id == -1:
            node= deref(self.c_factor_tree.root)
        elif id < self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.inner_nodes[id])
        elif id < 2*self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.leave_nodes[id-self.c_factor_tree.inner_nodes.size()])
        else:
            raise IndexError('id out of bounds')
        ch = []
        for i in range(node.children.size()):
            ch.append(deref(node.children[i].lock()).id)
        return ch
    
    # Set parameters by hand. Params is a list of lists with the parameters
    # of the nodes
    def set_parameters(self, list nodes, list params):
        cdef vector[double] param_vector 
        for node_i, pari in zip(nodes,params):
            param_vector.clear()
            for parij in pari:
                param_vector.push_back(parij)
            deref(deref(self.c_factor_tree.leave_nodes[node_i]).factor).prob = param_vector

    # get parameters 
    def get_parameters(self, list nodes_in = None):
        cdef PyFactor factor
        cdef list probs = []
        cdef int num_nds = self.c_factor_tree.inner_nodes.size()
        if nodes_in is None:
            nodes = range(num_nds)
        else:
            nodes = nodes_in
        for node_i in nodes:
            factor = self.get_factor(node_i+num_nds)
            probs.append(factor.get_prob())
        return probs
            
            
    
    def learn_parameters(self, datat data, double alpha = 0.0):
        self.c_factor_tree.learn_parameters(data.dat_c, data.nrows, data.ncols, alpha)   
    
    
    # Debugging function
    def get_factor(self, int id): 
        if id == -1:
            node= deref(self.c_factor_tree.root)
        elif id < self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.inner_nodes[id])
        elif id < 2*self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.leave_nodes[id-self.c_factor_tree.inner_nodes.size()])
        else:
            raise IndexError('id out of bounds')
        factor = PyFactor([],[])
        factor.c_factor = deref(node.factor)
        return factor
    
    # Debugging function
    def get_joint_factor(self, int id): 
        if id == -1:
            node= deref(self.c_factor_tree.root)
        elif id < self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.inner_nodes[id])
        elif id < 2*self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.leave_nodes[id-self.c_factor_tree.inner_nodes.size()])
        else:
            raise IndexError('id out of bounds')
        factor = PyFactor([],[])
        factor.c_factor = deref(node.joint_factor)
        return factor  
    
    # Debugging function
    def sum_factor(self, int id, int xi): 
        if id < self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.inner_nodes[id])
        elif id < 2*self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.leave_nodes[id-self.c_factor_tree.inner_nodes.size()])
        else:
            raise IndexError('id out of bounds')
        a = sum_factor(deref(node.factor),xi)
        factor = PyFactor([],[])
        factor.c_factor = deref(a)
        return factor
        
    def prod_factor(self, list ids):
        cdef vector[shared_ptr[Factor]] factors
        for id in ids:        
            if id == -1:
                node= deref(self.c_factor_tree.root)
            elif id < self.c_factor_tree.inner_nodes.size() and id >= 0:
                node = deref(self.c_factor_tree.inner_nodes[id])
            elif id < 2*self.c_factor_tree.inner_nodes.size() and id >= 0:
                node = deref(self.c_factor_tree.leave_nodes[id-self.c_factor_tree.inner_nodes.size()])
            else:
                raise IndexError('id out of bounds')
            factors.push_back(node.factor)
        a = prod_factor(factors)
        
        factor = PyFactor([],[])
        factor.c_factor = deref(a)
        return factor
        
    def max_factor(self, int id, int xi): 
        if id < self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.inner_nodes[id])
        elif id < 2*self.c_factor_tree.inner_nodes.size() and id >= 0:
            node = deref(self.c_factor_tree.leave_nodes[id-self.c_factor_tree.inner_nodes.size()])
        else:
            raise IndexError('id out of bounds')
        a = max_factor(deref(node.factor),xi)
        factor = PyFactor([],[])
        factor.c_factor = deref(a)
        return factor
        
        
    def max_compute(self, int xi):
        self.c_factor_tree.max_compute(xi)
        
    def sum_compute(self, int xi):
        self.c_factor_tree.sum_compute(xi)
    
    def distribute_evidence(self):
        self.c_factor_tree.distribute_evidence()
    
    def distribute_leaves(self, list var, list val):
        cdef vector[int] variables
        cdef vector[int] values
        cdef int ncols = self.c_factor_tree.inner_nodes.size()
        for p in var:
            variables.push_back(p)
        for p in val:
            values.push_back(p)
        self.c_factor_tree.distribute_leaves(variables, values, ncols)
        
    def prod_compute(self, int xi):
        self.c_factor_tree.prod_compute(xi)
        
    def max_compute_tree(self):
        self.c_factor_tree.max_compute_tree()
        
    def sum_compute_tree(self):
        self.c_factor_tree.sum_compute_tree()
    
    def sum_compute_query(self, list var):
        cdef vector[bool] variables
        for p in var:
            variables.push_back(p)
        self.c_factor_tree.sum_compute_query(variables)
    
    def set_evidence(self, list var, list val):
        cdef vector[int] variables
        cdef vector[int] values
        for p in var:
            variables.push_back(p)
        for p in val:
            values.push_back(p)
        self.c_factor_tree.set_evidence(variables, values)

    def retract_evidence(self):
        self.c_factor_tree.retract_evidence()
        
    def mpe_data(self, datat data_complete, datat data_missing, list rows, return_prob = False, parallel=True):
        cdef vector[int] rows_v
        cdef vector[double] mpe_prob
        cdef vector[double].iterator it_prob
        cdef list l_prob = []

        for r in rows:
            rows_v.push_back(r)
        if parallel:
            self.c_factor_tree.mpe_data(data_complete.dat_c, data_missing.dat_c, data_complete.ncols, data_complete.nrows, rows_v, mpe_prob)
        else:
            self.c_factor_tree.mpe_data_oneproc(data_complete.dat_c, data_missing.dat_c, data_complete.ncols, data_complete.nrows, rows_v, mpe_prob)
        if return_prob:
             it_prob = mpe_prob.begin()
             while it_prob != mpe_prob.end():
                 l_prob.append(deref(it_prob))
                 inc(it_prob)
        return l_prob
        
    def se_data(self, datat data_missing, list rows, list weights, list cluster):
        cdef vector[int] rows_v, weights_v
        cdef vector[int] cluster_v
        cdef vector[double] se
        cdef vector[double].iterator it_prob
        cdef list l_prob = []

        for r in rows:
            rows_v.push_back(r)
        for w in weights:
            weights_v.push_back(w)
        for c in cluster:
            cluster_v.push_back(c)
        se = self.c_factor_tree.se_data(data_missing.dat_c, data_missing.ncols, data_missing.nrows, rows_v, weights_v, cluster_v)
        it_prob = se.begin()
        while it_prob != se.end():
            l_prob.append(deref(it_prob))
            inc(it_prob)
        return l_prob
    

    def EM_parameters_distribute(self, datat data_missing, PyFactorTree fiti_old, list rows = None, list weights=None, double alpha = 0.0):
        cdef vector[int] rows_v, weights_v
        cdef vector[int] cluster_v
        cdef vector[double] se
        cdef vector[vector[double]] v_se
        cdef list ll
        ll = []
        if rows is None:
             for i in range(data_missing.nrows):
                rows_v.push_back(i)       
        else:
            for r in rows:
                rows_v.push_back(r)
        if weights is None:
            for _ in range(data_missing.nrows):
                weights_v.push_back(1)
        else:
            for w in weights:
                weights_v.push_back(w)
        
        v_se= fiti_old.c_factor_tree.se_data_distribute(data_missing.dat_c, data_missing.ncols, data_missing.nrows, rows_v, weights_v)
        for xi in range(data_missing.ncols):
            ll.append(deref(deref(self.c_factor_tree.leave_nodes[xi]).factor).learn_params_se(v_se.at(xi), alpha))
        return ll
    
    def EM_parameters(self, datat data_missing, PyFactorTree fiti_old, list rows = None, list weights = None, double alpha = 0.0, list var_update = None, parallel = False):
        cdef vector[int] rows_v, weights_v
        cdef vector[int] cluster_v
        cdef vector[double] se
        cdef list list_se
        cdef list cluster
        cdef list ll
        cdef list vup
        ll = []
        list_se = []
        if rows is None:
             for i in range(data_missing.nrows):
                rows_v.push_back(i)       
        else:
            for r in rows:
                rows_v.push_back(r)
        if weights is None:
            for _ in range(data_missing.nrows):
                weights_v.push_back(1)
        else:
            for w in weights:
                weights_v.push_back(w)
        
        if var_update is None:
            vup = range(data_missing.ncols)
        else:
            vup = var_update
        
        for xi in vup:
            cluster = self.get_factor(data_missing.ncols+xi).get_variables()
            cluster_v.clear()
            for c in cluster:
                cluster_v.push_back(c)
            if parallel:
                print 'EMp size: ', rows_v.size()
                se = fiti_old.c_factor_tree.se_data_parallel(data_missing.dat_c, data_missing.ncols, data_missing.nrows, rows_v, weights_v, cluster_v)
            else:
                se = fiti_old.c_factor_tree.se_data(data_missing.dat_c, data_missing.ncols, data_missing.nrows, rows_v, weights_v, cluster_v)
            list_se.append(se)
        for i,xi in enumerate(vup):
            ll.append(deref(deref(self.c_factor_tree.leave_nodes[xi]).factor).learn_params_se(list_se[i], alpha))
        return ll
    
    
    
    def learn_parameters_se(self, datat data_missing, list rows, list weights, int xi, double alpha = 0.0):
        cdef vector[int] rows_v, weights_v
        cdef vector[int] cluster_v
        cdef vector[double] se
        cdef list cluster
        cdef double ll

        for r in rows:
            rows_v.push_back(r)
        for w in weights:
            weights_v.push_back(w)
        
        cluster = self.get_factor(data_missing.ncols+xi).get_variables()
        for c in cluster:
            cluster_v.push_back(c)
        
        se = self.c_factor_tree.se_data(data_missing.dat_c, data_missing.ncols, data_missing.nrows, rows_v, weights_v, cluster_v)
        
        ll = deref(deref(self.c_factor_tree.leave_nodes[xi]).factor).learn_params_se(se, alpha)
        return ll
        
             
    def log_lik_se(self, datat data_missing, list rows, list weights, int xi, double alpha = 0.0):
        cdef vector[int] rows_v, weights_v
        cdef vector[int] cluster_v
        cdef vector[double] se
        cdef list cluster
        cdef double ll
        
        for r in rows:
            rows_v.push_back(r)
        for w in weights:
            weights_v.push_back(w)
        
        cluster = self.get_factor(data_missing.ncols+xi).get_variables()
        for c in cluster:
            cluster_v.push_back(c)
        
        se = self.c_factor_tree.se_data(data_missing.dat_c, data_missing.ncols, data_missing.nrows, rows_v, weights_v, cluster_v)
        
        ll = deref(deref(self.c_factor_tree.leave_nodes[xi]).factor).log_lik_se(se, alpha)
        return ll   
    
    def log_lik_se_cluster(self, datat data_missing, list rows, list weights, list cluster, double alpha = 0.0):
        cdef vector[int] rows_v, weights_v
        cdef vector[int] cluster_v
        cdef vector[int] num_categories_v
        cdef vector[double] se
        cdef double ll
        
        num_categories_v = deref(deref(self.c_factor_tree.leave_nodes[0]).factor).num_categories
        for r in rows:
            rows_v.push_back(r)
        for w in weights:
            weights_v.push_back(w)
    
        for c in cluster:
            cluster_v.push_back(c)
        
        se = self.c_factor_tree.se_data(data_missing.dat_c, data_missing.ncols, data_missing.nrows, rows_v, weights_v, cluster_v)
        ll = log_lik_se(se, alpha, cluster_v, num_categories_v)
        return ll   
    
    def pred_data(self, datat data, list qvalues_l, list query_l, list evidence_l):
        cdef vector[int] qvalues, query, evidence
        cdef vector[ vector[double]] response
        cdef int x, i, j
        cdef np.ndarray resp
        resp = numpy.zeros((data.nrows, len(query_l)))
        
        for x in qvalues_l:
            qvalues.push_back(x)
        for x in query_l:
            query.push_back(x)
        for x in evidence_l:
            evidence.push_back(x)
        
        response = self.c_factor_tree.pred_data(data.dat_c, data.ncols, data.nrows, qvalues, query, evidence)
        
        for i in range(data.nrows):
            for j in range(len(query_l)): 
                resp[i,j] = response[j][i]
        return resp        
    
    
    def cll(self, datat data, list query_l, list weights = None, parallel = False):
        cdef vector[int] query, weights_v
        if weights is None:
            for _ in range(data.nrows):
                weights_v.push_back(1)
        else:
            for w in weights:
                weights_v.push_back(w)
        for x in query_l:
            query.push_back(x)
        if parallel:
            return self.c_factor_tree.cll_parallel(data.dat_c, data.ncols, data.nrows, query, weights_v)
        else:
            return self.c_factor_tree.cll(data.dat_c, data.ncols, data.nrows, query, weights_v)

    def ll(self, datat data):
#        print 'll size: ', data.nrows
        return self.c_factor_tree.ll_parallel(data.dat_c, data.ncols, data.nrows)     

    def pred_data_distribute(self, datat data, list query_l, list evidence_l):
        cdef vector[int] query, evidence
        cdef vector[ vector[vector[double]]] response
        cdef int x, i, j
        cdef list resp = list()
        
        for x in query_l:
            query.push_back(x)
        for x in evidence_l:
            evidence.push_back(x)
        
        response = self.c_factor_tree.pred_data_distribute(data.dat_c, data.ncols, data.nrows, query, evidence)
        
        for i in range(data.nrows):
            resp.append(list())
            for j in range(len(query_l)):
                resp[i].append(list())
                for k in range(response[i][j].size()):
                    resp[i][j].append(response[i][j][k])
        return resp      

    def pseudo_likelihood(self, datat data, int xi, list ch_xi_l):
        cdef vector[int] ch_xi
        cdef int x
        for x in ch_xi_l:
            ch_xi.push_back(x)
        
        return self.c_factor_tree.pseudo_likelihood(data.dat_c, data.ncols, data.nrows, xi, ch_xi)
    
    def pseudo_cll_score(self, datat data, int change_type,int xout, int xin, list class_vars_l, list pseudocll_old_l):
        cdef vector[int] ch_xi, class_vars
        cdef vector[double] pseudocll_old
        cdef vector[double] pseudocll
        cdef vector[vector[int]] children
        cdef int x, i, xj
        cdef double pcll
        cdef list vxi
        cdef list pseudocll_l = []
        for x in class_vars_l:
            class_vars.push_back(x)
        for pcll in pseudocll_old_l:
            pseudocll_old.push_back(pcll)

        children.resize(data.ncols)
        
        for i in range(data.ncols):
            vxi = self.get_factor(data.ncols+i).get_variables()
            for xj in vxi:
                if xj!=i:
                    children[xj].push_back(i)
                    
        pseudocll = self.c_factor_tree.pseudo_cll_score(change_type,xout, xin, data.dat_c, data.ncols, data.nrows, children, class_vars, pseudocll_old)
        for i in range(pseudocll.size()):
            pseudocll_l.append(pseudocll[i])
        return pseudocll_l
        



cdef class PyFactor:
#    cdef Factor c_factor      # hold a C++ instance which we're wrapping
    
    def __cinit__(self,list variables, list num_categories):
        cdef vector[int] variables_c
        cdef vector[int] num_categories_c
        for v in variables:
            variables_c.push_back(v)
        for c in num_categories:
            num_categories_c.push_back(c)
        self.c_factor = Factor(variables_c, num_categories_c)
    
    def learn_params(self, datat data, double alpha = 0.0):
        self.c_factor.learn_params(data.dat_c, data.nrows, data.ncols, alpha)
        
    # Get variables in the domain of the factor
    def get_variables(self):
        cdef vector[int].iterator it_variables = self.c_factor.variables.begin()
        cdef list l_variables = []
        while it_variables != self.c_factor.variables.end():
            l_variables.append(deref(it_variables))
            inc(it_variables)
        return l_variables
        
    # Get list of probabilities of the factor
    def get_prob(self):
        cdef vector[double].iterator it_prob = self.c_factor.prob.begin()
        cdef list l_prob = []
        while it_prob != self.c_factor.prob.end():
            l_prob.append(deref(it_prob))
            inc(it_prob)
        return l_prob
            
    # Get current mpe from the factor
    def get_mpe(self):
        cdef vector[vector[UINT]].iterator it_mpe = self.c_factor.mpe.begin()
        cdef vector[UINT].iterator it_mpe_i
        cdef list l_mpe = []
        cdef list l_mpe_i
        while it_mpe != self.c_factor.mpe.end():
            it_mpe_i = deref(it_mpe).begin()
            l_mpe_i = []
            while it_mpe_i != deref(it_mpe).end():
                l_mpe_i.append(deref(it_mpe_i))
                inc(it_mpe_i)
            l_mpe.append(l_mpe_i)
            inc(it_mpe)
        return l_mpe
    
    # Get the number of categories of the variables in the factor
    def get_num_categories(self):
        cdef vector[int].iterator it_variables = self.c_factor.variables.begin()
        cdef list l_num_categories = []
        while it_variables != self.c_factor.variables.end():
            l_num_categories.append(self.c_factor.num_categories[deref(it_variables)])
            inc(it_variables)
        return l_num_categories
    



cdef np.ndarray get_adjacency_from_et(object et):
    cdef int i,j
    cdef int num_nds = et.nodes.num_nds
    cdef np.ndarray adj = numpy.zeros([num_nds,num_nds],dtype = numpy.int32)
    for i in range(et.nodes.num_nds):
        for j in et.nodes[i].parents.display():
            adj[i,j] = 1
    return adj


# Get matrix with allowed changes. TODO: Check for more efficient ways if necessary
def get_possible_changes(et, list forbidden_parents, int u=10, add_only=False):
    # Possible arc additions
    cdef int num_nds = et.nodes.num_nds
    cdef int i,j
    cdef list pred = [et.get_preds_bn(i) for i in range(num_nds)]
    cdef list desc = [et.get_desc_bn(i) for i in range(num_nds)]
    add_mat = numpy.zeros([num_nds,num_nds],dtype = numpy.int32) + 1
    remove_mat = numpy.zeros([num_nds,num_nds],dtype = numpy.int32)
    reverse_mat = numpy.zeros([num_nds,num_nds],dtype = numpy.int32)
    numpy.fill_diagonal(add_mat,0)

    for i in xrange(num_nds):
        prd = list(pred[i])
        add_mat[i,prd] = 0
        add_mat[et.nodes[i].parents.display(),i] = 0
        add_mat[i,i] = 0
        if et.nodes[i].parents.len_py() >= u:
            add_mat[:,i] = 0
        else:
            add_mat[forbidden_parents[i],i] = 0
    if not add_only:
        # 2. Possible edges to remove
#        remove_mat = get_adjacency_from_et(et)
        for i in xrange(et.nodes.num_nds):
            for j in et.nodes[i].parents.display():
                remove_mat[j,i] = 1
        
            # 3. Possible edges to reverse

        for i in range(et.nodes.num_nds):
            for j in et.nodes[i].parents.display():
                if len(desc[j].intersection(pred[i]))==0:
                    reverse_mat[j,i] = 1
        for i in range(et.nodes.num_nds):
            if et.nodes[i].parents.len_py() >= u:
                reverse_mat[i,:] = 0
            else:
                reverse_mat[i,forbidden_parents[i]] = 0
    return [add_mat, remove_mat, reverse_mat]

    
# Return matrix with required and cached results
def get_changes_from_matrix(lop_mats):
    [add_mat, remove_mat, reverse_mat] = lop_mats
    additions = numpy.argwhere(add_mat)
    removals = numpy.argwhere(remove_mat)
    reversals = numpy.argwhere(reverse_mat)
    return [additions, removals, reversals]

def get_indexes_int(long[:] names, long[:] index, int l_idx):
    cdef long[:] new_names = numpy.zeros(l_idx, dtype=int)
    cdef int i
    for i in range(l_idx):
        new_names[i] = names[index[i]]
    return numpy.array(new_names, dtype=int)

def get_indexes_double(double[:] names, long[:] index, int l_idx):
    cdef double[:] new_names = numpy.zeros(l_idx, dtype=int)
    cdef int i
    for i in range(l_idx):
        new_names[i] = names[index[i]]
    return numpy.array(new_names, dtype=numpy.double)


def get_indexes_int2d(long[:,:] names, long[:] index, int l_idx):
    cdef long[:,:] new_names = numpy.zeros((l_idx,2), dtype=int)
    cdef int i
    for i in range(l_idx):
        new_names[i,:] = names[index[i],:]
    return numpy.array(new_names, dtype=int)    

def sort_and_filter_op(lop, op_names, score_difs, filter_l0 = True):
    if filter_l0:
        flag = [score_difs > 0]
        score_difs_f = score_difs[flag]
        lop_f = lop[flag]
        op_names_f = op_names[flag]
    else: 
        score_difs_f = score_difs
        lop_f = lop
        op_names_f = op_names
    ord_op = numpy.argsort(score_difs_f)[::-1]
    lop1 = lop_f[:,0]
    lop2 = lop_f[:,1]
    lop_f2= numpy.array([lop1[ord_op],lop2[ord_op]]).transpose()
    return lop_f2, op_names_f[ord_op], score_difs_f[ord_op]
 
    
cdef double penalization(int nrows, vector[int] variables, vector[int] ncat, int stype) nogil:
    cdef double pen
    cdef long b
    cdef int i
    pen = 0
    
    # Obtain parameters
    if stype == 0 or stype == 1:
        b = ncat[variables[0]] - 1
        for i in range(1,variables.size()):
            b *= ncat[variables[i]]
        #aic
        if stype == 0:
            pen =  b
        else: 
            pen = (1.0 / 2.0) * log(nrows) * b
    return pen


cdef vector[int] list_2_vector(list l):
    cdef vector[int] v
    for li in l:
        v.push_back(li)
    return v
    
cdef vector[vector[int]] list_2_vector_2d(list l):
    cdef vector[vector[int]] v
    v.resize(len(l))
    for i,li in enumerate(l):
        for lij in li:
            v[i].push_back(lij)
    return v




def compute_scores_pseudo_cll(data data, object et, object score_old, object additions_in, object removals_in, object reversals_in , object cache_in, str stype_s, list class_vars): 
    cdef double scorei, sum_score_old, sum_score_new
    cdef int nvars = data.ncols
    cdef int nrows = data.nrows
    cdef int* dat_c = data.dat_c
    cdef int i, j, xj 
    cdef int its_add, its_remove, its_reverse
    cdef int xin, xout, idx_xout
    cdef int stype
    cdef vector[int] ncat, variables_pen
    cdef np.ndarray score_difs = numpy.zeros(len(additions_in) + len(removals_in) + len(reversals_in), dtype = numpy.double)
    cdef double [:] score_mv = score_difs
    cdef np.ndarray score_old_aux = numpy.array(score_old, dtype=numpy.double)
    cdef double [:] score_old_mv = score_old_aux
    cdef np.ndarray additions = numpy.array(additions_in, dtype=np.dtype("i"))
    cdef int [:,:] additions_mv
    cdef np.ndarray cache = numpy.array(cache_in, dtype=numpy.double)
    cdef double [:,:] cache_mv = cache
    cdef np.ndarray removals = numpy.array(removals_in, dtype=np.dtype("i"))
    cdef int [:,:] removals_mv
    cdef np.ndarray reversals = numpy.array(reversals_in, dtype=np.dtype("i"))
    cdef int [:,:] reversals_mv
    cdef vector[int] vxj, class_vars_v, vxi
    cdef double npmax = 1000000
    cdef vector[double] score_new, pseudocll_new, pseudocll_old
    cdef vector[vector[int]] children, variables
    cdef shared_ptr[Factor_Tree] ftp
    cdef Factor_Tree ft
#    cdef int cores

    pseudocll_new.resize(nvars)
    score_new.resize(nvars)
    if stype_s == 'aic':
        stype = 0
    elif stype_s == 'bic':
        stype = 1
    else: #ll
        stype = 2
        
#    cores=multiprocessing.cpu_count()
    # Init ft
    et_descriptor = [[et.nodes[i].parent_et for i in range(nvars)], [et.nodes[i].nFactor for i in range(nvars)], [[i] + et.nodes[i].parents.display() for i in range(nvars)], [len(c) for c in data.classes]]   
    ftp = make_shared[Factor_Tree](list_2_vector(et_descriptor[0]), list_2_vector(et_descriptor[1]), list_2_vector_2d(et_descriptor[2]),list_2_vector(et_descriptor[3]))
    deref(ftp).learn_parameters(dat_c, nrows, nvars, alpha=0)
    ft = deref(ftp)
    
    # Init children, variables, class_vars and pseudocll_old    
    children.resize(nvars)
    variables.resize(nvars)
    for i in range(nvars):
        vxi = deref(deref(ft.leave_nodes[i]).factor).variables
        for xj in vxi:
            variables[i].push_back(xj)
            if xj!=i:
                children[xj].push_back(i)
    for i in class_vars:
        class_vars_v.push_back(i)
    for i in range(nvars):
        pseudocll_old.push_back(0)
    for i in class_vars:
        pseudocll_old[i] = ft.pseudo_likelihood(dat_c, nvars, nrows, i, children[i])
    
        
    if len(additions_in) > 0:
        additions_mv = additions
    if len(removals_in) > 0:
        removals_mv = removals
    if len(reversals_in) > 0:
        reversals_mv = reversals
    
    for i in range(nvars):
        ncat.push_back(len(data.classes[i]))
    
    # Arc additions
    its_add = len(additions)
#    for i in prange(its_add, nogil=True, num_threads=cores):
    for i in range(its_add):
        xout = additions_mv[i,0]
        xin = additions_mv[i,1]
        
        if cache_mv[xout,xin] != npmax:
            score_mv[i] = cache_mv[xout,xin]
        else:
            #DEBUG
#            sum_score_old_old = 0
#            for j in range(nvars):
#                sum_score_old_old = sum_score_old_old + pseudocll_old[j]
            #DEBUG
#            openmp.omp_set_lock(&lock)
            pseudocll_new = pseudo_cll_score(ft,0,xout, xin, dat_c, nvars, nrows, children, class_vars_v, pseudocll_old)
#            openmp.omp_unset_lock(&lock)
            
            #Penalization
            for j in range(pseudocll_new.size()):
                variables_pen = variables[j]
                if j== xin:
                    variables_pen.push_back(xout)
                score_new[j] =  pseudocll_new[j] - penalization(nrows,variables_pen,ncat,stype) 
            
            #Difference with old score
            sum_score_new = 0
            sum_score_old = 0
            for j in range(pseudocll_new.size()):
                sum_score_new = sum_score_new + score_new[j]
                sum_score_old = sum_score_old + score_old_mv[j]

            # Create new factor for xin with xout as parent
            cache_mv[xout,xin] = sum_score_new - sum_score_old
            score_mv[i] = sum_score_new - sum_score_old 
            #DEBUG
#            if xout==1015 and xin==1044:
##                print "update 1015 -> 1044"
#                sum_cll_old = 0
#                sum_cll_new = 0
#                for xi in range(nvars):
#                    sum_cll_old += pseudocll_old[xi]
#                    sum_cll_new += pseudocll_new[xi]
#                
#                print "--------------"
#                print 'diff score: ', sum_score_new - sum_score_old
#                print 'diff ll: ', sum_cll_new - sum_cll_old
#                print 'diff old: ', sum_cll_old -sum_score_old_old
#                print "--------------"
        
        #DEBUG
#        sum_score_all = 0
#        for xi in class_vars:
#            pseudo = ft.pseudo_likelihood(dat_c, nvars, nrows, xi, children[xi])
#            sum_score_all += pseudo - penalization(nrows,variables[j],ncat,stype) 
#            print "pseudo new: ", pseudocll_new[xi], "pseudo all: ", pseudo
#        
#        print 'sum_score_new: ', sum_score_new
#        print 'sum_score_old: ', sum_score_old
#        print 'sum_score_all: ', sum_score_all
    
    # Arc removals
    its_remove = len(removals)
#    for i in prange(its_remove, nogil=True, num_threads=cores):
    for i in range(its_remove):
        xout = removals_mv[i,0]
        xin = removals_mv[i,1]
        
        if cache_mv[xout,xin] != npmax:
            score_mv[its_add + i] = cache_mv[xout,xin]
        else:
#            openmp.omp_set_lock(&lock)
            pseudocll_new = pseudo_cll_score(ft,1,xout, xin, dat_c, nvars, nrows, children, class_vars_v, pseudocll_old)
#            openmp.omp_unset_lock(&lock)
            
            #Penalization
            for j in range(pseudocll_new.size()):
                if j== xin:
                    variables_pen.clear()
                    for xj in variables[j]:
                        if xj!=xout:
                            variables_pen.push_back(xj)
                else:
                    variables_pen = variables[j]
                score_new[j] =  pseudocll_new[j] - penalization(nrows,variables_pen,ncat,stype) 
            
            #Difference with old score
            sum_score_new = 0
            sum_score_old = 0
            for j in range(pseudocll_new.size()):
                sum_score_new = sum_score_new + score_new[j]
                sum_score_old = sum_score_old + score_old_mv[j]

            # Create new factor for xin with xout as parent
            cache_mv[xout,xin] = sum_score_new - sum_score_old
            score_mv[its_add + i] = sum_score_new - sum_score_old 

    # Arc reversals
    its_reverse = len(reversals)
#    for i in prange(its_reverse, nogil=True, num_threads=cores):
    for i in range(its_reverse):

        xout = reversals_mv[i,0]
        xin = reversals_mv[i,1]
        
        if cache_mv[xin,xout] != npmax:
            score_mv[its_add + its_remove + i] = cache_mv[xout,xin]
        else:
#            openmp.omp_set_lock(&lock)
            pseudocll_new = pseudo_cll_score(ft,2,xout, xin, dat_c, nvars, nrows, children, class_vars_v, pseudocll_old)
#            openmp.omp_unset_lock(&lock)
            
            #Penalization
            for j in range(pseudocll_new.size()):
                variables_pen = variables[j]
                if j== xin:
                    variables_pen.clear()
                    for xj in variables[j]:
                        if xj!=xout:
                            variables_pen.push_back(xj)
                    
                elif j == xout:
                    variables_pen.push_back(xin)
                    
                score_new[j] =  pseudocll_new[j] - penalization(nrows,variables_pen,ncat,stype) 
            
            #Difference with old score
            sum_score_new = 0
            sum_score_old = 0
            for j in range(pseudocll_new.size()):
                sum_score_new = sum_score_new + score_new[j]
                sum_score_old = sum_score_old + score_old_mv[j]

            # Create new factor for xin with xout as parent
            cache_mv[xin,xout] = sum_score_new - sum_score_old
            score_mv[its_add + its_remove + i] = sum_score_new - sum_score_old 
    return score_difs, cache    

def best_pred_cache_pseudo_cll(data, class_vars, et, metric, score_old, forbidden_parents, cache, filter_l0 = True, u=10, add_only=False):
    lop_mats = get_possible_changes(et, forbidden_parents, u, add_only)
    lop = get_changes_from_matrix(lop_mats)
    
    score_difs, cache = compute_scores_pseudo_cll(data,et, score_old, lop[0], lop[1], lop[2] , cache, metric, class_vars) 
    
    op_add = numpy.full(len(lop[0]),0,dtype=int)
    op_remove = numpy.full(len(lop[1]),1,dtype=int)
    op_reverse = numpy.full(len(lop[2]),2,dtype=int)
    
    op_names = numpy.concatenate([op_add, op_remove, op_reverse])
    lop_o, op_names_o, score_difs_o = sort_and_filter_op(numpy.concatenate(lop), op_names, score_difs, filter_l0 = filter_l0)
    return lop_o, op_names_o, score_difs_o, cache

        
        
