"""
MBC learning methods and auxuliar functions

@author: Marco Benjumeda
"""


# -------------------------------------------------------#
# Learn structure of elimination tree
# -------------------------------------------------------#

from copy import deepcopy
import data_type
from scoring_functions import score_function
from elimination_tree import ElimTree
import multiprocessing
import numpy
from tree_width import greedy_tree_width
from export import get_adjacency_matrix_from_et
import numpy as np
from gs_structure import best_pred_cache, search_first_tractable_structure
from var_elim_data import PyFactorTree_data, Indicators_vector
from var_elim import PyFactorTree
from learn_structure_cache import hill_climbing_cache
from utils import get_subgraph


def learn_mbc_cll(data_frame, cll_query, et0=None, u=5, metric='bic', tw_bound_type='b', tw_bound=5,  cores=multiprocessing.cpu_count(), forbidden_parent=None, add_only = False, custom_classes=None, constraint = True, alpha = 1, verbose=False):
    """Discriminative learning of MBCs
    Args:
        data_frame (pandas.DataFrame): Input data
        cll_query: list of query variables, the rest are treated as evidence
        et0 (elimination_tree.ElimTree): Initial elimination tree (optional)
        u (int): maximum number of parents allowed
        metric (str): scoring functions
        tw_bound_type (str): 'b' bound, 'n' none
        tw_bound (float): tree-width bound
        cores (int): Number of cores
        forbidden_parent (list): blacklist with forbidden parents
        add_only (bool): If true, allow only arc additions
        custom_classes: If not None, the classes of each variable are set to custom_classes
        constraint: If true, the additions and reversals that exceed the treewidth bound are stored in a blacklist
        alpha: Dirichlet prior for Bayesian parameter estimation
        verbose (bool): If True, print details of the learning process 

    Returns:
        elimination_tree.ElimTree Learned MBC
    """
    #Initialization
    count_i = 0
    if custom_classes is None:
        data = data_type.data(data_frame)
    else:
        data = data_type.data(data_frame, classes= custom_classes)
    
    if et0 is None:
        et = ElimTree('et', data.col_names, data.classes)
    else:
        et = et0.copyInfo()
    
    data_cll = data_frame
    num_nds = et.nodes.num_nds
    data_complete = data_type.data(data_cll, classes= data.classes)
    #Current selected variables. For efficiency purpose, the variables that are part of the graph are not use to compute the score 
    sel_vars = cll_query + get_subgraph(et,cll_query)
    
    
    # Create initial FT
    ftd = PyFactorTree_data([et.nodes[i].parent_et for i in range(num_nds)], [et.nodes[i].nFactor for i in range(num_nds)], [[i]+et.nodes[i].parents.display() for i in range(num_nds)], [len(data.classes[i]) for i in range(num_nds)])
    ftd.learn_parameters(data_complete, alpha = 0) 
    score_best_cll = 0    
    
    # Initialize indicator vectors with all the data
    num_splits = numpy.ceil(data.nrows / (1000.0*8.0))*8
    indv = Indicators_vector(data_cll, data.classes, num_splits)  
    df_ev = data_cll.copy()      
    df_ev.iloc[:,cll_query] = np.nan
    ind_ev = Indicators_vector(df_ev, data.classes, num_splits)


    if forbidden_parent is None:
        forbidden_parents = [[] for _ in range(data.ncols)]
    else:
        forbidden_parents = forbidden_parent
    
    # Compute score of the first model
    score_best_cll = compute_cll(indv, ind_ev, et, ftd, cll_query, data.classes)
    
    for i in range(num_nds):
        score_best_cll += score_function(data, i, et.nodes[i].parents.display(), metric, [],ll_in = 0)      
    # BIC score
    score_best = numpy.array([score_function(data_complete, i, et.nodes[i].parents.display(), metric=metric) for i in range(num_nds)],dtype = numpy.double)
    # Cache
    cache = numpy.full([et.nodes.num_nds,et.nodes.num_nds],1000000,dtype=numpy.double)
    #loop hill-climbing
    ok_to_proceed_hc = True
    while ok_to_proceed_hc:
        count_i += 1
        # Get FP for the MBC
        forbidden_parents_mbc = forbidden_mbc(et, cll_query, forbidden_parents)

        # Input, Output and new
        lop_o, op_names_o, score_difs_o, cache =  best_pred_cache(data_complete, et, metric, score_best, forbidden_parents_mbc, cache, filter_l0 = True, add_only = add_only)
        
        ok_to_proceed_hc = False
        while len(lop_o) > 0:
            if tw_bound_type=='b':
                change_idx, et_new = search_first_tractable_structure(et, lop_o, op_names_o, tw_bound, forbidden_parents, constraint)
            else:
                change_idx, et_new = search_first_tractable_structure(et, lop_o, op_names_o, 10000)   
            if change_idx != -1:
                best_lop = lop_o[change_idx]
                xout = best_lop[0]
                xin = best_lop[1]
                best_op_names = op_names_o[change_idx]
                best_score_difs = score_difs_o[change_idx]
                                   
                #Compute parameters of the new ET
                ftd_aux = PyFactorTree_data([et_new.nodes[i].parent_et for i in range(et_new.nodes.num_nds)], [et_new.nodes[i].nFactor for i in range(et_new.nodes.num_nds)], [[i]+et_new.nodes[i].parents.display() for i in range(et_new.nodes.num_nds)], [len(c) for c in data.classes])
        
                if best_op_names == 2:
                    nodes_change = best_lop.tolist()
                else:
                    nodes_change = [best_lop[1]]

                nodes_copy = range(num_nds)
                for xi in nodes_change:
                    nodes_copy.remove(xi)
                ftd_aux.learn_parameters(data,alpha=0)
                 
                # Compute real score
                score_obs_new_real = compute_cll(indv, ind_ev, et_new, ftd_aux, cll_query, data.classes)
                for i in range(num_nds):
                    score_obs_new_real += score_function(data, i, et_new.nodes[i].parents.display(), metric, [],ll_in = 0) 
                if score_obs_new_real > score_best_cll:
                    ok_to_proceed_hc = True
                    ftd = ftd_aux
                    score_diff = score_obs_new_real - score_best_cll
                    score_best_cll = score_obs_new_real
                    et = et_new
                    # Update score and cache
                    if best_op_names == 2:
                        score_best[xin] += cache[xout, xin]
                        score_best[xout] += cache[xin, xout]
                        cache[:,xout]=1000000
                    else:
                        score_best[xin] += best_score_difs
                    cache[:,xin]=1000000
                    lop_o = []
                    sel_vars = cll_query + get_subgraph(et,cll_query)
                    if verbose:
                        print "iteration ", count_i, ', change: ', [best_op_names, xout, xin], ', tw: ', et.tw(), ', score_diff: ', score_diff,', len(sel_vars_aux): ', len(sel_vars)
                else:
#                        ok_to_proceed_hc = False
                    if best_op_names == 0:
                        forbidden_parents[best_lop[1]].append(best_lop[0])
                    elif best_op_names == 2:
                        forbidden_parents[best_lop[0]].append(best_lop[1])
                    
                lop_o = lop_o[(change_idx+1):] 
                op_names_o = op_names_o[(change_idx+1):] 
            else:
                ok_to_proceed_hc = False
                lop_o = []
    
    return et


def learn_mbc_generative(data_frame, query_cll, pruned=True, et0=None, u=5, metric='bic', tw_bound_type='b', tw_bound=5,  cores=multiprocessing.cpu_count(), forbidden_parent=None, add_only = False, custom_classes=None, constraint = True, verbose=False):
    """Learning MBCs with bounded treewidth (pruned graph)
    Args:
        data_frame (pandas.DataFrame): Input data
        query_cll: list of query variables, the rest are treated as evidence
        pruned (bool): if true, bound the pruned graph. Otherwise, bound the complete graph
        et0 (elimination_tree.ElimTree): Initial elimination tree (optional)
        u (int): maximum number of parents allowed
        metric (str): scoring functions
        tw_bound_type (str): 'b' bound, 'n' none
        tw_bound (float): tree-width bound
        cores (int): Number of cores
        forbidden_parent (list): blacklist with forbidden parents
        add_only (bool): If true, allow only arc additions
        custom_classes: If not None, the classes of each variable are set to custom_classes
        constraint: If true, the additions and reversals that exceed the treewidth bound are stored in a blacklist
        verbose (bool): If True, print details of the learning process 

    Returns:
        elimination_tree.ElimTree Learned MBC
    """    
    n = data_frame.shape[1]
    features = list(set([i for i in range(n)]).difference(query_cll))
    if forbidden_parent is None:
        forbidden_parents = [[] for _ in range(n)]
    else:
        forbidden_parents = forbidden_parent
    #learn class and bridge subgraph
    for xi in query_cll:
        forbidden_parents[xi] = list(set(forbidden_parents[xi] + features))
    for xj in features:
        forbidden_parents[xj] = list(set(forbidden_parents[xj] + features))
    et = hill_climbing_cache(data_frame, et0=et0, u=u, metric=metric, tw_bound_type=tw_bound_type, tw_bound=tw_bound,  cores=cores, forbidden_parent=forbidden_parents, add_only = add_only, custom_classes=custom_classes, constraint = constraint, verbose=verbose)
    if pruned:
        #Get topological ordering of the ET, for later compilation
        order_pruned = et.getDesc_py(-1)
        order_pruned.reverse()
        #Feature variables are positioned in the tail of the order
        for xi in features:
            order_pruned.remove(xi)
        order_pruned = order_pruned + features  
    
    # Learn feature subgraph
    if forbidden_parent is None:
        forbidden_parents = [[] for _ in range(n)]
    else:
        forbidden_parents = forbidden_parent
    for xi in query_cll:
        forbidden_parents[xi] = list(set(forbidden_parents[xi] + query_cll + features))
    for xi in features:
        forbidden_parents[xi] = list(set(forbidden_parents[xi] + query_cll))
    forbidden_parents = forbidden_mbc(et, query_cll, forbidden_parent = forbidden_parents)
    if pruned:
        et = hill_climbing_cache(data_frame, et0=et, u=u, metric=metric, tw_bound_type='n',  cores=cores, forbidden_parent=forbidden_parents, add_only = add_only, custom_classes=custom_classes, constraint = constraint)
        et.compile_ordering(order_pruned)
    else:
        et = hill_climbing_cache(data_frame, et0=et, u=u, metric=metric, tw_bound_type=tw_bound_type, tw_bound=tw_bound,  cores=cores, forbidden_parent=forbidden_parents, add_only = add_only, custom_classes=custom_classes, constraint = constraint, verbose=verbose)
    if tw_bound_type == 'n':
        adj = get_adjacency_matrix_from_et(et)
        order, _ = greedy_tree_width(adj, method='fill')
        et.compile_ordering(order.tolist())        
    return et

    
#Transforms vector so that it makes sense after prunning the model for sel_vars:
#All the values in v must be in sel_var
def transform_to_selvar(sel_var, v):
    laux = {}   
    for i,xi in enumerate(sel_var):
        laux[xi] = i
    v_res = [-1 for xi in v]
    for i,vi in enumerate(v):
        if vi != -1:
            v_res[i] = laux[vi]
    return v_res
    
# Filter dataset features according to score. Returns the filtered dataset, and the indexes of the variables from the original dataset
def filter_dataset(data_frame, cll_query, custom_classes , metric='bic',  cores=multiprocessing.cpu_count()):
    data = data_type.data(data_frame, classes= custom_classes)
    et = ElimTree('et', data.col_names, data.classes)
    num_nds = et.nodes.num_nds
    forbidden_parents = [[] for _ in range(data.ncols)]
    forbidden_parents = forbidden_mbc(et, cll_query, forbidden_parents)
    for i in range(len(forbidden_parents)):
        if i in cll_query:
            forbidden_parents[i] = forbidden_parents[i] + cll_query
    score_best = np.array([score_function(data, i, [], metric=metric) for i in range(num_nds)],dtype = np.double)
    cache = np.full([et.nodes.num_nds,et.nodes.num_nds],1000000,dtype=np.double)
    lop_o, op_names_o, score_difs_o, cache =  best_pred_cache(data, et, metric, score_best, forbidden_parents, cache, filter_l0 = True, add_only = True)
    
    selected = cll_query + np.unique(lop_o[:,1]).tolist()
    data_filtered = data_frame.iloc[:,selected]
    return data_filtered, selected


# Computes conditional log-likelihood, filtering those features that are not connected to the class variables
# indv is the indicator vector, ind_ev is the indicator vector of the feature variables, et is the elimination tree,
# ftd is the current factor tree over all variables (used to copy the parameters), and cll_query are the query variables
def compute_cll(indv, ind_ev, et, ftd_all, cll_query, classes):
    sel_vars = cll_query + get_subgraph(et,cll_query)
    sel_vars_aux = list(sel_vars)
    for xi in sel_vars_aux:
        sel_vars = list(set(sel_vars + et.nodes[xi].parents.display()))
    # Copy parameters in FT  with sel_var
    et_descriptor = [[et.nodes[i].parent_et for i in sel_vars], [et.nodes[i].nFactor for i in sel_vars], [[i]+et.nodes[i].parents.display() for i in sel_vars], [len(classes[i]) for i in sel_vars]]
    et_desc_new = [transform_to_selvar(sel_vars,et_descriptor[0]), transform_to_selvar(sel_vars,et_descriptor[1]), [transform_to_selvar(sel_vars,etd2) for etd2 in et_descriptor[2]],et_descriptor[3]]
    ftd = PyFactorTree_data(et_desc_new[0],et_desc_new[1],et_desc_new[2],et_desc_new[3])
    indv_aux = indv.subset_indicators(sel_vars)
    ind_ev_aux = ind_ev.subset_indicators(sel_vars)
    params_all = ftd_all.get_parameters()
    new_params = [params_all[xi] for xi in sel_vars]
    ftd.set_parameters(range(len(sel_vars)),new_params)    
    cll = ftd.log_likelihood_parallel(indv_aux) - ftd.log_likelihood_parallel(ind_ev_aux)
    return cll 



def prune_etind_query_cll(indv, ind_ev, et, ftd_all, cll_query, classes):
    sel_vars = cll_query + get_subgraph(et,cll_query)
    params_all = ftd_all.get_parameters()
    new_params = [params_all[xi] for xi in sel_vars]
    # Copy parameters in FT  with sel_var
    et_descriptor = [[et.nodes[i].parent_et for i in sel_vars], [et.nodes[i].nFactor for i in sel_vars], [[i]+et.nodes[i].parents.display() for i in sel_vars], [len(classes[i]) for i in sel_vars]]
    ftd = PyFactorTree_data(transform_to_selvar(sel_vars,et_descriptor[0]), transform_to_selvar(sel_vars,et_descriptor[1]), [transform_to_selvar(sel_vars,etd2) for etd2 in et_descriptor[2]],et_descriptor[3])
    ftd.set_parameters(range(len(sel_vars)),new_params)    
    indv_aux = indv.subset_indicators(sel_vars)
    ind_ev_aux = ind_ev.subset_indicators(sel_vars)
    return indv_aux, ind_ev_aux, ftd, sel_vars




def forbidden_mbc(et, query_vars, forbidden_parent = None):
    
    nvars = et.nodes.num_nds
    
    feature_nodes = list(set(range(nvars))-set(query_vars))
    if forbidden_parent is None:
        forbidden_parents = [[] for _ in range(nvars)]
    else:
        forbidden_parents = deepcopy(forbidden_parent)
        
    # Forbid parents from feature to class  
    for xi in query_vars:
        forbidden_parents[xi] = list(set(forbidden_parents[xi]).union(set(feature_nodes)))
    
    # Forbid parents between features that do not have a class as a parent
    unconnect_features = []
    connect_features = []
    for xi in feature_nodes:
        if len(set(et.nodes[xi].parents.display()).intersection(set(query_vars)))==0:
            unconnect_features.append(xi)
        else:
            connect_features.append(xi)
    for xi in connect_features:
        forbidden_parents[xi] = list(set(forbidden_parents[xi]).union(set(unconnect_features)))
    for xi in unconnect_features:
        forbidden_parents[xi] = list(set(forbidden_parents[xi]).union(set(feature_nodes)))
    
    return forbidden_parents


# Discriminative optimization of the parameters
def l_bfgs(data_frame, et, cll_query, custom_classes, alpha = 1.0):
    num_vars = data_frame.shape[1]
    et_descriptor = [[et.nodes[i].parent_et for i in range(num_vars)], [et.nodes[i].nFactor for i in range(num_vars)], [[i] + et.nodes[i].parents.display() for i in range(num_vars)], [len(c) for c in custom_classes]]
    etc_d = PyFactorTree_data(et_descriptor[0], et_descriptor[1], et_descriptor[2],et_descriptor[3])
    etc = PyFactorTree(et_descriptor[0], et_descriptor[1], et_descriptor[2],et_descriptor[3])
    dt = data_type.data(data_frame, custom_classes)
    etc_d.learn_parameters(dt, alpha = 1) 
    
    num_splits = np.ceil(data_frame.shape[0] / (1000.0*8.0))*8 # Data is splited for computing CLL to reduce memory requirements
    indv = Indicators_vector(data_frame, custom_classes, num_splits) # Indicator verctor for fast computations of the CLL
    data_ev = data_frame.copy()
    data_ev.iloc[:,cll_query] = np.nan 
    ind_ev = Indicators_vector(data_ev, custom_classes, num_splits) 
    
    etc_d.l_bfgs_b(indv,ind_ev,1.0)
    params = etc_d.get_parameters()
    nodes = [xi for xi in range(num_vars)]
    etc.set_parameters(nodes,params)
    return etc

