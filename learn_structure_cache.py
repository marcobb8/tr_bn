"""
greedy hill-climbing for learning BNs with bounded treewidth

@author: Marco Benjumeda
"""


# -------------------------------------------------------#
# Learn structure of elimination tree
# -------------------------------------------------------#

import sys
from time import time
import data_type
from scoring_functions import score_function
from elimination_tree import ElimTree
import multiprocessing
import numpy as np
from gs_structure import best_pred_cache, search_first_tractable_structure



# Hill climbing algorithm for ets
#
def hill_climbing_cache(data_frame, et0=None, u=5, metric='bic', tw_bound_type='b', tw_bound=5,  cores=multiprocessing.cpu_count(), forbidden_parent=None, add_only = False, custom_classes=None, constraint = True, verbose=False):
    """Learns a Bayesian network with bounded treewidth

    Args:
        data_frame (pandas.DataFrame): Input data
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
        elimination_tree.ElimTree Learned elimination tree
    """

    count_i = 0
    if custom_classes is None:
        data = data_type.data(data_frame)
    else:
        data = data_type.data(data_frame, classes= custom_classes)
    
    
    if et0 is None:
        et = ElimTree('et', data.col_names, data.classes)
    else:
        et = et0.copyInfo()
    
    if forbidden_parent is None:
        forbidden_parents = [[] for _ in range(data.ncols)]
    else:
        forbidden_parents = forbidden_parent
    ok_to_proceed = True
    num_nds = et.nodes.num_nds
    
    score_best = np.array([score_function(data, i, et.nodes[i].parents.display(), metric=metric) for i in range(num_nds)],dtype = np.double)
    # Cache
    cache = np.full([et.nodes.num_nds,et.nodes.num_nds],1000000,dtype=np.double)
    
    #loop
    while ok_to_proceed:
        count_i += 1
        
        ta= time()
        # Input, Output and new
        lop_o, op_names_o, score_difs_o, cache =  best_pred_cache(data, et, metric, score_best, forbidden_parents, cache, filter_l0 = True, u=u, add_only = add_only)
        tc = time()   
        if len(lop_o) > 0:
            if tw_bound_type=='b':
                change_idx, et_new = search_first_tractable_structure(et, lop_o, op_names_o, tw_bound, forbidden_parents, constraint)
                if change_idx == -1:
#                    print 'exit'
                    return et
            else:
                change_idx = 0
                xout = lop_o[change_idx][0]
                xin = lop_o[change_idx][1]
                lop_o[change_idx]
                et_new = et.copyInfo()
                if op_names_o[change_idx] == 0:
                    et_new.setArcBN_py(xout, xin)
                elif op_names_o[change_idx] == 1:
                    et_new.removeArcBN_py(xout, xin)
                else:
                    et_new.removeArcBN_py(xout, xin)
                    et_new.setArcBN_py(xin, xout)
            
            best_lop = lop_o[change_idx]
            xout = best_lop[0]
            xin = best_lop[1]
            best_op_names = op_names_o[change_idx]
            best_score_difs = score_difs_o[change_idx]
            et = et_new
            # Update score and cache
            if best_op_names == 2:
                score_best[xin] += cache[xout, xin]
                score_best[xout] += cache[xin, xout]
                cache[:,xout]=1000000
            else:
                score_best[xin] += best_score_difs
            cache[:,xin]=1000000
            tb= time()
            if verbose:
                if tw_bound_type=='b':
                    print 'it: ',count_i, ', change: ', [best_op_names, xout, xin], ', tw: ', et.tw(), ', time: ', tc-ta, ', timetw: ', tb-tc, ', best_score_difs: ', best_score_difs
                else:
                    print 'it: ',count_i, ', change: ', [best_op_names, xout, xin], ', time: ', tc-ta, ', timetw: ', tb-tc, ', best_score_difs: ', best_score_difs
        else:
            ok_to_proceed = False        
    return et
        




