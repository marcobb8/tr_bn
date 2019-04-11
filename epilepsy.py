"""
Code used to compare an MBC with nomograms for predicting Engel outcome 1, 2 and 5 years after surgery

@author: Marco Benjumeda
"""

import pandas
import numpy as np
import copy
import data_type
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from discretize import fixed_freq_disc, fixed_freq_disc_test
from learn_structure_cache import hill_climbing_cache
from var_elim import PyFactorTree
from utils import get_skfold_idx, get_kfold_idx
import matplotlib.pyplot as plt
import matplotlib

path = "data/"

# Nomogram predictions in the MNI dataset
# The np.nan values are produced when the number of GTC is not known (the nomograms do not allow missing values)
pred_y2_nom = [64.2,64.2,64.2,64.2,76.5,76.5,76.5,64.1,64.0,82.2,76.5,76.5,76.2,63.2,76.3,64.1,76.4,64.1,72.3,64.1,75.4,64.2,64.1,76.5,64.1,76.4,64.1,68.5,63.4,64.2,82.2,76.4,76.5,64.2,76.5,64.1,76.5,64.0,75.8,np.nan,68.5,76.5,64.0,76.3,76.5,np.nan,64.1,63.9,64.1,76.5,76.4,72.3,76.6,64.2,68.5,64.2,76.0,68.5,76.0,76.6,64.2,68.5,76.5,64.1]
pred_y5_nom = [55.9,55.9,55.9,55.9,70.4,70.4,70.4,55.8,55.7,77.4,70.4,70.4,70.0,54.8,70.1,55.8,70.2,55.8,65.3,55.8,69.0,55.9,55.8,70.3,55.8,70.3,55.8,60.8,55.0,55.9,77.4,70.2,70.3,55.9,70.4,55.8,70.3,55.7,69.5,np.nan,60.9,70.4,55.7,70.1,70.4,np.nan,55.8,55.6,55.8,70.4,70.3,65.3,70.4,55.9,60.9,55.9,69.7,60.8,69.7,70.4,55.9,60.8,70.4,55.8]
pred_y2_nom = [x/100 for x in pred_y2_nom]
pred_y5_nom = [x/100 for x in pred_y5_nom]


# Compare the MBC with nomograms in the MNI dataset
def compare_nomograms(path):
    data_train_disc, data_test_disc, q_vars_after_merge, cut_list = preprocess_data(path)
    sel_feat = select_variables_IG(path,data_train_disc, data_test_disc, q_vars_after_merge, cut_list)
    sel_feat = select_variables_IG(path,data_train_disc, data_test_disc, q_vars_after_merge, cut_list, nvars=31)
    alpha = 1.0
    datat = data_type.data(data_train_disc)
    classes_complete = datat.classes
    
    var_select = np.sort(sel_feat + [48,49,50]).tolist()
    data_train_disc_sel = data_train_disc.iloc[:,var_select]
    data_test_disc_sel = data_test_disc.iloc[:,var_select]

    
    num_nds = data_train_disc_sel.shape[1]
    forbidden_parent = [[] for _ in range(num_nds-3)]
    forbidden_parent.append(range(num_nds-3) + [num_nds-1,num_nds-2])
    forbidden_parent.append(range(num_nds-3) + [num_nds-1])
    forbidden_parent.append(range(num_nds-3))     
    classes_sel = [classes_complete[j] for j in var_select]
    datat = data_type.data(data_train_disc_sel, classes = classes_sel)

    # Results with GS+AIC
    et = hill_climbing_cache(data_train_disc_sel, metric = 'aic', tw_bound_type='b', custom_classes = datat.classes, forbidden_parent=forbidden_parent, constraint = False, add_only = True)    
    fiti = PyFactorTree([et.nodes[j].parent_et for j in range(et.nodes.num_nds)], [et.nodes[j].nFactor for j in range(et.nodes.num_nds)], [[j]+et.nodes[j].parents.display() for j in range(et.nodes.num_nds)], [len(c) for c in datat.classes])   
    fiti.learn_parameters(datat, alpha = alpha)
    
    
    results_mbc = validate_train_test(data_test_disc_sel,fiti,classes_sel) 
    
    preds_nom = [pred_y2_nom,pred_y5_nom]
    num_cols = data_test_disc_sel.shape[1]
    
    y_true_list = [data_test_disc_sel.iloc[:,num_cols-2],data_test_disc_sel.iloc[:,num_cols-1]]
    plot_roc_curves(results_mbc, preds_nom, y_true_list)


def validate_cv(df, custom_classes, k, tw_bound, pen, method="em", forbidden_parent = None, strat = True, alpha = 1.0, metric="aic"):
    num_cols = df.shape[1]
    if strat:
        folds = get_skfold_idx(df, range(num_cols-3,num_cols), k)    
    else:
        folds = get_kfold_idx(df, k)     
    responses1 = []
    responses2 = []
    responses3 = []
    
    tvalues1 = []
    tvalues2 = []
    tvalues3 = [] 
    
    for f in folds:
        df_train = df.loc[f[0],:]
        df_test = df.loc[f[1],:]
        data_train_t  = data_type.data(df_train, classes = custom_classes)
        data_test_t  = data_type.data(df_test, classes = custom_classes)
        et = hill_climbing_cache(df, metric = metric, tw_bound_type='b', custom_classes = custom_classes, forbidden_parent=forbidden_parent)    
        fiti = PyFactorTree([et.nodes[i].parent_et for i in range(et.nodes.num_nds)], [et.nodes[i].nFactor for i in range(et.nodes.num_nds)], [[i]+et.nodes[i].parents.display() for i in range(et.nodes.num_nds)], [len(c) for c in custom_classes])   
        fiti.learn_parameters(data_train_t, alpha = alpha)

        nan1 = ~pandas.isnull(df_test.iloc[:, num_cols-3])
        nan2 = ~pandas.isnull(df_test.iloc[:, num_cols-2])
        nan3 = ~pandas.isnull(df_test.iloc[:, num_cols-1])
    
        response = fiti.pred_data(data_test_t, [0,0,0], range(num_cols-3,num_cols), range(0,num_cols-3))
        responses1 = responses1 + response[nan1, 0].tolist()
        responses2 = responses2 + response[nan2, 1].tolist()
        responses3 = responses3 + response[nan3, 2].tolist()
    
        tvalues1 = tvalues1 + (df_test.iloc[np.where(nan1)[0],num_cols-3].values == 'I').tolist()
        tvalues2 = tvalues2 + (df_test.iloc[np.where(nan2)[0],num_cols-2].values == 'I').tolist()
        tvalues3 = tvalues3 + (df_test.iloc[np.where(nan3)[0],num_cols-1].values == 'I').tolist()
    
    fpr1, tpr1, _ = roc_curve(tvalues1, responses1)
    auc1 = auc(fpr1, tpr1)
    fpr2, tpr2, _ = roc_curve(tvalues2, responses2)
    auc2 = auc(fpr2, tpr2)
    fpr3, tpr3, _ = roc_curve(tvalues3, responses3)
    auc3 = auc(fpr3, tpr3)
    auc3 = roc_auc_score(tvalues3,responses3)
    
    return [[auc1, auc2, auc3],[responses1,responses2,responses3],[tvalues1,tvalues2,tvalues3]] 
    

        
def plot_roc_curves(results_mbc, preds_nom, y_true_list):
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 14}
    
    matplotlib.rc('font', **font)
    #Filter preds_nom and y_true for the nomograms. The predictionsof the mbc are already filtered
    nan2 = ~pandas.isnull(y_true_list[0])
    nan5 = ~pandas.isnull(y_true_list[1])
    nan2_preds = ~pandas.isnull(preds_nom[0])
    nan5_preds = ~pandas.isnull(preds_nom[1])
    nan2 = [x and y for x,y in zip(nan2,nan2_preds)]
    nan5 = [x and y for x,y in zip(nan5,nan5_preds)]
    
    preds_nom2 = np.array(preds_nom[0])[nan2]
    preds_nom5 = np.array(preds_nom[1])[nan5]
    y_true_nom2 = [x == 'I' for x in np.array(y_true_list[0])[nan2]]
    y_true_nom5 = [x == 'I' for x  in np.array(y_true_list[1])[nan5]]
    
    #Year 1s
    fpr_mbc, tpr_mbc, _ = roc_curve(results_mbc[2][1], results_mbc[1][1])
    
    plt.figure(figsize=(4,3.5))
    plt.plot(fpr_mbc, tpr_mbc, label='MBC', color='blue', linestyle='-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
#    plt.legend(loc="lower right")
    plt.show()

    idx_year = 0
    y_pred = [vi>0.5 for vi in results_mbc[1][idx_year]]
    y_true = results_mbc[2][idx_year]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()    
    acc = float(tp + tn) / (tn + fp + fn + tp)    
    tpr = float(tp) / (tp + fn)    
    tnr = float(tn) / (tn + fp)  
    print "Year1"
    print "MBC. auc: ", results_mbc[0][idx_year],", acc: ", acc,", tpr: ",tpr,", tnr: ",tnr
        
    #Year 2
    fpr_nom, tpr_nom, _ = roc_curve(y_true_nom2, preds_nom2)
    fpr_mbc, tpr_mbc, _ = roc_curve(results_mbc[2][1], results_mbc[1][1])
    
    plt.figure(figsize=(4,3.5))
    plt.plot(fpr_mbc, tpr_mbc, label='MBC', color='blue', linestyle='-', linewidth=2)
    plt.plot(fpr_nom, tpr_nom, label='Nomogram', color='red', linestyle='-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
#    plt.legend(loc="lower right")
    plt.show()
    
    idx_year = 1
    y_pred = [vi>0.5 for vi in results_mbc[1][idx_year]]
    y_true = results_mbc[2][idx_year]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()    
    acc = float(tp + tn) / (tn + fp + fn + tp)    
    tpr = float(tp) / (tp + fn)    
    tnr = float(tn) / (tn + fp)  
    print "Year2"
    print "MBC. auc: ", results_mbc[0][idx_year],", acc: ", acc,", tpr: ",tpr,", tnr: ",tnr
    y_pred = [vi>0.5 for vi in preds_nom2]
    y_true = y_true_nom2
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() 
    auc_nom = auc(fpr_nom, tpr_nom)
    acc = float(tp + tn) / (tn + fp + fn + tp)    
    tpr = float(tp) / (tp + fn)    
    tnr = float(tn) / (tn + fp)  
    print "Nomogram. auc: ", auc_nom,", acc: ", acc,", tpr: ",tpr,", tnr: ",tnr
    
    #Year 5
    fpr_nom, tpr_nom, _ = roc_curve(y_true_nom5, preds_nom5, drop_intermediate=False)
    fpr_mbc, tpr_mbc, _ = roc_curve(results_mbc[2][2], results_mbc[1][2])
    
    plt.figure(figsize=(4,3.5))
    plt.plot(fpr_mbc, tpr_mbc, label='MBC', color='blue', linestyle='-', linewidth=2)
    plt.plot(fpr_nom, tpr_nom, label='Nomogram', color='red', linestyle='-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
#    plt.legend(loc="lower right")
    plt.show()
    
    idx_year = 2
    y_pred = [vi>0.5 for vi in results_mbc[1][idx_year]]
    y_true = results_mbc[2][idx_year]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()    
    acc = float(tp + tn) / (tn + fp + fn + tp)    
    tpr = float(tp) / (tp + fn)    
    tnr = float(tn) / (tn + fp)  
    print "Year5"
    print "MBC. auc: ", results_mbc[0][idx_year],", acc: ", acc,", tpr: ",tpr,", tnr: ",tnr
    y_pred = [vi>0.56 for vi in preds_nom5]
    y_true = y_true_nom5
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() 
    auc_nom = auc(fpr_nom, tpr_nom)
    acc = float(tp + tn) / (tn + fp + fn + tp)    
    tpr = float(tp) / (tp + fn)    
    tnr = float(tn) / (tn + fp)  
    print "Nomogram. auc: ", auc_nom,", acc: ", acc,", tpr: ",tpr,", tnr: ",tnr
    plot_calibration(results_mbc[2],results_mbc[1],3)
    
def experimental_roc(y_true, preds):
    th = 0
    fpr_l = []
    tpr_l = []
    while th<=1:
        y_pred = [vi>th for vi in preds]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() 
        tpr = float(tp) / (tp + fn)    
        tnr = float(tn) / (tn + fp)  
        fpr_l.append(1-tnr)
        tpr_l.append(tpr)
        th+=0.001
    return fpr_l, tpr_l
    
    

# Select variables according to IG and cv-acc
def select_variables_IG(path,data_train_disc, data_test_disc, q_vars_after_merge, cut_list, nvars = None):
    path_ig = path + "info_gain.csv"
    ig_max = pandas.read_csv(path_ig).iloc[:,1].tolist()    
    
    order = np.argsort(ig_max)[::-1].tolist()
    
    order.remove(42)
    var_select = [42,48,49,50]
    
    datat_aux = data_type.data(data_train_disc)
    classes_complete = []
    for cl in datat_aux.classes:    
        cl2 = np.unique(cl).tolist()
        classes_complete.append(cl2)


    alpha = 1.0
    macro_auc_lst = []
    strat = True
    i = 0
    k = 10
    twb = 5
    if nvars is None:
        for xi in order:
            var_select = np.unique(np.sort(var_select + [xi])).tolist()
            data_train_disc_sel = data_train_disc.iloc[:,var_select]
            num_nds = data_train_disc_sel.shape[1]
            classes_sel = [classes_complete[j] for j in var_select]
            forbidden_parent = [[] for _ in range(num_nds-3)]
            forbidden_parent.append(range(num_nds-3) + [num_nds-1,num_nds-2])
            forbidden_parent.append(range(num_nds-3) + [num_nds-1])
            forbidden_parent.append(range(num_nds-3))    
            rescv = validate_cv(data_train_disc_sel,classes_sel,k, twb, 1, method = "ll", forbidden_parent= forbidden_parent, strat = strat, alpha = alpha)
            macro_auc = (rescv[0][0] + rescv[0][1] + rescv[0][2])/3.0
            macro_auc_lst.append(macro_auc)        
            
            i += 1
           
        xi = np.argmax(macro_auc_lst)
    else:
        xi = nvars-1
    sel_feat = np.sort([order[j] for j in range(xi)] + [42]).tolist()
    return sel_feat

# Preprocess epilepsy data
def preprocess_data(path):
    
    path_train = path + "dat_train.csv"
    data_train = pandas.read_csv(path_train)
    data_train = data_train.iloc[:,1:] # Remove row names
    path_test = path + "dat_test.csv"
    data_test = pandas.read_csv(path_test)
    data_test = data_test.iloc[:,1:] # Remove row names
    q_vars = [9,10,11,25,26,28,29,45,46]
    q_vars_after_merge = [9,10,11,24,25,27,28,44,45]
    freq = 30

    
    # Discretize variables
    _,  cut_list   =  fixed_freq_disc(data_train , q_vars, freq = freq)
    cut_list[2].remove(1.7)
    data_train_disc = fixed_freq_disc_test(data_train , q_vars, cut_list)
    data_test_disc = fixed_freq_disc_test(data_test , q_vars, cut_list)
    
    #Merge variables
    data_train_disc = merge_fsm(data_train_disc)
    data_test_disc = merge_fsm(data_test_disc) 
    
    
    path_train_prep = path + "dat_train_prep.csv"
    data_train_disc.to_csv(path_train_prep, index=False)
    path_test_prep = path + "dat_test_prep.csv"
    data_test_disc.to_csv(path_test_prep, index=False)
    return data_train_disc, data_test_disc, q_vars_after_merge, cut_list



def plot_calibration(tlabels, preds, nbins):
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}
    
    matplotlib.rc('font', **font)
    xv = []
    yv = []
    for j in range(3):
        xj = []
        yj = []
        dpreds = pandas.qcut(np.array(preds[j]),nbins, duplicates='drop')    
        dpreds_u = np.unique(dpreds)
        for inter in dpreds_u:
            idx, = np.where([dp == inter for dp in dpreds])
            xj.append(np.mean([preds[j][i] for i in idx]))
            yj.append(np.mean([tlabels[j][i] for i in idx]))
        xv.append(xj)
        yv.append(yj)
    

    fig, ax = plt.subplots()
    # only these two lines are calibration curves
    plt.plot(xv[0],yv[0], marker='x', linewidth=1, label='Y1')
    plt.plot(xv[1],yv[1], marker='o', linewidth=1, label='Y2')
    plt.plot(xv[2],yv[2], marker='s', linewidth=1, label='Y5')
    plt.plot([0,1], [0,1], linestyle='--', color='black')

    ax.set_xlabel('Predicted probability of Engel I')
    ax.set_ylabel('Measured probability of Engel I')
    plt.show()    

def validate_train_test(data_test, fiti, custom_classes): 
    data_test_t  = data_type.data(data_test, classes = custom_classes)  
    num_cols =  data_test_t.ncols   
    
    nan1 = ~pandas.isnull(data_test.iloc[:, num_cols-3])
    nan2 = ~pandas.isnull(data_test.iloc[:, num_cols-2])
    nan3 = ~pandas.isnull(data_test.iloc[:, num_cols-1])
    
    response = fiti.pred_data(data_test_t, [0,0,0], range(num_cols-3,num_cols), range(0,num_cols-3))
    responses1 = response[nan1, 0]
    responses2 = response[nan2, 1]
    responses3 = response[nan3, 2]
    
    tvalues1 = data_test.iloc[np.where(nan1)[0],num_cols-3].values == 'I'
    tvalues2 = data_test.iloc[np.where(nan2)[0],num_cols-2].values == 'I'
    tvalues3 = data_test.iloc[np.where(nan3)[0],num_cols-1].values == 'I'
    
    fpr1, tpr1, _ = roc_curve(tvalues1, responses1)
    auc1 = auc(fpr1, tpr1)
    fpr2, tpr2, _ = roc_curve(tvalues2, responses2)
    auc2 = auc(fpr2, tpr2)
    fpr3, tpr3, _ = roc_curve(tvalues3, responses3)
    auc3 = auc(fpr3, tpr3)
    auc3 = roc_auc_score(tvalues3,responses3)
    
    return [[auc1, auc2, auc3],[responses1,responses2,responses3],[tvalues1,tvalues2,tvalues3]]
    
# Merge function for variables Automatisms.as.first.sz.manifestation (AUT) and Tonic.clonic as first.. (TC)
# AUT: (1Y, 2N)
# TC: (1Y, 2N)
#
# First seizure freedom manifestation: 1 AUT, 2 TC, 3 other 
def merge_fsm(data):    
    data_m = copy.deepcopy(data)
    aut_row = data_m.loc[:,"Automatisms.as.first.sz.manifestation..1Y.2N."]
    tc_row = data_m.loc[:,"Tonic.clonic.hyperkinetic.movements.as.first.sz.manifestation..1Y.2N."]
    fsm_row = [merge_fsm_aux(aut,tc) for aut,tc in zip(aut_row,tc_row)]
    data_m.loc[:,"Automatisms.as.first.sz.manifestation..1Y.2N."] =  fsm_row  
    colnames = list(data_m)    
    colnames[colnames.index("Automatisms.as.first.sz.manifestation..1Y.2N.")] = "First clinical manifestation during a seizure..1Automatisms2Tonic-clonic or hyperkinetic movements3Other"
    data_m.columns = colnames  
    colnames.remove("Tonic.clonic.hyperkinetic.movements.as.first.sz.manifestation..1Y.2N.")
    data_m = data_m.loc[:,colnames]
    return data_m
        
    

def merge_fsm_aux(aut,tc):
    if not pandas.isnull(aut):
        if aut == 1:
            return 1
    if pandas.isnull(tc):
        return tc
    if tc == 1:
        return 2
    if pandas.isnull(aut): 
        return aut
    return 3