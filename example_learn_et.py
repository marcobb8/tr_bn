import pandas
from learn_structure_cache import hill_climbing_cache
from export import get_adjacency_matrix_from_et
from var_elim import PyFactorTree
from data_type import data as datat

# Load ASIA dataframe
file_name = "data/asia.csv"
data = pandas.read_csv(file_name)
var_classes = [['yes','no'] for _ in range(8)]

# ----LEARNING BAYESIAN NETWORKS WITH BOUNDED TREEWIDTH---- #
# Learn elimination tree (ET) with hc-et, using a tw bound of 3 and BIC as the objective score
et = hill_climbing_cache(data, metric = 'bic', tw_bound = 3, custom_classes=var_classes)
# Learn ET with hc-et-poly, using a tw bound of 3 and BIC as the objective score
et2 = hill_climbing_cache(data, metric = 'bic', tw_bound = 3, custom_classes=var_classes, add_only=True)

# Get adjacency matrix of the Bayesian network encoded by the ET et
adj_mat = get_adjacency_matrix_from_et(et)


# ----PARAMETER LEARNING---- #
num_vars = len(var_classes) #Number of variables

# Get cppet from from ET
et_descriptor = [[et.nodes[i].parent_et for i in range(num_vars)], [et.nodes[i].nFactor for i in range(num_vars)], [[i] + et.nodes[i].parents.display() for i in range(num_vars)], [len(c) for c in var_classes]]
cppet = PyFactorTree(et_descriptor[0], et_descriptor[1], et_descriptor[2],et_descriptor[3])
# Transfrom dataframe to data_type
data_p = datat(data,var_classes) 
# Learn parameters: alpha is the Dirichlet hyperparameter for the Bayesian estimation. 
#                   If alpha=0, the Maximum likelihood parameters are obtained 
cppet.learn_parameters(data_p, alpha = 1)


# Obtaining the parameters of node Tub
xi = 1 #Tub is node 1
factor = cppet.get_factor(num_vars + xi)
parameters = factor.get_prob()

# ----INFERENCE---- #
# Evidence propagation 
# Set evidence. For example, Asia = 'yes' and Lung Cancer = 'no'
cppet.set_evidence([0,3],[0,1])
# Propagate evidence
cppet.sum_compute_tree()
# Get factor with results
factor = cppet.get_factor(-1)
prob = factor.get_prob() # The result is the probability of the evidence 
# Retract evidence
cppet.retract_evidence()

# Obtaining most probable explanations (MPEs)
# Set evidence. For example, Asia = 'yes' and Lung Cancer = 'no'
cppet.set_evidence([0,3],[0,1])
# Compute MPE
cppet.max_compute_tree()
# Get factor with results
factor = cppet.get_factor(-1)
mpe_idx = factor.get_mpe()[0] # Get the MPE 
mpe = [var_classes[i][ci] for i,ci in enumerate(mpe_idx)]

prob = factor.get_prob() # Get probability of the MPE 
# Retract evidence
cppet.retract_evidence()