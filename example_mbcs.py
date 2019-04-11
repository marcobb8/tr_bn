# Import packages
import pandas
from mbc import learn_mbc_cll, learn_mbc_generative, l_bfgs
from export import get_adjacency_matrix_from_et
from var_elim import PyFactorTree
from data_type import data as datat

# Load ASIA dataframe
file_name = "data/asia.csv"
data = pandas.read_csv(file_name)
var_classes = [['yes','no'] for _ in range(8)] # Categories of the variables
cll_query = [0,1,2] #Class variables in the dataset
features = [3,4,5,6,7]

# ----LEARNING FROM INCOMPLETE DATASETS---- #
# Learn model with GS-pruned (Generative learning)
mbc_gen = learn_mbc_generative(data, cll_query, metric='bic', tw_bound=5, custom_classes = var_classes)
# Learn model with DGS (Discriminative learning)
mbc_disc = learn_mbc_cll(data, cll_query, metric='bic', tw_bound=5, custom_classes = var_classes)



# ----LEARN PARAMETERS---- #
num_vars = len(var_classes) #Number of variables

# Get cppet from from MBC
et = mbc_gen
et_descriptor = [[et.nodes[i].parent_et for i in range(num_vars)], [et.nodes[i].nFactor for i in range(num_vars)], [[i] + et.nodes[i].parents.display() for i in range(num_vars)], [len(c) for c in var_classes]]
mbc_gen_cpp = PyFactorTree(et_descriptor[0], et_descriptor[1], et_descriptor[2],et_descriptor[3])

# Transfrom dataframe to data_type
data_p = datat(data,var_classes) 
# Learn parameters: alpha is the Dirichlet hyperparameter for the Bayesian estimation. 
#                   If alpha=0, the Maximum likelihood parameters are obtained 
mbc_gen_cpp.learn_parameters(data_p, alpha = 1)

# (Optional) Optimize conditional likelihood of the parameters using l_bfgs
mbc_gen_cpp = l_bfgs(data, mbc_gen, cll_query, var_classes)


# ----INTERPRETING THE BAYESIAN NETWORK---- #
# Obtaining the MBC adjacency matrix from cppet
num_nodes = data.shape[1]
adj_mat = get_adjacency_matrix_from_et(mbc_disc)
# Obtaining the parameters of node Tub
xi = 1 #Tub is node 1
factor = mbc_gen_cpp.get_factor(num_nodes + xi)
parameters = factor.get_prob()


# ----Multidimensional classification---- #
# Obtaining most probable explanations (MPEs)
# Set evidence. For example, Asia = 'yes' and Lung Cancer = 'no'
mbc_gen_cpp.set_evidence(features,[0,1,0,0,1])
# Compute MPE
mbc_gen_cpp.max_compute_tree()
# Get factor with results
factor = mbc_gen_cpp.get_factor(-1)
mpe_idx = factor.get_mpe()[0] # Get the MPE 
mpe = [var_classes[i][ci] for i,ci in enumerate(mpe_idx)]
prob = factor.get_prob() # Get probability of the MPE 
# Retract evidence
mbc_gen_cpp.retract_evidence()

# Compute marginals of the class variables for each instance in the dataset
probs = mbc_gen_cpp.pred_data_distribute(data_p, cll_query, features)
#The result is a 3D list
# If we access probs[i][j][k], i is the row in the dataset, j is the index of the class variables in cll_query, and k is the category (index in var_classes)

