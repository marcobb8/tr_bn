# Import packages
import pandas
from utils import insert_nan_df
from sem_soft import tsem_cache
from export import get_adjacency_matrix_from_et

# Load ASIA dataframe
file_name = "data/asia.csv"
data = pandas.read_csv(file_name)
var_classes = [['yes','no'] for _ in range(8)]

# Insert missing values to the dataset
per_nan = 0.3 #percentage of missing values between 0 and 1
data_incomplete = insert_nan_df(data,per_nan)

# ----LEARNING FROM INCOMPLETE DATASETS---- #
# Learn model with TSEM
et, _, cppet, df_completed, _ = tsem_cache(data_incomplete, metric='bic', tw_bound=5, custom_classes = var_classes, add_only = False)
# Learn model with TSEM-poly
et, _, cppet, df_completed, _ = tsem_cache(data_incomplete, metric='bic', tw_bound=5, custom_classes = var_classes, add_only = True)

# cppet is the c++ elimination tree. It can be used to perform inference
# df_complete is the completed dataset (with hard assignments)

# ----INTERPRETING THE BAYESIAN NETWORK---- #
# Obtaining the BN adjacency matrix from cppet
num_nodes = data.shape[1]
adj_mat = get_adjacency_matrix_from_et(et)


# Obtaining the parameters of node Tub
xi = 1 #Tub is node 1
factor = cppet.get_factor(num_nodes + xi)
parameters = factor.get_prob()
# Assuming that the only parent of Tub is asia, parameters would be a vector of 4 elements such that
# parameters[0] = P(Tub='yes'|Asia='yes'), parameters[1] = P(Tub='no'|Asia='yes'), parameters[2] = P(Tub='yes'|Asia='no'),
# and parameters[3] = P(Tub='no'|Asia='no')

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






