import argparse, os, sys
import pickle as pkl
from scipy.stats import chisquare
import random
import pandas as pd
import numpy as np
import copy
import time
import csv


'''
TreeNode represents a node in your decision tree
TreeNode can be:
    - A non-leaf node: 
        - data: contains the feature number this node is using to split the data
        - children[0]-children[4]: Each correspond to one of the values that the feature can take
        
    - A leaf node:
        - data: 'T' or 'F' 
        - children[0]-children[4]: Doesn't matter, you can leave them the same or cast to None.

'''

sys.setrecursionlimit(100000)
# internal nodes
internal = 0
# leaf nodes
leaf = 0

# parse the command line arguments    
parser = argparse.ArgumentParser()
parser.add_argument('-p', help='specify p-value threshold', dest='pvalue', action='store', default='0.005')
parser.add_argument('-f1', help='specify training dataset path', dest='train_dataset', action='store', default='')
parser.add_argument('-f2', help='specify test dataset path', dest='test_dataset', action='store', default='')
parser.add_argument('-o', help='specify output file', dest='output_file', action='store', default='')
parser.add_argument('-t', help='specify decision tree', dest='decision_tree', action='store', default='') 
   
args = vars(parser.parse_args())

#read the train data
X_train = args['train_dataset']
y_train = args['train_dataset'].split('.')[0] + '_label.csv'
sample_space = pd.read_csv(X_train, header=None, sep=" ") 
train_output_values = pd.read_csv(y_train, header=None)
sample_space['output_value'] = train_output_values[0]
attribute_count = sample_space.shape[1] - 1
attributes = [i for i in range(attribute_count)]
pvalue = float(args['pvalue'])

# Chi-square implementation
def chisquare_criterion(sample_space, selection):    
    actual = []
    pred = []
    # negative samples    
    n = (sample_space['output_value'] == 0).sum()
    # positive samples
    p = (sample_space['output_value'] == 1).sum()
    N = n + p    
    r1 = calculate(p,N)
    r2 = calculate(n,N)
    atrs = sample_space[selection].unique()    
    # Calculating the pred and actual number of pos & neg
    for value in atrs:                
        attri = sample_space.filter([selection,'output_value'],axis=1)
        attri = attri.loc[(attri[selection]==value)]
        T_i = attri['output_value'].count()                                        
        p_dash_i = calculatePDash(r1,T_i)
        n_dash_i = calculatePDash(r2,T_i)
        p_i = calculatepi(attri)
        n_i = calculateni(attri)        
        if p_dash_i != 0: 
            pred.append(p_dash_i) 
            actual.append(p_i)
        if n_dash_i != 0:
            pred.append(n_dash_i)        
            actual.append(n_i)
    # calculate the chi-square value
    c, p = chisquare(actual, pred)          
    return p

def calculate(p,N):
    return float(p) / N

def calculatepi(attri):
    return float((attri['output_value'] == 1).sum())

def calculateni(attri):
    return float((attri['output_value'] == 0).sum())

def calculatePDash(r1,T_i):
    return float(r1) * T_i
    

    
class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data

    def save_tree(self,filename):
        obj = open(filename,'wb')
        pkl.dump(self,obj)

# Next selection 
def select_atr(sample_space, attributes):    
    H_min = None
    atrb = None    
    # Information Gain
    for attribute in attributes:                 
        rows_count = sample_space[attribute].count()
        atrs = sample_space[attribute].unique()        
        entropy_value = 0    
        for value in atrs:
            count = (sample_space[attribute] == value).sum()
            p = float(count) / rows_count            
            attr_sample_space = sample_space.filter([attribute, 'output_value'], axis=1)            
            attr_sample_space = attr_sample_space.loc[(attr_sample_space[attribute] == value)]
            attr_sample_space_rows_count = attr_sample_space['output_value'].count()            
            true = (attr_sample_space['output_value'] == 1).sum()
            tr_pr = float(true) / attr_sample_space_rows_count
            false = (attr_sample_space['output_value'] == 0).sum()
            fa_pr = float(false) / attr_sample_space_rows_count
            if tr_pr == 0:
                H_tr = 0
            else:
                H_tr = tr_pr * (np.log2(tr_pr))
            if fa_pr == 0:
                H_fa = 0
            else:
                H_fa = fa_pr * (np.log2(fa_pr))                            
            total_entropy = -(H_fa + H_tr)        
            entropy_value += p * total_entropy
        if H_min == None or entropy_value < H_min:
            atrb = attribute
            H_min = entropy_value   
    return atrb
       
def grow_DT(sample_space, attributes, pvalue):
    global internal, leaf

    calculateTreeNode(sample_space,leaf)
    calculateTreeNode1(sample_space,leaf)
    calculateLeaf(sample_space,leaf,attributes)
         
    # Attribute selection based on max info gain
    selection = select_atr(sample_space, attributes)    
    attributes.remove(selection)
    node = None
    chi = chisquare_criterion(sample_space, selection) 
    # build the node if the chi sqaure value if less than p_value, else terminate the node   
    if chi < pvalue:        
        node = TreeNode(selection + 1)
        internal += 1
        uniqueValues = sample_space[selection].unique()
        i = 1
        tr_miss = -1
        fa_miss = -1
        while i < 6:
            if i in uniqueValues:
                sample_space_subset = sample_space.loc[sample_space[selection] == i]
                if sample_space_subset.empty:
                    
                    false , true = calculateTrueFalse(sample_space_subset)
                    calculateNodes(true,false,node,leaf)
                else:       
                    attri = copy.deepcopy(attributes)                                
                    is_node = grow_DT(sample_space_subset, attributes, pvalue)
                    if is_node:
                        node.nodes[i - 1] = is_node
                    else:
                        
                        false , true = calculateTrueFalse(sample_space_subset)
                        if true >= false:
                            leaf += 1
                            node.nodes[i - 1]= TreeNode()
                        else:
                            leaf += 1
                            node.nodes[i - 1] = TreeNode('F')
            else:
                if tr_miss == -1 and fa_miss == -1:
                    fa_miss , tr_miss = calculateTrueFalse(sample_space)
                leaf, node.nodes[i-1] = calculateNodes(tr_miss,fa_miss,node,leaf)
                
            i += 1
    else:                       
        return None
    return node  

def calculateTreeNode(sample_space,leaf):
    
    # Sample with positive vals   
    if (sample_space['output_value'] == 1).sum() == sample_space['output_value'].count():
        leaf += 1
        return TreeNode()  

def calculateTreeNode1(sample_space,leaf):
        
        # Sample with negative vals
    if (sample_space['output_value'] == 0).sum() == sample_space['output_value'].count():
        leaf += 1
        return TreeNode('F')

def calculateLeaf(sample_space,leaf,attributes):
     
    if len(attributes) == 0:
        true = 0
        false = 0
        true = (sample_space['output_value'] == 1).sum()
        false = (sample_space['output_value'] == 0).sum()
        if true >= false:
            leaf += 1
            return TreeNode()            
        else:
            leaf += 1
            return TreeNode('F')  
        
def calculateTrueFalse(sample_space_subset):
    return (sample_space_subset['output_value'] == 0).sum(), (sample_space_subset['output_value'] == 1).sum()

def calculateNodes(tr_miss,fa_miss,node,leaf):
    if tr_miss >= fa_miss:
        return (leaf+1), TreeNode() 
    else:
        return (leaf+1), TreeNode('F')

# traverse the tree and return the best possible value for the test data        
def classify(root, datapoint):
    if root.data == 'T': return 1
    if root.data =='F': return 0
    return classify(root.nodes[datapoint[int(root.data)-1]-1], datapoint)  

# build the tree using the train data
root = grow_DT(sample_space, attributes, pvalue)   
root.save_tree(args['decision_tree'])  

# read test data
test_data_file = args['test_dataset']
X_test = pd.read_csv(test_data_file, header=None, sep=" ")
test_row_count = X_test.shape[0]
predictions = []

# get the best possible output value for the test data
for i in range(test_row_count):
    predictions.append([classify(root, X_test.loc[i])])

# write the test output to a file
output_file = args['output_file']
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerows(predictions)
print("Internal nodes: ", internal, " Leaf nodes: ", leaf)
