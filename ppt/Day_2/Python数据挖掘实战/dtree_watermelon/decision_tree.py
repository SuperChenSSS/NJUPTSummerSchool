#coding=utf-8
class Node(object):
    """
    Tree Node, each with a parent node and child node.

    attr: the attribute to split on.
    branch: dict, splitting value as key and corresponding  value as value. If leaf, branch=None
    label: the classification label of current based on label votes
    """

    def __init__(self, attr, branch, label):
        self.attr = attr
        self.branch = branch
        self.label = label

import pandas as pd
with open("./watermelon2.0.csv", mode = 'r', encoding='utf-8') as data_file:
    df_global = pd.read_csv(data_file)

def tree_gen(df):
    """
    Tree generation method given data df.

    Inputs：
            df, the DataFrame object of the data
    Outputs:
            root, Root Node
    """
    root = Node(None, {}, None)  # Init a node
    labels = df.values[:,-1] # get the last columns as the label array
    label_dist = node_dist(labels)
    if label_dist:
        # find out label of current node 
        root.label = max(label_dist, key=label_dist.get)
        
        # if all instances belong to one class 
        # or attributes run out
        if len(label_dist)==1 or len(df.columns[1:-1]) == 0:
            return root       
        # find out the optimal splitting attribute and splitting value(continuous)
        root.attr, split_val = opt_attr(df)
        # print("best attr %s" %root.attr)
        # if split_val !=0:
        #     print("连续属性:%s, 划分值：%f" %(root.attr, split_val))
        
        if split_val == 0:  # categorical attribute
            # get branch distribution 
            attr_dist = node_dist(df[root.attr])         
            # create branch for each attr val
            print("Attr list in tree generation %s" %sorted(set(df_global[root.attr])))
            for val in sorted(set(df_global[root.attr])):
                # if no instances in the branch, make leaf and set label as label vote in parent node
                if val not in attr_dist.keys():
                    root.label = max(label_dist, key=label_dist.get)
                    root.branch[val] = Node(None, {}, root.label)
                else:
                    df_branch = df[ df[root.attr] == val ]  # get the rows corr. to val 
                    df_branch = df_branch.drop(root.attr, axis=1)   #remove the attribute(categorical)
                    root.branch[val] = tree_gen(df_branch)
                
        else:  # continuous variable # left and right child
            val_l = "<=%.3f" % split_val
            val_r = ">%.3f" % split_val
            df_v_l = df[ df[root.attr] <= split_val ]  # get sub set
            df_v_r = df[ df[root.attr] > split_val ]
 
            root.branch[val_l] = tree_gen(df_v_l)
            root.branch[val_r] = tree_gen(df_v_r)
            
        return root


def opt_attr(df):
    """
    Find out the optimal splitting attribute acc. certain evaluation criterion.

    Inputs：
            df, the DataFrame object of the data
    Outputs:
            split_attr, the best attribute to split on 
            split_val, the split value for continuous attributes
    """
    # attribute evaluation criterion 
    crton = float('-Inf')
    # get all attribute index (columns)
    for attr_index in list(df.columns[1:-1]):
        crton_tmp, split_val_tmp = infogain_cal(df, attr_index)
        if  crton_tmp > crton: 
            crton = crton_tmp
            split_attr = attr_index
            split_val = split_val_tmp
    print("Chosen %s" %split_attr)
    return split_attr, split_val


def infogain_cal(df, attr):
    """
    calculating the information gain of an attribute
     
    Inputs: 
            df:      dataframe, the pandas dataframe of the data_set
            attr: the target attribution in df
    Outputs: 
            info_gain: the information gain of current attribution
            split_val: for discrete variable, split_val = 0
                       for continuous variable, value = the division value
    """
    attr_col = df[attr]
    labels = df.values[:,-1]
    info_gain = entropy_cal(labels)  # Entropy for the current data 
    
    split_val = 0.0  # best splitting for continuous attribute
    
    n = len(df[attr])  # the number of instances
    # 1.for continuous variable using method of bisection
    if attr_col.dtype == (float, int):
        branch_ent = {}  # store the div_value (div) and it's subset entropy
        
        df_sorted = df.sort_values([attr], ascending=1)  # sort column in ascending order
        df_sorted = df_sorted.reset_index(drop=True)   #reset data index 
        labels_sorted = df_sorted.values[:,-1]
        val_attr = df_sorted[attr]   #obtain sorted attribute values 
            
        for i in range(n-1):
            div = (val_attr[i] + val_attr[i+1]) / 2
            branch_ent[div] = ( (i+1)/n * entropy_cal(labels_sorted[0:i+1])  ) \
                              + ( (n-i-1)/n * entropy_cal(labels_sorted[i+1:]) )
        # the best split value gives the minimal entropy
        split_val = min(branch_ent, key=branch_ent.get)
        best_ent = branch_ent[split_val]
        info_gain -= best_ent
        
    # 2.for categorical attributes
    else:
        # get distribution of different attribute values
        branches= node_dist(attr_col)
        
        # calculate entropy of each partition
        for key in branches.keys():
            label_split = labels[attr_col == key]
            info_gain -= branches[key]/n * entropy_cal(label_split) 
    return round(info_gain, 3), split_val


def entropy_cal(labels):
    '''
    calculating the information entropy of an attribute
     
    Inputs:
            labels: ndarray, class label array
    Outputs:
            entropy: the information entropy of partition
    ''' 
    try :
        from math import log2
    except ImportError :
        print("module math.log2 not found")
    
    entropy = 0.0
    n = len(labels)
    label_count = node_dist(labels)
    
    for key in label_count:
        entropy -= ( label_count[key] / n ) * log2( label_count[key] / n )
    return entropy


def gini_cal(labels):
    gini = 1.0
    n = len(labels)
    label_count = node_dist(labels)
    
    for key in label_count:
        gini -= ( label_count[key] / n ) ** 2
    return gini


def node_dist(labels):
    """
    count each label within each node
    Inputs:
            labels, the label column
    Outputs:
            dict, key: label; value: count
    """
    val_dist = {}
    for label in labels:
        if label in val_dist.keys():
            val_dist[label] += 1
        else:
            val_dist[label] = 1
    return val_dist


def draw_tree(root, out_file):
    '''
    visualization of decision tree.
    Inputs:
            root: Node, the root node for tree.
            out_file: str, file path
    '''
    try:
        from pydotplus import graphviz
    except ImportError:
        print("module pydotplus.graphviz not found")
        
    g = graphviz.Dot()  # generation of new dot   

    tree2graph(0, g, root)
    g2 = graphviz.graph_from_dot_data( g.to_string() )
    
    g2.write_png(out_file) 
    
def tree2graph(i, g, root):
    '''
    build a graph from root
    Inputs:
            i: node id in this tree
            g: pydotplus.graphviz.Dot() object
            root: the root node
    
    Outputs:
            i: node id after modified  
            g: pydotplus.graphviz.Dot() object after modified
            g_node: the current root node in graphviz
    '''
    try:
        from pydotplus import graphviz
    except ImportError:
        print("module pydotplus.graphviz not found")

    if root.attr == None:
        g_node_label = "Node:%d\n好瓜:%s" % (i, root.label)
    else:
        g_node_label = "Node:%d\n好瓜:%s\n属性:%s" % (i, root.label, root.attr)
    g_node = i
    g.add_node( graphviz.Node( g_node, label = g_node_label ) )
    
    for value in sorted(root.branch):
        i, g_child = tree2graph(i+1, g, root.branch[value])
        g.add_edge( graphviz.Edge(g_node, g_child, label = value) ) 

    return i, g_node    



def predict(root, df_sample):
    """
    traverse the generated tree root with the given sample 
    Inputs:
            root, tree root
            df_sample, an instance DataFrame
    Outputs:
            root.label, classification result based on the generated tree and sample
    """  
    try :
        import re # using Regular Expression to get the number in string
    except ImportError :
        print("module re not found")
    
    while root.attr != None :        
        # continuous variable
        if df_sample[root.attr].dtype == (float, int):
            # traverse branches
            for key in list(root.branch):
                num = re.findall(r"\d+\.?\d*",key)
                split_val = float(num[0])
                break
            if df_sample[root.attr].values[0] <= split_val:
                key = "<=%.3f" % split_val
                root = root.branch[key]
            else:
                key = ">%.3f" % split_val
                root = root.branch[key]
                
        # categoric variable
        else:  
            key = df_sample[root.attr].values[0]
            # check whether the attr_value in the child branch
            if key in root.branch: 
                root = root.branch[key]
            else: 
                break
    return root.label


def acc_cal(root, df_test):  
    '''
    Given test set, calculate the acc of correctly classified samples
    Inputs:
            root, tree root
            df_test, test set 
    Outputs:
            pred_true / len(df_test.index), accuracy
    '''
    if len(df_test.index) == 0: 
        return 0
    # the count of correctly classified instances
    pred_true = 0
    for i in df_test.index:
        label = predict(root, df_test[df_test.index == i])
        if label == df_test[df_test.columns[-1]][i]:
            pred_true += 1
    return round(pred_true / len(df_test.index), 3)


def pre_pruning(df_train, df_valdt):
    '''
    pre-prune the tree based on df_valdt while tree generation
    Inputs:
            df_train: dataframe, the training set
            df_valdt: dataframe, the validation set for pruning decision
    Outputs: 
            root, tree generated with pre-pruning 
    '''
    # generating a new root node
    root = Node(None, {}, None)
    labels = df_train.values[:,-1]
    
    label_dist = node_dist(labels)
    if label_dist:  
        root.label= max(label_dist, key=label_dist.get) 
     
        # end if there is only 1 class in current node data
        # end if attribution array is empty
        if len(label_dist) == 1 or len(df_train.columns[1:-1]) == 0:
            return root
        
        acc_unsplit = acc_cal(root, df_valdt)
        # get the optimal attribution for a new branching
        root.attr, split_val = opt_attr(df_train)  # via Gini index 
        
        if split_val == 0:  # categorical attribute
            attr_dist = node_dist(df_train[root.attr])         
            # create branch for each attr val
            for val in sorted(set(df_global[root.attr])):
                # if no instances in the branch, make leaf and set label as label vote in parent node
                root.label = max(label_dist, key=label_dist.get)
                root.branch[val] = Node(None, {}, root.label)
            
            # calculating to check whether need further branching
            acc_split = acc_cal(root, df_valdt) 
            if acc_split > acc_unsplit:  # need branching
                df_branch = df_train[ df_train[root.attr] == val ]  # get the rows corr. to val 
                df_branch = df_branch.drop(root.attr, axis=1)   #remove the attribute(categorical)
                df_valdt_branch = df_valdt[df_valdt[root.attr] == val]
                df_valdt_branch = df_valdt_branch.drop(root.attr, axis=1)
                root.branch[val] = pre_pruning(df_branch, df_valdt_branch)
        else:  # continuous variable # left and right child
            val_l = "<=%.3f" % split_val
            val_r = ">%.3f" % split_val
            df_v_l = df_train[ df_train[root.attr] <= split_val ]  # get sub set
            df_v_r = df_train[ df_train[root.attr] > split_val ]
            
            # pre-branch the current node 
            labels_l = node_dist(df_v_l.values[:,-1])
            labels_r = node_dist(df_v_r.values[:,-1])
            node_l.label = max(labels_l, key=labels_l.get)
            node_r.label = max(labels_r, key=labels_r.get)
            root.branch[val_l] = Node(None, {}, node_l.label)
            root.branch[val_r] = Node(None, {}, node_r.label)
            
            acc_split = acc_cal(root, df_valdt)
            if acc_split > acc_unsplit:  # need branching
                df_v_l = df_v_l.drop(root.attr, axis=1)
                df_v_r = df_v_r.drop(root.attr, axis=1)
                df_valdt_l = df_valdt[ df_valdt[root.attr] <= split_val ]
                df_valdt_r = df_valdt[ df_valdt[root.attr] > split_val ]
                df_valdt_l = df_valdt_l.drop(root.attr, axis=1)
                df_valdt_r = df_valdt_r.drop(root.attr, axis=1)
                root.branch[val_l] = pre_pruning(df_v_l, df_valdt_l)
                root.branch[val_r] = pre_pruning(df_v_r, df_valdt_r)
                
    return root

            
def post_pruning(root, df_valdt):
    '''
    post-prune the tree based on df_valdt while tree generation
    Inputs:
            root: generated tree 
            df_valdt: dataframe, the validation set for pruning decision
    Outputs: 
            accuracy 
    '''
    # if leaf node, return accuracy
    if root.attr == None:
        return acc_cal(root, df_valdt)
    
    # calculating the test accuracy on children node
    acc_unpruned = 0
    attr_dist = node_dist(df_valdt[root.attr]) 
    for val in sorted(attr_dist):
        df_valdt_v = df_valdt[ df_valdt[root.attr]==val ]  # get sub set
        if val in root.branch:  # root has the value
            acc_v = post_pruning(root.branch[val], df_valdt_v)
        else:  # root doesn't have value
            acc_v = acc_cal(root, df_valdt_v)
        if acc_v == -1:  # -1 means no pruning back
            return -1
        else:
            # weighted accuracy
            acc_unpruned += acc_v * len(df_valdt_v.index) / len(df_valdt.index)
            
    # make leaf and calculate the validation accuracy on this node   
    node = Node(None, {}, root.label)
    acc_pruned = acc_cal(node, df_valdt)
    print("Attr %s After post-pruning %f; Before post-pruning %f" %(root.attr, acc_pruned, acc_unpruned))
    # check if need pruning
    if acc_pruned >= acc_unpruned:
        # print("Prune %s" %root.attr)
        root.attr = None
        root.branch = {}
        return acc_pruned
    else: 
        print("Attr %s needs no pruning" %root.attr)
        return -1


def test():
    import pandas as pd
    
    with open("./watermelon2.0.csv", mode = 'r', encoding='utf-8') as data_file:
        df = pd.read_csv(data_file)

    index_train = [0,1,2,5,6,9,13,14,15,16]

    df_train = df.iloc[index_train]
    df_test  = df.drop(index_train)

    # generate a full tree
    root = tree_gen(df)
    draw_tree(root, "dt_full.png")
    
    root_unpruned = tree_gen(df_train)
    print("accuracy of unpruned tree: %.3f" % acc_cal(root_unpruned, df_test))
    draw_tree(root_unpruned, "dt_unpruned.png")
    
    root_pre = pre_pruning(df_train, df_test)
    print("accuracy of pre-pruned tree: %.3f" % acc_cal(root_pre, df_test))
    draw_tree(root_pre, "dt_pre_pruning.png")
    
    root_post = root_unpruned
    post_pruning(root_post, df_test)
    print("accuracy of post-pruned tree: %.3f" % acc_cal(root_post, df_test))
    print("Branches %s" %root_post.branch)
    draw_tree(root_post, "dt_post_pruning.png")
      

if __name__ == '__main__':
    test()
            
