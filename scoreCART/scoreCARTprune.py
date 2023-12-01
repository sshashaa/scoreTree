import numpy as np
from scoreCART.newsplit import sse_for_new_split, crps_for_new_split, dss_for_new_split, is1_for_new_split, crpsnew_for_new_split
from scoreCART.accuracy import accuracy_sse, accuracy_crps, accuracy_dss, accuracy_is1, accuracy_crpsnew
import copy

class scoreCART():
    def __init__(self, method, train_set, tol, max_depth, min_node_size, num_quantiles, alpha, prune_thr, args):

        self.train_set = train_set
        self.x_dim = len(train_set[0]) - 1
        self.method = method
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.num_quantiles = num_quantiles
        self.args = args
        self.new_split_funcs = eval(method + '_for_new_split')
        self.is_cat = args['is_cat']
        self.cov_uniqvals = args['cov_uniqvals']
        self.tol = tol
        self.alpha = alpha
        self.prune_thr = prune_thr
                
    # Find the empirical cdf of a sample, Outcome: quantiles and cumulative probabilities
    def ecdf(self, sample):
        sample = np.atleast_1d(sample)
        quantiles, counts = np.unique(sample, return_counts=True)
        cumprob = np.cumsum(counts).astype(np.double) / sample.size
        return quantiles, cumprob

    def column(self, matrix, i):
        return [row[i] for row in matrix]

    def test_split(self, index, value, train_set, equal):
        left, right = list(), list()
        if equal:
            for row in train_set:
                if row[index] == value:
                    left.append(row)
                else:
                        right.append(row)
        else:
            for row in train_set:
                if row[index] < value:
                    left.append(row)
                else:
                    right.append(row)
        return left, right

    def get_split(self, nodetr):
        #train_set = self.train_set 
        args = {}
        args['alpha'] = self.alpha
        x_dim = self.x_dim 
        num_quantiles = self.num_quantiles
        min_node_size = self.min_node_size 
        b_index, b_value, b_groups = 999, 999, None
        b_score = self.new_split_funcs(nodetr, 0, args)
        b_parent = copy.copy(b_score)
        #print('b_parent:', b_parent)
        first_val = 0  
        split_occurs = 0
        
        for index in range(x_dim):
            qe, pe = self.ecdf(self.column(nodetr, index))
            if self.is_cat[index]:# and len(unique_vals) <= 25:
                tocheck_val = qe
                equal = 1
            elif len(qe) < num_quantiles:
                tocheck_val = qe
                equal = 0
            else:
                inc_p = 1/(num_quantiles+1)
                inds = [next(x[0] for x in enumerate(pe) if x[1] > i*inc_p) for i in range(1,(num_quantiles+1))] 
                tocheck_val = list(sorted(set([qe[i] for i in inds])))
                equal = 0        
            for val in tocheck_val:
                groups = self.test_split(index, val, nodetr, equal)
                if len(groups[0]) >= min_node_size and len(groups[1]) >= min_node_size:
                    measure = self.new_split_funcs(groups, 1, args)
                    if not first_val:
                        first_val = 1
                        if b_score < measure:
                            print("monotonicity violated - " + str(self.method) + " - variable "+str(index))
                            print(b_score)
                            print(measure)
                            # log_file.write("monotonicity violated - "+ str(self.method) + " - variable "+str(val))
                        b_score = max(b_score, measure)
                    if split_occurs:
                        check_tol = 0
                    else:
                        check_tol = self.tol
    
                    if measure <= b_score*(1 - check_tol):                    
                        split_occurs = 1
                        b_index, b_value, b_score, b_groups = index, val, measure, groups
        if not split_occurs:
            print("no improvement - " + str(self.method))
            # log_file.write("no improvement - " + str(self.method))
        
        node_size = sum([len(g) for g in b_groups])
        #print('b_score:', b_score)
        # print('improvement:', (b_parent-b_score)/node_size)
        return {'index':b_index, 'value':b_value, 'groups':b_groups, 'score': b_score, 'improvement': (b_parent-b_score)/node_size}  

    # Return the observaions in the leaf    
    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return outcomes
    
    def split(self, node, depth):

        max_depth = self.max_depth
        min_node_size = self.min_node_size 

        if node['groups']:
            left, right = node['groups']
            del(node['groups'])
        else:
            print('NOTHING')
            
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        
        #if depth == 1:
            # process left child
        if len(left) < 3*min_node_size:
            node['left'] = self.to_terminal(left)
        else:
            left_split = self.get_split(left)
            
            # if left_split['score'] < (self.prune_thr)*(self.initialscore):
            if left_split['improvement'] < (self.prune_thr)*(self.initialimpr):
                node['left'] = self.to_terminal(left)
            else:
                node['left'] = left_split
                self.split(node['left'], depth+1)
                
            # process right child
        if len(right) < 3*min_node_size:
            node['right'] = self.to_terminal(right)
        else:
            right_split = self.get_split(right)
            
            #if right_split['score'] < (self.prune_thr)*(self.initialscore):      
            if right_split['improvement'] < (self.prune_thr)*(self.initialimpr): 
                node['right'] = self.to_terminal(right)
            else:
                node['right'] = right_split
                self.split(node['right'], depth+1)

    # Print a decision tree
    def print_tree(self, node, depth=0):
        is_cat = self.is_cat
        if isinstance(node, dict):
            if is_cat[node['index']]:
                print('%s[X%d = %d]' % ((depth*' ', (node['index']+1), int(node['value']))))
            else:
                print('%s[X%d < %.4f]' % ((depth*' ', (node['index']+1), node['value'])))

            self.print_tree(node['left'], depth+1)
            self.print_tree(node['right'], depth+1)
        else:
            print('%s[%s]' % ((depth*' ', len(node)))) 
                
    def build_tree(self):
        #C0 = self.new_split_funcs(self.train_set, 0, args=None)
        root = self.get_split(self.train_set)
        self.initialscore = root['score']
        self.initialimpr = root['improvement']
        self.split(root, 1)
        #print("tree_method " + self.method + "\n###########################")
        #self.print_tree(root, depth=0)
        self.fittedtree = root
        
    # All the functions below for evalution
    # List of data points in all leaves 
    def leaves_list(self, node, depth=0):
        #leaves = []
        if isinstance(node, dict):
            self.leaves_list(node['left'], depth+1)
            self.leaves_list(node['right'], depth+1)
        else:
            self.leaves.append(node)
        #return leaves
        
    def predict(self, node, test_data_row):
    	if test_data_row[node['index']] < node['value']:
    		if isinstance(node['left'], dict):
    			return self.predict(node['left'], test_data_row)
    		else:
    			return node['left']
    	else:
    		if isinstance(node['right'], dict):
    			return self.predict(node['right'], test_data_row)
    		else:
    			return node['right']

    def tree_preds(self, test_set):
        self.leaves = []
        self.leaves_list(self.fittedtree, 0)
        predictions = list()
        for row in test_set:
            prediction = self.predict(self.fittedtree, row)
            predictions.append(self.leaves.index(prediction))
        return predictions        
    
    def accuracy_val(self, xtest, ytest, xtrain, ytrain, metrics=['sse']):
        args = {}
        args['alpha'] = self.alpha
        
        # sort data once (sort x based on y, then sort y)
        xtest = [xtest for _, xtest in sorted(zip(ytest, xtest))]
        ytest = sorted(ytest)
        
        xtrain = [xtrain for _, xtrain in sorted(zip(ytrain, xtrain))]
        ytrain = sorted(ytrain)
        
        
        # predictions show the leaf id falling
        predictions = self.tree_preds(xtest)
        predictions_in = self.tree_preds(xtrain)

        leaf_dict = dict((str(l),[]) for l in self.leaves)
        leaf_dict_in = dict((str(l),[]) for l in self.leaves)
        
        #print(self.leaves)
        
        for l in range(len(self.leaves)):
            leaf_dict[str(self.leaves[l])] = [ytest[i] for i in range(len(ytest)) if predictions[i] == l]
            leaf_dict_in[str(self.leaves[l])] = [ytrain[i] for i in range(len(ytrain)) if predictions_in[i] == l]
        
 
        evals_dict = {}
        for eval_method in metrics:
            self.accuracy_func = eval('accuracy_' + eval_method)

            eval_new = [self.accuracy_func(leaf_dict, args)]
            #print(eval_new)
            eval_new += [self.accuracy_func(leaf_dict_in, args)]
            #print(eval_new)
            evals_dict[eval_method] = eval_new
            
        return evals_dict
        
