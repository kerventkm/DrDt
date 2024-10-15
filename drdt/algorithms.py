import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from drdt.helper_functions import Reduction, R_SR, R_AD, SAlphaStep, SPlus, SMax, NCover, NGreedy


########################################### Algorithms based on the path #######################################################
class A_C_N:
    def __init__(self, C="EAR", N="cover"):
        """
        C - type of problem:
            "AR" - All Rules
            "EAR" - Extended All Rules
            "SR" - Some Rules
            "ESR" - Extended Some Rules
            "AD" - All Decisions
            "EAD" - Extended All Decisions
        N - type of Node Cover Algorithm:
            "cover"
            "greedy"
        """
        self.C = C
        self.N = N
        self.depth = 0
        self.rules = None
        self.R_C = Reduction(C)
        
    def solve(self, S, delta):
        """
        S - System of Decision Rules (pandas DataFrame)
        delta - tuple of attribute values from the set V_C (a row of a pandas df, without any NaN values)
        """
        num_of_features = len(S.columns)-1
        
        # AR - All Rules problem, EAR - Extended All Rules problem
        if self.C in ["AR", "EAR"]:
            if self.N == "cover": # with "cover" NodeCover method
                Q = S.copy()
                while True:
                    # Step 1
                    P = self.R_C(Q)
                    if (P.empty or P.iloc[:, :-1].isna().all().all()):
                        if P.empty:
#                             print("There is no such rule")
                            return self.depth, self.rules
                        else:
                            row_indecies = P.index.tolist()
                            self.rules = S.loc[row_indecies]
                            return self.depth, self.rules
                       
                    # Step 2
                    else:
                        P_plus = SPlus(P)
                        B = NCover(P_plus)
                        
                        if len(B) == num_of_features:    # Checking, if they are same no need to run further
                            return num_of_features, "when we focus just depth"
                        
                        for attr in B:
                            self.depth += 1
                            alpha = (attr, delta[attr])
                            P = SAlphaStep(P, alpha)
                        Q = P
                            
            elif self.N == "greedy": # with "cover" NodeCover method
                Q = S.copy()
                while True:
                    # Step 1
                    P = self.R_C(Q)
                    if (P.empty or P.iloc[:, :-1].isna().all().all()):
                        if P.empty:
#                             print("There is no such rule")
                            return self.depth, self.rules
                        else:
                            row_indecies = P.index.tolist()
                            self.rules = S.loc[row_indecies]
                            return self.depth, self.rules
                    # Step 2
                    else:
                        P_plus = SPlus(P)
                        B = NGreedy(P_plus)
                        
                        if len(B) == num_of_features:    # Checking, if they are same no need to run further
                            return num_of_features, "when we focus just depth"
                        
                        for attr in B:
                            self.depth += 1
                            alpha = (attr, delta[attr])
                            P = SAlphaStep(P, alpha)
                        Q = P
            
            else:
                raise ValueError("N must be 'cover' or 'greedy'.")
            
                
        # SR - Some Rules problem, ESR - Extended Some Rules problem
        elif self.C in ["SR", "ESR"]:
            if self.N == "cover": # with "cover" NodeCover method
                Q = S.copy()
                while True:
                    # Step 1
                    P = self.R_C(Q)
                    if (P.empty or P.iloc[:, :-1].isna().all().all()):
                        if P.empty:
#                             print("There is no such rule")
                            return self.depth, self.rules
                        else:
                            row_indecies = P.index.tolist()
                            self.rules = S.loc[row_indecies]
                            return self.depth, self.rules
                       
                    # Step 2
                    else:
                        P_plus = SPlus(P)
                        B = NCover(P_plus)
                        
                        if len(B) == num_of_features:    # Checking, if they are same no need to run further
                            return num_of_features, "when we focus just depth"
                        
                        for attr in B:
                            self.depth += 1
                            alpha = (attr, delta[attr])
                            P = SAlphaStep(P, alpha)
                        Q = P
                            
            elif self.N == "greedy": # with "cover" NodeCover method
                Q = S.copy()
                while True:
                    # Step 1
                    P = self.R_C(Q)
                    if (P.empty or P.iloc[:, :-1].isna().all().all()):
                        if P.empty:
#                             print("There is no such rule")
                            return self.depth, self.rules
                        else:
                            row_indecies = P.index.tolist()
                            self.rules = S.loc[row_indecies]
                            return self.depth, self.rules
                       
                    # Step 2
                    else:
                        P_plus = SPlus(P)
                        B = NGreedy(P_plus)
                        
                        if len(B) == num_of_features:    # Checking, if they are same no need to run further
                            return num_of_features, "when we focus just depth"
                        
                        for attr in B:
                            self.depth += 1
                            alpha = (attr, delta[attr])
                            P = SAlphaStep(P, alpha)
                        Q = P
            
            else:
                raise ValueError("N must be 'cover' or 'greedy'.")
                
        # AD - All Decisions problem, EAD - Extended All Decisions problem
        elif self.C in ["AD", "EAD"]:
            if self.N == "cover": # with "cover" NodeCover method
                Q = S.copy()
                while True:
                    # Step 1
                    P = self.R_C(Q)
                    if (P.empty or P.iloc[:, :-1].isna().all().all()):
                        if P.empty:
#                             print("There is no such rule")
                            return self.depth, self.rules
                        else:
                            row_indecies = P.index.tolist()
                            self.rules = S.loc[row_indecies]
                            return self.depth, self.rules
                       
                    # Step 2
                    else:
                        P_plus = SPlus(P)
                        B = NCover(P_plus)
                        
                        if len(B) == num_of_features:    # Checking, if they are same no need to run further
                            return num_of_features, "when we focus just depth"
                        
                        for attr in B:
                            self.depth += 1
                            alpha = (attr, delta[attr])
                            P = SAlphaStep(P, alpha)
                        Q = P
                            
            elif self.N == "greedy": # with "cover" NodeCover method
                Q = S.copy()
                while True:
                    # Step 1
                    P = self.R_C(Q)
                    if (P.empty or P.iloc[:, :-1].isna().all().all()):
                        if P.empty:
#                             print("There is no such rule")
                            return self.depth, self.rules
                        else:
                            row_indecies = P.index.tolist()
                            self.rules = S.loc[row_indecies]
                            return self.depth, self.rules
                       
                    # Step 2
                    else:
                        P_plus = SPlus(P)
                        B = NGreedy(P_plus)
                        
                        if len(B) == num_of_features:    # Checking, if they are same no need to run further
                            return num_of_features, "when we focus just depth"
                        
                        for attr in B:
                            self.depth += 1
                            alpha = (attr, delta[attr])
                            P = SAlphaStep(P, alpha)
                        Q = P
            
            else:
                raise ValueError("N must be 'cover' or 'greedy'.")
                
        # Wrong problem type      
        else: 
            raise ValueError("C must be one of {'AR', 'EAR', 'SR', 'ESR', 'AD', 'EAD'}")
            
#################################### Start Greedy Debug #################################################################

class A_C_G:
    def __init__(self, C="EAR"):
        """
         For each of the considered six problems, this is a polynomial time algorithm
         that, for a given tuple of attribute values, describes the work on this tuple 
         of a decision tree, which solves the problem. This algorithm is a completely 
         greedy algorithm by nature.
         
         C - type of problem:
            "AR" - All Rules
            "EAR" - Extended All Rules
            "SR" - Some Rules
            "ESR" - Extended Some Rules
            "AD" - All Decisions
            "EAD" - Extended All Decisions
        """
        self.C = C
        self.depth = 0
        self.rules = None
        self.R_C = Reduction(C)
        self.attributes_order = []
        
    def solve(self, S, delta):
        """
        S - System of Decision Rules (pandas DataFrame)
        delta - tuple of attribute values from the set V_C (a row of a pandas df, without any NaN values)
        """
        
        # AR - All Rules problem, EAR - Extended All Rules problem
        if self.C in ["AR", "EAR"]:
            Q = S.copy()
            while True:
                # Step 1
                P = self.R_C(Q)
                if (P.empty or P.iloc[:, :-1].isna().all().all()):
                    if P.empty:
#                             print("There is no such rule")
                        return self.depth, self.rules
                    else:
                        row_indecies = P.index.tolist()
                        self.rules = S.loc[row_indecies]
                        return self.depth, self.rules

                # Step 2
                else:
#                     for column in P.columns:       # We choose an attribute ai ∈ A(P) with the minimum index i
#                         if P[column].notna().any():
#                             attr = column
#                             break
                    self.depth += 1
                    attr = P.iloc[:, :-1].count().idxmax() # Find the column name with the maximum number of non-NaN values
                    alpha = (attr, delta[attr])
                    Q = SAlphaStep(P, alpha)
                    self.attributes_order.append(attr)
           
                
            
        # SR - Some Rules problem, ESR - Extended Some Rules problem
        elif self.C in ["SR", "ESR"]:
            Q = S.copy()
            while True:
                # Step 1
                P = self.R_C(Q)
                if (P.empty or P.iloc[:, :-1].isna().all().all()):
                    if P.empty:
#                             print("There is no such rule")
                        return self.depth, self.rules
                    else:
                        row_indecies = P.index.tolist()
                        self.rules = S.loc[row_indecies]
                        return self.depth, self.rules

                # Step 2
                else:
#                     for column in P.columns:       # We choose an attribute ai ∈ A(P) with the minimum index i
#                         if P[column].notna().any():
#                             attr = column
#                             break
                    self.depth += 1
                    attr = P.iloc[:, :-1].count().idxmax() # Find the column name with the maximum number of non-NaN values
                    alpha = (attr, delta[attr])
                    Q = SAlphaStep(P, alpha)
                    self.attributes_order.append(attr)
                
        # AD - All Decisions problem, EAD - Extended All Decisions problem
        elif self.C in ["AD", "EAD"]:
            Q = S.copy()
            while True:
                # Step 1
                P = self.R_C(Q)
                if (P.empty or P.iloc[:, :-1].isna().all().all()):
                    if P.empty:
#                             print("There is no such rule")
                        return self.depth, self.rules
                    else:
                        row_indecies = P.index.tolist()
                        self.rules = S.loc[row_indecies]
                        return self.depth, self.rules

                # Step 2
                else:
#                     for column in P.columns:       # We choose an attribute ai ∈ A(P) with the minimum index i
#                         if P[column].notna().any():
#                             attr = column
#                             break
                    self.depth += 1
                    attr = P.iloc[:, :-1].count().idxmax() # Find the column name with the maximum number of non-NaN values
                    alpha = (attr, delta[attr])
                    Q = SAlphaStep(P, alpha)
                    self.attributes_order.append(attr)
                
        # Wrong problem type      
        else: 
            raise ValueError("C must be one of {'AR', 'EAR', 'SR', 'ESR', 'AD', 'EAD'}")
            
            
#################################### End Greedy Debug #################################################################


# class A_C_G:
#     def __init__(self, C="EAR"):
#         """
#          For each of the considered six problems, this is a polynomial time algorithm
#          that, for a given tuple of attribute values, describes the work on this tuple 
#          of a decision tree, which solves the problem. This algorithm is a completely 
#          greedy algorithm by nature.
         
#          C - type of problem:
#             "AR" - All Rules
#             "EAR" - Extended All Rules
#             "SR" - Some Rules
#             "ESR" - Extended Some Rules
#             "AD" - All Decisions
#             "EAD" - Extended All Decisions
#         """
#         self.C = C
#         self.depth = 0
#         self.rules = None
#         self.R_C = Reduction(C)
        
#     def solve(self, S, delta):
#         """
#         S - System of Decision Rules (pandas DataFrame)
#         delta - tuple of attribute values from the set V_C (a row of a pandas df, without any NaN values)
#         """
        
#         # AR - All Rules problem, EAR - Extended All Rules problem
#         if self.C in ["AR", "EAR"]:
#             Q = S.copy()
#             while True:
#                 # Step 1
#                 P = self.R_C(Q)
#                 if (P.empty or P.iloc[:, :-1].isna().all().all()):
#                     if P.empty:
# #                             print("There is no such rule")
#                         return self.depth, self.rules
#                     else:
#                         row_indecies = P.index.tolist()
#                         self.rules = S.loc[row_indecies]
#                         return self.depth, self.rules

#                 # Step 2
#                 else:
# #                     for column in P.columns:       # We choose an attribute ai ∈ A(P) with the minimum index i
# #                         if P[column].notna().any():
# #                             attr = column
# #                             break
#                     self.depth += 1
#                     attr = P.iloc[:, :-1].count().idxmax() # Find the column name with the maximum number of non-NaN values
#                     alpha = (attr, delta[attr])
#                     Q = SAlphaStep(P, alpha)
            
#         # SR - Some Rules problem, ESR - Extended Some Rules problem
#         elif self.C in ["SR", "ESR"]:
#             Q = S.copy()
#             while True:
#                 # Step 1
#                 P = self.R_C(Q)
#                 if (P.empty or P.iloc[:, :-1].isna().all().all()):
#                     if P.empty:
# #                             print("There is no such rule")
#                         return self.depth, self.rules
#                     else:
#                         row_indecies = P.index.tolist()
#                         self.rules = S.loc[row_indecies]
#                         return self.depth, self.rules

#                 # Step 2
#                 else:
# #                     for column in P.columns:       # We choose an attribute ai ∈ A(P) with the minimum index i
# #                         if P[column].notna().any():
# #                             attr = column
# #                             break
#                     self.depth += 1
#                     attr = P.iloc[:, :-1].count().idxmax() # Find the column name with the maximum number of non-NaN values
#                     alpha = (attr, delta[attr])
#                     Q = SAlphaStep(P, alpha)
                
#         # AD - All Decisions problem, EAD - Extended All Decisions problem
#         elif self.C in ["AD", "EAD"]:
#             Q = S.copy()
#             while True:
#                 # Step 1
#                 P = self.R_C(Q)
#                 if (P.empty or P.iloc[:, :-1].isna().all().all()):
#                     if P.empty:
# #                             print("There is no such rule")
#                         return self.depth, self.rules
#                     else:
#                         row_indecies = P.index.tolist()
#                         self.rules = S.loc[row_indecies]
#                         return self.depth, self.rules

#                 # Step 2
#                 else:
# #                     for column in P.columns:       # We choose an attribute ai ∈ A(P) with the minimum index i
# #                         if P[column].notna().any():
# #                             attr = column
# #                             break
#                     self.depth += 1
#                     attr = P.iloc[:, :-1].count().idxmax() # Find the column name with the maximum number of non-NaN values
#                     alpha = (attr, delta[attr])
#                     Q = SAlphaStep(P, alpha)
                
#         # Wrong problem type      
#         else: 
#             raise ValueError("C must be one of {'AR', 'EAR', 'SR', 'ESR', 'AD', 'EAD'}")
            
            
########################################### Algorithms For Full Decision Tree ###################################################

class DynamicProgrammingAlgorithms:
    def __init__(self, C="AR"):
        """
        C - type of problem:
            "AR" - All Rules
            "EAR" - Extended All Rules
            "SR" - Some Rules
            "ESR" - Extended Some Rules
            "AD" - All Decisions
            "EAD" - Extended All Decisions
        """
        self.C = C
                
        self.R_C = Reduction(C)
            
            
    def DAG_C(self, S):
        """Construct the DAG based on the decision rule system S."""

        DAG = nx.MultiDiGraph()
        DAG.add_node(id(S), data=S, processed=False, H=None, best_attr=None)

        def process_node(Q, DAG):
            """Process a node by checking applicable rules and updating the DAG."""
            if (self.R_C(Q).empty or self.R_C(Q).iloc[:, :-1].isna().all().all()):
                DAG.nodes[id(Q)]['processed'] = True
                DAG.nodes[id(Q)]['H'] = 0
                return

            for a in Q.columns[Q.notna().any()].tolist()[:-1]: # Finding columns with at least one non-NaN value
                for delta in S[a].dropna().unique():  # Find unique non-NaN values in column a
                    new_node_data = SAlphaStep(Q, (a, delta if type(delta) in [str, np.str_] else float(delta)))
                    not_equal = True
                    for node in DAG.nodes:
                        if new_node_data.equals(DAG.nodes[node]['data']):
                            DAG.add_edge(id(Q), node, label=(a,delta))
                            not_equal = False
                            break
                    if not_equal:
                        DAG.add_node(id(new_node_data), data=new_node_data, processed=False, H=None, best_attr=None)
                        DAG.add_edge(id(Q), id(new_node_data), label=(a,delta))
            # Mark the current node as processed
            DAG.nodes[id(Q)]['processed'] = True

        while not all(nx.get_node_attributes(DAG, 'processed').values()):
            # Find a node that is not processed
            for node in DAG.nodes:
                if not DAG.nodes[node]['processed']:
                    process_node(DAG.nodes[node]['data'], DAG)
                    break # Move to next iteration after processing a node
        return DAG
    
    
    def DAG_EC(self, S):
        """Construct the DAG based on the decision rule system S."""

        DAG = nx.MultiDiGraph()
        DAG.add_node(id(S), data=S, processed=False, H=None, best_attr=None)

        def process_node(Q, DAG):
            """Process a node by checking applicable rules and updating the DAG."""
            if (self.R_C(Q).empty or self.R_C(Q).iloc[:, :-1].isna().all().all()):
                DAG.nodes[id(Q)]['processed'] = True
                DAG.nodes[id(Q)]['H'] = 0
                return

            for a in Q.columns[Q.notna().any()].tolist()[:-1]: # Finding columns with at least one non-NaN value
                for delta in np.append(S[a].dropna().unique(), "*"):  # Find unique non-NaN values in column a
#                     try:
#                         delta = float(delta)
#                     except ValueError:
#                         pass 
                    new_node_data = SAlphaStep(Q, (a, delta))# if type(delta) in [str, np.str_] else int(delta)))
                    not_equal = True
                    for node in DAG.nodes:
                        if new_node_data.equals(DAG.nodes[node]['data']):
                            DAG.add_edge(id(Q), node, label=(a,delta))
                            not_equal = False
                            break
                    if not_equal:
                        DAG.add_node(id(new_node_data), data=new_node_data, processed=False, H=None, best_attr=None)
                        DAG.add_edge(id(Q), id(new_node_data), label=(a,delta))
            # Mark the current node as processed
            DAG.nodes[id(Q)]['processed'] = True

        while not all(nx.get_node_attributes(DAG, 'processed').values()):
            # Find a node that is not processed
            for node in DAG.nodes:
                if not DAG.nodes[node]['processed']:
                    process_node(DAG.nodes[node]['data'], DAG)
                    break # Move to next iteration after processing a node
        return DAG
    
    
    def DAG_update(self, DAG, node):

        # Step 1: If "H" is already a number, return it
        if DAG.nodes[node]['H']:
            return DAG.nodes[node]['H']

        # Step 2: Process the node based on its children
        children = list(DAG.successors(node))
        if not children:  # Terminal node
            DAG.nodes[node]['H'] = 0
        else:
            # Calculate the H value based on children and edge labels
            attribute_H_values = []
            for child in children:
                edges_data = DAG.get_edge_data(node, child)
                for edge in edges_data:
                    a, delta = edges_data[edge]['label']  # Assuming the edge label is stored as a tuple (a, delta)
                    child_H = DAG.nodes[child]['H'] if DAG.nodes[child].get('H') is not None else self.DAG_update(DAG, child)
                    attribute_H_values.append((a, child_H))

            # Group by attribute and find the min of the max H values for each attribute
            attribute_to_H = {}
            for a, h in attribute_H_values:
                if a in attribute_to_H:
                    attribute_to_H[a] = max(attribute_to_H[a], h)
                else:
                    attribute_to_H[a] = h

            # Now calculate the final H value for the node
            best_attr, min_H = min(attribute_to_H.items(), key=lambda x: x[1])

            DAG.nodes[node]['H'] = 1 + min_H
            DAG.nodes[node]['best_attr'] = best_attr

        return DAG.nodes[node]['H']
    
    def A_DP(self, S):
        
        if self.C in ["AR", "SR", "AD"]:
#             if self.C == "AR" and (not S.dropna().empty):
#                 return len(S.columns) - 1
            if self.C == "SR" and (S.iloc[:, :-1].isna().all(axis=1).any()):
                return 0
            DAG = self.DAG_C(S)
        else:
#             if self.C == "EAR" and (not S.dropna().empty):
#                 return len(S.columns) - 1
            if self.C == "ESR" and (S.iloc[:, :-1].isna().all(axis=1).any()):
                return 0
            DAG = self.DAG_EC(S)

        return self.DAG_update(DAG, id(S))
    
    def create_decision_tree(self, DAG, node):
        """
        Creates a decision tree from a given DAG and initial node.

        Args:
        DAG: The Directed Acyclic Graph from which to create the decision tree.
        node: The initial node in the DAG to start creating the decision tree.

        Returns:
        A decision tree represented as a nested dictionary.
        """
        # Base case: if the node is a terminal node, return its data as the leaf of the decision tree
        if DAG.nodes[node]['best_attr'] is None:
            return {"Result": set(self.R_C(DAG.nodes[node]['data']).index)} # It is mandatory for SR, ESR, AD, EAD problems

        # Initialize the decision tree node with the best attribute
        best_attr = DAG.nodes[node]['best_attr']
        decision_tree_node = {best_attr: {}}

        # Iterate over outgoing edges of the node, but only consider edges that match the best_attr
        for _, child, edge_data in DAG.out_edges(node, data=True):
            a, delta = edge_data['label']
            if a == best_attr:
                # Recursively build the decision tree for the child node
                decision_tree_node[best_attr][delta] = self.create_decision_tree(DAG, child)

        return decision_tree_node
    
    def DT(self, S):
        
        if self.C in ["AR", "SR", "AD"]:
            DAG = self.DAG_C(S)
        else:
            DAG = self.DAG_EC(S)
            
        self.DAG_update(DAG, id(S))

        return self.create_decision_tree(DAG, id(S))
