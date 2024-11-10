import numpy as np 
import pandas as pd
from tqdm import tqdm
import random
import itertools


def Reduction(problem):
    if (problem=="AR" or problem=="EAR"):
        return R_AR
    elif (problem=="SR" or problem=="ESR"):
        return R_SR
    elif (problem=="AD" or problem=="EAD"):
        return R_AD
    else:
        raise ValueError("C must be one of {'AR', 'EAR', 'SR', 'ESR', 'AD', 'EAD'}")


def R_AR(S):
    """
    input: S - system of decision rules (pandas DataFrame)
    output: S - system of decision rules (pandas DataFrame)
    """
    return S

# 1203 seconds
# def R_SR(S):
#     """
#     input: S - system of decision rules (pandas DataFrame)
#     output: subset of S which has reduced by SR reduction (pandas DataFrame)
#     """
#     to_remove = set()
#     n = len(S)

#     for i in range(n):
#         for j in range(i+1, n):
#             if i in to_remove or j in to_remove:
#                 continue
#             r1, r2 = S.iloc[i][:-1], S.iloc[j][:-1] # Exclude the last column
#             # Check if r1 is a subset of r2
#             if all(pd.isna(r1[k]) or r1[k] == r2[k] for k in S.columns[:-1]):
#                 to_remove.add(j)
#             # Check if r2 is a subset of r1
#             elif all(pd.isna(r2[k]) or r2[k] == r1[k] for k in S.columns[:-1]):
#                 to_remove.add(i)

#     return S.drop(S.index[list(to_remove)])

# 297 seconds
# def R_SR(S):
#     """
#     input: S - pandas DataFrame
#     output: pandas DataFrame - subset of S which has reduced by SR reduction 
#     """
#     to_remove = set()
#     data = S.iloc[:, :-1]  # exclude the last column

#     for i in range(len(data)):
#         if i in to_remove:
#             continue
#         row_i = data.iloc[i]
#         for j in range(i + 1, len(data)):
#             if j in to_remove:
#                 continue
#             row_j = data.iloc[j]
            
#             # Check if row_i is a subset of row_j
#             if all(pd.isna(row_i[k]) or row_i[k] == row_j[k] for k in data.columns):
#                 to_remove.add(j)
#             # Check if row_j is a subset of row_i
#             elif all(pd.isna(row_j[k]) or row_j[k] == row_i[k] for k in data.columns):
#                 to_remove.add(i)

#     return S.drop(S.index[list(to_remove)])

# 17 seconds but gives wrong result
# def R_SR(S):
#     """
#     input: S - pandas DataFrame
#     output: pandas DataFrame - subset of S which has reduced by SR reduction 
#     """
#     data = S.iloc[:, :-1].to_numpy()  # Convert to NumPy array, excluding the last column for faster operations
#     to_remove = np.zeros(len(S), dtype=bool)  # Boolean array to mark rows for removal

#     for i in range(len(data)):
#         if to_remove[i]:
#             continue
#         row_i = data[i]

#         # Apply broadcasting with Pandas `isna` to handle NaN and non-numeric types
#         mask = np.all((pd.isna(row_i) | (data == row_i)), axis=1)
        
#         # Mark rows as subset of row_i if mask is True, excluding itself
#         to_remove |= (mask & (np.arange(len(data)) > i))

#     # Select rows not marked for removal
#     return S[~to_remove].reset_index(drop=True)

# 62 seconds
# def R_SR(S):
#     """
#     input: S - pandas DataFrame
#     output: pandas DataFrame - subset of S which has reduced by SR reduction 
#     """
#     data = S.iloc[:, :-1].to_numpy()  # Exclude the last column and convert to NumPy array for faster processing
#     to_remove = np.zeros(len(S), dtype=bool)  # Boolean array to mark rows for removal

#     for i in range(len(data)):
#         if to_remove[i]:
#             continue
#         row_i = data[i]
        
#         for j in range(i + 1, len(data)):
#             if to_remove[j]:
#                 continue
#             row_j = data[j]
            
#             # Check if row_i is a subset of row_j
#             if np.all(pd.isna(row_i) | (row_i == row_j)):
#                 to_remove[j] = True
#             # Check if row_j is a subset of row_i
#             elif np.all(pd.isna(row_j) | (row_j == row_i)):
#                 to_remove[i] = True
#                 break  # Since row_i is now marked, no need to check it against further rows

#     # Select rows not marked for removal
#     return S[~to_remove].reset_index(drop=True)

# 20 seconds
def R_SR(S):
    """
    input: S - pandas DataFrame
    output: pandas DataFrame - subset of S which has reduced by SR reduction 
    """
    data = S.iloc[:, :-1].to_numpy()  # Convert to NumPy array, excluding the last column
    to_remove = np.zeros(len(S), dtype=bool)  # Boolean array to mark rows for removal

    for i in range(len(data)):
        if to_remove[i]:
            continue
        row_i = data[i]

        # Compare row_i to rows below it for subset checks
        mask = np.all(pd.isna(row_i) | (data[i + 1:] == row_i), axis=1)
        
        # Mark rows as subset of row_i if mask is True
        to_remove[i + 1:] |= mask
        
        # Check if any row below row_i is a subset of row_i
        subset_mask = np.all(pd.isna(data[i + 1:]) | (data[i + 1:] == row_i), axis=1)
        
        if np.any(subset_mask):
            to_remove[i] = True

    # Select rows not marked for removal
    return S[~to_remove].reset_index(drop=True)

# 765 seconds
# def R_AD(S):
#     """
#     input: S - system of decision rules (pandas DataFrame)
#     output: subset of S which has reduced by AD reduction (pandas DataFrame)
#     """
#     to_remove = set()
#     n = len(S)

#     for i in range(n):
#         for j in range(i+1, n):
#             if i in to_remove or j in to_remove:
#                 continue
#             r1, r2 = S.iloc[i], S.iloc[j]

#             if all(pd.isna(r1[k]) or r1[k] == r2[k] for k in S.columns):
#                 to_remove.add(j)
#             # Check if r2 is a subset of r1
#             elif all(pd.isna(r2[k]) or r2[k] == r1[k] for k in S.columns):
#                 to_remove.add(i)

#     return S.drop(S.index[list(to_remove)])

# 44 seconds
# def R_AD(S):
#     """
#     input: S - system of decision rules (pandas DataFrame)
#     output: subset of S which has reduced by AD reduction (pandas DataFrame)
#     """
#     to_remove = set()
#     data = S.values  # convert entire DataFrame to a numpy array for faster processing

#     for i in range(len(data)):
#         if i in to_remove:
#             continue
#         row_i = data[i]
#         for j in range(i + 1, len(data)):
#             if j in to_remove:
#                 continue
#             row_j = data[j]
            
#             # Check if row_i is a subset of row_j or equal
#             if all(pd.isna(row_i[k]) or row_i[k] == row_j[k] for k in range(len(S.columns))):
#                 to_remove.add(j)
#             # Check if row_j is a subset of row_i or equal
#             elif all(pd.isna(row_j[k]) or row_j[k] == row_i[k] for k in range(len(S.columns))):
#                 to_remove.add(i)

#     return S.drop(S.index[list(to_remove)])

def R_AD(S):
    """
    input: S - system of decision rules (pandas DataFrame)
    output: subset of S which has reduced by AD reduction (pandas DataFrame)
    """
    data = S.to_numpy()  # Convert entire DataFrame to NumPy array for faster processing
    to_remove = np.zeros(len(S), dtype=bool)  # Boolean array to mark rows for removal

    for i in range(len(data)):
        if to_remove[i]:
            continue
        row_i = data[i]

        # Compare row_i to rows below it for subset or equality checks
        mask = np.all(pd.isna(row_i) | (data[i + 1:] == row_i), axis=1)
        
        # Mark rows as subset of row_i if mask is True
        to_remove[i + 1:] |= mask
        
        # Check if any row below row_i is a subset of row_i or equal
        subset_mask = np.all(pd.isna(data[i + 1:]) | (data[i + 1:] == row_i), axis=1)
        
        if np.any(subset_mask):
            to_remove[i] = True

    # Select rows not marked for removal
    return S[~to_remove].reset_index(drop=True)


def SAlphaStep(S, alpha):
    """
    input: S - system of decision rules (pandas DataFrame)
           alpha - s tuple of the form (a_i, delta_j)
    output: S_alpha - subset of S as defined in the paper.(just for 1 attribute) (pandas DataFrame)
    """
    
    attr, value = alpha

    # Keep rows where the attr is NaN or equals the specified value
    S = S[(S[attr].isna()) | (S[attr] == value)]
    
    #Make NaN the values
#     S.loc[~S[attr].isna(), attr] = np.nan
    S_copy = S.copy()
    S_copy.loc[~S_copy[attr].isna(), attr] = np.nan
    S = S_copy

    return S


def SPlus(S):
    """
    input: S - system of decision rules (pandas DataFrame)
    output: S_plus - subset of S as defined in the paper. (pandas DataFrame)
    """
    non_nan_counts = S.notna().sum(axis=1)

    max_non_nan = non_nan_counts.max()

    S_plus = S[non_nan_counts == max_non_nan]
    
    return S_plus


def SMax(S_plus):
    """
    input: S_plus - system of decision rules with length d (pandas DataFrame)
    output: S_max - subset of S as defined in the paper. (pandas DataFrame)
    """

    columns_to_check = S_plus.columns[:-1]

    S_max = S_plus.drop_duplicates(subset=columns_to_check, keep='first')
    
    return S_max


def DecisionRuleCreatorFromDecisionTable(DecisionTable):
    mask = []

#     def setCover(S, A):
#         """
#         Find a subset of S that covers all elements in A using a greedy approach.
#         S - set of sets
#         A - set of elements to be covered
#         """
#         SetCover = set()
#         remainingElements = set(A)
#         usedSubsets = set()

#         def most_covering_subset(S, remainingElements, usedSubsets):
#             max_covered = -1
#             MostCoveringSubset = None
#             for idx, subset in enumerate(S):
#                 if idx not in usedSubsets:
#                     covered = len(remainingElements & subset)
#                     if covered > max_covered:
#                         max_covered = covered
#                         MostCoveringSubset = idx
#             return MostCoveringSubset

#         while remainingElements:
#             idx = most_covering_subset(S, remainingElements, usedSubsets)
#             if idx is not None:
#                 SetCover.add(idx)
#                 remainingElements -= S[idx]
#                 usedSubsets.add(idx)
                
#         return SetCover
    
    def setCover(S, A):
        """
        Find a subset of S that covers all elements in A using a greedy approach.
        S - set of sets
        A - set of elements to be covered
        """
        SetCover = set()
        remainingElements = set(A)
        usedSubsets = set()

        while remainingElements:
            max_covered = -1
            MostCoveringSubset = None
            for idx, subset in enumerate(S):
                if idx not in usedSubsets:
                    covered = len(remainingElements & subset)
                    if covered > max_covered:
                        max_covered = covered
                        MostCoveringSubset = idx
            if MostCoveringSubset is not None:
                SetCover.add(MostCoveringSubset)
                remainingElements -= S[MostCoveringSubset]
                usedSubsets.add(MostCoveringSubset)
            else:
                break  # No further subsets cover remaining elements

        return SetCover
    
    def OneRule(r):
        """
        r - decision rule (a row of pandas data)
        """
        A = DecisionTable[DecisionTable["class"] != r["class"]]
        A_indecies = set(A.index)

        S = []

        for feature in range(len(r)-1):
            S.append(set(A[A.iloc[:, feature] != r[feature]].index))

        indecies_needed = setCover(S, A_indecies)

        # Create a boolean mask
        mask_row = [value not in indecies_needed for value in range(len(r))]
        mask_row[-1] = False

        # Set values to NaN where the mask is True
        r[mask_row] = np.nan

        return r, mask_row
    

    for i in tqdm(range(len(DecisionTable))):
        _, mask_row = OneRule(DecisionTable.iloc[i])
        
        mask.append(mask_row)
        
    DecisionTable[np.array(mask)] = np.nan 
    
    DecisionTable = DecisionTable.dropna(axis=1, how='all')

    return DecisionTable


def NCover(S_plus):
    """ 
    input: S_plus - system of decision rules with length d (pandas DataFrame)
    output: Node cover of S_plus, set of columns that covers all rows
    """
    B = set()

    while not S_plus.iloc[:, :-1].isna().all().all():
        # Select the first row
        r1 = S_plus.iloc[0][:-1]
        # Create A_r1
        A_r1 = set(r1[r1.notna()].index)
        # Add A_r1 to B
        B.update(A_r1)
        
        # Drop rows that have common elements in their index set with A_r1
        rows_to_drop = []
        for index, row in S_plus.iterrows():
            row = row[:-1]
            A_r = set(row[row.notna()].index)
            if A_r1 & A_r:  # Check for common elements
                rows_to_drop.append(index)
        
        S_plus = S_plus.drop(rows_to_drop)

    return B


def NGreedy(S_plus):
    """
    input: S_plus - subset of S as defined in the paper. (pandas DataFrame)
    output: Node cover of S_plus, set of indecies of columns that covers all rows
    """
    S_max = SMax(S_plus)
    
    B = set()
    uncovered_rows = set(S_max.index)

    while uncovered_rows:
        # Find the column that covers the maximum number of uncovered rows
        max_cover = 0
        max_col = None

        for col in S_max.columns[:-1]: # Exclude the last column
            if col in B:
                continue  # Skip columns that have already been chosen
            cover = S_max.index[S_max[col].notna()].intersection(uncovered_rows)
            if len(cover) > max_cover:
                max_cover = len(cover)
                max_col = col

        if max_col is None:
            # No more columns to cover rows, break out
            break

        # Add the column to B and remove covered rows from consideration
        B.add(max_col)
        uncovered_rows -= set(S_max.index[S_max[max_col].notna()])

    return B
