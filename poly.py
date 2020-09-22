# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:21:21 2020

@author: Yi-Tun Lin
"""

import numpy as np
from itertools import combinations_with_replacement


def get_polynomial_terms(num_of_var, highest_order, root):
    
    if highest_order == 1:
        all_set = np.eye(num_of_var)
        final_set = [tuple(all_set[i, :]) for i in range(num_of_var)]
        
        return final_set
    
    final_set = set()   # save the set of polynomial terms
    index_of_variables = [i for i in range(num_of_var)]
    
    for order in range(1,highest_order+1):  # consider all higher order terms from order 1, excluding the constant term
        
        # Each list member: one composition of the term of the assigned order, in terms of variable indices      
        curr_polynomial_terms = list(combinations_with_replacement(index_of_variables,order))   
        
        for t in range(len(curr_polynomial_terms)):
            curr_term = curr_polynomial_terms[t]
            mapped_term = np.zeros(num_of_var)       # save the index value of each variables
            
            for var in curr_term:
                if root:
                    mapped_term[var] += 1./order
                else:
                    mapped_term[var] += 1.
                    
            final_set.add(tuple(mapped_term))
        
    return list(sorted(final_set))