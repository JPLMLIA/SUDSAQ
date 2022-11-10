#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:15:44 2022

@author: marchett
"""
import merge_model_output as merge
import summarize_residuals as residuals 
import produce_correlations as correlations


def main(sub_dir):
    
    merge.main('all', sub_dir)
    
    explanations.main()
    
    residuals.main(sub_dir)
    
    correlations.main(sub_dir, 'all')
    
    
    
    