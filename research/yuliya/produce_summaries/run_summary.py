#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:15:44 2022

@author: marchett
"""
import merge_model_output as merge
import merge_model_yearly as merge_years
import summarize_performance as evaluate 
import produce_correlations as correlations
import make_ml_data as extract
import summarize_explanations as explain


def main(sub_dir, months = 'all'):
    
    #--merge all the cross val years
    merge.main(months, sub_dir)
    
    #--make and save ml-ready data
    extract.main(sub_dir, months)
    
    #--extract all importances and plot
    explain.main(sub_dir, a = 20)
    
    #--performance and residual analysis
    evaluate.main(sub_dir)
    
    #--produce raw feature correlations (toar locs only)
    correlations.main(sub_dir, months)
    
    #--merge all the cross val years
    merge_years.main(sub_dir)
    
    
    
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_dir', type=str, default = '/bias/local/8hr_median/v1/')
    parser.add_argument('--months', default = 'all', nargs = '*', type=str)
    #parser.add_argument('--parameter', type=str, default=None)

    args = parser.parse_args()
    main(**vars(args)) 
    
    
    