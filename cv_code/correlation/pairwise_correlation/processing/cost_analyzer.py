"""
Cost matrix analysis and thresholding utilities.
"""

import numpy as np


class CostAnalyzer:
    """Class for analyzing cost matrices and determining thresholds."""
    
    def __init__(self, config):
        """
        Initialize cost analyzer with configuration.
        
        Args:
            config: CorrelationConfig object
        """
        self.config = config
    
    def calculate_global_cost_threshold(self, all_match_costs):
        """
        Calculate global cost threshold based on percentile.
        
        Args:
            all_match_costs: List of all match costs
            
        Returns:
            Global cost threshold value
        """
        if all_match_costs:
            global_cost_threshold = np.percentile(all_match_costs, self.config.COST_PERCENTILE_THRESHOLD)
            return global_cost_threshold
        else:
            return float('inf')
    
    def filter_matches_by_threshold(self, matches, cost_matrix, threshold):
        """
        Filter matches based on cost threshold.
        
        Args:
            matches: List of (i, j) match pairs
            cost_matrix: Cost matrix
            threshold: Cost threshold value
            
        Returns:
            Filtered list of matches
        """
        return [(i, j) for i, j in matches if cost_matrix[i, j] <= threshold]
    
    def collect_match_costs(self, matches, cost_matrix):
        """
        Collect costs for all matches.
        
        Args:
            matches: List of (i, j) match pairs
            cost_matrix: Cost matrix
            
        Returns:
            List of match costs
        """
        return [cost_matrix[i, j] for i, j in matches]
    
    def find_min_match_cost(self, matches, cost_matrix):
        """
        Find the minimum match cost for a frame.
        
        Args:
            matches: List of (i, j) match pairs
            cost_matrix: Cost matrix
            
        Returns:
            Minimum match cost or None if no matches
        """
        if matches:
            return min(cost_matrix[i, j] for i, j in matches)
        return None
