def compute_f05_per_entity(true_matches: set[str], pred_matches: set[str]) -> float:
    """Compute F_0.5 score for a single Source 1 entity.
    
    Formula:
      F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
      
    Rules:
      - If true_matches is empty:
          scores 1.0 if pred_matches is also empty, 0.0 otherwise.
      - If true_matches is not empty:
          scores 0.0 if pred_matches is empty or no true positives.
          otherwise standard F_0.5 formula.
    """
    if len(true_matches) == 0:
        return 1.0 if len(pred_matches) == 0 else 0.0
    
    if len(pred_matches) == 0:
        return 0.0
        
    tp = len(true_matches & pred_matches)
    if tp == 0:
        return 0.0
        
    precision = tp / len(pred_matches)
    recall = tp / len(true_matches)
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom

def compute_macro_f05(ground_truth_dict: dict[str, set[str]], predictions_dict: dict[str, set[str]]) -> float:
    """Compute Macro F_0.5 across all Source 1 entities in ground_truth_dict."""
    if not ground_truth_dict:
        return 0.0
        
    total_score = 0.0
    for s1_id, true_set in ground_truth_dict.items():
        pred_set = predictions_dict.get(s1_id, set())
        total_score += compute_f05_per_entity(true_set, pred_set)
        
    return total_score / len(ground_truth_dict)
