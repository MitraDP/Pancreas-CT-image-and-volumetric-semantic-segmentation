import torch

class performance_metrics():
    #calculates specificity, sensitivity, precision, F1_score, F2_score, and DSC per batch
    def __init__(self, smooth = 0.00000001):
        super().__init__()
        self.smooth = smooth

    def __call__(self, y_pred, y_true):
        # Flatten both prediction and GT tensors
        y_pred_flat = torch.flatten(y_pred)
        y_true_flat = torch.flatten(y_true)
        
        tp = (y_pred_flat * y_true_flat).sum()
        tn = ((1 - y_pred_flat) * (1- y_true_flat)).sum()
        fp = (y_pred_flat * (1 - y_true_flat)).sum()
        fn = ((1 - y_pred_flat) * y_true_flat).sum()
        #calculate the metrics
        specificity = tn / (tn + fp + self.smooth)
        sensitivity = tp / (tp + fn + self.smooth)
        precision =  tp / (tp + fp + self.smooth)
        F1_score = (2 * precision * sensitivity + self.smooth) / (precision + sensitivity  + self.smooth)
        #ref: Salehi arXiv:1706.05721v1
        F2_score = (5*tp + self.smooth)/(5*tp + 4*fn +  fp + self.smooth)
        #Dice_Similarity_Coefficient
        DSC = (2*tp + self.smooth)/(2*tp + fn + fp + self.smooth)
        
        return specificity, sensitivity, precision, F1_score, F2_score, DSC

