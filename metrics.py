#-----------------------------------------------------------------------#
#                          Library imports                              #
#-----------------------------------------------------------------------#
import torch

#-----------------------------------------------------------------------#
#                         performance_metrics                           #
#                    Calculate performance metrics                      #
#-----------------------------------------------------------------------#
# Reference:                                                            #
# Salehi et al. “Tversky Loss Function for Image Segmentation           #
# Using 3D Fully Convolutional Deep Networks.”                          #
# ArXiv abs/1706.05721 (2017)                                           #
#-----------------------------------------------------------------------#
# Returns: specificity, sensitivity, precision, F1_score, F2_score,     #
# and Dice similarity coefficient per batch as a numpy array            #
#-----------------------------------------------------------------------#
# smooth: a very small number added to the denomiators to               #
#         prevent the division by zero                                  #
# tp:     number of true positives                                      #
# fp:     number of false positives                                     #
# tn:     number of true negatives                                      #
# fn:     number of false negatives                                     #
# DSC:    Dice_Similarity_Coefficient                                   #
#-----------------------------------------------------------------------#
class performance_metrics():
    def __init__(self, smooth = 1e-10):
        super().__init__()
        self.smooth = smooth

    def __call__(self, y_pred, y_true):
        # Flatten both prediction and GT tensors
        y_pred_flat = torch.flatten(y_pred)
        y_true_flat = torch.flatten(y_true)
        # calculate the parameters
        tp = (y_pred_flat * y_true_flat).sum()
        tn = ((1 - y_pred_flat) * (1- y_true_flat)).sum()
        fp = (y_pred_flat * (1 - y_true_flat)).sum()
        fn = ((1 - y_pred_flat) * y_true_flat).sum()
        # continue the calculation in numpy
        tp = tp.cpu().detach().numpy()
        fp = fp.cpu().detach().numpy()
        tn = tn.cpu().detach().numpy()
        fn = fn.cpu().detach().numpy()
        #calculate the metrics
        if tn == 0.0:
            specificity = 0.0
        else:
            specificity = tn / (tn + fp)
        if tp == 0:
            sensitivity = 0.0
            precision = 0.0
            F2_score = 0.0
            DSC = 0.0
        else:
            sensitivity = tp/(tp + fn)
            precision =  tp/(tp + fp)
            F2_score = (5*tp)/(5*tp + 4*fn +  fp)
            DSC = (2*tp)/(2*tp + fn + fp)
        if precision==0 or sensitivity==0:
            F1_score = 0.0 
        else:
            F1_score = (2 * precision * sensitivity) / (precision + sensitivity)
        return specificity, sensitivity, precision, F1_score, F2_score, DSC
