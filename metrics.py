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
        specificity = tn / (tn + fp + self.smooth)
        sensitivity = tp/(tp + fn + self.smooth)
        precision =  tp/(tp + fp + self.smooth)
        F2_score = (5*tp + self.smooth)/(5*tp + 4*fn +  fp + self.smooth)
        DSC = (2*tp + self.smooth)/(2*tp + fn + fp + self.smooth)
        F1_score = (2 * precision * sensitivity + self.smooth) / (precision + sensitivity + self.smooth)
        return specificity, sensitivity, precision, F1_score, F2_score, DSC
