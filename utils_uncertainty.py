#!/usr/bin/env python

"""
Description: This file contains the implementation of the following uncertainty metrics:
- Expected Calibration Error (ECE)
- Entropy
- Area under the Receiving Operator Characteristic (ROC) curve (AUROC)
- Kullback-Leibler Divergence (KLD)


Source:
https://github.com/prs-eth/FILM-Ensemble/blob/main/utils_uncertainty.py
Turkoglu, M. O., Becker, A., Gündüz, H. A., Rezaei, M., Bischl, B., Daudt, R. C.,
D'Aronco, S., Wegner, J. D., & Schindler, K. (2022). FiLM-Ensemble: Probabilistic
Deep Learning via Feature-wise Linear Modulation. In Advances in Neural Information Processing Systems.
"""

### IMPORTS ##
# Built-in imports
from typing import Union
import pathlib

# Lib imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

# Custom imports
import const


### CLASS DEFINITIONS ###
class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = self.bin_boundaries[:-1]
        self.bin_uppers = self.bin_boundaries[1:]

    def forward(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            plot: bool = False,
            file_name: Union[str, pathlib.Path] = None,
            threshold: float = None
    ):
        """
        Calculate the Expected Calibration Error (ECE) and plot the reliability diagram if needed

        Parameters
        ----------
        logits: torch.Tensor
            The logits of the model
        labels: torch.Tensor
            The labels of the data
        plot: bool, optional
            Whether to plot the reliability diagram
            Default: False
        file_name: str, optional
            The name of the file to save the plot to
            Must be provided if plot is True
            Default: None

        Returns
        -------
        ece: torch.Tensor
            The Expected Calibration Error
        accs: List
            List of accuracies per bin
        confs: List
            List of confidences per bin
        avg_acc: float
            The samplewise average accuracy
        avg_conf: float
            The samplewise average confidence
        """

        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        avg_conf = confidences.float().mean().item()
        avg_acc = accuracies.float().mean().item()

        ece = torch.zeros(1, device=logits.device)
        accs = list()
        accs_all = list()
        confs = list()
        confs_all = list()
        counts = list()
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):

            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            counts.append(in_bin.sum().item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                accs.append(accuracy_in_bin)
                accs_all.append(accuracy_in_bin.item())
                confs.append(avg_confidence_in_bin)
                confs_all.append(avg_confidence_in_bin.item())
            else:
                accs_all.append(0)
                confs_all.append(0)

        counts = np.array(counts)

        if plot:
            self.plot_reliability_diagram(accs_all, confs_all, counts, avg_conf, avg_acc, file_name)

        return ece, accs, confs, avg_acc, avg_conf

    def plot_reliability_diagram(
            self,
            accs: list,
            confs: list,
            counts: np.ndarray,
            avg_conf: float,
            avg_acc: float,
            file_name: str
        ) -> None:
        """
        Plots the reliability diagram

        Parameters
        ----------
        accs: List
            List of accuracies
        confs: List
            List of confidences
        counts: np.ndarray
            Number of elements per bin
        avg_conf: float
            Average confidence across all samples
        avg_acc: float
            Average accuracy across all samples
        file_name: pathlib.Path
            The name of the file to save the plot to

        Added by M. Halbheer and D. Mühlematter.
        """

        # Convert to numpy arrays
        accs, confs = np.array(accs), np.array(confs)

        # Set the font sizes
        title_fontsize = 24
        label_fontsize = 20
        tick_fontsize = 18
        legend_fontsize = 18

        # Create the figure
        fig, ax0 = plt.subplots(1, 1, sharex="col", figsize=(10, 8))

        # Calculate the bin size and the positions of the bars
        bin_size = 1 / len(counts)
        positions = self.bin_boundaries.numpy()[:-1] + bin_size / 2

        # Define the width of the bars
        width = bin_size

        # Plot the accuracy bars
        ax0.bar(positions, width=width, edgecolor='black', height=accs,
                color='lightgray', label='Accuracy', linewidth=2)
        # Plot the gap to a well calibrated model
        gap_positions = positions[counts > 0]
        gap_accs = accs[counts > 0]
        ax0.bar(gap_positions, height=np.abs(gap_accs - gap_positions),
                bottom=np.minimum(gap_accs, gap_positions), width=width,
                edgecolor='darkred', color='red', alpha=0.4, linewidth=1, label="Gap")

        # Plot the diagonal (perfectly calibrated model)
        ax0.set_aspect("equal", adjustable='box')
        ax0.plot([0, 1], [0, 1], linestyle="--", color="gray")

        # Set the axis limits
        ax0.set_xlim(0, 1)
        ax0.set_ylim(0, 1)

        # Format the axes
        ax0.set_ylabel("Accuracy", fontsize=label_fontsize)
        ax0.set_yticks(np.linspace(0, 1, 11))
        ax0.tick_params(axis='y', which='major', labelsize=tick_fontsize)

        # Enable grid
        ax0.grid()

        # Set the title
        ax0.set_title("Reliability Diagram", fontsize=title_fontsize)

        # Calculate the percentage of samples in each bin
        percentage_counts = counts / np.sum(counts)

        # Conversion functions for the secondary y-axis
        def percentage_to_norm(x):
            return x / 100

        def norm_to_percentage(x):
            return x * 100

        # Secondary y-axis for the percentage of samples in each bin
        secax = ax0.secondary_yaxis('right', functions=(norm_to_percentage, percentage_to_norm))
        secax.set_yticks(np.linspace(0, 100, 11))
        secax.set_ylabel('% of samples', fontsize=label_fontsize)
        secax.tick_params(axis='y', which='major', labelsize=tick_fontsize)

        # Plot the histogram of the confidences
        ax0.bar(positions, 0, bottom=percentage_counts, width=0.9*width,
               edgecolor="darkslategray", color="darkslategray", alpha=1.0, linewidth=3,
               label="% of samples in bin")

        # Format the axes
        ax0.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # Add vertical lines for the average accuracy and confidence
        ax0.axvline(x=avg_conf, color='darkgoldenrod', linestyle='--', label=f'Avg. Confidence {avg_conf * 100:.1f}%')
        ax0.axvline(x=avg_acc, color='darkgoldenrod', ls='solid', label=f'Avg. Accuracy {avg_acc * 100:.1f}%')

        # Add a legend
        ax0.legend(fontsize=legend_fontsize)

        # Format the entire plot layout
        plt.tight_layout()
        plt.subplots_adjust(hspace=-0.02)

        # Create the plot directory if it does not exist
        if not const.PLOT_DIR.exists():
            const.PLOT_DIR.mkdir(parents=True)

        # Set absolute plot size
        fig.set_size_inches(8, 8)

        # Save the plot
        png_name = f"{file_name}.png"
        pdf_name = f"{file_name}.pdf"
        plt.savefig(const.PLOT_DIR.joinpath(pdf_name), bbox_inches='tight')
        plt.savefig(const.PLOT_DIR.joinpath(png_name), bbox_inches='tight')
        plt.close()


class Entropy(nn.Module):
    """
    Calculates the entropy of the distribution and means over batch dimension
    """
    def __init__(self, softmax=True):
        super(Entropy, self).__init__()
        self.softmax = softmax

    def forward(self, logits):
        if self.softmax:
            logits = F.softmax(logits, dim=1)

        entropy = Categorical(logits=logits).entropy().mean()

        return entropy


class AUROC(nn.Module):
    """
    Calculates the AUROC
    (Area under the Receiving Operator Characteristic (ROC) curve)
    for out-of-distribution (OOD) detection
    """

    def __init__(self, softmax=True, equal_size=True):
        super(AUROC, self).__init__()
        self.softmax = softmax
        self.equal_size = equal_size

    def forward(self, id_logits, ood_logits):
        if self.softmax:
            id_logits = F.softmax(id_logits, dim=1)
            ood_logits = F.softmax(ood_logits, dim=1)

        if self.equal_size:
            min_size = np.min((id_logits.shape[0], ood_logits.shape[0]))
            id_logits = id_logits[:min_size,...]
            ood_logits = ood_logits[:min_size,...]

        id_conf_scores, _ = torch.max(id_logits, dim=1, keepdim=False)
        ood_conf_scores, _ = torch.max(ood_logits, dim=1, keepdim=False)

        id_targets = torch.ones_like(id_conf_scores)
        od_targets = torch.zeros_like(ood_conf_scores)

        y_pred = torch.cat((id_conf_scores, ood_conf_scores), dim=0).cpu().data.numpy()
        y_target = torch.cat((id_targets, od_targets), dim=0).cpu().data.numpy()

        score = roc_auc_score(y_target, y_pred)

        return score


class function_space_analysis(nn.Module):
    """
    Calculates the disagreement and distance between two models.
    As distance measure the Jensen-Shannon divergence is used, potentially 
    the square root is taken.

    Paramters:
    ----------
    w_softmax: bool
        apply softmax to the input logits
    square_root: bool
        take the square root of the Jensen-Shannon divergence

    Returns:
    --------
    disagreement: float
        disagreement between the two models between 0 and 1
    distance: float
        distance between the two models
    """

    def __init__(self, w_softmax=True, square_root=True):
        super(function_space_analysis, self).__init__()
        self.w_softmax = w_softmax
        self.square_root = square_root
        self.lossFn = nn.KLDivLoss(reduction='sum', log_target=True)

    def forward(self, logits_1, logits_2):

        # check if input is tensor
        if not torch.is_tensor(logits_1):
            logits_1 = torch.tensor(logits_1)
            logits_2 = torch.tensor(logits_2)

        # apply softmax if necessary
        if self.w_softmax:
            logits_1 = F.log_softmax(logits_1, dim=1)
            logits_2 = F.log_softmax(logits_2, dim=1)

        # Jensen-Shannon divergence / symmetric Kullback-Leibler divergence
        distance = 0.5 *(self.lossFn(logits_1,logits_2) + self.lossFn(logits_2,logits_1))

        # square root if necessary
        if self.square_root:
            distance = torch.sqrt(distance)

        # get predictions
        pred_1 = logits_1.max(1, keepdim=True)[1]
        pred_2 = logits_2.max(1, keepdim=True)[1]

        # calculate disagreement
        disagreement = torch.sum(pred_1 != pred_2) # /pred_2.shape[0]

        return disagreement, distance


def precision(predictions, labels):
    """
    Calculate the precision of the model

    Parameters
    ----------
    predictions: torch.Tensor
        The predictions of the model
    labels: torch.Tensor
        The labels of the data

    Returns
    -------
    prec: float
        The precision of the model
    """

    n = predictions.shape[0]

    pred_classes = predictions.sum(dim=0)
    true_positives = (predictions * labels).sum(dim=0)

    prec = (true_positives / pred_classes).sum(dim=1)

    return prec
