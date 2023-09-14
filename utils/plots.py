import wandb
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


def plot_model_output_histogram(
    pred_logits, signal_vectors, mask, epoch, dataset_split
):
    """
    Plotting function which takes model predictions, mask, and ground truth values, and
    creates a side-by-side histogram of ground truth and predicted values.

    Plots are saved both as images and
    Args:
        pred_logits:    numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
        signal_vectors: numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
        mask:           binary mask of shape [batch_size, num_voxels, num_tokens]
        epoch:          training epoch
        dataset_split:  train, val, or test
    """
    gt_vals_list = []
    pred_vals_list = []
    for sample_idx in range(signal_vectors.shape[0]):
        for voxel_idx in range(signal_vectors.shape[1]):
            gt_vals_list += list(
                signal_vectors[sample_idx, voxel_idx][
                    mask[sample_idx, voxel_idx]
                ].flatten()
            )
            pred_vals_list += list(
                pred_logits[sample_idx, voxel_idx][
                    mask[sample_idx, voxel_idx]
                ].flatten()
            )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.hist(gt_vals_list, bins=20)
    ax1.set_xlabel("Ground Truth Signal Value", fontsize=16)
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax2.hist(pred_vals_list, bins=20)
    ax2.set_xlabel("Ground Truth Signal Value", fontsize=16)
    ax2.tick_params(axis="both", which="major", labelsize=16)
    ax2.set_xlabel("Predicted Signal Value", fontsize=16)
    fig.suptitle(
        "Ground Truth vs Predicted Signal Histograms\n({} split, epoch {})".format(
            dataset_split, epoch
        ),
        fontsize=18,
    )
    wandb.log({"plots/pred_gt_histogram_{}".format(dataset_split): wandb.Image(plt)})
    plt.close()


def plot_scatterplot(pred_logits, signal_vectors, mask, epoch, dataset_split):
    """
    Plotting function which takes model predictions, mask, and labels, and creates a scatter
    plot of ground truth vs predicted gene expression values.

    Plots are saved to wandb run.

    Args:
        pred_logits:    numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
        signal_vectors: numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
        mask:           binary mask of shape [batch_size, num_voxels, num_tokens]
        epoch:          training epoch
        dataset_split:  string containing train, val, or test
    """
    gt_vals_list = []
    pred_vals_list = []
    for sample_idx in range(signal_vectors.shape[0]):
        for voxel_idx in range(signal_vectors.shape[1]):
            gt_vals_list += list(
                signal_vectors[sample_idx, voxel_idx][
                    mask[sample_idx, voxel_idx]
                ].flatten()
            )
            pred_vals_list += list(
                pred_logits[sample_idx, voxel_idx][
                    mask[sample_idx, voxel_idx]
                ].flatten()
            )

    # Create Pandas DataFrame
    plot_df = pd.DataFrame(
        {"Ground Truth Signal": gt_vals_list, "Predicted Signal": pred_vals_list}
    )

    # Calculate slope of fitted line and R-squared
    lm = LinearRegression(
        fit_intercept=False
    )  # We expect no y-intercept, start at (0, 0)
    gt_vals_np = np.array(gt_vals_list).reshape(-1, 1)  # shape [num_vals, 1]
    pred_vals_np = np.array(pred_vals_list).reshape(-1, 1)  # shape [num_vals, 1]
    lm.fit(X=gt_vals_np, y=pred_vals_np)

    slope = lm.coef_[0][0].item()
    r2_sklearn_score = lm.score(X=gt_vals_np, y=pred_vals_np)
    if r2_sklearn_score < 0.0:
        r2_sklearn_score = 0.0

    pearson = pearsonr(x=np.squeeze(gt_vals_np), y=np.squeeze(pred_vals_np))
    p = pearson.statistic
    if p < 0.0:
        p = 0.0

    # Seaborn plot
    sns.scatterplot(data=plot_df, x="Ground Truth Signal", y="Predicted Signal", alpha=0.5, s=5)
    plt.title(
        "Ground Truth vs Predicted Signal\nR2 Value {:.5f}, Pearson r {:.5f}, Fitted Line Slope {:.5f}\n({} Split, Epoch {})".format(
            r2_sklearn_score, p, slope, dataset_split, epoch
        )
    )
    wandb.log({"plots/scatterplot_{}".format(dataset_split): wandb.Image(plt)})
    plt.close()


def plot_masked_pred_trends_one_sample(
    pred_logits: np.array,
    signal_vectors: np.array,
    mask: np.array,
    sample_idx: int,
    node_idxs: np.array,
    dataset_split: str,
    epoch: int,
):
    """
    Function to plot timeseries of model predictions as continuation of input data compared to
    ground truth. A grid of sub-lineplots will be logged to wandb, with len(sample_idxs) rows
    and len(node_idxs) columns
    Args:
        pred_logits:    numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
        signal_vectors: numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
        mask:           binary mask of shape [batch_size, num_voxels, num_tokens]
        sample_idx:     index of sample to plot; one per figure
        node_idxs:      indices of voxels to plot; affects how many columns in plot grid there will be
        dataset_split:  train, val, or test
        epoch:          training epoch
    Returns:
    """
    fig, axes = plt.subplots(nrows=len(node_idxs), ncols=1, sharex=True, sharey=True)
    fig.set_figwidth(25)
    fig.set_figheight(3 * len(node_idxs))

    batch_size, num_voxels, num_tokens, time_patch_preds = pred_logits.shape

    # --- Plot Figure ---#
    for row_idx, node_idx in enumerate(node_idxs):
        ax = axes[row_idx]

        input_data_vals = []
        input_data_timepoints = []
        for token_idx in range(signal_vectors.shape[2]):
            input_data_vals += signal_vectors[sample_idx, node_idx, token_idx].tolist()
            start_timepoint = time_patch_preds * token_idx
            end_timepoint = start_timepoint + time_patch_preds
            input_data_timepoints += list(range(start_timepoint, end_timepoint))

            if mask[sample_idx, node_idx, token_idx] == 1:
                model_pred_vals = pred_logits[sample_idx, node_idx, token_idx].tolist()
                model_pred_timepoints = list(range(start_timepoint, end_timepoint))
                ax.plot(
                    model_pred_timepoints,
                    model_pred_vals,
                    marker=".",
                    markersize=3,
                    label="Masked Predictions",
                    color="orange",
                )

        ax.plot(
            input_data_timepoints,
            input_data_vals,
            marker=".",
            markersize=3,
            label="Input Data",
            color="green",
        )
        ax.set_title("Sample {}, Voxel {}".format(sample_idx, node_idx))
        ax.axhline(y=0.0, color="gray", linestyle="--", markersize=2)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.9))
    plt.tight_layout(rect=[0.03, 0.03, 0.95, 0.95])
    fig.supxlabel("Timepoint")
    fig.supylabel("Prediction Value")
    plt.suptitle(
        "Ground Truth Signal vs Masked Prediction\n({} Split, Epoch {})".format(
            dataset_split, epoch
        )
    )
    wandb.log(
        {
            "trends/pred_trend_{}_sample{}".format(
                dataset_split, sample_idx
            ): wandb.Image(fig)
        }
    )
    plt.close()


def plot_future_timepoint_trends(
    preds: np.array,
    ground_truth: np.array,
    input_data: np.array,
    sample_idxs: np.array,
    node_idxs: np.array,
    dataset_split: str,
    epoch: int,
):
    """
    Function to plot timeseries of model predictions as continuation of input data compared to
    ground truth. A grid of sub-lineplots will be logged to wandb, with len(sample_idxs) rows
    and len(node_idxs) columns
    Args:
        preds: [num_samples, num_nodes, pred_timepoints]
        ground_truth: [num_samples, num_nodes, pred_timepoints]
        input_data: [num_samples, num_nodes, timepoints_per_node]
        sample_idxs: indices of samples to plot; affects how many rows in plot grid there will be
        node_idxs: indices of voxels to plot; affects how many columns in plot grid there will be
        dataset_split: train, val, or test
        epoch: training epoch
    Returns:
    """
    fig, axes = plt.subplots(
        nrows=len(sample_idxs), ncols=len(node_idxs), sharex=True, sharey=True
    )
    fig.set_figwidth(3.5 * len(node_idxs))
    fig.set_figheight(3 * len(sample_idxs))

    # --- Plot Figure ---#
    for row_idx, sample_idx in enumerate(sample_idxs):
        for col_idx, node_idx in enumerate(node_idxs):
            ax = axes[row_idx][col_idx]

            # Get values
            input_data_vals = input_data[sample_idx, node_idx, :].tolist()
            input_timepoints = list(range(1, len(input_data_vals) + 1))
            model_preds = [input_data_vals[-1]]
            model_preds += preds[sample_idx, node_idx, :].tolist()
            pred_timepoints = list(
                range(len(input_data_vals), len(input_data_vals) + len(model_preds))
            )
            ground_truth_vals = [input_data_vals[-1]]
            ground_truth_vals += ground_truth[sample_idx, node_idx, :].tolist()
            gt_timepoints = list(
                range(
                    len(input_data_vals), len(input_data_vals) + len(ground_truth_vals)
                )
            )

            # Plot
            ax.plot(input_timepoints, input_data_vals, marker="o", label="Input Data")
            ax.plot(pred_timepoints, model_preds, marker="o", label="Model Predictions")
            ax.plot(gt_timepoints, ground_truth_vals, marker="o", label="Ground Truth")
            ax.set_title("Sample {}, Voxel {}".format(sample_idx, node_idx))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.9))
    plt.tight_layout(rect=[0.03, 0.03, 0.95, 0.95])
    fig.supxlabel("Timepoint")
    fig.supylabel("Prediction Value")
    plt.suptitle(
        "Ground Truth vs Predicted Signal\n({} Split, Epoch {})".format(
            dataset_split, epoch
        )
    )
    wandb.log({"plots/pred_trend_{}".format(dataset_split): wandb.Image(fig)})
    plt.close()


def plot_cls_token_2d_umap(cls_tokens, age_labels, epoch, dataset_split):
    """
    This function plots a UMAP embedding figure with model predictions and input data points
    concatenated, so both appear on the UMAP embedding together. This plot shows if model
    predictions/reconstruction of input data is close to the original input data.

    Args:
        cls_tokens:   numpy array of shape [num_samples, hidden_size]
        age_labels:      numpy array of shape [num_cells, num_genes]
        epoch:          training epoch
        dataset_split:  string containing train, val, or test
    """
    # Apply UMAP
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(cls_tokens)

    age_labels = [float(age) for age in age_labels]
    # missing age labels are given as -1. Replace here back with nan values
    age_labels = [val if val != -1 else np.nan for val in age_labels]
    colormap = mcol.LinearSegmentedColormap.from_list("Blue-Red-Colormap", ["b", "r"])
    colormap.set_bad("gray")

    # Plot
    plt.rcParams.update({"font.size": 18})
    sns.set(style="white", context="notebook", rc={"figure.figsize": (8, 6)})
    sc = plt.scatter(
        embedding[:, 0].tolist(),
        embedding[:, 1].tolist(),
        alpha=0.5,
        c=age_labels,
        plotnonfinite=True,
        cmap=colormap,
    )
    plt.title(
        "CLS Token Embedding UMAP Colored\nBy Age (Missing Ages Grayed), Epoch {}".format(
            epoch
        ),
        fontsize=16,
    )
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.colorbar(sc)
    wandb.log({"umap/cls_tokens_{}".format(dataset_split): wandb.Image(plt)})
    plt.close()
