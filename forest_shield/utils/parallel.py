"""
Utilities for parallel processing.
"""

import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from typing import Optional, Union, List, Dict, Any, Tuple, Callable
import time


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    n_jobs = min(n_jobs, n_estimators)

    # Compute the number of jobs
    if n_jobs < 0:
        n_jobs = max(multiprocessing.cpu_count() + 1 + n_jobs, 1)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs, dtype=int)
    n_estimators_per_job[: n_estimators % n_jobs] += 1
    starts = np.cumsum(np.hstack([[0], n_estimators_per_job[:-1]]))
    ends = np.cumsum(n_estimators_per_job)

    return n_jobs, starts, ends


def parallel_build_trees(
    tree_builder, n_estimators, X, y, sample_weight, seeds, n_jobs, verbose=0
):
    """Build trees in parallel.

    Parameters
    ----------
    tree_builder : callable
        Function to build a single tree.
    n_estimators : int
        Number of trees to build.
    X : array-like of shape (n_samples, n_features)
        The training input samples.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    seeds : list of int
        Random seeds for each tree.
    n_jobs : int
        Number of jobs to run in parallel.
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    Returns
    -------
    trees : list
        List of fitted trees.
    """
    n_jobs, starts, ends = _partition_estimators(n_estimators, n_jobs)

    if verbose > 0:
        print(f"=== Starting Random Forest training ===")
        print(f"Training {n_estimators} trees using {n_jobs} parallel jobs")
        print(f"Data shape: {X.shape}, Classes: {np.unique(y).shape[0]}")

        # Format the trees per job output properly
        trees_per_job = [ends[i] - starts[i] for i in range(n_jobs)]
        print(f"Trees per job: {trees_per_job}")

        print(f"Starting parallel training...")
        start_time = time.time()

    all_trees = []

    # Parallel loop
    all_trees = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(parallel_helper)(
            tree_builder, X, y, sample_weight, seeds[starts[i] : ends[i]], verbose
        )
        for i in range(n_jobs)
    )

    # Reduce
    trees = [tree for job_trees in all_trees for tree in job_trees]

    if verbose > 0:
        total_time = time.time() - start_time
        print(f"=== Random Forest training completed ===")
        print(f"Built {len(trees)} trees in {total_time:.4f} seconds")
        print(f"Average time per tree: {total_time/max(1, len(trees)):.4f} seconds")

        # Print some statistics about the trees if available
        try:
            depths = [tree.get_depth() for tree in trees if hasattr(tree, "get_depth")]
            if depths:
                print(
                    f"Tree depth stats: min={min(depths)}, max={max(depths)}, avg={sum(depths)/len(depths):.1f}"
                )

            nodes = [tree.tree_.node_count for tree in trees if hasattr(tree, "tree_")]
            if nodes:
                print(
                    f"Tree node stats: min={min(nodes)}, max={max(nodes)}, avg={sum(nodes)/len(nodes):.1f}"
                )
        except Exception as e:
            print(f"Note: Could not compute all tree statistics: {str(e)}")

    return trees


def parallel_helper(tree_builder, X, y, sample_weight, seeds, verbose=0):
    """Helper to parallelize tree building."""
    trees = []
    n_trees = len(seeds)

    if verbose > 0:
        job_start_time = time.time()
        print(f"Job started: Building {n_trees} trees in a single job...")

    for i, seed in enumerate(seeds):
        if verbose > 0:
            print(f"  Training estimator {i+1}/{n_trees} with seed {seed}")

        start_time = time.time()
        try:
            tree = tree_builder(X, y, sample_weight, seed)
            train_time = time.time() - start_time

            if verbose > 0:
                print(
                    f"  Estimator {i+1}/{n_trees} completed in {train_time:.4f} seconds"
                )

                # Print some basic tree info if available
                try:
                    if hasattr(tree, "tree_") and hasattr(tree.tree_, "node_count"):
                        if hasattr(tree, "get_depth"):
                            print(
                                f"  Tree stats: {tree.tree_.node_count} nodes, max depth {tree.get_depth()}"
                            )
                        else:
                            print(f"  Tree stats: {tree.tree_.node_count} nodes")
                except Exception as e:
                    print(f"  Note: Could not get tree statistics: {str(e)}")

            trees.append(tree)
        except Exception as e:
            print(f"  Error training estimator {i+1}/{n_trees}: {str(e)}")

    if verbose > 0:
        job_time = time.time() - job_start_time
        print(
            f"Job completed: Built {len(trees)} trees successfully in {job_time:.4f} seconds"
        )

    return trees


def parallel_predict(predict_function, trees, X, n_jobs, verbose=0):
    """Make predictions in parallel.

    Parameters
    ----------
    predict_function : callable
        Function to make predictions with a single tree.
    trees : list
        List of fitted trees.
    X : array-like of shape (n_samples, n_features)
        The input samples.
    n_jobs : int
        Number of jobs to run in parallel.
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    Returns
    -------
    y_pred : ndarray
        The predicted values.
    """
    n_estimators = len(trees)

    if verbose > 0:
        print(f"\n=== Starting prediction with {n_estimators} trees ===")
        print(f"Input data shape: {X.shape}")
        start_time = time.time()

    if n_estimators == 0:
        # Handle the case when there are no trees
        print("Warning: No trees available for prediction. Returning zeros.")
        return np.zeros((X.shape[0],))

    # For small number of trees, it's more efficient to predict sequentially
    if n_estimators <= 10:
        if verbose > 0:
            print(f"Predicting with {n_estimators} trees sequentially...")

        # Get the first tree's prediction to determine the shape
        if verbose > 0:
            print(f"  Processing tree 1/{n_estimators}...")
            tree_start = time.time()

        first_pred = predict_function(trees[0], X)

        if verbose > 0:
            tree_time = time.time() - tree_start
            print(f"  Tree 1 prediction completed in {tree_time:.4f} seconds")

        # Initialize predictions with the correct shape for multi-class
        if len(first_pred.shape) > 1:
            n_classes = first_pred.shape[1]
            predictions = np.zeros((X.shape[0], n_classes))
            if verbose > 0:
                print(f"  Multi-class prediction with {n_classes} classes")
        else:
            predictions = np.zeros((X.shape[0],))
            if verbose > 0:
                print(f"  Single-class prediction")

        # Add predictions from all trees
        predictions += first_pred
        for i, tree in enumerate(trees[1:], 1):
            if verbose > 0:
                print(f"  Processing tree {i+1}/{n_estimators}...")
                tree_start = time.time()

            predictions += predict_function(tree, X)

            if verbose > 0:
                tree_time = time.time() - tree_start
                print(f"  Tree {i+1} prediction completed in {tree_time:.4f} seconds")

        if verbose > 0:
            total_time = time.time() - start_time
            print(
                f"Finished processing all {n_estimators} trees in {total_time:.4f} seconds"
            )
            print(f"Average time per tree: {total_time/n_estimators:.4f} seconds")

        return predictions / n_estimators

    # For larger number of trees, use parallel processing
    n_jobs, starts, ends = _partition_estimators(n_estimators, n_jobs)

    if verbose > 0:
        print(f"Predicting with {n_estimators} trees using {n_jobs} parallel jobs...")
        print(f"Trees per job: {[ends[i] - starts[i] for i in range(n_jobs)]}")

    # Parallel loop
    all_predictions = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(parallel_predict_helper)(
            predict_function, trees[starts[i] : ends[i]], X, verbose
        )
        for i in range(n_jobs)
    )

    # Check if all predictions have the same shape
    shapes = [pred.shape for pred in all_predictions]
    if not all(len(shape) == len(shapes[0]) for shape in shapes):
        # Handle different shapes by converting to the most common shape
        if verbose > 0:
            print(
                "Warning: Predictions have different shapes. Converting to consistent format."
            )

        # Find the most common shape length
        shape_lengths = [len(shape) for shape in shapes]
        most_common_length = max(set(shape_lengths), key=shape_lengths.count)

        # Convert all predictions to the most common shape
        for i, pred in enumerate(all_predictions):
            if len(pred.shape) != most_common_length:
                if most_common_length == 2:
                    # Convert 1D to 2D
                    n_samples = pred.shape[0]
                    n_classes = max(
                        2, max(shape[1] for shape in shapes if len(shape) > 1)
                    )
                    new_pred = np.zeros((n_samples, n_classes))
                    for j, p in enumerate(pred):
                        new_pred[j, int(p)] = 1.0
                    all_predictions[i] = new_pred
                else:
                    # Convert 2D to 1D
                    all_predictions[i] = np.argmax(pred, axis=1)

    # Sum the predictions
    predictions = sum(all_predictions) / len(all_predictions)

    if verbose > 0:
        total_time = time.time() - start_time
        print(
            f"Finished predictions from all {n_jobs} jobs in {total_time:.4f} seconds"
        )
        print(f"Average time per tree: {total_time/n_estimators:.4f} seconds")
        print(f"=== Prediction complete ===")

    return predictions


def parallel_predict_helper(predict_function, trees, X, verbose=0):
    """Helper to parallelize prediction."""
    n_samples = X.shape[0]
    n_trees = len(trees)

    if n_trees == 0:
        # Handle the case when there are no trees
        print("Warning: No trees available for prediction. Returning zeros.")
        return np.zeros((n_samples,))

    # Get the first tree's prediction to determine the shape
    first_pred = predict_function(trees[0], X)

    # Initialize predictions with the correct shape for multi-class
    if len(first_pred.shape) > 1:
        n_classes = first_pred.shape[1]
        predictions = np.zeros((n_samples, n_classes))
    else:
        predictions = np.zeros((n_samples,))

    # Add predictions from all trees
    predictions += first_pred

    if verbose > 0:
        print(f"Processing predictions for {n_trees} trees...")

    for i, tree in enumerate(trees[1:], 1):
        predictions += predict_function(tree, X)
        if verbose > 0 and i % 10 == 0:
            print(f"Processed {i+1}/{n_trees} trees")

    if verbose > 0:
        print(f"Finished processing all {n_trees} trees")

    return predictions / n_trees
