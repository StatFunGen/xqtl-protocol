import os
import sys
import argparse

import pandas as pd
from dask import dataframe as dd
from dask.diagnostics import ProgressBar


pbar = ProgressBar(dt=1)
pbar.register()
# pbar.unregister()

import time
from sklearn.model_selection import cross_val_score, BaseCrossValidator
from sklearn.model_selection import LeaveOneGroupOut

import optuna
from optuna.samplers import TPESampler, CmaEsSampler, GPSampler

from tqdm import tqdm

tqdm.pandas()

import requests
import numpy as np
from os import walk
import os
import dask

import pickle

import json
import pysam
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn import metrics
import sklearn
from sklearn.linear_model import SGDClassifier
import yaml
import pickle
import torch
import random
from sklearn.calibration import CalibratedClassifierCV
import optunahub
import joblib

def make_variant_features(df):
    """
    Generate variant-level features from variant ID.

    Args:
        df: DataFrame with variant_id column

    Returns:
        DataFrame with added variant features
    """
    #split variant_id by
    df[['chr','pos','ref','alt']] = df['variant_id'].str.split(':', expand=True)
    # calculate difference between length ref and length alt
    df['length_diff'] = df['ref'].str.len() - df['alt'].str.len()
    df['is_SNP'] = df['length_diff'].apply(lambda x: 1 if x == 0 else 0)
    df['is_indel'] = df['length_diff'].apply(lambda x: 1 if x != 0 else 0)
    df['is_insertion'] = df['length_diff'].apply(lambda x: 1 if x > 0 else 0)
    df['is_deletion'] = df['length_diff'].apply(lambda x: 1 if x < 0 else 0)
    df.drop(columns=['chr','pos','ref','alt'], inplace=True)
    #make label the last column
    cols = df.columns.tolist()
    cols.insert(len(cols)-1, cols.pop(cols.index('label')))
    df = df.loc[:, cols]
    return df


def train_model(args):
    """
    Train the GEMS model using CatBoost classifier.

    Args:
        args: Namespace containing training arguments (cohort, chromosome, data_config, model_config)
    """
    cohort = args.cohort
    chromosome = args.chromosome
    data_config_path = args.data_config
    model_config_path = args.model_config

    # Load configuration files
    data_config = yaml.safe_load(open(data_config_path))
    model_config = yaml.safe_load(open(model_config_path))
    
    # Configure dask temporary directory
    dask.config.set({"temporary_directory": model_config["system"]["temp_directory"]})
    
    # Load configurations
    
    # Set random seeds from configuration
    torch.manual_seed(model_config['system']['random_seeds']['torch_seed'])
    np.random.seed(model_config['system']['random_seeds']['numpy_seed'])
    random.seed(model_config['system']['random_seeds']['random_seed'])
    
    # Set dask temporary directory from configuration
    
    # Extract paths from data config
    gene_lof_file = data_config['feature_data']['gene_constraint']['file_path']
    maf_file_pattern = data_config['feature_data']['population_genetics']['file_pattern']
    data_dir_pattern = data_config['training_data']['base_dir']
    
    NPR_tr = model_config['experiment']['sampling_parameters']['npr_train']
    NPR_te = model_config['experiment']['sampling_parameters']['npr_test']
    
    # Normalize chromosome format - remove 'chr' prefix if present, then add it consistently
    chromosome_clean = chromosome.replace('chr', '')
    chromosome_out = f'chr{chromosome_clean}'
    
    # Set up chromosomes for proper train/test split to avoid data leakage
    # Use provided chromosome for training, and different chromosomes for testing
    train_chromosomes = [chromosome_out]
    
    # Define test chromosomes (use different chromosomes to avoid leakage)
    # Available chromosomes in dataset: chr1, chr2, chr3, chr5
    available_chromosomes = ['1', '2', '3', '5']
    test_chromosome_candidates = [c for c in available_chromosomes if c != chromosome_clean]
    
    if len(test_chromosome_candidates) == 0:
        raise ValueError(f"No different chromosomes available for testing. Only chr{chromosome} found.")
    
    # Use the first available different chromosome for testing
    test_chromosomes = [f'chr{test_chromosome_candidates[0]}']
    
    num_train_chromosomes = len(train_chromosomes)
    
    print(f"Using chromosome-based train/test split to prevent data leakage:")
    print(f"Training chromosomes: {train_chromosomes}")
    print(f"Testing chromosomes: {test_chromosomes}")
    print(f"This ensures no overlap between training and testing data.")
    
    # Load gene constraint data with configurable sheet name
    # Load gene constraint data generically
    constraint_config = data_config["feature_data"]["gene_constraint"]
    gene_lof_df = pd.read_excel(constraint_config["file_path"], constraint_config["xlsx_sheet"])
    
    # Use column mapping from constraint config
    constraint_mapping = constraint_config["column_mapping"]
    source_gene_col = constraint_mapping["source_gene_id"]
    target_gene_col = constraint_mapping["target_gene_id"]
    source_value_col = constraint_mapping["source_value"]
    target_value_col = constraint_mapping["target_value"]
    
    gene_lof_df = gene_lof_df[[source_gene_col, source_value_col]]
    gene_lof_df = gene_lof_df.rename(columns={source_gene_col: target_gene_col, source_value_col: target_value_col})
    
    gene_lof_df[target_value_col] = np.log2(gene_lof_df[target_value_col])
    
    
    maf_files = maf_file_pattern.format(chromosome=chromosome_clean)
    maf_df = dd.read_csv(maf_files, sep='\t')
    
    # Use population genetics column mapping
    pop_gen_config = data_config["feature_data"]["population_genetics"]
    pop_gen_mapping = pop_gen_config["column_mapping"]
    variant_id_col = pop_gen_mapping["variant_id"]
    target_maf_col = pop_gen_mapping["target_value"]
    
    maf_df = maf_df[[variant_id_col, target_maf_col]].compute()
    
    data_dir = data_config['training_data']['base_dir'].format(cohort=cohort)
    write_dir = data_config['output']['base_dir'].format(cohort=cohort)
    
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    
    predictions_dir = f"{write_dir}/{data_config['output']['predictions_dir']}"
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    
    # Use configurable columns dictionary file
    columns_dict_file = data_config['feature_data']['distance_features']['columns_dict_file']
    
    # open pickle file as column_dict
    with open(columns_dict_file, 'rb') as f:
        column_dict = pickle.load(f)
    
    
    def make_variant_features(df):
        #split variant_id by
        df[['chr','pos','ref','alt']] = df['variant_id'].str.split(':', expand=True)
        # calculate difference between length ref and length alt
        df['length_diff'] = df['ref'].str.len() - df['alt'].str.len()
        df['is_SNP'] = df['length_diff'].apply(lambda x: 1 if x == 0 else 0)
        df['is_indel'] = df['length_diff'].apply(lambda x: 1 if x != 0 else 0)
        df['is_insertion'] = df['length_diff'].apply(lambda x: 1 if x > 0 else 0)
        df['is_deletion'] = df['length_diff'].apply(lambda x: 1 if x < 0 else 0)
        df.drop(columns=['chr','pos','ref','alt'], inplace=True)
        #make label the last column
        cols = df.columns.tolist()
        cols.insert(len(cols)-1, cols.pop(cols.index('label')))
        df = df.loc[:, cols]
        return df
    
    
    # Load columns to remove from configuration
    columns_to_remove = data_config['feature_data']['distance_features']['columns_to_remove']
    
    #######################################################STANDARD TRAINING DATA#######################################################
    columns_to_remove = data_config['feature_data']['distance_features']['columns_to_remove']
    
    #######################################################STANDARD TRAINING DATA#######################################################
    train_files = []
    valid_train_chromosomes = []
    
    for i in train_chromosomes:
        train_dir = data_config['training_data']['train_dir_pattern'].format(
            npr_tr=NPR_tr,
    	pos_threshold=model_config['experiment']['classification_thresholds']['train']['positive_class_threshold'],
    	neg_threshold=model_config['experiment']['classification_thresholds']['train']['negative_class_threshold']
    
        )
        file_pattern = data_config['training_data']['file_pattern'].format(cohort=cohort, chromosome=i)
        file_path = f'{data_dir}/{train_dir}/{file_pattern}'
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist. Skipping chromosome {i}.")
            continue
        # Add the file to the list
        train_files.append(file_path)
        valid_train_chromosomes.append(i)
    
    if not train_files:
        raise ValueError("No training files found for any chromosome. Cannot proceed.")
    
    # Read standard training data
    train_df = dd.read_parquet(train_files, engine='pyarrow')
    train_df = train_df.compute()
    
    #train_df = train_df.drop(columns=residual_cols)
    
    train_df = make_variant_features(train_df)
    
    
    train_df = train_df.merge(gene_lof_df, on=target_gene_col, how='left')
    train_df = train_df.merge(maf_df, on=variant_id_col, how='left')
    
    #find rows with missing gene_lof
    
    # Calculate imputation statistics from training data only to prevent data leakage
    train_gene_lof_median = train_df[target_value_col].median()
    train_gnomad_maf_median = train_df[target_maf_col].median()
    
    print(f"Training data imputation - {target_value_col} median: {train_gene_lof_median:.6f}, {target_maf_col} median: {train_gnomad_maf_median:.6f}")
    
    # Apply imputation using training statistics
    train_df[target_value_col] = train_df[target_value_col].fillna(train_gene_lof_median)
    train_df[target_maf_col] = train_df[target_maf_col].fillna(train_gnomad_maf_median)
    
    #######################################################TEST DATA#######################################################
    test_files = []
    for i in test_chromosomes:
        test_dir = data_config['training_data']['test_dir_pattern'].format(
            npr_te=NPR_te,
            pos_threshold=model_config['experiment']['classification_thresholds']['test']['positive_class_threshold'],
            neg_threshold=model_config['experiment']['classification_thresholds']['test']['negative_class_threshold']
        )
        file_pattern = data_config['training_data']['file_pattern'].format(cohort=cohort, chromosome=i)
        file_path = f'{data_dir}/{test_dir}/{file_pattern}'
        # Check if file exists
        if os.path.exists(file_path):
            test_files.append(file_path)
        else:
            print(f"Warning: Test file {file_path} does not exist. Skipping chromosome {i}.")
    
    if not test_files:
        raise ValueError("No test files found. Cannot proceed.")
    
    test_df = dd.read_parquet(test_files, engine='pyarrow')
    test_df = test_df.compute()
    
    #test_df = test_df.drop(columns=residual_cols)
    
    test_df = make_variant_features(test_df)
    
    
    test_df = test_df.merge(gene_lof_df, on=target_gene_col, how='left')
    test_df = test_df.merge(maf_df, on=variant_id_col, how='left')
    
    # Apply imputation using training data statistics (prevent data leakage)
    test_df[target_value_col] = test_df[target_value_col].fillna(train_gene_lof_median)
    test_df[target_maf_col] = test_df[target_maf_col].fillna(train_gnomad_maf_median)
    
    print(f"Applied training imputation statistics to test data")
    
    ##############################################################################################################
    # Calculate class-balanced weights (FIXED: No longer using PIP to avoid target leakage)
    train_class_0 = train_df[train_df['label'] == 0].shape[0]
    train_class_1 = train_df[train_df['label'] == 1].shape[0]
    
    # Use standard class balancing instead of PIP-based weighting to avoid target leakage
    # Standard approach: inverse class frequency weighting
    train_total = train_class_0 + train_class_1
    weight_class_0 = train_total / (2 * train_class_0) if train_class_0 > 0 else 1.0
    weight_class_1 = train_total / (2 * train_class_1) if train_class_1 > 0 else 1.0
    
    # Create balanced weights (not using PIP anymore)
    train_df['weight'] = np.where(train_df['label'] == 0, weight_class_0, weight_class_1)
    
    # Calculate weights for test data using same approach
    test_class_0 = test_df[test_df['label'] == 0].shape[0]
    test_class_1 = test_df[test_df['label'] == 1].shape[0]
    test_total = test_class_0 + test_class_1
    test_weight_class_0 = test_total / (2 * test_class_0) if test_class_0 > 0 else 1.0
    test_weight_class_1 = test_total / (2 * test_class_1) if test_class_1 > 0 else 1.0
    
    # Create balanced weights for test data
    test_df['weight'] = np.where(test_df['label'] == 0, test_weight_class_0, test_weight_class_1)
    
    # Check weight distribution
    print("Standard training data weight distribution:")
    print(train_df.groupby('label')['weight'].sum())
    
    print("Test data weight distribution:")
    print(test_df.groupby('label')['weight'].sum())
    
    ##############################################################################################################
    # Load meta data columns from configuration
    meta_data = data_config['training_data']['metadata_columns']
    
    # Prepare standard training data
    X_train = train_df.drop(columns=meta_data)
    Y_train = train_df['label']
    weight_train = train_df['weight']
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_train = X_train.fillna(0)
    
    cols_order = X_train.columns.tolist()
    
    # Prepare test data
    X_test = test_df.drop(columns=meta_data)
    Y_test = test_df['label']
    weight_test = test_df['weight']
    X_test = X_test.replace([np.inf, -np.inf], 0)
    X_test = X_test.fillna(0)
    
    # Remove gene_id if present
    if target_gene_col in X_train.columns:
        X_train = X_train.drop(columns=[target_gene_col])
        X_test = X_test.drop(columns=[target_gene_col])
    
    # Print class distributions
    print("Standard training data class distribution:")
    print(Y_train.value_counts())
    
    print("Test data class distribution:")
    print(Y_test.value_counts())
    
    ##############################################################################################################
    # Create subset of columns based on column_dict keys - load from configuration
    # Combine subset_keys from different feature sections
    subset_keys = []
    subset_keys.extend(data_config['feature_data']['distance_features']['subset_keys'])
    subset_keys.extend(data_config['feature_data']['regulatory_features']['subset_keys'])
    subset_keys.extend(data_config['feature_data']['deep_learning_features']['subset_keys'])
    
    # Extract columns for each subset
    subset_cols = []
    for key in subset_keys:
        if key in column_dict:
            subset_cols.extend(column_dict[key])
    
    # Keep only columns that exist in the dataframes
    subset_cols = [col for col in subset_cols if col in X_train.columns]
    
    # Add variant features to subset columns - load from configuration
    variant_features = data_config['feature_data']['variant_features']['generated_columns']
    subset_cols.extend(variant_features)
    
    # Apply absolute value to configured columns
    columns_to_abs = []
    absolute_value_columns = data_config['feature_data']['deep_learning_features']['transformations']['absolute_value']
    for col in absolute_value_columns:
        if col in X_train.columns:
            columns_to_abs.append(col)
    
    # Create subset dataframes with absolute values applied
    X_train_subset = X_train[subset_cols].copy()
    X_test_subset = X_test[subset_cols].copy()
    
    # Apply absolute values only to the specified columns (not to variant features)
    for col in columns_to_abs:
        if col in X_train_subset.columns:
            X_train_subset[col] = X_train_subset[col].abs()
            X_test_subset[col] = X_test_subset[col].abs()
    
    # Drop columns from configuration
    columns_to_drop = data_config['feature_data']['distance_features']['columns_to_remove']
    for col in columns_to_drop:
        if col in X_train_subset.columns:
            X_train_subset = X_train_subset.drop(columns=[col])
        if col in X_test_subset.columns:
            X_test_subset = X_test_subset.drop(columns=[col])
        if col in X_train.columns:
            X_train = X_train.drop(columns=[col])
        if col in X_test.columns:
            X_test = X_test.drop(columns=[col])
    
    
    ##############################################################################################################
    from catboost import CatBoostClassifier
    
    # Load model parameters from config
    original_params = model_config['algorithm']['parameter_sets']['standard']
    
    # Create feature weight dictionary from configuration
    feature_weights = {}
    default_weight = model_config['feature_weighting']['default_weight']
    high_weight_value = model_config['feature_weighting']['high_priority_patterns']['weight']
    high_priority_patterns = model_config['feature_weighting']['high_priority_patterns']['feature_patterns']
    
    # Set default weight for all features
    for col in X_train_subset.columns:
        feature_weights[col] = default_weight
    
    # Set high weight for priority features based on patterns
    for col in X_train_subset.columns:
        if col in columns_to_abs:
            if any(pattern in col for pattern in high_priority_patterns):
                feature_weights[col] = high_weight_value
    
    print("Feature weights distribution:")
    print(f"Number of features with weight {high_weight_value}: {sum(value == high_weight_value for value in feature_weights.values())}")
    print(f"Number of features with weight {default_weight}: {sum(value == default_weight for value in feature_weights.values())}")
    
    # Model 5: Standard data, subset features (original params) with feature weighting
    cat_standard_subset_weighted = CatBoostClassifier(
        **original_params,
        feature_weights=feature_weights,
        name="Standard-Subset-Weighted"
    )
    
    # Train model 5
    print("Training model 5: Standard data, subset features (original params) with feature weighting")
    cat_standard_subset_weighted.fit(X_train_subset, Y_train, sample_weight=weight_train)
    
    # Calculate predictions for model 5
    preds_standard_subset_weighted = cat_standard_subset_weighted.predict_proba(X_test_subset)[:, 1]
    
    # Calculate metrics for model 5
    ap_score = metrics.average_precision_score(Y_test, preds_standard_subset_weighted)
    auc_score = metrics.roc_auc_score(Y_test, preds_standard_subset_weighted)
    
    # Print metrics for model 5
    print("\nTest Set Metrics:")
    print(f"5. Standard data, subset features (original) weighted - AP: {ap_score:.4f}, AUC: {auc_score:.4f}")
    
    # Save model 5
    joblib.dump(cat_standard_subset_weighted, f'{write_dir}/model_standard_subset_weighted_chr_{chromosome_out}_NPR_{NPR_tr}.joblib')
    
    # Get feature importances for model 5
    importances = cat_standard_subset_weighted.feature_importances_
    features = X_train_subset.columns
    
    # Create feature importance dataframe
    feature_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    })
    feature_df = feature_df.sort_values(by='importance', ascending=False)
    
    # Print top 20 features
    print(f"\nTop 20 features for model 5 (feature-weighted):")
    for i, (feature, importance) in enumerate(zip(feature_df['feature'][:20], feature_df['importance'][:20])):
        print(f"{i + 1}. {feature}: {importance:.6f}")
    
    # Save feature importance
    feature_df.to_csv(f'{write_dir}/features_importance_model5_chr_{chromosome_out}_NPR_{NPR_tr}.csv', index=False)
    
    # Create summary dictionary for model 5
    summary_dict = {
        'CatBoost': {
            'standard_subset_weighted': {
                'AP_test': ap_score,
                'AUC_test': auc_score,
                'params': original_params,
                'feature_weights': f'{", ".join(high_priority_patterns)} features set to {high_weight_value}, others to {default_weight}'
            },
            'test_num_positive_labels': Y_test.value_counts().get(1, 0),
            'test_num_negative_labels': Y_test.value_counts().get(0, 0),
            'train_standard_num_positive_labels': Y_train.value_counts().get(1, 0),
            'train_standard_num_negative_labels': Y_train.value_counts().get(0, 0)
        }
    }
    
    print('Writing results to file')
    
    # Write summary_dict to pickle file
    with open(f'{write_dir}/summary_dict_catboost_weighted_model_chr_{chromosome_out}_NPR_{NPR_tr}.pkl', 'wb') as f:
        pickle.dump(summary_dict, f)
    
    # Add predictions to test_df and actual labels
    test_df['standard_subset_weighted_pred_prob'] = preds_standard_subset_weighted
    test_df['standard_subset_weighted_pred_label'] = cat_standard_subset_weighted.predict(X_test_subset)
    test_df['actual_label'] = Y_test
    
    # Save the test_df with model 5 predictions
    test_df.to_csv(f'{write_dir}/predictions_parquet_catboost/predictions_weighted_model_chr{chromosome}.tsv', sep='\t',
                   index=False)
    
    # Save the feature weights dictionary for reference
    with open(f'{write_dir}/feature_weights_chr_{chromosome}_NPR_{NPR_tr}.pkl', 'wb') as f:
        pickle.dump(feature_weights, f)
    
    # Save the subset columns list for future reference
    with open(f'{write_dir}/subset_columns_chr_{chromosome}_NPR_{NPR_tr}.pkl', 'wb') as f:
        pickle.dump({
            'subset_columns': subset_cols,
            'abs_columns': columns_to_abs
        }, f)
    
    print("\nTraining and evaluation complete for feature-weighted CatBoost model (5).")
    
    ##############################################################################################################
    # Additional Cross-Validation for More Robust Evaluation (ADDED TO PREVENT OVERFITTING)
    ##############################################################################################################
    print("\n" + "="*80)
    print("PERFORMING CROSS-VALIDATION ON TRAINING DATA FOR ROBUST EVALUATION")
    print("="*80)
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import average_precision_score, roc_auc_score
    
    # Perform 5-fold cross-validation on training data
    cv_folds = 5
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=model_config['system']['random_seeds']['numpy_seed'])
    
    cv_ap_scores = []
    cv_auc_scores = []
    
    # Set feature weights flag for cross-validation
    use_feature_weights = True
    
    print(f"Performing {cv_folds}-fold cross-validation on training chromosome(s): {train_chromosomes}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_subset, Y_train)):
        print(f"\nFold {fold + 1}/{cv_folds}:")
    
        # Split training data into train/validation for this fold
        X_train_fold = X_train_subset.iloc[train_idx]
        X_val_fold = X_train_subset.iloc[val_idx]
        Y_train_fold = Y_train.iloc[train_idx]
        Y_val_fold = Y_train.iloc[val_idx]
        weight_train_fold = weight_train.iloc[train_idx]
    
        # Train model on this fold - fix verbose parameter conflict
        fold_params = original_params.copy()
        fold_params['verbose'] = False
    
        # Note: CatBoost constructor takes feature_weights, not fit method
        if use_feature_weights:
            fold_model = CatBoostClassifier(**fold_params, feature_weights=feature_weights)
        else:
            fold_model = CatBoostClassifier(**fold_params)
    
        fold_model.fit(X_train_fold, Y_train_fold, sample_weight=weight_train_fold)
    
        # Predict on validation fold
        val_preds = fold_model.predict_proba(X_val_fold)[:, 1]
    
        # Calculate metrics
        fold_ap = average_precision_score(Y_val_fold, val_preds)
        fold_auc = roc_auc_score(Y_val_fold, val_preds)
    
        cv_ap_scores.append(fold_ap)
        cv_auc_scores.append(fold_auc)
    
        print(f"  Validation AP: {fold_ap:.4f}, AUC: {fold_auc:.4f}")
    
    # Calculate cross-validation statistics
    cv_ap_mean = np.mean(cv_ap_scores)
    cv_ap_std = np.std(cv_ap_scores)
    cv_auc_mean = np.mean(cv_auc_scores)
    cv_auc_std = np.std(cv_auc_scores)
    
    print(f"\n{cv_folds}-Fold Cross-Validation Results:")
    print(f"Average Precision: {cv_ap_mean:.4f} ± {cv_ap_std:.4f}")
    print(f"AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
    
    # Update summary with cross-validation results
    summary_dict['CatBoost']['standard_subset_weighted']['cross_validation'] = {
        'cv_folds': cv_folds,
        'cv_ap_scores': cv_ap_scores,
        'cv_auc_scores': cv_auc_scores,
        'cv_ap_mean': cv_ap_mean,
        'cv_ap_std': cv_ap_std,
        'cv_auc_mean': cv_auc_mean,
        'cv_auc_std': cv_auc_std
    }
    
    # Save updated summary
    with open(f'{write_dir}/model_5_summary_chr_{chromosome_out}_NPR_{NPR_tr}.pkl', 'wb') as f:
        pickle.dump(summary_dict, f)
    
    print(f"\n" + "="*80)
    print("FIXED DATA LEAKAGE ISSUES:")
    print("1. ✅ Using different chromosomes for train/test")
    print("2. ✅ Removed PIP-based weighting (was causing target leakage)")
    print("3. ✅ Added cross-validation for robust evaluation")
    print("4. ✅ Expect more realistic performance scores (typically 60-80% AUC)")
    print("="*80)


def predict_model(args):
    """
    Generate predictions using a trained GEMS model.

    Args:
        args: Namespace containing prediction arguments (cohort, chromosome, model_path, data_config)
    """
    print("="*80)
    print("GEMS PREDICTION MODE")
    print("="*80)
    print("\nThis functionality is not yet implemented.")
    print("\nPlanned features:")
    print("  - Load a trained GEMS model from the specified path")
    print("  - Apply the model to new genomic data")
    print("  - Generate expression modifier scores for variants")
    print("  - Export predictions in standard formats")
    print("\nFor now, please use the 'train' subcommand to train models.")
    print("="*80)


def main():
    """
    Main entry point for the GEMS pipeline.

    Parses command-line arguments and routes to the appropriate subcommand.
    """
    parser = argparse.ArgumentParser(
        description="GEMS Pipeline: Generalized Expression Modifier Scores\n\n"
                    "A machine learning pipeline for predicting genetic variants that modify gene expression.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train subcommand
    train_parser = subparsers.add_parser(
        'train',
        help='Train a GEMS prediction model',
        description='Train a CatBoost model to predict expression modifier variants'
    )
    train_parser.add_argument('cohort', type=str, help='Cohort/project name (e.g., Mic_mega_eQTL)')
    train_parser.add_argument('chromosome', type=str, help='Chromosome number (e.g., 2)')
    train_parser.add_argument('--data_config', type=str, required=True, help='Path to data configuration YAML')
    train_parser.add_argument('--model_config', type=str, required=True, help='Path to model configuration YAML')

    # Predict subcommand
    predict_parser = subparsers.add_parser(
        'predict',
        help='Generate predictions using a trained model',
        description='Apply a trained GEMS model to generate expression modifier scores'
    )
    predict_parser.add_argument('cohort', type=str, help='Cohort/project name (e.g., Mic_mega_eQTL)')
    predict_parser.add_argument('chromosome', type=str, help='Chromosome number (e.g., 2)')
    predict_parser.add_argument('--model_path', type=str, required=True, help='Path to trained model file')
    predict_parser.add_argument('--data_config', type=str, required=True, help='Path to data configuration YAML')

    args = parser.parse_args()

    # Route to appropriate function based on subcommand
    if args.command == 'train':
        train_model(args)
    elif args.command == 'predict':
        predict_model(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
