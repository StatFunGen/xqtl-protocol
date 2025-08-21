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

dask.config.set({'temporary_directory': '/nfs/scratch'})

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

torch.manual_seed(9448)
np.random.seed(9448)
random.seed(9448)

parser = argparse.ArgumentParser(description="Train eQTL prediction model")
parser.add_argument("cohort", type=str, help="Cohort/project name (e.g., Mic_mega_eQTL)")
parser.add_argument("chromosome", type=str, help="Chromosome number (e.g., 2)")
parser.add_argument("--gene_lof_file", type=str, required=True, help="Path to Excel file (e.g., 41588_2024_1820_MOESM4_ESM.xlsx)")
parser.add_argument("--yaml_path", type=str, default="pipeline/5_model_training/data_params.yaml", help="Path to data_params.yaml")

args = parser.parse_args()

cohort = args.cohort
chromosome = args.chromosome
gene_lof_file = args.gene_lof_file
yaml_path = args.yaml_path

params_data = yaml.safe_load(open(yaml_path))

chromosome_out = f'chr{chromosome}'




#cohort = 'Mic_mega_eQTL'

NPR_tr = 10
NPR_te = 10

chromosome_out = f'chr{chromosome}'

chromosomes = [f'chr{x}' for x in range(1, 23)]
train_chromosomes = [x for x in chromosomes if x != chromosome_out]
test_chromosomes = [chromosome_out]
num_train_chromosomes = len(train_chromosomes)




gene_lof_df = pd.read_excel(gene_lof_file, "Supplementary Table 1")

gene_lof_df = gene_lof_df[['ensg','post_mean']]

gene_lof_df = gene_lof_df.rename(columns={'ensg': 'gene_id', 'post_mean': 'gene_lof'})

gene_lof_df['gene_lof'] = np.log2(gene_lof_df['gene_lof'])


maf_files = f'../../data/gnomad_MAF_chr{chromosome}.tsv'
maf_df = dd.read_csv(maf_files, sep='\t')

maf_df = maf_df[['variant_id', 'gnomad_MAF']].compute()

data_dir = f'../../data/{cohort}'
write_dir = f'{data_dir}/model_results'

if not os.path.exists(write_dir):
    os.makedirs(write_dir)

if not os.path.exists(f'{write_dir}/predictions_parquet_catboost'):
    os.makedirs(f'{write_dir}/predictions_parquet_catboost')

columns_dir = 'data'

# open pickle file as column_dict
with open(f'{columns_dir}/columns_dict.pkl', 'rb') as f:
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


residual_cols = ['length_diff', 'is_snp', 'var_type_snp', 'var_type_ins',
       'var_type_del']

#######################################################STANDARD TRAINING DATA#######################################################
train_files = []
valid_train_chromosomes = []

for i in train_chromosomes:
    file_path = f'{data_dir}/training_data/train_NPR_{NPR_tr}_PIP_{params_data["train"]["positive_class_threshold"]}_{params_data["train"]["negative_class_threshold"]}/annotated_data_{cohort}_{i}.parquet'
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


train_df = train_df.merge(gene_lof_df, on='gene_id', how='left')
train_df = train_df.merge(maf_df, on='variant_id', how='left')

#find rows with missing gene_lof

#fill missing gene_lof with median

train_df['gene_lof'] = train_df['gene_lof'].fillna(train_df['gene_lof'].median())

train_df['gnomad_MAF'] = train_df['gnomad_MAF'].fillna(train_df['gnomad_MAF'].median())

#######################################################TEST DATA#######################################################
test_files = []
for i in test_chromosomes:
    file_path = f'{data_dir}/training_data/test_NPR_{NPR_te}_PIP_{params_data["test"]["positive_class_threshold"]}_{params_data["test"]["negative_class_threshold"]}/annotated_data_{cohort}_{i}.parquet'
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


test_df = test_df.merge(gene_lof_df, on='gene_id', how='left')
test_df = test_df.merge(maf_df, on='variant_id', how='left')

#find rows with missing gene_lof
#fill missing gene_lof with median
test_df['gene_lof'] = test_df['gene_lof'].fillna(test_df['gene_lof'].median())
test_df['gnomad_MAF'] = test_df['gnomad_MAF'].fillna(test_df['gnomad_MAF'].median())

##############################################################################################################
# Calculate weights for standard training data
train_class_0 = train_df[train_df['label'] == 0].shape[0]
train_class_1 = train_df[train_df['label'] == 1].shape[0]
train_total_pip = train_df[train_df['label'] == 1].pip.sum()
train_pip_percent = train_class_0 / train_total_pip if train_total_pip > 0 else 1

# Create a column called weight where everything with label = 0 has weight 1 and label = 1 has weight pip * train_pip_percent
train_df['weight'] = np.where(train_df['label'] == 0, 1, train_df['pip'] * train_pip_percent)

# Calculate weights for test data
test_class_0 = test_df[test_df['label'] == 0].shape[0]
test_class_1 = test_df[test_df['label'] == 1].shape[0]
test_total_pip = test_df[test_df['label'] == 1].pip.sum()
test_pip_percent = test_class_0 / test_total_pip if test_total_pip > 0 else 1

# Create a column called weight where everything with label = 0 has weight 1 and label = 1 has weight pip * test_pip_percent
test_df['weight'] = np.where(test_df['label'] == 0, 1, test_df['pip'] * test_pip_percent)

# Check weight distribution
print("Standard training data weight distribution:")
print(train_df.groupby('label')['weight'].sum())

print("Test data weight distribution:")
print(test_df.groupby('label')['weight'].sum())

##############################################################################################################
meta_data = ['variant_id', 'pip', 'CHR', 'BP', 'REF', 'ALT', 'SNP', 'label', 'weight']

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
if 'gene_id' in X_train.columns:
    X_train = X_train.drop(columns=['gene_id'])
    X_test = X_test.drop(columns=['gene_id'])

# Print class distributions
print("Standard training data class distribution:")
print(Y_train.value_counts())

print("Test data class distribution:")
print(Y_test.value_counts())

##############################################################################################################
# Create subset of columns based on column_dict keys
subset_keys = ['distance', 'ABC', 'celltype', 'baseline', 'chrombpnet_positive', 'diff', 'tf_positive']

# Extract columns for each subset
subset_cols = []
for key in subset_keys:
    if key in column_dict:
        subset_cols.extend(column_dict[key])

# Keep only columns that exist in the dataframes
subset_cols = [col for col in subset_cols if col in X_train.columns]

# Add variant features to subset columns
variant_features = ['length_diff', 'is_SNP', 'is_indel', 'is_insertion', 'is_deletion', 'gene_lof', 'gnomad_MAF']
subset_cols.extend(variant_features)

# Apply absolute value to diff, tf_positive, and chrombpnet_positive columns
columns_to_abs = []
for key in ['diff', 'tf_positive', 'chrombpnet_positive']:
    if key in column_dict:
        columns_to_abs.extend([col for col in column_dict[key] if col in X_train.columns])

# Create subset dataframes with absolute values applied
X_train_subset = X_train[subset_cols].copy()
X_test_subset = X_test[subset_cols].copy()

# Apply absolute values only to the specified columns (not to variant features)
for col in columns_to_abs:
    if col in X_train_subset.columns:
        X_train_subset[col] = X_train_subset[col].abs()
        X_test_subset[col] = X_test_subset[col].abs()


X_train_subset = X_train_subset.drop(columns=['abs_distance_TSS', 'distance_TSS'])
X_test_subset = X_test_subset.drop(columns=['abs_distance_TSS', 'distance_TSS'])
X_train = X_train.drop(columns=['abs_distance_TSS', 'distance_TSS'])
X_test = X_test.drop(columns=['abs_distance_TSS', 'distance_TSS'])


##############################################################################################################
from catboost import CatBoostClassifier

# Original parameters
original_params = {
    'depth': 6,
    'iterations': 1000,
    'learning_rate': 0.03,
    'l2_leaf_reg': 3.0,
    'min_data_in_leaf': 10,
    'verbose': True
}

# More conservative parameters (to reduce overfitting)
conservative_params = {
    'depth': 5,
    'iterations': 1000,
    'learning_rate': 0.03,
    'l2_leaf_reg': 5.0,
    'min_data_in_leaf': 10,
    'bagging_temperature': 1.0,
    'leaf_estimation_method': 'Newton',
    'leaf_estimation_iterations': 10,
    'verbose': True
}

# Create feature weight dictionary
feature_weights = {}

# Set default weight of 1.0 for all features
for col in X_train_subset.columns:
    feature_weights[col] = 1.0

# Set weight of 10.0 for chrombpnet_positive, tf_positive, and diff features
for col in X_train_subset.columns:
    if col in columns_to_abs:
        if any(key in col for key in ['chrombpnet_positive', 'tf_positive', 'diff']):
            feature_weights[col] = 10.0

print("Feature weights distribution:")
print(f"Number of features with weight 10.0: {sum(value == 10.0 for value in feature_weights.values())}")
print(f"Number of features with weight 1.0: {sum(value == 1.0 for value in feature_weights.values())}")

# Initialize 4 models (only standard training data models)
# Model 1: Standard data, subset features (original params)
cat_standard_subset = CatBoostClassifier(
    **original_params,
    loss_function='Logloss',
    name="Standard-Subset"
)

# Model 3: Standard data, subset features (conservative params)
cat_standard_subset_conservative = CatBoostClassifier(
    **conservative_params,
    loss_function='Logloss',
    name="Standard-Subset-Conservative"
)

# Model 5: Standard data, subset features (original params) with feature weighting
cat_standard_subset_weighted = CatBoostClassifier(
    **original_params,
    loss_function='Logloss',
    feature_weights=feature_weights,
    name="Standard-Subset-Weighted"
)

# Model 7: Standard data, subset features (conservative params) with feature weighting
cat_standard_subset_conservative_weighted = CatBoostClassifier(
    **conservative_params,
    loss_function='Logloss',
    feature_weights=feature_weights,
    name="Standard-Subset-Conservative-Weighted"
)

# Train all models
print("Training model 1: Standard data, subset features (original params)")
cat_standard_subset.fit(X_train_subset, Y_train, sample_weight=weight_train)

print("Training model 3: Standard data, subset features (conservative params)")
cat_standard_subset_conservative.fit(X_train_subset, Y_train, sample_weight=weight_train)

print("Training model 5: Standard data, subset features (original params) with feature weighting")
cat_standard_subset_weighted.fit(X_train_subset, Y_train, sample_weight=weight_train)

print("Training model 7: Standard data, subset features (conservative params) with feature weighting")
cat_standard_subset_conservative_weighted.fit(X_train_subset, Y_train, sample_weight=weight_train)

# Calculate predictions for all models
preds_standard_subset = cat_standard_subset.predict_proba(X_test_subset)[:, 1]
preds_standard_subset_conservative = cat_standard_subset_conservative.predict_proba(X_test_subset)[:, 1]
preds_standard_subset_weighted = cat_standard_subset_weighted.predict_proba(X_test_subset)[:, 1]
preds_standard_subset_conservative_weighted = cat_standard_subset_conservative_weighted.predict_proba(X_test_subset)[:, 1]

# Calculate metrics for all models
metrics_dict = {
    'standard_subset': {
        'AP': metrics.average_precision_score(Y_test, preds_standard_subset),
        'AUC': metrics.roc_auc_score(Y_test, preds_standard_subset)
    },
    'standard_subset_conservative': {
        'AP': metrics.average_precision_score(Y_test, preds_standard_subset_conservative),
        'AUC': metrics.roc_auc_score(Y_test, preds_standard_subset_conservative)
    },
    'standard_subset_weighted': {
        'AP': metrics.average_precision_score(Y_test, preds_standard_subset_weighted),
        'AUC': metrics.roc_auc_score(Y_test, preds_standard_subset_weighted)
    },
    'standard_subset_conservative_weighted': {
        'AP': metrics.average_precision_score(Y_test, preds_standard_subset_conservative_weighted),
        'AUC': metrics.roc_auc_score(Y_test, preds_standard_subset_conservative_weighted)
    }
}

# Print metrics for all models
print("\nTest Set Metrics:")
print(f"1. Standard data, subset features (original) - AP: {metrics_dict['standard_subset']['AP']:.4f}, AUC: {metrics_dict['standard_subset']['AUC']:.4f}")
print(f"3. Standard data, subset features (conservative) - AP: {metrics_dict['standard_subset_conservative']['AP']:.4f}, AUC: {metrics_dict['standard_subset_conservative']['AUC']:.4f}")
print(f"5. Standard data, subset features (original) weighted - AP: {metrics_dict['standard_subset_weighted']['AP']:.4f}, AUC: {metrics_dict['standard_subset_weighted']['AUC']:.4f}")
print(f"7. Standard data, subset features (conservative) weighted - AP: {metrics_dict['standard_subset_conservative_weighted']['AP']:.4f}, AUC: {metrics_dict['standard_subset_conservative_weighted']['AUC']:.4f}")

# Save all models
joblib.dump(cat_standard_subset, f'{write_dir}/model_standard_subset_chr_{chromosome_out}_NPR_{NPR_tr}.joblib')
joblib.dump(cat_standard_subset_conservative, f'{write_dir}/model_standard_subset_conservative_chr_{chromosome_out}_NPR_{NPR_tr}.joblib')
joblib.dump(cat_standard_subset_weighted, f'{write_dir}/model_standard_subset_weighted_chr_{chromosome_out}_NPR_{NPR_tr}.joblib')
joblib.dump(cat_standard_subset_conservative_weighted, f'{write_dir}/model_standard_subset_conservative_weighted_chr_{chromosome_out}_NPR_{NPR_tr}.joblib')

# Get feature importances for all models
feature_importances = {
    'standard_subset': {
        'importances': cat_standard_subset.feature_importances_,
        'features': X_train_subset.columns
    },
    'standard_subset_conservative': {
        'importances': cat_standard_subset_conservative.feature_importances_,
        'features': X_train_subset.columns
    },
    'standard_subset_weighted': {
        'importances': cat_standard_subset_weighted.feature_importances_,
        'features': X_train_subset.columns
    },
    'standard_subset_conservative_weighted': {
        'importances': cat_standard_subset_conservative_weighted.feature_importances_,
        'features': X_train_subset.columns
    }
}

# Create individual dataframes for each model
feature_dfs = []
for model_name, model_data in feature_importances.items():
    features = model_data['features']
    importances = model_data['importances']
    model_df = pd.DataFrame({
        'feature': features,
        'importance': importances,
        'model': model_name
    })
    model_df = model_df.sort_values(by='importance', ascending=False)
    feature_dfs.append(model_df)
    # Print top 20 features for each model
    print(f"\nTop 20 features for {model_name}:")
    for i, (feature, importance) in enumerate(zip(model_df['feature'][:20], model_df['importance'][:20])):
        print(f"{i + 1}. {feature}: {importance:.6f}")

# Combine all feature importance dataframes
all_features_df = pd.concat(feature_dfs, ignore_index=True)

# Save the combined feature importance dataframe
all_features_df.to_csv(f'{write_dir}/features_importance_4models_chr_{chromosome_out}_NPR_{NPR_tr}.csv',
                       index=False)

# Create summary dictionary with all models' parameters and metrics
summary_dict = {
    'CatBoost': {
        'standard_subset': {
            'AP_test': metrics_dict['standard_subset']['AP'],
            'AUC_test': metrics_dict['standard_subset']['AUC'],
            'params': original_params
        },
        'standard_subset_conservative': {
            'AP_test': metrics_dict['standard_subset_conservative']['AP'],
            'AUC_test': metrics_dict['standard_subset_conservative']['AUC'],
            'params': conservative_params
        },
        'standard_subset_weighted': {
            'AP_test': metrics_dict['standard_subset_weighted']['AP'],
            'AUC_test': metrics_dict['standard_subset_weighted']['AUC'],
            'params': original_params,
            'feature_weights': 'chrombpnet_positive, tf_positive, and diff features set to 10.0, others to 1.0'
        },
        'standard_subset_conservative_weighted': {
            'AP_test': metrics_dict['standard_subset_conservative_weighted']['AP'],
            'AUC_test': metrics_dict['standard_subset_conservative_weighted']['AUC'],
            'params': conservative_params,
            'feature_weights': 'chrombpnet_positive, tf_positive, and diff features set to 10.0, others to 1.0'
        },
        'test_num_positive_labels': Y_test.value_counts().get(1, 0),
        'test_num_negative_labels': Y_test.value_counts().get(0, 0),
        'train_standard_num_positive_labels': Y_train.value_counts().get(1, 0),
        'train_standard_num_negative_labels': Y_train.value_counts().get(0, 0)
    }
}

print('Writing results to file')

# Write summary_dict to pickle file
with open(f'{write_dir}/summary_dict_catboost_4models_chr_{chromosome_out}_NPR_{NPR_tr}.pkl', 'wb') as f:
    pickle.dump(summary_dict, f)

# Add predictions to test_df and actual labels
test_df['standard_subset_pred_prob'] = preds_standard_subset
test_df['standard_subset_pred_label'] = cat_standard_subset.predict(X_test_subset)

test_df['standard_subset_conservative_pred_prob'] = preds_standard_subset_conservative
test_df['standard_subset_conservative_pred_label'] = cat_standard_subset_conservative.predict(X_test_subset)

test_df['standard_subset_weighted_pred_prob'] = preds_standard_subset_weighted
test_df['standard_subset_weighted_pred_label'] = cat_standard_subset_weighted.predict(X_test_subset)

test_df['standard_subset_conservative_weighted_pred_prob'] = preds_standard_subset_conservative_weighted
test_df['standard_subset_conservative_weighted_pred_label'] = cat_standard_subset_conservative_weighted.predict(
    X_test_subset)

test_df['actual_label'] = Y_test  # Add actual labels to test_df

# Save the test_df with all 4 models' predictions
test_df.to_csv(f'{write_dir}/predictions_parquet_catboost/predictions_4models_chr{chromosome}.tsv', sep='\t',
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

print("\nTraining and evaluation complete for 4 standard models (1, 3, 5, 7).")
