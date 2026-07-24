"""Notebook tier: ems_training.ipynb / ems_prediction.ipynb (GEMS / EMS).

Worker: code/script/xqtl_modifier_score/gems_pipeline.R (train / predict).

Fixtures (tests/fixtures/ems_training), from the MWE:
  protocol_example.gene_constraint.xlsx        gene-constraint (LOEUF-like) table
  protocol_example.gnomad_MAF_chr{1,2}.tsv     population MAF
  protocol_example.columns_dict.json           feature-group -> columns (converted once from .pkl)
  protocol_example/training_data/{train,test}_.../annotated_data_*.parquet
  model_config.yaml
  expected/features_importance_model5_chr_chr2_NPR_1.csv   Python-reference importances

R catboost and Python CatBoost share the libcatboost backend and train deterministically here,
so the model reproduces the Python reference NUMERICALLY: feature importances match to ~1e-6 and
the test AP/AUC match. (The 5-fold CV differs — sklearn StratifiedKFold shuffle RNG != R's.)
predict is a stub (mirrors the Python), so ems_prediction has no trainable target.
"""
from __future__ import annotations

import json

WORKER = "code/script/xqtl_modifier_score/gems_pipeline.R"
FIX = "tests/fixtures/ems_training"
REF_AP, REF_AUC = 0.49887491976172904, 0.5061220393416299   # from the MWE summary pickle


def _data_config(fix, out):
    """A data_config.yaml wired to the committed fixtures (absolute paths)."""
    return f"""feature_data:
  gene_constraint:
    file_path: "{fix}/protocol_example.gene_constraint.xlsx"
    xlsx_sheet: "Supplementary Table 1"
    column_mapping: {{source_gene_id: ensg, target_gene_id: gene_id, source_value: post_mean, target_value: gene_lof}}
  population_genetics:
    file_pattern: "{fix}/protocol_example.gnomad_MAF_chr{{chromosome}}.tsv"
    column_mapping: {{variant_id: variant_id, target_value: gnomad_MAF}}
  distance_features:
    columns_dict_file: "{fix}/protocol_example.columns_dict.json"
    subset_keys: ["distance"]
    columns_to_remove: ["abs_distance_TSS", "distance_TSS"]
  regulatory_features: {{subset_keys: ["ABC", "celltype", "baseline"]}}
  deep_learning_features:
    subset_keys: ["chrombpnet_positive", "diff", "tf_positive"]
    transformations: {{absolute_value: ["diff", "tf_positive", "chrombpnet_positive"]}}
  variant_features:
    generated_columns: ["length_diff", "is_SNP", "is_indel", "is_insertion", "is_deletion", "gene_lof", "gnomad_MAF"]
training_data:
  base_dir: "{fix}/{{cohort}}/training_data"
  file_pattern: "annotated_data_{{cohort}}_{{chromosome}}.parquet"
  train_dir_pattern: "train_NPR_{{npr_tr}}_PIP_{{pos_threshold}}_{{neg_threshold}}"
  test_dir_pattern: "test_NPR_{{npr_te}}_PIP_{{pos_threshold}}_{{neg_threshold}}"
  metadata_columns: ["variant_id", "pip", "CHR", "BP", "REF", "ALT", "SNP", "label", "weight"]
output:
  base_dir: "{out}/{{cohort}}/model_results"
  predictions_dir: "predictions_parquet_catboost"
"""


def _importances(text):
    rows = [ln.split(",") for ln in text.strip().splitlines()[1:]]
    return {f.strip('"'): float(v) for f, v in rows}


def test_ems_train(run_r, repo_root, tmp_path):
    fix = repo_root / FIX
    cfg = tmp_path / "data_config.yaml"
    cfg.write_text(_data_config(fix, tmp_path / "output"))
    p = run_r(repo_root / WORKER, ["--step", "train", "--cohort", "protocol_example",
                                   "--chromosome", "2", "--data-config", cfg,
                                   "--model-config", fix / "model_config.yaml"])
    assert p.returncode == 0, p.stdout + p.stderr
    res = tmp_path / "output" / "protocol_example" / "model_results"

    # feature importances reproduce the Python reference (same libcatboost, deterministic)
    got = _importances((res / "features_importance_model5_chr_chr2_NPR_1.csv").read_text())
    exp = _importances((fix / "expected" / "features_importance_model5_chr_chr2_NPR_1.csv").read_text())
    assert set(got) == set(exp)
    for feat, val in exp.items():
        assert abs(got[feat] - val) < 1e-4, f"{feat}: {got[feat]} vs {val}"

    # test-set AP/AUC match the reference metrics
    summary = json.loads((res / "model_5_summary_chr_chr2_NPR_1.json").read_text())
    s = summary["CatBoost"]["standard_subset_weighted"]
    assert abs(s["AP_test"] - REF_AP) < 1e-4
    assert abs(s["AUC_test"] - REF_AUC) < 1e-4

    # native model + predictions written
    assert (res / "model_standard_subset_weighted_chr_chr2_NPR_1.cbm").exists()
    pred = (res / "predictions_parquet_catboost" / "predictions_weighted_model_chr2.tsv").read_text().splitlines()
    assert pred[0].split("\t")[-3:] == [
        "standard_subset_weighted_pred_prob", "standard_subset_weighted_pred_label", "actual_label"]
    assert len(pred) - 1 == 200                                   # chr1 test rows


def test_ems_train_via_sos(run_sos, repo_root, tmp_path):
    # Exercise the SoS wiring: `sos run ems_training.ipynb train` -> R worker -> outputs.
    fix = repo_root / FIX
    cfg = tmp_path / "data_config.yaml"
    cfg.write_text(_data_config(fix, tmp_path / "output"))
    p = run_sos(repo_root / "pipeline/ems_training.ipynb", "train", {
        "cohort": "protocol_example", "chromosome": "2", "data_config": cfg,
        "model_config": fix / "model_config.yaml", "cwd": tmp_path,
        "modular_script_dir": repo_root / "code/script"})
    assert p.returncode == 0, p.stdout + p.stderr
    assert (tmp_path / "protocol_example_chr2_scEEMS_model.done").exists()
    res = tmp_path / "output" / "protocol_example" / "model_results"
    got = _importances((res / "features_importance_model5_chr_chr2_NPR_1.csv").read_text())
    exp = _importances((fix / "expected" / "features_importance_model5_chr_chr2_NPR_1.csv").read_text())
    for feat, val in exp.items():
        assert abs(got[feat] - val) < 1e-4


def test_ems_predict_stub(run_r, repo_root, tmp_path):
    # predict is intentionally unimplemented (mirrors the Python); it must exit 0
    # with the "not yet implemented" message rather than erroring.
    p = run_r(repo_root / WORKER, ["--step", "predict", "--cohort", "protocol_example",
                                   "--chromosome", "2", "--model-path", tmp_path / "m.cbm",
                                   "--data-config", tmp_path / "none.yaml"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert "not yet implemented" in p.stdout.lower()
