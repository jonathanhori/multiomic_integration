"""
Write, for each train/test split used by compare_methods.py, the gene names of
the model's features (after preprocessing) and the training/test sample IDs, so
factors fitted on different splits can be lined up by gene and joined to
clinical data.

Outputs: results/splits/seed<k>/genes_view<l>.txt, train_samples.txt, test_samples.txt

usage: python export_split_info.py
"""
import os
import sys
import warnings

import torch

sys.path.insert(1, os.path.expanduser("~/multiomic_integration/python"))
from coadread_data import load_coadread, make_view_list, make_feature_names
from run_methods import process_data
from compare_methods import SPLIT_SEEDS, VIEWS, N_RNA, N_METH, K, RESULTS

warnings.filterwarnings("ignore")

for seed in SPLIT_SEEDS:
    data = load_coadread(top_rna_genes=N_RNA, top_meth_genes=N_METH, log1p_tmb=True,
                         train_frac=0.8, random_seed=seed, preprocess_on_train_only=True)
    real = lambda split, y: {"X_l": make_view_list(data, VIEWS, split=split), "y": y,
                             "K": torch.tensor(K), "Z": None}
    _, _, dp = process_data(real("train", data["y"]), real("test", data["y_test"]), model_type="shared")
    out = os.path.join(RESULTS, "splits", f"seed{seed}")
    os.makedirs(out, exist_ok=True)
    for l, (genes, keep) in enumerate(zip(make_feature_names(data, VIEWS), dp["X_l_list_column_filters"])):
        with open(os.path.join(out, f"genes_view{l}.txt"), "w") as f:
            f.write("\n".join(g for g, k in zip(genes, keep.tolist()) if k))
    for split in ("train", "test"):
        with open(os.path.join(out, f"{split}_samples.txt"), "w") as f:
            f.write("\n".join(data[f"sample_ids_{split}"]))
    print(f"seed {seed}: done", flush=True)
