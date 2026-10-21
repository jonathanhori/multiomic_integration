# Real-data analyses

Analyses of real (non-simulated) datasets live here, one folder per analysis.
Raw inputs stay in `data/`; every output of an analysis (fitted models, tables,
figures) is written inside its own folder, never to `sim/`.

Each analysis folder contains:

- a notebook (`.qmd`) that runs the whole analysis — data processing, model
  training, prediction, interpretation — and reads the comparison results;
- scripts for systematic evaluations (e.g. comparison with other methods over
  several train/test splits), which the notebook reads;
- `results/` — tables and fitted models;
- `figures/` — plots.

| Folder | Data | Question |
|---|---|---|
| `tcga_coadread/` | TCGA colon and rectal adenocarcinoma (PanCancer Atlas 2018): gene expression + TSS200 promoter methylation | Predict tumour mutational burden; interpret the shared expression–methylation factors |

To render a notebook (from the `multiomic` conda environment):

```bash
cd analysis/tcga_coadread && quarto render tcga_coadread.qmd
```
