# --- Dependencies ---
library(TCGAretriever)
library(data.table)  # fread
library(dplyr)
library(tidyr)
# (ggplot2/reshape2 not used below; load them only if you plot later)
# library(ggplot2)
# library(reshape2)

# --- Load gene panel definitions (Cancer Gene List TSV) ---
CGL <- data.table::fread("data/tcga/cancerGeneList.tsv", sep = "\t")
q_genes_MSK   <- CGL[`MSK-IMPACT` == "Yes", `Hugo Symbol`]
q_genes_F1CDX <- CGL[`FOUNDATION ONE` == "Yes", `Hugo Symbol`]

# --- TCGA studies (published only) ---
all_studies  <- get_cancer_studies()
keep         <- grepl("tcga_pub$", all_studies[,"studyId"])
tcga_studies <- all_studies[keep, ]

# Drop studies without suitable molecular data (as noted)
drop_ids   <- c("lgggbm_tcga_pub", "stes_tcga_pub", "pcpg_tcga_pub")
all_csids  <- setdiff(tcga_studies$studyId, drop_ids)

# --- Helper to download/assemble per-study TMB + gene expression ---
get_TCGA_TMB_datasets <- function(all_csids, q_genes) {
  all_TCGA_TMB_data <- vector("list", length(all_csids))
  names(all_TCGA_TMB_data) <- all_csids

  for (source_csid in all_csids) {
    # Available genetic profiles
    source_gen <- get_genetic_profiles(csid = source_csid)

    # Case list (sequenced)
    q_cases <- paste0(source_csid, "_sequenced")

    # Expression profile id (try RNA-Seq V2 first, fallback to older mrna)
    rna_v2_id <- paste0(source_csid, "_rna_seq_v2_mrna")
    rna_id    <- paste0(source_csid, "_mrna")
    rna_pfa   <- if (rna_v2_id %in% source_gen$molecularProfileId) rna_v2_id else rna_id

    # Clinical
    source_cli <- get_clinical_data(csid = source_csid, case_list_id = q_cases)

    # Expression (wide)
    if (length(q_genes) <= 500) {
      source_RNA_wide <- get_molecular_data(
        case_list_id = q_cases,
        gprofile_id  = rna_pfa,
        glist        = q_genes
      ) %>%
        pivot_longer(-c("entrezGeneId", "hugoGeneSymbol", "type"),
                     names_to = "sampleId", values_to = "exp") %>%
        select(-entrezGeneId, -type) %>%
        pivot_wider(names_from = hugoGeneSymbol, values_from = exp)
    } else {
      source_RNA_wide <- fetch_all_tcgadata(
        case_list_id = q_cases,
        gprofile_id  = rna_pfa
      ) %>%
        filter(hugoGeneSymbol %in% q_genes) %>%
        pivot_longer(-c("entrezGeneId", "hugoGeneSymbol", "type"),
                     names_to = "sampleId", values_to = "exp") %>%
        select(-entrezGeneId, -type) %>%
        pivot_wider(names_from = hugoGeneSymbol, values_from = exp)
    }

    # Merge clinical + expression by sampleId
    source_RNA_final <- left_join(source_cli, source_RNA_wide, by = "sampleId")

    all_TCGA_TMB_data[[source_csid]] <- source_RNA_final
    message("Finished: ", source_csid)
  }

  # Harmonize on the intersection of columns across studies
  covs              <- lapply(all_TCGA_TMB_data, names)
  intersected_covs  <- Reduce(intersect, covs)
  all_TCGA_TMB_data <- lapply(all_TCGA_TMB_data, \(x) x[, intersected_covs])

  all_TCGA_TMB_data
}

# === Fetch FoundationOne-sized panel across TCGA ===
TMB_F1CDX <- get_TCGA_TMB_datasets(all_csids, q_genes_F1CDX)

# Genes actually present after the intersection step
present_gene_cols <- intersect(q_genes_F1CDX, colnames(TMB_F1CDX[[1]]))

# Keep complete cases of TMB + gene expression only
# (do not rely on a fixed column index; use explicit names)
TMB_F1CDX <- lapply(TMB_F1CDX, function(df) {
  keep_cols <- c("sampleId", "TMB_NONSYNONYMOUS", present_gene_cols)
  keep_cols <- keep_cols[keep_cols %in% names(df)]
  out <- df[, keep_cols, drop = FALSE]
  out[complete.cases(out), , drop = FALSE]
})

# Identify genes with >80% zeros (exclude TMB_NONSYNONYMOUS from this check)
bad_gene_names_list <- lapply(TMB_F1CDX, function(df) {
  gene_cols <- setdiff(colnames(df), c("sampleId", "TMB_NONSYNONYMOUS"))
  if (length(gene_cols) == 0) return(character(0))
  zero_frac <- vapply(df[gene_cols], function(g) mean(g == 0, na.rm = TRUE), numeric(1))
  names(zero_frac[zero_frac > 0.80])
})
bad_gene_names <- Reduce(union, bad_gene_names_list)

# Drop zero-heavy genes (if any)
if (length(bad_gene_names) > 0) {
  TMB_F1CDX <- lapply(TMB_F1CDX, function(df) {
    drop_cols <- intersect(bad_gene_names, colnames(df))
    dplyr::select(df, -dplyr::all_of(drop_cols))
  })
}

# Save to RDS
saveRDS(TMB_F1CDX, file = "TMB_F1CDX_data.rds")
