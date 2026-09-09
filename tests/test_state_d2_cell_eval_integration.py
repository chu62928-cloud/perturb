import numpy as np
import pandas as pd
import pytest


def test_cell_eval_082_perfect_synthetic_matches_contract(tmp_path):
    ad = pytest.importorskip("anndata")
    pytest.importorskip("cell_eval")
    from cell_eval import MetricsEvaluator

    genes = ["A", "B", "C", "D"]
    perturbations = ["A", "B"]
    target = np.asarray(
        [[2.0, 1.0, 0.0, 0.0], [0.0, 0.0, 3.0, 1.0]], dtype=np.float32
    )
    control = np.zeros((3, len(genes)), dtype=np.float32)
    cells = np.concatenate([np.repeat(target[:, None, :], 3, axis=1).reshape(-1, 4), control], axis=0)
    labels = np.concatenate([np.repeat(perturbations, 3), np.repeat("NTC", 3)])
    obs = {"perturbation": labels}
    real = ad.AnnData(X=cells.copy(), obs=obs, var=pd.DataFrame(index=genes))
    pred = ad.AnnData(X=cells.copy(), obs=obs, var=pd.DataFrame(index=genes))
    real.var_names = genes
    pred.var_names = genes
    evaluator = MetricsEvaluator(
        adata_pred=pred,
        adata_real=real,
        control_pert="NTC",
        pert_col="perturbation",
        outdir=str(tmp_path),
        prefix="synthetic",
        pdex_kwargs={"exp_post_agg": True, "is_log1p": True},
        num_threads=1,
    )
    rows, _aggregate = evaluator.compute(
        profile="full",
        skip_metrics=["pearson_edistance", "clustering_agreement"],
        write_csv=False,
        break_on_error=True,
    )
    records = rows.to_dicts()
    expected = {
        "pearson_delta", "pr_auc", "de_spearman_sig", "de_spearman_lfc_sig",
        "overlap_at_N", "mae", "mse", "mae_delta", "mse_delta",
        "discrimination_score_l1", "discrimination_score_l2", "discrimination_score_cosine",
    }
    assert records
    assert expected.issubset(records[0])
    assert [row["pearson_delta"] for row in records] == pytest.approx([1.0, 1.0])
    assert [row["discrimination_score_l1"] for row in records] == pytest.approx([1.0, 1.0])
