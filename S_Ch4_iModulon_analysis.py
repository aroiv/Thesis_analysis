"""
Port of TM_analysis_clean.m to Python.

Inputs (expected in the same folder as this script, or adjust paths):
- S.csv                       (gene x iModulon matrix; first col Geneid)
- log_tpm_norm.csv            (gene x compendium expression; first col Geneid)
- TL_data_processed_463_7942.xlsx (must contain sheets TPM_norm_mapped_WT and TPM_norm_mapped_BRT)

Outputs:
- iModulon_results.xlsx
- iModulon_activities_centered_WT_T1.csv
- iModulon_differential_sig_filtered.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import erf, sqrt

MOD_NAMES = [
    'ccm-2','PSII','ribosome-1','SG_1','SG_2','kaiABC','RpaB','unchar_1','unchar_2','SG_3','SG_4',
    'unchar_3','anL','ccm-1','sigA2/rpoD2','livJHMGF','Cytc_oxidases','groELS','iron','OXPHOS','unchar_4',
    'NtcA-1','Biofilm-1','prophage','unchar_5','CysR','HSR','IdiB','phosphate','SG_5','SG_6','membrane',
    'Biofilm-2','RpaA','competence','SG_7','phototaxis','oxi_stress_tolerance','Photosystems','SG_8',
    'NtcA-2','SG_9','unchar_6','SG_10','RpaC','unchar_7','SG_11','sps','unchar_8','SG_12','SG_13',
    'ribosome-2','unchar_9','unchar_10','sufR','unchar_11','SG_14'
]

CONDITIONS = ['WT_T1','WT_T2','WT_T3','WT_T4','BRT_T1','BRT_T2','BRT_T3','BRT_T4']
LAST_24_NAMES = (
    [f"WT_T{t}_{r}" for t in range(1,5) for r in range(1,4)]
    + [f"BRT_T{t}_{r}" for t in range(1,5) for r in range(1,4)]
)

COMPARISONS = [
    (1,2),(1,3),(1,4),(2,3),(2,4),(3,4),  # WT
    (5,6),(5,7),(5,8),(6,7),(6,8),(7,8),  # BRT
    (1,5),(2,6),(3,7),(4,8)               # WT vs BRT same T
]
SAMPLE_MAP = {1:'WT_T1',2:'WT_T2',3:'WT_T3',4:'WT_T4',5:'BRT_T1',6:'BRT_T2',7:'BRT_T3',8:'BRT_T4'}


def norm_cdf(z: float) -> float:
    return 0.5*(1.0 + erf(z/sqrt(2.0)))


def lognorm_sf(x, mu: float, sigma: float):
    """Survival function (1-CDF) for lognormal with ln-params mu, sigma."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    mask = x > 0
    z = (np.log(x[mask]) - mu) / sigma
    out[mask] = 1.0 - np.vectorize(norm_cdf)(z)
    return out


def benjamini_hochberg(pvals_flat):
    p = np.asarray(pvals_flat, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n+1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    q_out = np.empty_like(q)
    q_out[order] = q
    return q_out


def main(
    s_csv="S.csv",
    log_tpm_csv="log_tpm_norm.csv",
    xlsx="TL_data_processed_463_7942.xlsx",
    out_prefix="iModulon_TM_analysis_results"
):
    s_csv = Path(s_csv)
    log_tpm_csv = Path(log_tpm_csv)
    xlsx = Path(xlsx)

    # Load matrices
    S_df = pd.read_csv(s_csv)
    log_df = pd.read_csv(log_tpm_csv)

    wt_mapped = pd.read_excel(xlsx, sheet_name="TPM_norm_mapped_WT")
    brt_mapped = pd.read_excel(xlsx, sheet_name="TPM_norm_mapped_BRT")

    # Sanity: gene ordering must match
    genes = S_df["Geneid"].astype(str).values
    assert np.array_equal(genes, log_df["Geneid"].astype(str).values), "Gene order mismatch: S vs log_tpm"
    assert np.array_equal(genes, wt_mapped["Geneid"].astype(str).values), "Gene order mismatch: S vs WT mapped"
    assert np.array_equal(genes, brt_mapped["Geneid"].astype(str).values), "Gene order mismatch: S vs BRT mapped"

    S = S_df.drop(columns=["Geneid"]).to_numpy(float)                # genes x 57
    X_comp = log_df.drop(columns=["Geneid"]).to_numpy(float)         # genes x N
    X_wt = wt_mapped.drop(columns=["Geneid"]).to_numpy(float)        # genes x 12
    X_brt = brt_mapped.drop(columns=["Geneid"]).to_numpy(float)      # genes x 12
    X_exp = np.concatenate([X_wt, X_brt], axis=1)                    # genes x 24
    X = np.concatenate([X_comp, X_exp], axis=1)                      # genes x (N+24)

    # Activities
    A = np.linalg.pinv(S) @ X                                        # 57 x (N+24)
    A_exp = A[:, -24:]                                               # 57 x 24
    A_exp_df = pd.DataFrame(A_exp, index=MOD_NAMES, columns=LAST_24_NAMES)

    # Center to WT_T1 mean (optional but often expected)
    A_centered = A_exp - A_exp[:, :3].mean(axis=1, keepdims=True)
    A_centered_df = pd.DataFrame(A_centered, index=MOD_NAMES, columns=LAST_24_NAMES)

    # Condition stats
    cond_cols = {c: [n for n in LAST_24_NAMES if n.startswith(c+"_")] for c in CONDITIONS}
    avg_df = pd.DataFrame({c: A_exp_df[cols].mean(axis=1) for c, cols in cond_cols.items()}, index=MOD_NAMES)
    sd_df  = pd.DataFrame({c: A_exp_df[cols].std(axis=1, ddof=1) for c, cols in cond_cols.items()}, index=MOD_NAMES)

    avg_cent_df = pd.DataFrame({c: A_centered_df[cols].mean(axis=1) for c, cols in cond_cols.items()}, index=MOD_NAMES)
    sd_cent_df  = pd.DataFrame({c: A_centered_df[cols].std(axis=1, ddof=1) for c, cols in cond_cols.items()}, index=MOD_NAMES)

    # Background noise distribution from within-triplicate diffs
    diffs = []
    for c in CONDITIONS:
        r1, r2, r3 = cond_cols[c]
        v1, v2, v3 = A_exp_df[r1].values, A_exp_df[r2].values, A_exp_df[r3].values
        diffs.append(np.abs(v2 - v1))
        diffs.append(np.abs(v3 - v1))
        diffs.append(np.abs(v3 - v2))
    diffs = np.concatenate(diffs)
    mu = np.log(diffs).mean()
    sigma = np.log(diffs).std(ddof=0)

    # Comparisons (avg diffs + pvals via lognormal tail prob)
    dif_cols = [f"{SAMPLE_MAP[a]}_vs_{SAMPLE_MAP[b]}_diff" for a,b in COMPARISONS]
    p_cols   = [f"{SAMPLE_MAP[a]}_vs_{SAMPLE_MAP[b]}_pval" for a,b in COMPARISONS]

    avg_dif = pd.DataFrame(index=MOD_NAMES, columns=dif_cols, dtype=float)
    pvals   = pd.DataFrame(index=MOD_NAMES, columns=p_cols, dtype=float)

    for (a,b), dcol, pcol in zip(COMPARISONS, dif_cols, p_cols):
        ca, cb = SAMPLE_MAP[a], SAMPLE_MAP[b]
        dif = avg_df[cb] - avg_df[ca]
        avg_dif[dcol] = dif
        pvals[pcol] = lognorm_sf(np.abs(dif.values), mu, sigma)

    # BH q-values (across all modulons and comparisons)
    q = benjamini_hochberg(pvals.to_numpy().ravel()).reshape(pvals.shape)
    qvals = pd.DataFrame(q, index=MOD_NAMES, columns=p_cols)

    # Filter significant diffs: if none survive FDR, fall back to p<=0.05 (matches MATLAB logic)
    sig = avg_dif.copy()
    if (qvals <= 0.05).any().any():
        for dcol, pcol in zip(dif_cols, p_cols):
            sig.loc[qvals[pcol] > 0.05, dcol] = 0.0
    else:
        for dcol, pcol in zip(dif_cols, p_cols):
            sig.loc[pvals[pcol] > 0.05, dcol] = 0.0

    # Outputs
    out_xlsx = Path(f"{out_prefix}.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        A_exp_df.to_excel(w, sheet_name="activities_last24")
        A_centered_df.to_excel(w, sheet_name="activities_centered_WT_T1")
        avg_df.to_excel(w, sheet_name="avg_by_condition")
        sd_df.to_excel(w, sheet_name="sd_by_condition")
        avg_cent_df.to_excel(w, sheet_name="avg_centered_by_condition")
        sd_cent_df.to_excel(w, sheet_name="sd_centered_by_condition")
        avg_dif.to_excel(w, sheet_name="diff_avg")
        pvals.to_excel(w, sheet_name="diff_pvals")
        qvals.to_excel(w, sheet_name="diff_qvals")
        sig.to_excel(w, sheet_name="diff_sig_filtered")
        pd.DataFrame({"mu":[mu],"sigma":[sigma],"n_diffs":[diffs.size]}).to_excel(w, sheet_name="noise_model", index=False)

    A_centered_df.to_csv("iModulon_activities_centered_WT_T1.csv")
    sig.to_csv("iModulon_differential_sig_filtered.csv")

    # Heatmap of condition means (centered)
    fig, ax = plt.subplots(figsize=(10, 14))
    im = ax.imshow(avg_cent_df.values, aspect="auto")
    ax.set_yticks(np.arange(len(MOD_NAMES)))
    ax.set_yticklabels(MOD_NAMES, fontsize=7)
    ax.set_xticks(np.arange(len(CONDITIONS)))
    ax.set_xticklabels(CONDITIONS, rotation=45, ha="right")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Activity (centered to WT_T1 mean)")
    ax.set_title("iModulon activities (condition means), centered to WT_T1")
    fig.tight_layout()
    fig.savefig("iModulon_heatmap_avg_centered.png", dpi=200)
    plt.close(fig)

    print(f"Wrote: {out_xlsx.resolve()}")
    print("Wrote: iModulon_activities_centered_WT_T1.csv")
    print("Wrote: iModulon_differential_sig_filtered.csv")
    print("Wrote: iModulon_heatmap_avg_centered.png")


if __name__ == "__main__":
    main()
