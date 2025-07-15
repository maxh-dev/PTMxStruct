# builtin
import json
import os
import socket
import re
from itertools import groupby
import urllib.request
import random
import logging
import ssl
import tempfile
import requests

# external
import numba
import numpy as np
import pandas as pd
import tqdm
import h5py
import statsmodels.stats.multitest
from statsmodels.stats.multitest import multipletests
import Bio.PDB.MMCIF2Dict
import scipy.stats
import sys

import plotly.express as px

from structuremap.plotting import plot_enrichment, scale_pvals

def perform_enrichment_analysis_by_site(df: pd.DataFrame,
                                ptm_types: list,
                                rois: list,
                                quality_cutoffs: list,
                                ptm_site_dict: dict,
                                multiple_testing: bool = True) -> pd.DataFrame:
    # Adapted from structuremap 0.0.10, original function: structuremap.processing.perform_enrichment_analysis
    """
    Get enrichment p-values for selected PTMs acros regions of interest (ROIs).

    Parameters
    ----------
    df : pd.DataFrame
        pd.DataFrame of formatted AlphaFold data across various proteins.
    ptm_types: list
        List of PTM modifications for which to perform the enrichment analysis.
    rois : list
        List of regions of interest (one hot encoded columns in df) for which
        to perform the enrichment analysis.
    quality_cutoffs : list
        List of quality cutoffs (AlphaFold pLDDT values) to filter for.
    ptm_site_dict : dict
        Dictionary containing the possible amino acid sites for each PTM.
    multiple_testing : bool
        Bool if multiple hypothesis testing correction should be performed.
        Default is 'True'.

    Returns
    -------
    : pd.DataFrame
        Dataframe reporting p-values for the enrichment of all selected
        ptm_types across selected rois.
    """
    enrichment = []
    for q_cut in quality_cutoffs:
        seq_ann_qcut = df[df.quality >= q_cut]
        for ptm in ptm_types:
            seq_ann_qcut_aa = seq_ann_qcut[seq_ann_qcut.AA.isin(ptm_site_dict[ptm])]
            for roi in rois:
                seq_ann_qcut_aa_roi1 = seq_ann_qcut_aa[roi] == 1
                seq_ann_qcut_aa_roi0 = seq_ann_qcut_aa[roi] == 0
                seq_ann_qcut_aa_ptm1 = seq_ann_qcut_aa[ptm] >= 1 # This line was changed, compared to the originals == 1. 
                seq_ann_qcut_aa_ptm0 = seq_ann_qcut_aa[ptm] == 0
                n_ptm_in_roi = seq_ann_qcut_aa[seq_ann_qcut_aa_roi1 & seq_ann_qcut_aa_ptm1].shape[0]
                n_ptm_not_in_roi = seq_ann_qcut_aa[seq_ann_qcut_aa_roi0 & seq_ann_qcut_aa_ptm1].shape[0]
                n_naked_in_roi = seq_ann_qcut_aa[seq_ann_qcut_aa_roi1 & seq_ann_qcut_aa_ptm0].shape[0]
                n_naked_not_in_roi = seq_ann_qcut_aa[seq_ann_qcut_aa_roi0 & seq_ann_qcut_aa_ptm0].shape[0]
                fisher_table = np.array([[n_ptm_in_roi, n_naked_in_roi], [n_ptm_not_in_roi, n_naked_not_in_roi]])
                oddsr, p = scipy.stats.fisher_exact(fisher_table,
                                                    alternative='two-sided')
                res = pd.DataFrame({'quality_cutoff': [q_cut],
                                    'ptm': [ptm],
                                    'roi': [roi],
                                    'n_aa_ptm':  np.sum(seq_ann_qcut_aa_ptm1),
                                    'n_aa_roi':  np.sum(seq_ann_qcut_aa_roi1),
                                    'n_ptm_in_roi': n_ptm_in_roi,
                                    'n_ptm_not_in_roi': n_ptm_not_in_roi,
                                    'n_naked_in_roi': n_naked_in_roi,
                                    'n_naked_not_in_roi': n_naked_not_in_roi,
                                    'oddsr': [oddsr],
                                    'p': [p]})
                enrichment.append(res)
    enrichment_df = pd.concat(enrichment)
    if multiple_testing:
        enrichment_df['p_adj_bf'] = statsmodels.stats.multitest.multipletests(
            pvals=enrichment_df.p, alpha=0.01, method='bonferroni')[1]
        enrichment_df['p_adj_bh'] = statsmodels.stats.multitest.multipletests(
            pvals=enrichment_df.p, alpha=0.01, method='fdr_bh')[1]
    return(enrichment_df)


def structuremap_plot_enrichment(
    data: pd.DataFrame,
    ptm_select: list = None,
    roi_select: list = None,
    plot_width: int = None,
    plot_height: int = None,
):
    plot_enrichment(data, ptm_select, roi_select, plot_width, plot_height)


def plot_enrichment(
    data: pd.DataFrame,
    ptm_select: list = None,
    roi_select: list = None,
    plot_width: int = None,
    plot_height: int = None,
    facet_spacing: float = 0.05,
    plot_title: str = "",
    ptm_rename_dict: dict = None, 
    roi_rename_dict: dict = None,
    y_range: tuple = None
):
    # Adapted from structuremap 0.0.10, original function: structuremap.plotting.plot_enrichment
    df = data.copy(deep=True)
    df['ptm'] = [re.sub('_', ' ', p) for p in df['ptm']]

    category_dict = {}

    if ptm_select is not None:
        ptm_select = [re.sub('_', ' ', p) for p in ptm_select]
        df = df[df.ptm.isin(ptm_select)]
        category_dict['ptm'] = ptm_select

    if roi_select is not None:
        df = df[df.roi.isin(roi_select)]
        category_dict['roi'] = roi_select

    df['log_odds_ratio'] = np.log(df['oddsr'])
    df['neg_log_adj_p'] = -np.log10(df.p_adj_bh)
    df['neg_log_adj_p_round'] = scale_pvals(df.neg_log_adj_p)

    category_dict['neg_log_adj_p_round'] = list(reversed([
        '> 1000', '> 100', '> 50', '> 10', '> 5', '> 2', '> 0']))

    color_dict = {
        '> 1000': 'rgb(120,0,0)',
        '> 100': 'rgb(177, 63, 100)',
        '> 50': 'rgb(221, 104, 108)',
        '> 10': 'rgb(241, 156, 124)',
        '> 5': 'rgb(245, 183, 142)',
        '> 2': 'rgb(246, 210, 169)',
        '> 0': 'grey'
    }

    if not ptm_rename_dict == None:
        df['ptm'] = df['ptm'].replace(ptm_rename_dict)

    fig = px.bar(
        df,
        x='ptm',
        y='log_odds_ratio',
        labels={
            'ptm': 'PTM',
            'log_odds_ratio': 'log odds ratio',
            'neg_log_adj_p_round': '-log10 (adj. p-value)'
        },
        color='neg_log_adj_p_round',
        facet_col='roi',
        facet_col_spacing=facet_spacing,
        hover_data=['oddsr', 'p_adj_bh'],
        category_orders=category_dict,
        color_discrete_map=color_dict,
        template="simple_white",
        title=plot_title,
    )
    if not roi_rename_dict == None:
        fig.for_each_annotation(lambda a: a.update(text=roi_rename_dict.get(a.text.split('=')[-1], a.text)))

    # Dynamic plot sizing
    if plot_width is None:
        p_width = max(500, 400 + len(df['ptm'].unique()) * 70 + len(df['roi'].unique()) * 70)
    elif plot_width > 0:
        p_width = plot_width
    else:
        raise ValueError("plot_width must be a positive integer.")

    if plot_height is None:
        p_height = 500
    elif plot_height > 0:
        p_height = plot_height
    else:
        raise ValueError("plot_height must be a positive integer.")

    if y_range is not None:
        fig.update_yaxes(range=y_range)
    fig.update_layout(
        autosize=False,
        width=p_width,
        height=p_height,
        margin=dict(
            autoexpand=False,
            l=100,
            r=220,
            b=150,
            t=80,
        ),
        legend=dict(
            title_font=dict(size=12),
            font=dict(size=11),
            tracegroupgap=5,
        )
    )


    config = {'toImageButtonOptions': {
        'format': 'svg', 'filename': 'structure ptm enrichment'}}

    return fig.show(config=config)


def enrichment_in_region(df, ptm_type, roi, group1, group2, method_str):
    if method_str == "by_site":
        method = 'count'
    elif method_str == "by_occurence":
        method = 'sum'
    else:
        raise ValueError(f"Method {method} not vaild.")
    seletced_cols = []
    seletced_cols.append(f"{group1}_{ptm_type}")
    seletced_cols.append(f"{group2}_{ptm_type}")

    all_cols = ["protein_id", "AA", "position"]
    all_cols.append(roi)
    all_cols.extend(seletced_cols)

    df_cols_removed = df[all_cols]
    df_zero_rows_removed = df_cols_removed[~(df_cols_removed[seletced_cols].eq(0).all(axis=1))].copy()

    aa_in_roi = df_zero_rows_removed[roi] == 1
    aa_not_in_roi = df_zero_rows_removed[roi] == 0
    group1_aa_ptm = df_zero_rows_removed[f"{group1}_{ptm_type}"] >= 1
    group2_aa_ptm = df_zero_rows_removed[f"{group2}_{ptm_type}"] >= 1

    group1_ptm_in_roi = df_zero_rows_removed[aa_in_roi & group1_aa_ptm][f"{group1}_{ptm_type}"].agg(method)
    group2_ptm_in_roi = df_zero_rows_removed[aa_in_roi & group2_aa_ptm][f"{group2}_{ptm_type}"].agg(method)
    group1_ptm_not_in_roi = df_zero_rows_removed[aa_not_in_roi & group1_aa_ptm][f"{group1}_{ptm_type}"].agg(method)
    group2_ptm_not_in_roi = df_zero_rows_removed[aa_not_in_roi & group2_aa_ptm][f"{group2}_{ptm_type}"].agg(method)

    fisher_table = np.array(
        [[group1_ptm_in_roi, group2_ptm_in_roi],
         [group1_ptm_not_in_roi, group2_ptm_not_in_roi]]
    )
    oddsr, p = scipy.stats.fisher_exact(fisher_table, alternative='two-sided')

    return oddsr, p, str(fisher_table.tolist())

def enrichment_across_regions(df, ptm_types, rois, group1, group2, method):
    result_dict = {
        "ptm": [],
        "roi": [],
        "group1": [],
        "group2": [],
        "table": [],
        "p_val": [],
        "oddsr": [],
        "p_adj_bh": None,
        "p_adj_bf": None
    }
    
    for ptm_type in ptm_types:
        for roi in rois:
            oddsr, p, fisher_table_str = enrichment_in_region(df, ptm_type, roi, group1, group2, method)
            result_dict["ptm"].append(ptm_type)
            result_dict["roi"].append(roi)
            result_dict["group1"].append(group1)
            result_dict["group2"].append(group2)
            result_dict["table"].append(fisher_table_str)
            result_dict["p_val"].append(p)
            result_dict["oddsr"].append(oddsr)

    _, p_adj_fdr_bh, _, _ = multipletests(result_dict["p_val"], method="fdr_bh")
    _, p_adj_bonferroni , _, _ = multipletests(result_dict["p_val"], method="bonferroni")
    result_dict["p_adj_bh"] = p_adj_fdr_bh
    result_dict["p_adj_bf"] =p_adj_bonferroni


    result_df = pd.DataFrame.from_dict(result_dict)
    return result_df

