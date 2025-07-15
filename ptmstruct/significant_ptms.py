import os 
import pandas as pd 
from scipy.stats import shapiro
from scipy import stats
import numpy as np
import scipy.stats as stats
from scipy.stats import power_divergence
from statsmodels.stats.multitest import multipletests
import json
import operator as op
import plotly.express as px
import plotly.graph_objects as go

def compute_significant_ptms(df, ptm_type, groups, sig_thresh, group_total):
    # assums that both groups have the same number of smaples
    seletced_cols = []
    if len(groups) != 2:
        return None
    
    group_trg = None 
    group_ctg = None
    for group in groups:
        seletced_cols.append(f"{group}_{ptm_type}_distinct")
        seletced_cols.append(f"{group}_{ptm_type}")
        if group == "TRG":
            group_trg = f"{group}_{ptm_type}_distinct"
        elif group == "CTG":
            group_ctg = f"{group}_{ptm_type}_distinct"
        else:
            return None 

    all_cols = ["protein_id", "AA", "position"]

    all_cols.extend(seletced_cols)
    df_cols_removed = df[all_cols]
    df_zero_rows_removed = df_cols_removed[~(df_cols_removed[seletced_cols].eq(0).all(axis=1))].copy()
    
    p_values = []
    oddsrs = []
    for index, row in df_zero_rows_removed.iterrows():
        table = [[row[group_trg], row[group_ctg]], 
                [group_total - row[group_trg], group_total - row[group_ctg]]]
        
        oddsr, p = stats.fisher_exact(table, alternative='two-sided')
        p_values.append(p)
        oddsrs.append(oddsr)

    _, p_adj, _, _ = multipletests(p_values, method="fdr_bh")
    
    df_zero_rows_removed[f"{ptm_type}_oddsrs"] = oddsrs
    df_zero_rows_removed[f"{ptm_type}_p_adj"] = p_adj
    df_zero_rows_removed[f"{ptm_type}_p_adj_sig"] = (df_zero_rows_removed[f"{ptm_type}_p_adj"] < sig_thresh).astype(int)

    sorted_df = df_zero_rows_removed.sort_values(by=f"{ptm_type}_p_adj")

    return sorted_df

def merge_significant_ptms_df_with_structure(structure_df, sig_ptm_df, ppse_column):
    ppse_bin_column = f"{ppse_column}_bin"
    if ppse_column not in structure_df.columns:
        raise ValueError(f"Column '{ppse_column}' not found in DataFrame")
    if ppse_bin_column not in structure_df.columns:
        raise ValueError(f"Column '{ppse_column}_bin' not found in DataFrame")
    
    merged_df = sig_ptm_df.merge(structure_df[['protein_id', 'AA', 'position', ppse_column, ppse_bin_column]], 
                                on=['protein_id', 'AA', 'position'], 
                                how='left')
    
    return merged_df, ppse_bin_column


def add_unique_count_to_significant_ptms(merged_df, ppse_bin_column, col1, col2):
    col1_name = f'{col1}_ptm_site_present'
    col2_name = f'{col2}_ptm_site_present'
    diff_col_name = "ptm_site_present_diff"

    merged_df[col1_name] = merged_df[col1].astype(bool)
    merged_df[col2_name] = merged_df[col2].astype(bool)

    ptm_summary_1 = merged_df.groupby(ppse_bin_column)[col1_name].sum().reset_index()
    ptm_summary_2 = merged_df.groupby(ppse_bin_column)[col2_name].sum().reset_index()

    ptm_summary = ptm_summary_1.merge(ptm_summary_2, on=[ppse_bin_column])
    ptm_summary[diff_col_name] = ptm_summary[col1_name] - ptm_summary[col2_name]
    return ptm_summary, col1_name, col2_name, diff_col_name


def add_occurence_count_to_significant_ptms(merged_df, ppse_bin_column, col1, col2):
    col1_name = f"{col1}_ptm_frequency"
    col2_name = f"{col2}_ptm_frequency"
    col1_perc_name = f"{col1_name}_percentage"
    col2_perc_name = f"{col2_name}_percentage"

    diff_col_name = "ptm_frequency_diff"
    diff_perc_col_name = "ptm_frequency_percentage_diff"

    ptm_summary_1 = merged_df.groupby(ppse_bin_column)[col1].sum().reset_index()
    ptm_summary_2 = merged_df.groupby(ppse_bin_column)[col2].sum().reset_index()
    ptm_summary_1[col1_perc_name] = round(ptm_summary_1[col1] / ptm_summary_1[col1].sum() * 100, 2)
    ptm_summary_2[col2_perc_name] = round(ptm_summary_2[col2] / ptm_summary_2[col2].sum() * 100, 2)
    ptm_summary = ptm_summary_1.merge(ptm_summary_2, on=[ppse_bin_column])
    ptm_summary = ptm_summary.rename(columns={
        col1: col1_name,
        col2: col2_name,
    })
    ptm_summary[diff_col_name] = ptm_summary[col1_name] - ptm_summary[col2_name]
    ptm_summary[diff_perc_col_name] = ptm_summary[col1_perc_name] - ptm_summary[col2_perc_name]
    return ptm_summary, col1_name, col2_name, col1_perc_name, col2_perc_name, diff_col_name, diff_perc_col_name


def create_bar_plot(ptm_summary, ppse_bin_column, yaxis_name, bar1_name, bar1_display, bar2_name, bar2_display, title = "", yaxis_range= None):
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=ptm_summary[ppse_bin_column],
        y=ptm_summary[bar1_name],
        name=bar1_display
    ))
    bar_fig.add_trace(go.Bar(
        x=ptm_summary[ppse_bin_column],
        y=ptm_summary[bar2_name],
        name=bar2_display
    ))
    layout_params = {
        'title': title,
        'xaxis_title': f'{ppse_bin_column}',
        'yaxis_title': yaxis_name,
        'barmode': 'group'
    }
    if yaxis_range:
        layout_params['yaxis'] = {'range': yaxis_range}

    bar_fig.update_layout(layout_params)
    bar_fig.show()

def create_line_plot(ptm_summary, ppse_bin_column, diff_name, title = "", yaxis_range= None):
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(
        x=ptm_summary[ppse_bin_column],
        y=ptm_summary[diff_name],
        mode='lines+markers',
        name='Difference'
    ))
    layout_params = {
        'title': title,
        'xaxis_title': f'{ppse_bin_column}',
        'yaxis_title': 'Difference',
        'barmode': 'group'
    }
    if yaxis_range:
        layout_params['yaxis'] = {'range': yaxis_range}

    line_fig.update_layout(layout_params)
    line_fig.show()


def plot_ptm_count_per_accessibility(structure_df, sig_ptm_df, ppse_column, col1, col2, plot_config):
    if plot_config["only_plot_significant"]:
        ptm_type = col1.split("_")[1]
        sig_ptm_df = sig_ptm_df[sig_ptm_df[f"{ptm_type}_p_adj_sig"] == 1]

    if plot_config["only_plot_unique_ptms"]:
        ptm_type = col1.split("_")[1]
        group_trg = f"TRG_{ptm_type}_distinct"
        group_ctg = f"CTG_{ptm_type}_distinct"

        condition_1 = (sig_ptm_df[group_trg] != 0) & (sig_ptm_df[group_ctg] == 0)
        condition_2 = (sig_ptm_df[group_trg] == 0) & (sig_ptm_df[group_ctg] != 0)

        sig_ptm_df = sig_ptm_df[condition_1 | condition_2]

    merged_df, ppse_bin_column = merge_significant_ptms_df_with_structure(structure_df, sig_ptm_df, ppse_column)
    if plot_config["count_method"] == "by_site":
        ptm_summary, col1_name, col2_name, diff_col_name = add_unique_count_to_significant_ptms(merged_df, ppse_bin_column, col1, col2)
    elif plot_config["count_method"] == "by_occurence":
        ptm_summary, col1_name, col2_name, col1_perc_name, col2_perc_name, diff_col_name, diff_perc_col_name = add_occurence_count_to_significant_ptms(merged_df, ppse_bin_column, col1, col2)
    else:
        raise ValueError(f"count_method {plot_config['count_method']} not valid.")
    

    if plot_config["plot_type"] == "count" and plot_config["display_mode"] == "total":
        create_bar_plot(
            ptm_summary,
            ppse_bin_column,
            "Count",
            col1_name, col1_name,
            col2_name, col2_name,
            title = f"PTMs present <br><span style='font-size:12px; color:gray'> ({col1_name} count vs {col2_name} count) by {ppse_bin_column}",
            yaxis_range=plot_config["plot_range"]
        )
    elif plot_config["plot_type"] == "count" and plot_config["display_mode"] == "percentage":
        create_bar_plot(
            ptm_summary,
            ppse_bin_column,
            "Percentage",
            col1_perc_name, col1_perc_name,
            col2_perc_name, col2_perc_name,
            title = f"PTMs present <br><span style='font-size:12px; color:gray'> ({col1_name} percentage vs {col2_name} percentage) by {ppse_bin_column}",
            yaxis_range=plot_config["plot_range"]
        )
    elif plot_config["plot_type"] == "difference" and plot_config["display_mode"] == "total":
        create_line_plot(
            ptm_summary,
            ppse_bin_column,
            diff_col_name,
            title = f"Difference in PTMs present <br><span style='font-size:12px; color:gray'> ({col1_name} count vs {col2_name} count) by {ppse_bin_column}",
            yaxis_range=plot_config["plot_range"]
        ) 
    elif plot_config["plot_type"] == "difference" and plot_config["display_mode"] == "percentage":
        create_line_plot(
            ptm_summary,
            ppse_bin_column,
            diff_perc_col_name,
            title = f"Difference in PTMs present <br><span style='font-size:12px; color:gray'> ({col1_name} percentage vs {col2_name} percentage) by {ppse_bin_column}",
            yaxis_range=plot_config["plot_range"]
        )
    else:
        raise ValueError(f"plot_type {plot_config['plot_type']} not valid.")

    return ptm_summary


def filter_significant_ptm_df_by_ppse(structure_df, sig_ptm_df, ptm_type, ppse_column, operator_str, threshold):
    ops = {
        '>': op.gt,
        '>=': op.ge,
        '<': op.lt,
        '<=': op.le,
        '==': op.eq,
        '!=': op.ne
    }
    if operator_str not in ops:
        raise ValueError(f"Unsupported operator: {operator_str}")
    
    merged_df, ppse_bin_column = merge_significant_ptms_df_with_structure(structure_df, sig_ptm_df, ppse_column)

    condition = ops[operator_str](merged_df[ppse_column], threshold)
    aa_low_p = merged_df[f"{ptm_type}_p_adj_sig"] == 1

    filtered_df = merged_df[condition & aa_low_p]
    sorted_df = filtered_df.sort_values(by=f"{ptm_type}_p_adj")
    return sorted_df