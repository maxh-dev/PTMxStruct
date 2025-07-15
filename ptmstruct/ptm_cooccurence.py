import random
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import json
from tqdm import tqdm
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import plotly.express as px
import numpy as np
import re


def apply_rus_to_ptms_for_proteins_df(ptms_for_proteins_df, num_proteins, expected_group_size):
    sample_group = ptms_for_proteins_df[['Sample', 'Group']].drop_duplicates()
    rus = RandomUnderSampler(random_state=42)
    X = sample_group['Sample'].values.reshape(-1, 1)
    y = sample_group['Group']
    X_res, y_res = rus.fit_resample(X, y)
    balanced_df = pd.DataFrame({
        'Sample': X_res.flatten(),
        'Group': y_res
    })
    balanced_df['Group'].value_counts()

    sampled_df = ptms_for_proteins_df.merge(balanced_df, on=["Sample", "Group"], how="right")

    if len(sampled_df) != num_proteins * (expected_group_size*2):
        raise ValueError("Error in sampling process")

    return sampled_df


def find_frequent_ptm_sets(df, group_name, min_support=0.2, method="apriori"):
    frequent_sets_per_protein = {}

    for protein in tqdm(df["Protein"].unique(), desc=f"Finding frequent ptm sets for {group_name}"):
        group_df = df[(df["Protein"] == protein) & (df["Group"] == group_name)]
        transactions = group_df["PTMs"].tolist()

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        ptm_df = pd.DataFrame(te_ary, columns=te.columns_)

        if method == "apriori":
            frequent_itemsets = apriori(ptm_df, min_support=min_support, use_colnames=True)
        elif method == "fpgrowth":
            frequent_itemsets = fpgrowth(ptm_df, min_support=min_support, use_colnames=True)
        else:
            raise ValueError(f"Method {method} is not valid.")

        frequent_sets_per_protein[protein] = frequent_itemsets

    return frequent_sets_per_protein


def test_significance_of_ptm_set(comparison_df, total_cancer_samples, total_control_samples):
    p_values = []
    
    for index, row in comparison_df.iterrows():
        ptm_set = row["itemsets"]
        cancer_support = row["TRG support"] * total_cancer_samples
        control_support = row["CTG support"] * total_control_samples
        
        cancer_absent = total_cancer_samples - cancer_support
        control_absent = total_control_samples - control_support
        
        contingency_table = [[cancer_support, control_support],
                             [cancer_absent, control_absent]]
        
        _, p_value = fisher_exact(contingency_table)
        p_values.append(p_value)
    
    comparison_df["p-value"] = p_values
    comparison_df["adjusted p-value"] = multipletests(comparison_df["p-value"], method="fdr_bh")[1]
    
    return comparison_df


def test_significance_of_frequent_ptm_sets(sampled_df, trg_cooccurrence, ctg_cooccurrence):
    final_comparison_df = [] 
    for protein in tqdm(sampled_df["Protein"].unique(), desc=f"Calculating log-fc and p-vals"):
        comparison_df = pd.merge(trg_cooccurrence[protein], ctg_cooccurrence[protein], on="itemsets", how="outer").fillna(0)
        if comparison_df.empty:
            continue
        comparison_df.rename(columns={'support_x': 'TRG support', 'support_y': 'CTG support'}, inplace=True)
        comparison_df["logfc"] = np.log2(comparison_df["TRG support"] + 1e-6) - np.log2(comparison_df["CTG support"] + 1e-6)
        comparison_df = test_significance_of_ptm_set(comparison_df, 60, 60)

        comparison_df["Protein"] = protein
        column_order = ["Protein"] + [col for col in comparison_df.columns if col != "Protein"]
        comparison_df = comparison_df[column_order]

        final_comparison_df.append(comparison_df)

    final_comparison_df = pd.concat(final_comparison_df, ignore_index=True)
    final_comparison_df["itemset_length"] = final_comparison_df["itemsets"].apply(len)
    return final_comparison_df


def enrich_comparison_df_itemset_info(comparison_df):
    comparison_df["phospho_in_set"] = None
    comparison_df["acetyl_in_set"] = None
    phospho_in_set = []
    acetyl_in_set = []

    ptm_pattern = r'([A-Za-z]+)\(([A-Za-z])\)@(\d+)'

    for index, row in tqdm(comparison_df.iterrows(), desc=f"Enriching comparison_df with itemset info"):
        phospho_present = False
        acetyl_present = False
        for item in row["itemsets"]:
            match = re.match(ptm_pattern, item)
            if match:
                ptm_type = match.group(1) 
                if ptm_type == "Phospho":
                    phospho_present = True
                elif ptm_type == "Acetyl":
                    acetyl_present = True
        if phospho_present:
            phospho_in_set.append(1)
        else:
            phospho_in_set.append(0)

        if acetyl_present:
            acetyl_in_set.append(1)
        else:
            acetyl_in_set.append(0)

    comparison_df["phospho_in_set"] = phospho_in_set
    comparison_df["acetyl_in_set"] = acetyl_in_set

    return comparison_df


def enrich_comparison_df_with_metric(comparison_df, protein_structure_ptms_df, metric_name):
    comparison_df["metric_value"] = None
    comparison_df["metric_value_avg"] = None
    comparison_df["metric_name"] = metric_name
    metric_list = []
    metric_avg_list = []
    for index, row in tqdm(comparison_df.iterrows(), desc=f"Enriching comparison_df with structual metric"):
        protein_id = row["Protein"]
        metric_for_protein_list = []
        for ptm_str in row["itemsets"]:
            ptm_pos = int(ptm_str.split("@")[1])
            filtered_row = protein_structure_ptms_df[
                (protein_structure_ptms_df["protein_id"] == protein_id) & (protein_structure_ptms_df["position"] == ptm_pos)
            ]
            metric_for_protein_list.append(filtered_row[metric_name])
        metric_avg_list.append(np.array(metric_for_protein_list).mean())
        metric_list.append(metric_for_protein_list)
    comparison_df["metric_value"] = metric_list
    comparison_df["metric_value_avg"] = metric_avg_list
    return comparison_df


def compute_ptm_cooccurence(ptms_for_proteins_df, protein_structure_ptms_df, expected_group_size, num_proteins, treatment_label, control_label, ppse_column):
    sampled_df = apply_rus_to_ptms_for_proteins_df(ptms_for_proteins_df, num_proteins, expected_group_size)
    trg_cooccurrence = find_frequent_ptm_sets(sampled_df, treatment_label)
    ctg_cooccurrence = find_frequent_ptm_sets(sampled_df, control_label)
    comparison_df = test_significance_of_frequent_ptm_sets(sampled_df, trg_cooccurrence, ctg_cooccurrence)
    comparison_df = enrich_comparison_df_itemset_info(comparison_df)
    comparison_df = enrich_comparison_df_with_metric(comparison_df, protein_structure_ptms_df, ppse_column)
    return comparison_df


def vulcano_plot_ptm_cooccurence(comparison_df,significance_threshold = 0.05, logfc_threshold = 1, min_itemset_length = 2):
    final_comparison_plot_df = comparison_df[comparison_df["itemset_length"]>=min_itemset_length].copy()
    final_comparison_plot_df["-log10(p-value)"] = -np.log10(final_comparison_plot_df["p-value"])
    final_comparison_plot_df["itemsets_str"] = final_comparison_plot_df["itemsets"].apply(lambda x: ", ".join(x) if isinstance(x, frozenset) else str(x))

    log_pval_thresh = -np.log10(significance_threshold)

    final_comparison_plot_df["color"] = "Non-Significant"
    final_comparison_plot_df.loc[(final_comparison_plot_df["p-value"] < significance_threshold) & (final_comparison_plot_df["logfc"] > logfc_threshold), "color"] = "Upregulated in TRG"
    final_comparison_plot_df.loc[(final_comparison_plot_df["p-value"] < significance_threshold) & (final_comparison_plot_df["logfc"] < -logfc_threshold), "color"] = "Downregulated in TRG"

    fig = px.scatter(
        final_comparison_plot_df, 
        x="logfc", 
        y="-log10(p-value)", 
        color="color", 
        hover_data=["Protein", "itemsets_str", "TRG support", "CTG support", "metric_name", "metric_value", "metric_value_avg"],
        title=f"Volcano Plot of PTM Co-occurrence Differences",
        labels={"logfc": "Log2 Fold Change", "-log10(p-value)": "-Log10(P-Value)"},
        color_discrete_map={
            "Upregulated in TRG": "red", 
            "Downregulated in TRG": "blue", 
            "Non-Significant": "gray"
        }
    )

    fig.add_vline(x=logfc_threshold, line_dash="dash", line_color="black")
    fig.add_vline(x=-logfc_threshold, line_dash="dash", line_color="black")
    fig.add_hline(y=log_pval_thresh, line_dash="dash", line_color="black")

    fig.add_annotation(
        text=f"Description: <br> significance_threshold: {significance_threshold} <br> logfc_threshold: > {logfc_threshold} <br> min_itemset_length: {min_itemset_length}",
        xref="paper", 
        yref="paper",
        x=1.17,  
        y=0, 
        showarrow=False,
        align="left",
        font=dict(size=12),
        bordercolor="black",
        borderwidth=1,
        bgcolor="white",
        opacity=0.8
    )

    fig.show()
