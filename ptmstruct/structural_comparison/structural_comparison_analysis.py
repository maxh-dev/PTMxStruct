import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull


def get_all_coords(structure):
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.get_coord())
        break
    return np.array(coords)

def get_ca_coords(structure):
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if "CA" in atom.name:
                        coords.append(atom.get_coord())
        break
    return np.array(coords)

def radius_of_gyration(com, coords):
    distances_from_com = coords - com
    R_g = np.sqrt(np.sum(distances_from_com**2) / len(distances_from_com))
    return R_g

def compute_global_metrics(protein_structure_dict):
    protein_metric_dict = {}
    for protein_id in protein_structure_dict:
        if not protein_id in protein_metric_dict:
            protein_metric_dict[protein_id] = {}

        densities = []
        radius_of_gyrations = []

        for protein_structure, com in protein_structure_dict[protein_id]:
            all_coords =  get_all_coords(protein_structure)
    
            hull = ConvexHull(all_coords)
            density = len(all_coords) / hull.volume

            densities.append(density)
            radius_of_gyrations.append(radius_of_gyration(com, all_coords))

        protein_metric_dict[protein_id]["packing_volume"] = np.array(densities).mean()
        protein_metric_dict[protein_id]["radius_of_gyration"] = np.array(radius_of_gyrations).mean()   

    return protein_metric_dict


def extract_ppse_metric_from_df(df_ppse, ppse_name = "nAA_12_70_pae_smooth10"):
    df_grouped = df_ppse.groupby('protein_id', as_index=False)[ppse_name].agg(list)
    group_dict = {}
    group_std_dict = {}
    group_avg_dict = {}

    for i, row in df_grouped.iterrows():
        protein_id = re.sub(r'_m\d+$', '', row["protein_id"])
        if protein_id not in group_dict:
            group_dict[protein_id] = []
        group_dict[protein_id].append(
            row[ppse_name]
        )

    for key in group_dict:
        group_dict[key] = np.array(group_dict[key])
        group_std_dict[key] = np.std(group_dict[key], axis=0) 
        group_avg_dict[key] = np.average(group_dict[key], axis=0) 

    return group_dict, group_std_dict, group_avg_dict


def mae(actual, predicted, round_res = False):
    mae = np.mean(np.abs(actual - predicted))
    if round_res:
        return round(mae, 2)
    else:
        return mae

def mean_error(actual, predicted, round_res = False):
    me = np.mean((actual - predicted))
    if round_res:
        return round(me, 2)
    else:
        return me

def create_comparison_list(df_structure_ptms, comparisons, top_n_df, group_avg_dict, raw_ppse_name="nAA_12_70_pae",smooth_ppse_name = "nAA_12_70_pae_smooth10"):
    comp_pairs = []
    for comparison in tqdm(comparisons):
        mod_id = comparison[0]
        transcript_id = mod_id.split("_")[0].strip()
        aa = mod_id.split("_")[2].strip()
        pos = mod_id.split("_")[3].strip()
        rows = top_n_df.loc[
            (top_n_df['protein_id'] == transcript_id) & 
            (top_n_df['AA'] == aa) & 
            (top_n_df['position'] == int(pos))
        ]
        if rows.shape[0] > 1: 
            raise ValueError("doublicate ptm")
        
        df_protein = df_structure_ptms[df_structure_ptms["protein_id"] == comparison[1]]
        
        comp_pairs.append({
            "ppse_bin": rows.iloc[0]["ppse_smooth_bin"],
            "ppse": rows.iloc[0][smooth_ppse_name],
            "ppse_raw": rows.iloc[0][raw_ppse_name],
            "pos": pos,
            "info": f"CRC: {rows.iloc[0]['TRG_ac']}, CTR: {rows.iloc[0]['CTG_ac']}",
            "idr": df_protein["IDR"].values,
            "comparison": comparison,
            "af3_mod": group_avg_dict[comparison[0]],
            "af3_no_mod": group_avg_dict[comparison[1]],
            "af2_no_mod": df_protein[smooth_ppse_name].values,
            "diff_af3_af2": (group_avg_dict[comparison[1]] - df_protein[smooth_ppse_name].values),
            "diff_af3_af3_mod": (group_avg_dict[comparison[1]] - group_avg_dict[comparison[0]]),
        })
    return comp_pairs

def plot_ppse_comparison_with_differences(comp_pair, comparison_labels=None, add_vertical_line=None, safe_to_file=False, file_path=None):
    """
    Plots PPSE comparisons and their differences using subplots,
    including an IDR indicator bar between plots.

    Parameters:
        comp_pair (dict): Must contain keys: 'af3_mod', 'af3_no_mod', 'af2_no_mod', 'IDR', 'comparison', 'ppse', 'info'
        comparison_labels (list of str): Optional custom labels for the three lines.
        add_vertical_line (bool): Optional, adds vertical marker at comp_pair["pos"] if True.
        safe_to_file (bool): Whether to save to file or show interactively.
        file_path (str): Path prefix for saving image if safe_to_file is True.
    """
    ppse_cutoff = 6
    af3_mod = comp_pair["af3_mod"]
    af3_no_mod = comp_pair["af3_no_mod"]
    af2_no_mod = comp_pair["af2_no_mod"]

    mae_af2 = mae(af3_no_mod, af2_no_mod, round_res=True)
    mae_mod = mae(af3_no_mod, af3_mod, round_res=True)
    me_af2 = mean_error(af3_no_mod, af2_no_mod, round_res=True)
    me_mod = mean_error(af3_no_mod, af3_mod, round_res=True)

    x = list(range(len(af3_mod)))
    labels = comparison_labels or ['AF3 (mod)', 'AF3 (no mod)', 'AF2 (no mod)']
    colors = px.colors.qualitative.Plotly
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(labels)}

    # Create subplots with 3 rows (Top: PPSE, Middle: IDR, Bottom: Differences)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.45, 0.05, 0.45],
        vertical_spacing=0.05,
        subplot_titles=("Smoothed pPSE Comparison", "", "Difference Plot")
    )

    # --- Top Plot: PAE Values ---
    for label, y_vals in zip(labels, [af3_mod, af3_no_mod, af2_no_mod]):
        fig.add_trace(go.Scatter(
            x=x,
            y=y_vals,
            mode='lines+markers',
            name=label,
            line=dict(color=color_map[label])
        ), row=1, col=1)

    # Optional vertical line in top plot
    if add_vertical_line and "pos" in comp_pair:
        fig.add_shape(
            type='line',
            x0=comp_pair["pos"], x1=comp_pair["pos"],
            y0=min(min(af3_mod), min(af3_no_mod), min(af2_no_mod)),
            y1=max(max(af3_mod), max(af3_no_mod), max(af2_no_mod)),
            line=dict(color="black", dash='dot'),
            xref='x', yref='y',
            row=1, col=1
        )

    # Optional pPSE cutoff line
    if any([max(af3_mod) > ppse_cutoff, max(af3_no_mod) > ppse_cutoff, max(af2_no_mod) > ppse_cutoff]):
        fig.add_shape(
            type='line',
            x0=min(x),
            x1=max(x),
            y0=ppse_cutoff,
            y1=ppse_cutoff,
            line=dict(color="red", dash='dash'),
            xref='x', yref='y',
            row=1, col=1
        )

    # --- Middle Plot: IDR Fill with Merged Regions ---
    idr_array = comp_pair["idr"]
    start = None
    for i in range(len(idr_array) + 1):
        if i < len(idr_array) and idr_array[i] == 1:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i - 1
                fig.add_trace(go.Scatter(
                    x=[start, end+1, end+1, start],
                    y=[0, 0, 1, 1],
                    fill='toself',
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='lightblue',
                    hoverinfo='skip',
                    showlegend=False
                ), row=2, col=1)
                start = None


    fig.add_annotation(
        text="IDRs",
        xref="paper", yref="paper",
        x=-0.05, y=0.5,  # Adjust x/y to position label correctly for row 2
        showarrow=False,
        font=dict(size=12),
        textangle=0  # Rotate for y-axis style
    )

    fig.update_yaxes(visible=False, row=2, col=1, range=[0, 1])
    fig.update_xaxes(visible=False, row=2, col=1)

    # --- Bottom Plot: Differences ---
    diff_af3_af2 = [a - b for a, b in zip(af3_no_mod, af2_no_mod)]
    diff_af3_af3 = [a - b for a, b in zip(af3_no_mod, af3_mod)]

    fig.add_trace(go.Scatter(
        x=x,
        y=diff_af3_af2,
        mode='lines+markers',
        name='AF3 (no mod) - AF2',
        line=dict(color='orange')
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=x,
        y=diff_af3_af3,
        mode='lines+markers',
        name='AF3 (no mod) - AF3 (mod)',
        line=dict(color='purple')
    ), row=3, col=1)

    # Annotation box with error metrics
    fig.add_annotation(
        text=f"Mean Errors: <br> ME Af3/Af2 or Boltz: {me_af2} <br> ME Af3/Af3 with Mod: {me_mod} <br> MAE Af3/Af2 or Boltz: {mae_af2} <br> MAE Af3/Af3 with Mod: {mae_mod}",
        xref="paper", 
        yref="paper",
        x=1.22,  
        y=0, 
        showarrow=False,
        align="left",
        font=dict(size=12),
        bordercolor="black",
        borderwidth=1,
        bgcolor="white",
        opacity=0.8
    )

    # Layout updates
    fig.update_layout(
        height=750,
        title_text=f"PPSE Comparison and Differences <br><span style='font-size:12px; color:gray;'> for {comp_pair['comparison'][0]} with smoothed pPSE {round(comp_pair['ppse'],2)}. {comp_pair['info']}",
        template='plotly_white'
    )

    fig.update_xaxes(title_text='Index', row=3, col=1)
    fig.update_yaxes(title_text='Smoothed pPSE', row=1, col=1)
    fig.update_yaxes(title_text='Difference', row=3, col=1)

    # Show or save figure
    if not safe_to_file:
        fig.show()
    else:
        fig.write_image(f"{file_path}pPSE_comp_{int(comp_pair['ppse_bin'])}_{comp_pair['comparison'][0]}.png", width=1200, height=800)
  

def mae_boxplot(comparison_list, plot=True):
    mae_data = {
        "low_acc": {"mae_af3_af2": [], "mae_af3_af3_mod": []},
        "high_acc": {"mae_af3_af2": [], "mae_af3_af3_mod": []}
    }

    plot_rows = []

    for entry in comparison_list:
        ppse = entry["ppse"]
        mae_af2 = entry["mae_af3_af2"]
        mae_mod = entry["mae_af3_af3_mod"]

        acc_label = "High Accessibility" if ppse < 6 else "Low Accessibility"
        acc_key = "high_acc" if ppse < 6 else "low_acc"

        mae_data[acc_key]["mae_af3_af2"].append(mae_af2)
        mae_data[acc_key]["mae_af3_af3_mod"].append(mae_mod)

        plot_rows.append({"Accessibility": acc_label, "Comparison": "Af3/Af2 or Boltz", "MAE": mae_af2})
        plot_rows.append({"Accessibility": acc_label, "Comparison": "Af3/Af3 with Mod", "MAE": mae_mod})

    if plot:
        df = pd.DataFrame(plot_rows)

        fig = px.box(
            df,
            x="Comparison",
            y="MAE",
            color="Accessibility",
            facet_col="Accessibility",
            boxmode="group",
            color_discrete_map={
                "Low Accessibility": "royalblue",
                "High Accessibility": "orange"
            },
            title="MAE Comparison by Accessibility"
        )

        fig.update_layout(
            yaxis_title="Mean Absolute Error",
            xaxis_title="",
            title_x=0.5,
            uniformtext_minsize=8,
            uniformtext_mode='hide',
        )

        # Remove "Accuracy=" prefix in facet titles
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
        fig.update_xaxes(tickvals=["Af3/Af2 or Boltz", "Af3/Af3 with Mod"], ticktext=["Af3/Af2 or Boltz", "Af3/Af3 with Mod"], matches=None)
        fig.for_each_xaxis(lambda axis: axis.update(title=''))

        fig.show()

    return mae_data


def error_diff_boxplot(comparison_list, plot_type="MAE", plot=True):
    error_data = {
        "low_acc": {"mae_diff": [], "me_diff": []},
        "high_acc": {"mae_diff": [], "me_diff": []}
    }

    plot_rows = []

    for entry in comparison_list:
        ppse = entry["ppse"]
        mae_diff = mae(entry["diff_af3_af2"], entry["diff_af3_af3_mod"])
        me_diff = mean_error(entry["diff_af3_af2"], entry["diff_af3_af3_mod"])

        acc_label = "High Accessibility" if ppse < 6 else "Low Accessibility"
        acc_key = "high_acc" if ppse < 6 else "low_acc"

        error_data[acc_key]["mae_diff"].append(mae_diff)
        error_data[acc_key]["me_diff"].append(me_diff)

        if plot_type == "MAE":
            plot_rows.append({"Accessibility": acc_label, "Comparison": "MAE((Af3 - Af2 or Boltz), (Af3 - Af3 with Mod))", "Error Type": "MAE", "Error Value": mae_diff})
        elif plot_type == "ME":
            plot_rows.append({"Accessibility": acc_label, "Comparison": "ME((Af3 - Af2 or Boltz), (Af3 - Af3 with Mod))", "Error Type": "ME", "Error Value": me_diff})

    if plot:
        # Create DataFrame for plotting
        df = pd.DataFrame(plot_rows)

        fig = px.box(
            df,
            x="Comparison",
            y="Error Value",
            color="Accessibility",
            boxmode="group",
            color_discrete_map={
                "Low Accessibility": "dimgray",
                "High Accessibility": "orange"
            },
            title=f"{plot_type} Comparison of Difference between Predictions by Accessibility"
        )

        fig.update_layout(
            yaxis_title=f"Error Value ({plot_type})",
            xaxis_title="",
            title_x=0.5,
            uniformtext_minsize=8,
            uniformtext_mode='hide',
            width=800 
        )

        fig.show()

    return error_data


def error_boxplot(comparison_list, plot_type="MAE", plot=True):
    error_data = {
        "low_acc": {"mae_af3_af2": [], "mae_af3_af3_mod": [], "me_af3_af2": [], "me_af3_af3_mod": []},
        "high_acc": {"mae_af3_af2": [], "mae_af3_af3_mod": [], "me_af3_af2": [], "me_af3_af3_mod": []}
    }

    plot_rows = []

    for entry in comparison_list:
        ppse = entry["ppse"]
        mae_af2 = mae(entry["af3_no_mod"], entry["af2_no_mod"])
        mae_mod = mae(entry["af3_no_mod"], entry["af3_mod"])
        me_af2 = mean_error(entry["af3_no_mod"], entry["af2_no_mod"])
        me_mod = mean_error(entry["af3_no_mod"], entry["af3_mod"])

        acc_label = "High Accessibility" if ppse < 6 else "Low Accessibility"
        acc_key = "high_acc" if ppse < 6 else "low_acc"

        error_data[acc_key]["mae_af3_af2"].append(mae_af2)
        error_data[acc_key]["mae_af3_af3_mod"].append(mae_mod)
        error_data[acc_key]["me_af3_af2"].append(me_af2)
        error_data[acc_key]["me_af3_af3_mod"].append(me_mod)

        if plot_type == "MAE":
            plot_rows.append({"Accessibility": acc_label, "Comparison": "MAE(Af3, Af2 or Boltz)", "Error Type": "MAE", "Error Value": mae_af2})
            plot_rows.append({"Accessibility": acc_label, "Comparison": "MAE(Af3, Af3 with Mod)", "Error Type": "MAE", "Error Value": mae_mod})
        elif plot_type == "ME":
            plot_rows.append({"Accessibility": acc_label, "Comparison": "ME(Af3, Af2 or Boltz)", "Error Type": "ME", "Error Value": me_af2})
            plot_rows.append({"Accessibility": acc_label, "Comparison": "ME(Af3, Af3 with Mod)", "Error Type": "ME", "Error Value": me_mod})

    if plot:
        df = pd.DataFrame(plot_rows)

        fig = px.box(
            df,
            x="Comparison",
            y="Error Value",
            color="Accessibility",
            facet_col="Accessibility",
            boxmode="group",
            color_discrete_map={
                "Low Accessibility": "dimgray",
                "High Accessibility": "orange"
            },
            title=f"{plot_type} Comparison by Accessibility"
        )

        fig.update_layout(
            yaxis_title=f"Error Value ({plot_type})",
            xaxis_title="",
            title_x=0.5,
            uniformtext_minsize=8,
            uniformtext_mode='hide',
            width=1000 
        )

        # Remove "Accessibility=" prefix in facet titles
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
        #fig.update_xaxes(tickvals=["Af3/Af2 or Boltz", "Af3/Af3 with Mod"], ticktext=["Af3/Af2 or Boltz", "Af3/Af3 with Mod"], matches=None)
        #fig.for_each_xaxis(lambda axis: axis.update(title=''))

        fig.show()

    return error_data


"""def global_metric_boxplot(comparison_list, protein_metric_dict, metric_name="packing_volume", plot=True):
    # Organize metric data consistently
    metric_data = {
        "no_mod": [v[metric_name] for k, v in protein_metric_dict.items() if '_ac' not in k],
        "mod": [v[metric_name] for k, v in protein_metric_dict.items() if '_ac' in k],
        "mod_low_acc": [],
        "mod_high_acc": [],
    }

    # Fill in mod_low_acc and mod_high_acc lists
    for entry in comparison_list:
        ppse = entry["ppse"]
        acc_key = "mod_high_acc" if ppse < 6 else "mod_low_acc"
        protein_mod_id = entry["comparison"][0]

        # Avoid KeyError if the ID isn't found
        if protein_mod_id in protein_metric_dict:
            metric_data[acc_key].append(protein_metric_dict[protein_mod_id][metric_name])

    if plot:
        color_map = {
            "no_mod": "red",
            "mod": "blue",
            "mod_low_acc": "lightblue",
            "mod_high_acc": "darkblue"
        }

        traces = [
            go.Box(y=metric_data["no_mod"], name="No Modification",
                   marker=dict(color=color_map["no_mod"])),
            go.Box(y=metric_data["mod"], name="Modification",
                   marker=dict(color=color_map["mod"])),
            go.Box(y=metric_data["mod_low_acc"], name="Modification in Low Accessibility Region",
                   marker=dict(color=color_map["mod_low_acc"])),
            go.Box(y=metric_data["mod_high_acc"], name="Modification in High Accessibility Region",
                   marker=dict(color=color_map["mod_high_acc"])),
        ]

        layout = go.Layout(
            title=f"{metric_name.replace('_', ' ').title()} of Proteins Comparison",
            yaxis=dict(title=metric_name.replace('_', ' ').title()),
            xaxis=dict(
                title=None,
                showticklabels=False,  # <--- This hides the x-axis category labels
                showgrid=False
            ),
            boxmode='group',
            width=1000
        )

        fig = go.Figure(data=traces, layout=layout)
        fig.show()
"""
