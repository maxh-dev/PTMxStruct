import numpy as np
import pandas as pd
import plotly.express as px

def plot_pPSE_cutoff(df, ppse_column, title="pPSE Exposure Cutoff"):
    if ppse_column not in df.columns:
        raise ValueError(f"Column '{ppse_column}' not found in DataFrame")
    values = df[df.IDR == 0][ppse_column].dropna().values
    if len(values) == 0:
        raise ValueError("No data found after filtering for IDR == 0")


    bincount = np.unique(values, return_counts=True)
    if len(bincount[0]) == 0:
        raise ValueError("No bins found to plot")

    bincount_df = pd.DataFrame({'pPSE': bincount[0], 'count': bincount[1]})
    bincount_df['cutoff'] = np.where(bincount_df.pPSE <= 5, 'High Accessibility', 'Low Accessibility')

    pPSE_cut = px.bar(
        bincount_df,
        x='pPSE',
        y='count',
        color='cutoff',
        color_discrete_map={
            'High Accessibility': 'orange',
            'Low Accessibility': 'dimgrey'
        },
        template="simple_white",
        width=500,
        height=300,
        title=f"{title} <br><span style='font-size:12px; color:gray'> for {ppse_column}</span>"
    )

    pPSE_cut = pPSE_cut.update_layout(
        legend=dict(
            title='',
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    config = {
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'pPAE_cutoff'
        }
    }

    pPSE_cut.show(config=config)

