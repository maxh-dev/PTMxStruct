from Bio.PDB import MMCIFParser, Superimposer
import numpy as np
from scipy.spatial.distance import cdist
import plotly.express as px
from Bio.PDB import is_aa
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist


def load_structure(file_path, structure_id):
    parser = MMCIFParser(QUIET=True)
    return parser.get_structure(structure_id, file_path)

def extract_ca_atoms_and_coords(structure):
    ca_atoms = []
    ca_coords = []
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    atom = residue['CA']
                    ca_atoms.append(atom)
                    ca_coords.append(atom.get_coord())
                    residues.append(residue.get_id())
        break
    return ca_atoms, np.array(ca_coords), residues

def compare_coords(coords1, coords2):
    return np.linalg.norm(coords1 - coords2, axis=1)  # Per-residue RMSD


def extract_ca_cb_vectors(structure):
    ca_cb_vectors = []
    valid_residues = []
    residue_count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True) and 'CA' in residue and 'CB' in residue:
                    residue_count += 1
                    ca = residue['CA'].get_coord()
                    cb = residue['CB'].get_coord()
                    vector = cb - ca
                    norm = np.linalg.norm(vector)
                    if norm != 0:
                        ca_cb_vectors.append(vector / norm)  # Normalize
                        valid_residues.append(residue_count)
            break  # only first model
        break
    return np.array(ca_cb_vectors), valid_residues


def align_structures(atoms1, atoms2, structure2):
    # Align only matching residues
    min_len = min(len(atoms1), len(atoms2))
    sup = Superimposer()
    sup.set_atoms(atoms1[:min_len], atoms2[:min_len])
    sup.apply(structure2.get_atoms()) 

    # Compute aligned coordinates manually
    coords2_aligned = np.array([atom.get_coord() for atom in atoms2[:min_len]])

    # Report match status
    if len(atoms1) == len(atoms2):
        print(f"✅ Full match: {len(atoms1)} Cα atoms aligned.")
    else:
        print(f"⚠️ Partial match: {min_len} out of {len(atoms1)} (structure1) and {len(atoms2)} (structure2) Cα atoms aligned.")

    return min_len, coords2_aligned

def compute_distances(coords1, coords2_aligned, min_len):
    distance_matrix1 = cdist(coords1[:min_len], coords1[:min_len])
    distance_matrix2 = cdist(coords2_aligned, coords2_aligned)
    difference_matrix = np.abs(distance_matrix1 - distance_matrix2)

    return distance_matrix1, distance_matrix2, difference_matrix

def plot_distances(structure1, structure2, distance_matrix1, distance_matrix2, difference_matrix):
    # Set up subplots with 1 row and 3 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Distance Matrix: {structure1.id}", f"Distance Matrix: {structure2.id}"),
        horizontal_spacing=0.05
    )

    # Plot 1: distance_matrix1
    fig.add_trace(go.Heatmap(
        z=distance_matrix1,
        colorscale='Viridis',
        colorbar=dict(title="Å"),
        showscale=True
    ), row=1, col=1)

    # Plot 2: distance_matrix2
    fig.add_trace(go.Heatmap(
        z=distance_matrix2,
        colorscale='Viridis',
        colorbar=dict(title="Å"),
        showscale=False  # Hide duplicate colorbars
    ), row=1, col=2)

    # Update layout
    fig.update_layout(
        title_text="Pairwise Cα Distance Matrices",
        width=1200,
        height=500
    )

    fig.show()


    fig = px.imshow(
        difference_matrix,
        labels=dict(x="Residue Index", y="Residue Index", color="Distance Δ (Å)"),
        color_continuous_scale='Viridis',
        title=f"Difference in Pairwise Cα Distances for <br> {structure1.id} - {structure2.id}"
    )
    fig.update_layout(width=600, height=600)
    fig.show()

def compute_consecutive_distance(coords):
    return np.linalg.norm(coords[1:] - coords[:-1], axis=1)

def compute_consecutive_distances(coords1, coords2_aligned, min_len):
    distances1 = compute_consecutive_distance(coords1[:min_len])
    distances2 = compute_consecutive_distance(coords2_aligned)

    return distances1, distances2

def plot_consecutive_distances(structure1, structure2, distances1, distances2, min_len):
    residue_indices = np.arange(1, min_len) 

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=residue_indices,
        y=distances1,
        mode='lines+markers',
        name=f'{structure1.id}',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=residue_indices,
        y=distances2,
        mode='lines+markers',
        name=f'{structure2.id} (Aligned)',
        line=dict(color='red')
    ))

    fig.update_layout(
        title=f'Distance Between Consecutive Cα Atoms for <br> {structure1.id} and {structure2.id}',
        xaxis_title='Residue Index (i)',
        yaxis_title='Distance between Residue i and i+1 (Å)',
        legend=dict(x=0.01, y=0.99),
        width=1000,
        height=500
    )

    fig.show()


def get_vector_angles(vec1, vec2):
    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angle_rad = np.arccos(cos_sim)
    angle_deg = np.degrees(angle_rad)

    return angle_deg, angle_rad


def compute_vectors(structure1, structure2):
    vectors1, resids1 = extract_ca_cb_vectors(structure1)
    vectors2, resids2 = extract_ca_cb_vectors(structure2)
    common_residues = sorted(set(resids1) & set(resids2))

    vecs1_matched, vecs2_matched = [], []
    for resid in common_residues:
        vecs1_matched.append(vectors1[resids1.index(resid)])
        vecs2_matched.append(vectors2[resids2.index(resid)])
    vecs1_matched = np.array(vecs1_matched)
    vecs2_matched = np.array(vecs2_matched)

    def vector_angles(vectors, axis=np.array([0, 0, 1])):
        dots = np.dot(vectors, axis)
        dots = np.clip(dots, -1.0, 1.0)
        return np.arccos(dots) * (180 / np.pi)

    angles1 = vector_angles(vecs1_matched)
    angles2 = vector_angles(vecs2_matched)

    angle_diff = []
    for i in range(len(vecs1_matched)):
        angle_diff.append(get_vector_angles(vecs1_matched[i], vecs2_matched[i])[0])

    return angles1, angles2, angle_diff


def plot_orientation_vectors(structure1, structure2, angles1, angles2, angle_diff):
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[
            "Cα→Cβ Orientation Angles (Z-Axis Projection)",
            "Angle Difference Between Matched Vectors"
        ]
    )

    # Subplot 1: angles for both structures
    fig.add_trace(go.Scatter(
        x=np.arange(len(angles1)), y=angles1,
        mode='lines+markers',
        name=f'{structure1.id}',
        line=dict(color='blue')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=np.arange(len(angles2)), y=angles2,
        mode='lines+markers',
        name=f'{structure2.id}',
        line=dict(color='orange')
    ), row=1, col=1)

    # Subplot 2: angle difference
    fig.add_trace(go.Scatter(
        x=np.arange(len(angle_diff)), y=angle_diff,
        mode='lines+markers',
        name='Angle Difference',
        line=dict(color='green')
    ), row=2, col=1)

    # Layout
    fig.update_layout(
        height=700, width=1300,
        title="Comparison of Cα→Cβ Vector Orientations",
        xaxis_title="Matched Residue Index",
        yaxis_title="Angle (degrees)",
        showlegend=True
    )
    fig.update_yaxes(title_text="Angle (°)", row=1, col=1)
    fig.update_yaxes(title_text="Angle Difference (°)", row=2, col=1)

    fig.show()