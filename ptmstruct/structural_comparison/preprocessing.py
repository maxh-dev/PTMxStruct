import os
import re
import json
import shutil
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import Bio.PDB.MMCIF2Dict

import structuremap.utils
structuremap.utils.set_logger()
from structuremap.processing import (
    download_alphafold_cif,
    download_alphafold_pae,
    format_alphafold_data,
    annotate_accessibility,
    get_smooth_score,
    annotate_proteins_with_idr_pattern,
    get_extended_flexible_pattern
)


from Bio.PDB import MMCIFParser, PDBIO
import MDAnalysis as mda
from io import StringIO

## Custom

def load_structure(file_path, structure_id):
    parser = MMCIFParser(QUIET=True)
    return parser.get_structure(structure_id, file_path)


def get_center_of_mass(structure):
    buffer = StringIO()

    io = PDBIO()
    io.set_structure(structure)
    io.save(buffer)  

    buffer.seek(0)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w+') as tmp:
        tmp.write(buffer.getvalue())
        tmp.flush()
        
        u = mda.Universe(tmp.name)
        com = u.atoms.center_of_mass()
        return com
    

def load_protein_strctures(proteins, cif_dir):
    protein_structure_dict = {}

    for file in tqdm.tqdm(sorted(os.listdir(cif_dir))):
        if file.endswith("cif"):
            filepath = os.path.join(cif_dir, file)

            protein_prediction_id = re.sub(r'.cif', '', file)
            if protein_prediction_id not in proteins:
                continue

            protein_id = re.sub(r'_m\d+$', '', protein_prediction_id)
            
            if not protein_id in protein_structure_dict:
                protein_structure_dict[protein_id] = []

            protein_structure = load_structure(filepath, protein_prediction_id)
            
            protein_structure_dict[protein_id].append((protein_structure, get_center_of_mass(protein_structure)))    

    return protein_structure_dict

# StructureMap

def format_alphafold_data(
    directory: str,
    protein_ids: list,
) -> pd.DataFrame:
    """
    Function to import structure files and format them into a combined dataframe.

    Parameters
    ----------
    directory : str
        Path to the folder with all .cif files.
    proteins : list
        List of UniProt protein accessions to create an annotation table.
        If an empty list is provided, all proteins in the provided directory
        are used to create the annotation table.

    Returns
    -------
    : pd.DataFrame
        A dataframe with structural information presented in following columns:
        ['protein_id', 'protein_number', 'AA', 'position', 'quality',
        'x_coord_c', 'x_coord_ca', 'x_coord_cb', 'x_coord_n', 'y_coord_c',
        'y_coord_ca', 'y_coord_cb', 'y_coord_n', 'z_coord_c', 'z_coord_ca',
        'z_coord_cb', 'z_coord_n', 'secondary_structure', 'structure_group',
        'BEND', 'HELX', 'STRN', 'TURN', 'unstructured']
    """

    alphafold_annotation_l = []
    protein_number = 0

    for file in tqdm.tqdm(sorted(os.listdir(directory))):

        if file.endswith("cif"):
            filepath = os.path.join(directory, file)

            protein_id = re.sub(r'.cif', '', file)

            if ((protein_id in protein_ids) or (len(protein_ids) == 0)):

                protein_number += 1

                structure = Bio.PDB.MMCIF2Dict.MMCIF2Dict(filepath)

                df = pd.DataFrame({'protein_id': protein_id, # differnt to v2
                                   'protein_number': protein_number,
                                   'AA': structure['_atom_site.label_comp_id'], # differnt to v2
                                   'position': structure['_atom_site.label_seq_id'],
                                   'quality': structure['_atom_site.B_iso_or_equiv'],
                                   'atom_id': structure['_atom_site.label_atom_id'],
                                   'x_coord': structure['_atom_site.Cartn_x'],
                                   'y_coord': structure['_atom_site.Cartn_y'],
                                   'z_coord': structure['_atom_site.Cartn_z']})

                df = df[df.atom_id.isin(['CA', 'CB', 'C', 'N'])].reset_index(drop=True)
                df = df.apply(pd.to_numeric, errors='ignore')

                mean_quality = (
                    df.groupby(['protein_id', 'protein_number', 'AA', 'position'])['quality']
                    .mean()
                    .reset_index()
                )
                df = df.drop('quality', axis=1)

                df = df.pivot(index=['protein_id',
                                     'protein_number',
                                     'AA', 'position'],
                              columns="atom_id")

                df.columns = ['_'.join((col[0], col[1].lower())).strip('_') for col in df.columns.values]

                df = pd.DataFrame(df.to_records())
                df = pd.merge(df, mean_quality, on=['protein_id', 'protein_number', 'AA', 'position'], how='left')
                df['secondary_structure'] = 'unstructured'
                alphafold_annotation_l.append(df)

    alphafold_annotation = pd.concat(alphafold_annotation_l)
    alphafold_annotation = alphafold_annotation.sort_values(
        by=['protein_number', 'position']).reset_index(drop=True)

    return(alphafold_annotation)


def load_protein_data(proteins, cif_dir, pae_dir):
    valid_proteins_cif, invalid_proteins_cif, existing_proteins_cif = download_alphafold_cif(
        proteins=proteins,
        out_folder=cif_dir)

    valid_proteins_pae, invalid_proteins_pae, existing_proteins_pae = download_alphafold_pae(
        proteins=proteins,
        out_folder=pae_dir,
        )

    alphafold_annotation = format_alphafold_data(
        directory=cif_dir,
        protein_ids=proteins
    )

    return valid_proteins_cif, invalid_proteins_cif, alphafold_annotation


def load_protein_data_from_file(proteins : list, cif_dir: str):
    alphafold_annotation = format_alphafold_data(
        directory=cif_dir,
        protein_ids=proteins
    )

    return alphafold_annotation


def annotate_pPSE_values(alphafold_annotation, pae_dir):
    full_sphere_exposure = annotate_accessibility(
        df=alphafold_annotation,
        max_dist=24,
        max_angle=180,
        error_dir=pae_dir)

    alphafold_accessibility = alphafold_annotation.merge(
        full_sphere_exposure, how='left', on=['protein_id', 'AA', 'position'])

    part_sphere_exposure = annotate_accessibility(
        df=alphafold_annotation,
        max_dist=12,
        max_angle=70,
        error_dir=pae_dir)

    alphafold_accessibility = alphafold_accessibility.merge(
        part_sphere_exposure, how='left', on=['protein_id', 'AA', 'position'])

    alphafold_accessibility['high_acc_5'] = np.where(alphafold_accessibility.nAA_12_70_pae <= 5, 1, 0)
    alphafold_accessibility['low_acc_5'] = np.where(alphafold_accessibility.nAA_12_70_pae > 5, 1, 0)

    return alphafold_accessibility

def annotate_IDRs(alphafold_accessibility):
    alphafold_accessibility_smooth_1 = get_smooth_score(
        alphafold_accessibility,
        np.array(['nAA_12_70_pae']),
        [10])
    
    alphafold_accessibility_smooth_2 = get_smooth_score(
        alphafold_accessibility_smooth_1,
        np.array(['nAA_24_180_pae']),
        [10])

    alphafold_accessibility_smooth_2['IDR'] = np.where(
        alphafold_accessibility_smooth_2['nAA_24_180_pae_smooth10'] <= 34.27, 1, 0)

    return alphafold_accessibility_smooth_2


def annotate_short_IDRs(alphafold_accessibility_smooth):
    alphafold_accessibility_smooth_pattern = annotate_proteins_with_idr_pattern(
        alphafold_accessibility_smooth,
        min_structured_length=80,
        max_unstructured_length=20)

    alphafold_accessibility_smooth_pattern_ext = get_extended_flexible_pattern(
        alphafold_accessibility_smooth_pattern,
        ['flexible_pattern'], [5])

    return alphafold_accessibility_smooth_pattern_ext


def perform_annotations(proteins, cif_dir, pae_dir):
    alphafold_annotation = load_protein_data_from_file(proteins, cif_dir)
    alphafold_accessibility = annotate_pPSE_values(alphafold_annotation, pae_dir)
    alphafold_accessibility_smooth = annotate_IDRs(alphafold_accessibility)
    alphafold_accessibility_smooth_pattern_ext = annotate_short_IDRs(alphafold_accessibility_smooth)

    return alphafold_accessibility_smooth_pattern_ext


# Independent 


