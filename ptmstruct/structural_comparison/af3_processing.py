import os
import json
import shutil
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import h5py

def copy_and_rename_file(source_file, destination_path, new_name):
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    destination_file = os.path.join(destination_path, new_name)

    shutil.copy2(source_file, destination_file)


def extract_af3_model_path(alphafold_model_path, res_number):
    """Allways takes model 0"""
    full_data_path = ""
    model_path = ""
    for file_name in sorted(os.listdir(alphafold_model_path)):
        if (f"full_data_{res_number}" in file_name) and ("Zone.Identifier" not in file_name):
            full_data_path = os.path.join(alphafold_model_path, file_name)
            continue
        if (f"model_{res_number}" in file_name) and ("Zone.Identifier" not in file_name):
            model_path = os.path.join(alphafold_model_path, file_name)

    return full_data_path, model_path


def extract_pae_and_plddts(input_path, output_path, plddts_list):
    with open(input_path) as f:
        full_data = json.load(f)

    plddts_list.append(full_data['atom_plddts'])

    pae_matrix = np.array(full_data['pae'])
    token_res_ids = full_data['token_res_ids']

    unique_res_ids = sorted(set(token_res_ids))
    new_pae_matrix = np.zeros((len(unique_res_ids), len(unique_res_ids)))

    for i, res_i in enumerate(unique_res_ids):
        for j, res_j in enumerate(unique_res_ids):
            original_i = token_res_ids.index(res_i)
            original_j = token_res_ids.index(res_j)
            new_pae_matrix[i, j] = pae_matrix[original_i, original_j]

    matrix_flat = np.array(new_pae_matrix).flatten()

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('dist', data=matrix_flat)


def load_af3_data(alphafold_path, data_dir, af3_name_mapping, use_present_files = True):
    cif_dir = os.path.join(data_dir, "cif")
    pae_dir = os.path.join(data_dir, "pae")
    
    if use_present_files:
        if not os.path.exists(cif_dir):
            raise FileNotFoundError(f"cif_dir '{cif_dir}' does not exist.")
        
        if not os.path.exists(pae_dir):
            raise FileNotFoundError(f"pae_dir '{pae_dir}' does not exist.")
        
        selected_proteins = [
            filename.replace(".cif", "") 
            for filename in os.listdir(cif_dir) 
            if filename.endswith(".cif")
        ]

        with open(os.path.join(data_dir, 'plddts_dict.json')) as f:
            plddts_dict = json.load(f)
    
        return cif_dir, pae_dir, selected_proteins, plddts_dict
    
    pae_dir = os.path.join(data_dir, "pae")
    if os.path.exists(pae_dir):
        # If it exists, delete it
        shutil.rmtree(pae_dir)
        print(f"Directory '{pae_dir}' already existed and was deleted.")
    os.makedirs(pae_dir)

    cif_dir = os.path.join(data_dir, "cif")
    if os.path.exists(cif_dir):
        # If it exists, delete it
        shutil.rmtree(cif_dir)
        print(f"Directory '{cif_dir}' already existed and was deleted.")
    os.makedirs(cif_dir)

    """name_mapping_path = os.path.join(alphafold_path, 'name_mapping.json')

    with open(name_mapping_path, 'r') as file:
        af3_name_mapping = json.load(file)"""

    selected_proteins = []

    plddts_dict = {}

    for af3_dir_name in tqdm.tqdm(sorted(os.listdir(alphafold_path)), desc= "Converting Af3 predictions"):
        if not af3_dir_name in af3_name_mapping:
            continue
        protein_name = af3_name_mapping[af3_dir_name]
        alphafold_model_path = os.path.join(alphafold_path, af3_dir_name)
        plddts_dict[protein_name] = []
        for i in range(5):
            #print(f"Processing {protein_name} Model {i}")
            full_data_path, model_path = extract_af3_model_path(alphafold_model_path, res_number=i)
            af3_hdf = os.path.join(pae_dir, f"pae_{protein_name}_m{i}.hdf")
            extract_pae_and_plddts(full_data_path, af3_hdf, plddts_dict[protein_name])
            copy_and_rename_file(source_file=model_path, destination_path=cif_dir, new_name=f"{protein_name}_m{i}.cif")
            selected_proteins.append(f"{protein_name}_m{i}")

    with open(os.path.join(data_dir, 'plddts_dict.json'), 'w') as f:
        json.dump(plddts_dict, f)
     

    return cif_dir, pae_dir, selected_proteins, plddts_dict