def create_af3_job(job_name, sequence, modifications = []):
    sequence_dict = {
        "proteinChain": {
            "sequence": sequence,
            "count": 1,
            "maxTemplateDate": "2018-01-20"
        }
    }
    if modifications: 
        sequence_dict["proteinChain"]["modifications"] = modifications

    base_job_dict = {
        "name": job_name,
        "modelSeeds": [],
        "sequences": [sequence_dict],
        "dialect": "alphafoldserver",
        "version": 1
    }

    return base_job_dict


def create_af3_comparison_job(transcript_id, structure_reference_dict, ptms = None):
    job_name = f"SysComp Job {transcript_id}"
    modifications = []
    if ptms:
        for ptm in ptms:
            ptm_location, ptm_aa, ptm_type = ptm 
            ptm_name = f"_{ptm_type}_{ptm_aa}_{ptm_location}"

            if ptm_aa == "S" and ptm_type == "p":
                ptm_code = "CCD_SEP"
            elif ptm_aa == "T" and ptm_type == "p":
                ptm_code = "CCD_TPO"
            elif ptm_aa == "Y" and ptm_type == "p":
                ptm_code = "CCD_PTR"
            elif ptm_aa == "K" and ptm_type == "ac":
                ptm_code = "CCD_ALY"
            else: 
                print(f"PTM not supported {ptm_name}")

            job_name += ptm_name

            mod_dict = {
                "ptmType": ptm_code,
                "ptmPosition": ptm_location
            }
            modifications.append(mod_dict)


    sequence = structure_reference_dict["present"][transcript_id]["sequence"]

    if ptms != None and len(modifications) == 0:
        print("Nothing to compare")
        return None
    
    return job_name, create_af3_job(job_name, sequence, modifications = modifications)


def create_af3_comparison_jobs(job_requests, structure_reference_dict):
    all_job_list = []
    base_job_names = {}
    name_napping = {}
    comps = []

    for transcript_id, ptms in job_requests:
        base_job_names[transcript_id] = None

    for transcript_id in base_job_names:
        job_name, res_job = create_af3_comparison_job(transcript_id, structure_reference_dict, ptms = None)
        if res_job:
            af3_job_name = job_name.lower().replace(" ", "_")
            name_napping[af3_job_name] = job_name.replace("SysComp Job ", "")
            all_job_list.append(res_job)

    for transcript_id, ptms in job_requests:
        if not ptms:
            continue
        job_name, res_job = create_af3_comparison_job(transcript_id, structure_reference_dict, ptms = ptms)
        if res_job:
            af3_job_name = job_name.lower().replace(" ", "_")
            name_napping[af3_job_name] = job_name.replace("SysComp Job ", "")
            all_job_list.append(res_job)
        comps.append((job_name.replace("SysComp Job ", ""), job_name.replace("SysComp Job ", "").split("_")[0]))
    return all_job_list, name_napping, comps


def create_job_request(top_n_df):
    job_requests = []
    for i, row in top_n_df.iterrows():
        p_id = row["protein_id"]
        aa = row["AA"]
        pos = row["position"]
        job_requests.append((p_id, [(pos, aa, "ac")]))
    return job_requests