import json
import sys
import pathlib as pl
import arviz
import pandas as pd
from input_related import InputManager


def write_summary_table(
        map_estimate_file: str, sampling_estimate_file: str,
        subjects: list, output_filename_without_extension: str):
    """
    Reading results of MAP estimation and NUTS sampling and writes a summary table in CSV format.
    :param pathlib.Path|str map_estimate_file: JSON file containing results of MAP estimation
    :param pathlib.Path|str sampling_estimate_file: NetCDF(.NC) file containing results of NUTS sampling
    :param list subjects: list of subject labels used when fitting
    :param pathlib.Path|str output_filename_without_extension: output will be written to this path with .CSV extension
    :return: None
    """

    with open(map_estimate_file) as fp:
        map_estimates = json.load(fp)

    traces = arviz.from_netcdf(sampling_estimate_file)

    traces_summary = arviz.summary(traces)

    group_level_parameters = [
        "mu_alpha", "sigma_alpha",
        "mu_beta", "sigma_beta"
    ]

    subject_level_parameters = {
        "alpha": lambda subject: (f"alpha_{subject}", None),
        "beta2": lambda subject: (f'beta_{subject}', 0),
        "beta_mb": lambda subject: (f'beta_{subject}', 1),
        "beta_mf0": lambda subject: (f'beta_{subject}', 2),
        "beta_mf1": lambda subject: (f'beta_{subject}', 3),
        "beta_st": lambda subject: (f'beta_{subject}', 4)
    }

    metadata_df = pd.DataFrame(dtype=str)
    data_df = pd.DataFrame(dtype=float)

    for glp in group_level_parameters:

        temp_s = pd.Series(dtype=str)
        temp_s["Parameter"] = glp
        temp_s["Subject/Group-level"] = "Group-level"

        metadata_df = metadata_df.append(pd.DataFrame(temp_s).T, ignore_index=True)

        temp_f = pd.Series(dtype=float)
        temp_f['Map Estimate'] = map_estimates[glp]
        temp_f['Sampling Estimate:\nPosterior Mean'] = traces_summary.loc[glp, "mean"]
        temp_f['Sampling Estimate:\nPosterior Standard Deviation'] = traces_summary.loc[glp, "sd"]
        temp_f['Gelman-Rubin diagnostic'] = traces_summary.loc[glp, "r_hat"]

        data_df = data_df.append(pd.DataFrame(temp_f).T, ignore_index=True)

    for subject in subjects:

        for slp, slp_name_func in subject_level_parameters.items():
            temp_s = pd.Series(dtype=str)
            temp_s["Parameter"] = slp
            temp_s["Subject/Group-level"] = subject

            metadata_df = metadata_df.append(pd.DataFrame(temp_s).T, ignore_index=True)

            temp_f = pd.Series(dtype=float)
            var_name, index = slp_name_func(subject)
            if index is not None:
                traces_var_name = f"{var_name}[{index}]"
                temp_f['Map Estimate'] = map_estimates[var_name][index]
            else:
                traces_var_name = var_name
                temp_f["Map Estimate"] = map_estimates[var_name]

            temp_f['Sampling Estimate:\nPosterior Mean'] = \
                traces_summary.loc[traces_var_name, "mean"]
            temp_f['Sampling Estimate:\nPosterior Standard Deviation'] = \
                traces_summary.loc[traces_var_name, "sd"]
            temp_f['Gelman-Rubin diagnostic'] = \
                traces_summary.loc[traces_var_name, "r_hat"]

            data_df = data_df.append(pd.DataFrame(temp_f).T, ignore_index=True)

    op_df = pd.concat([metadata_df, data_df], axis=1).set_index("Parameter")
    op_df.to_csv(f"{output_filename_without_extension}.csv", float_format='%.3f')


if __name__ == '__main__':

    assert len(sys.argv) == 4, \
        f"Could not understand usage! Please use as\n" \
        f"python {__file__} <folder containing raw csvs> <number of subjects> <number of trials>"

    ip_dir_path = pl.Path(sys.argv[1])
    n_subjects = sys.argv[2]
    n_trials = sys.argv[3]

    input_manager = InputManager(ip_dir_path)
    subjects = [x.stem for x in input_manager.input_files[:int(n_subjects)]]

    op_folder = input_manager.get_op_folder()
    suffix = f"multilevel_{n_subjects}subjects_{n_trials}trials"

    write_summary_table(
        map_estimate_file=op_folder / f"map_estimate_{suffix}.json",
        sampling_estimate_file=op_folder / f"sampling_results_{suffix}.nc",
        subjects=subjects,
        output_filename_without_extension=op_folder / f"summary_{suffix}"
    )
