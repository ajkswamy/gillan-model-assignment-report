import pymc3 as pm
import arviz
import numpy as np
from input_related import InputManager
import sys
import pandas as pd
import datetime
import json
import pathlib as pl
import gc


def logisitic(x):

    return 1 / (1 + pm.math.exp(x))


def get_value_diff_0m1(q_arr):

    return q_arr[0] - q_arr[1]


def fit_multilevel_model(data_df: pd.DataFrame, op_dir: pl.Path, n_trials: int = 200):
    """
    Constructs the model from Gillan et al. eLife 2016;5:e11305. DOI: 10.7554/eLife.11305, pg. 19-20 using implementational
    details from the supplementary information of Otto et al. PNAS 2013; DOI: 10.1073/pnas.1312011110.
    Runs MAP estimation and NUTS sampling. Results are written into <op_dir>. Output of NUTS sampling is stored
    in a NetCDF file that can be read and analyzed using the package `arviz`
    :param pandas.DataFrame data_df: Concatenation of DataFrames returned by input_related.read_single_data_file
    :param pathlib.Path op_dir: path of output folder; must exist
    :param int n_trials: number of trials to fit
    :return: None
    """
    unique_subjects = pd.unique(data_df["subject_id"])
    n_subjects = unique_subjects.shape[0]
    data_df.sort_values(by=["subject_id", "trial_number"], inplace=True)

    print(f"Using data of {n_subjects} subjects: {unique_subjects.tolist()}")

    with pm.Model() as multilevel_model:

        # Priors for group level params
        mu_alpha = pm.Uniform(name='mu_alpha', lower=0, upper=1, transform=None)
        sigma_alpha_log = pm.Exponential(name="sigma_alpha_log", lam=1.5, transform=None)
        sigma_alpha = pm.Deterministic(name='sigma_alpha', var=pm.math.exp(sigma_alpha_log))
        mu_beta = pm.Normal(name="mu_beta", mu=0, sigma=100)
        sigma_beta_log = pm.Exponential(name="sigma_beta_log", lam=1.5, transform=None)
        sigma_beta = pm.Deterministic(name='sigma_beta', var=pm.math.exp(sigma_beta_log))

        for subject_ind, (subject_id, subject_df) in enumerate(data_df.groupby("subject_id")):

            alpha = pm.Beta(name=f'alpha_{subject_id}',
                            alpha=mu_alpha * sigma_alpha, beta=(1 - mu_alpha) * sigma_alpha)
            beta = pm.Normal(name=f"beta_{subject_id}", mu=mu_beta, sigma=sigma_beta, shape=5)
            beta2, beta_mb, beta_mf0, beta_mf1, beta_st = beta[0], beta[1], beta[2], beta[3], beta[4]

            print(f"{datetime.datetime.now()} Doing {subject_id}")
            choice1_repeated = subject_df["choice1"].astype(bool).diff().fillna(False).astype(int)
            data_df.loc[subject_df.index.values, "choice1_repeated"] = choice1_repeated

            # Value function
            # first dimension state2, second dimension choice2
            q2_arr = np.zeros((2, 2), dtype=object)
            # dimension is choice1
            qmb_arr = np.zeros((2,), dtype=object)
            qmf0_arr = np.zeros((2,), dtype=object)
            qmf1_arr = np.zeros((2,), dtype=object)

            for trial_number, trial_df in subject_df.groupby("trial_number"):

                if trial_number > n_trials:
                    continue

                subject_trial_row = trial_df.iloc[0]
                st = subject_trial_row["state2"]
                c2t = subject_trial_row["choice2"]
                c1t = subject_trial_row["choice1"]
                rt = subject_trial_row["reward"]
                repeated_choice = choice1_repeated[trial_df.index.values[0]]

                p1 = logisitic(beta2 * get_value_diff_0m1(q2_arr[st, :]))

                # observations
                c2t_rv = pm.Bernoulli(
                    name=f'c2_{trial_number}_{subject_ind}', observed=c2t,
                    p=p1
                )

                q2_arr[st, c2t] = (1 - alpha) * q2_arr[st, c2t] + rt

                p1 = logisitic(
                    beta_mb * get_value_diff_0m1(qmb_arr) +
                    beta_mf0 * get_value_diff_0m1(qmf0_arr) +
                    beta_mf1 * get_value_diff_0m1(qmf1_arr) +
                    beta_st * repeated_choice
                    )
                c1t_rv = pm.Bernoulli(
                    name=f'c1_{trial_number}_{subject_ind}', observed=c1t,
                    p=p1
                )

                qmb_arr[0] = pm.math.maximum(q2_arr[0, 0], q2_arr[0, 1])
                qmb_arr[1] = pm.math.maximum(q2_arr[1, 0], q2_arr[1, 1])

                qmf0_arr[c1t] = (1 - alpha) * qmf0_arr[c1t] + q2_arr[st, c2t]
                qmf1_arr[c1t] = (1 - alpha) * qmf1_arr[c1t] + rt

        gc.collect()

        start = {
            "mu_alpha": 0.5, "sigma_alpha_log": 1,
            "mu_beta": 0.5, "sigma_beta_log": 1}

        print(f"{datetime.datetime.now()} MAP estimation started")
        map_estimate = pm.find_MAP(
            model=multilevel_model, progressbar=False, start=start
        )
        print(f"{datetime.datetime.now()} MAP estimation done")
        print(map_estimate)
        with open(op_dir / f"map_estimate_multilevel_{n_subjects}subjects_{n_trials}trials.json", "w") as fp:
            json.dump({k: v.tolist() for k, v in map_estimate.items()}, fp)

        print(f"{datetime.datetime.now()} sampling started")
        trace = pm.sample(
            draws=2000, tune=1000, return_inferencedata=False,
            compute_convergence_checks=True, progressbar=False, cores=4, start=start
        )
        print(f"{datetime.datetime.now()} sampling done")
        print(arviz.summary(trace, round_to=2))
        arviz.to_netcdf(
            data=trace, filename=op_dir / f"sampling_results_multilevel_{n_subjects}subjects_{n_trials}trials.nc")


if __name__ == '__main__':

    assert len(sys.argv) == 2, \
        f"Could not understand usage. Please use as:\n" \
        f"python {__name__} <folder containing raw csvs>"

    n_trials = 20
    n_subjects = 5

    print(f"Using {n_trials} trials")

    input_manager = InputManager(data_folder=sys.argv[1])
    sample_df = input_manager.get_data_top(n_subjects)

    op_dir = input_manager.get_op_folder()
    op_dir.mkdir(exist_ok=True)
    fit_multilevel_model(data_df=sample_df, op_dir=op_dir, n_trials=n_trials)

