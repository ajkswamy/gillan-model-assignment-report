import pymc3
import arviz
import numpy as np
from input_related import InputManager
import sys
import math


def fit_stage2_params(
        choice2_arr, state2_arr, rewards_arr):

    assert len(choice2_arr) == len(state2_arr) == len(rewards_arr), \
        f"Size of inputs inconsistent: " \
        f"choice2 ({len(choice2_arr)}, state2 ({len(state2_arr)}), rewards_arr ({len(rewards_arr)})"

    n_trials = len(choice2_arr)

    with pymc3.Model() as stage2_model:

        # Priors for parameters
        alpha = pymc3.Normal(name='alpha', sigma=0.5 / 3, mu=0.5)
        beta = pymc3.Normal(name='beta', sigma=0.5 / 3, mu=0.5)

        # Value function
        # first dimension state2, second dimension choice
        q2t_arr = np.empty((2, 2, n_trials + 1), dtype=object)

        for c in (0, 1):
            for s in (0, 1):
                q2t_arr[s, c, 0] = math.log(0.5) / beta

        ct_arr = np.empty((n_trials,), dtype=object)
        for trial_ind, (st, rt, ct2) in enumerate(zip(state2_arr, rewards_arr, choice2_arr)):

            llc0_unnormed = pymc3.math.exp(q2t_arr[st, 0, trial_ind] * beta)
            llc1_unnormed = pymc3.math.exp(q2t_arr[st, 1, trial_ind] * beta)
            norm = llc0_unnormed + llc1_unnormed
            llc0 = llc0_unnormed / norm
            # llc1 = llc1_unnormed / norm

            # observation
            ct_arr[trial_ind] = pymc3.Bernoulli(name=f'c2{trial_ind}', p=llc0, observed=choice2_arr[trial_ind])

            for c in (0, 1):
                for s in (0, 1):
                    if c == ct2 and s == st:
                        q2t_arr[s, c, trial_ind + 1] = (1 - alpha) * q2t_arr[st, ct2, trial_ind] + rt
                    else:
                        q2t_arr[s, c, trial_ind + 1] = q2t_arr[st, ct2, trial_ind]

        # map_estimate = pymc3.find_MAP(model=stage2_model)
        # print(map_estimate)

        trace = pymc3.sample(500, return_inferencedata=False, cores=3)
        print(arviz.summary(trace, round_to=2))


if __name__ == '__main__':

    assert len(sys.argv) == 2, \
        f"Could not understand usage. Please use as:\n" \
        f"python {__name__} <data_folder>"

    input_manager = InputManager(data_folder=sys.argv[1])
    random_sample_df = input_manager.get_random_sample()
    fit_stage2_params(
        choice2_arr=random_sample_df["choice2"].values,
        state2_arr=random_sample_df["state2"].values,
        rewards_arr=random_sample_df["reward"].values
    )
