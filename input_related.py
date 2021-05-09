import pathlib as pl
import pandas as pd
import random


def read_single_data_file(csv_file):

    column_names = [
        "trial_number", "drift1", "drift2", "drift3", "drift4",
        "choice1", "stimulus1", "reaction_time1", "is_transition_common",
        "choice2", "stimulus2", "state2", "reaction_time2", "reward", "dummy"
    ]

    df = pd.read_csv(csv_file, header=None, index_col=None, names=column_names)
    
    inds_to_drop = []
    for ind, row in df.iterrows():
        inds_to_drop.append(ind)
        if row.iloc[2] == 'twostep_instruct_9':
            break
    else:
        raise IOError(
            f"No trail data found in file {csv_file}! "
            f"I could not find any row with 'twostep_instruct_9' in its third column!")
    
    trial_data_df = df.drop(index=inds_to_drop)
    trial_data_df["choice1"] = trial_data_df["choice1"].apply(lambda x: 0 if x == "left" else 1)
    trial_data_df["choice2"] = trial_data_df["choice2"].apply(lambda x: 0 if x == "left" else 1)
    trial_data_df["state2"] = trial_data_df["state2"].apply(lambda x: 0 if x == 2 else 1)

    return trial_data_df


class InputManager(object):

    def __init__(self, data_folder):

        data_folder_path = pl.Path(data_folder)
        assert data_folder_path.is_dir(), f"Input data folder {data_folder} not found!"

        self.input_files = [x for x in data_folder_path.iterdir() if x.suffix == ".csv"]

    def get_iterator(self):

        for csv in self.input_files:

            yield read_single_data_file(csv)

    def get_random_sample(self):

        return read_single_data_file(random.sample(self.input_files, 1)[0])

