import pathlib as pl
import pandas as pd
import random


def read_single_data_file(csv_file):
    """
    Read data from single CSV file. Values of columns for "choice1" and "choice2" are mapped as: left -> 0, right -> 1
    Raises IOError if 'twostep_instruct_9' is not found in the third column of the file.
    :param csv_file: input file
    :rtype: pandas.DataFrame
    :return: Dataframe containing the data read
    """

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
    trial_data_df["subject_id"] = pl.Path(csv_file).stem
    trial_data_df["trial_number"] = pd.to_numeric(trial_data_df["trial_number"])

    return trial_data_df


class InputManager(object):
    """
    Class for managing input and output paths and files for model fitting
    """

    def __init__(self, data_folder: str):
        """
        Inspect and note available CSV files
        :param str data_folder: path of the folder containing input CSV files
        """
        self.data_folder_path = pl.Path(data_folder)
        assert self.data_folder_path.is_dir(), f"Input data folder {data_folder} not found!"

        self.input_files = sorted([x for x in self.data_folder_path.iterdir() if x.suffix == ".csv"])

    def get_iterator_all_csv(self):
        """
        Returns an iterator generator over all CSV files
        """

        for csv in self.input_files:

            yield read_single_data_file(csv)

    def get_iterator_random_n_csv(self, n: int):
        """
        Returns an iterator generator over `n` available CSV files, chosen randomly
        :param int n: number of files
        """

        for csv in random.sample(self.input_files, n):

            yield read_single_data_file(csv)

    def get_data_top(self, n: int = 1):
        """
        Return the data of `n` CSV files concatenated into single pandas Dataframe
        :param int n: number of files
        """

        return pd.concat([read_single_data_file(csv) for csv in self.input_files[:n]], axis=0)

    def get_op_folder(self):
        """
        Return the output folder, which is a sibling of the folder containing CSV files and with the name "Results"
        """

        return self.data_folder_path.parent / "Results"
