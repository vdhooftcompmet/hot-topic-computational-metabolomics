import os
import random
import pickle
import numpy as np
from calculate_binned_average_rmse import calculate_binned_average_rmse


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def create_random_subsets(testing_spectra, nr_of_splits, tanimoto_score_df, ms2ds_model_file):
    random.seed(42)
    random.shuffle(testing_spectra)
    set_size = len(testing_spectra) / nr_of_splits
    rmses = []
    for i in np.linspace(0, len(testing_spectra) - set_size, nr_of_splits):
        start = int(i)
        end = int(i + set_size)
        binned_average_RMSE = calculate_binned_average_rmse(testing_spectra[start:end], tanimoto_score_df, ms2ds_model_file)
        print(binned_average_RMSE)
        print(i)
        rmses.append(binned_average_RMSE)
    return rmses


if __name__ == "__main__":
    path_root = os.path.dirname(os.getcwd())
    path_files_folder = os.path.join(path_root, "../../data/hot_topics_metabolomics/")
    testing_spectra = load_pickled_file(os.path.join(path_files_folder, "all_testing_spectra.pickle"))
    tanimoto_score_df = load_pickled_file(os.path.join(path_root,
                                                       "../../data/libraries_and_models/gnps_15_12_2021/in_between_files/GNPS_15_12_2021_pos_tanimoto_scores.pickle"))
    ms2ds_model_file = os.path.join(path_files_folder, "ms2deepscore_model_with_spectrumbinner.hdf5")
    print(create_random_subsets(testing_spectra, 10,
                                tanimoto_score_df,
                                ms2ds_model_file))
    print(create_random_subsets(testing_spectra, 100,
                                tanimoto_score_df,
                                ms2ds_model_file))
    print(create_random_subsets(testing_spectra, 1000,
                                tanimoto_score_df,
                                ms2ds_model_file))
