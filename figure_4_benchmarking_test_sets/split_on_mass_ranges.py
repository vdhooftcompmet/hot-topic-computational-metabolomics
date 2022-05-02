import random
import os
import pickle
from calculate_binned_average_rmse import calculate_binned_average_rmse


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def create_stratified_test_set(all_test_spectra):
    bins = [0, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 5000]
    spectra_split_on_mass = {}
    for i in range(len(bins)-1):
        min = bins[i]
        max = bins[i+1]
        selected_spectra = [spectrum for spectrum in all_test_spectra if spectrum.get("parent_mass") > min and spectrum.get("parent_mass") <= max]
        spectra_split_on_mass[f"mass_{min}_{max}"] = random.sample(selected_spectra, k=1500)
    return spectra_split_on_mass


if __name__ == "__main__":
    path_root = os.path.dirname(os.getcwd())
    path_files_folder = os.path.join(path_root, "../../../data/hot_topics_metabolomics/")
    testing_spectra = load_pickled_file(os.path.join(path_files_folder, "all_testing_spectra.pickle"))
    spectra_split_on_mass = create_stratified_test_set(testing_spectra)

    tanimoto_score_df = load_pickled_file(os.path.join(path_root,
                                                       "../../../data/libraries_and_models/gnps_15_12_2021/in_between_files/GNPS_15_12_2021_pos_tanimoto_scores.pickle"))
    ms2ds_model_file = os.path.join(path_files_folder, "ms2deepscore_model_with_spectrumbinner.hdf5")
    for mass_range in spectra_split_on_mass:
        testing_spectra = spectra_split_on_mass[mass_range]
        binned_average_rmse = calculate_binned_average_rmse(testing_spectra, tanimoto_score_df, ms2ds_model_file)
        print(mass_range, ": ", binned_average_rmse)
