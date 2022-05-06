import os
import pickle
import random
import pandas as pd
from calculate_binned_average_rmse import calculate_binned_average_rmse


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def create_super_class_dict(classifiers_file, test_spectra):
    """Creates a dictionary with spectra sorted per superclass"""
    classifiers_df = pd.read_csv(classifiers_file, sep="\t")[["cf_superclass", "inchi_key"]]

    superclasses = classifiers_df["cf_superclass"].unique()
    superclass_dict = {}
    for super_class in superclasses:
        inchikeys = list(classifiers_df[classifiers_df["cf_superclass"] == super_class]["inchi_key"])
        inchikeys_14 = [inchikey[:14] for inchikey in inchikeys]
        unique_inchikeys = set(inchikeys_14)
        superclass_dict[super_class] = unique_inchikeys
    spectra_per_class = {}
    for super_class in superclass_dict:
        # select spectra
        spectra_in_class = [spectrum for spectrum in test_spectra
                            if spectrum.get("inchikey")[:14] in superclass_dict[super_class]]
        spectra_per_class[super_class] = spectra_in_class
    return spectra_per_class


if __name__ == "__main__":
    path_root = os.path.dirname(os.getcwd())
    path_files_folder = os.path.join(path_root, "../../../data/hot_topics_metabolomics/")
    test_spectra = load_pickled_file(os.path.join(path_files_folder, "all_testing_spectra.pickle"))

    classifiers_csv_file = "C:/Users/jonge094/PycharmProjects/PhD_MS2Query/ms2query/data/libraries_and_models/gnps_09_04_2021/ALL_GNPS_210409_positive_processed_annotated_CF_NPC_classes.txt"
    tanimoto_score_df = load_pickled_file(os.path.join(path_root,
                                                       "../../../data/libraries_and_models/gnps_15_12_2021/in_between_files/GNPS_15_12_2021_pos_tanimoto_scores.pickle"))
    ms2ds_model_file = os.path.join(path_files_folder, "ms2deepscore_model_with_spectrumbinner.hdf5")

    spectra_per_class = create_super_class_dict(classifiers_csv_file, test_spectra)

    for super_class in spectra_per_class:
        if len(spectra_per_class[super_class]) > 1500:
            testing_spectra = random.sample(spectra_per_class[super_class], k=1500)
            binned_average_rmse = calculate_binned_average_rmse(testing_spectra, tanimoto_score_df, ms2ds_model_file)
            print(super_class, ": ", binned_average_rmse)
