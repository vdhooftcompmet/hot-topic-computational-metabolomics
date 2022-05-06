import numpy as np
import pickle

# Import data
def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object

def split_into_train_and_val(all_spectra, nr_of_val, nr_of_test):
    n_spectra = len(all_spectra)
    n_train = n_spectra - nr_of_val - nr_of_test

    spectrum_ids = np.arange(n_spectra)
    # Select training, validation, and test IDs:
    train_spectrumIDs = np.random.choice(spectrum_ids, n_train, replace=False)
    val_spectrumIDs = np.random.choice(list(set(spectrum_ids) - set(train_spectrumIDs)), nr_of_val, replace=False)
    test_spectrumIDs = list(set(spectrum_ids) - set(train_spectrumIDs) - set(val_spectrumIDs))
    train_split = [all_spectra[i] for i in train_spectrumIDs]
    val_split = [all_spectra[i] for i in val_spectrumIDs]
    test_split = [all_spectra[i] for i in test_spectrumIDs]
    return train_split, val_split, test_split

library_spectra = load_pickled_file("C:/Users/jonge094/PycharmProjects/PhD_MS2Query/ms2query/data/libraries_and_models/gnps_15_12_2021/in_between_files/ALL_GNPS_15_12_2021_positive_annotated.pickle")

train, val, test = split_into_train_and_val(library_spectra, 10000, 100000)

pickle.dump(train, open("C:/Users/jonge094/PycharmProjects/PhD_MS2Query/ms2query/data/hot_topics_metabolomics/training_spectra.pickle","wb"))
pickle.dump(val, open("C:/Users/jonge094/PycharmProjects/PhD_MS2Query/ms2query/data/hot_topics_metabolomics/training_spectra.pickle","wb"))
pickle.dump(test, open("C:/Users/jonge094/PycharmProjects/PhD_MS2Query/ms2query/data/hot_topics_metabolomics/training_spectra.pickle","wb"))
