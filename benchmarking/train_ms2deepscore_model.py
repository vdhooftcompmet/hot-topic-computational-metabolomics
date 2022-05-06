import os
import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Union
import h5py
from tensorflow import keras
from tensorflow.keras.callbacks import (  # pylint: disable=import-error
    EarlyStopping, ModelCheckpoint)
from tensorflow.keras.optimizers import Adam  # pylint: disable=import-error
from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import DataGeneratorAllInchikeys
from ms2deepscore.models import SiameseModel


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def train_ms2deepscore_model(spectrums_training, spectrums_val, tanimoto_df, output_folder):
    # Create binned spectra
    spectrum_binner = SpectrumBinner(10000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5,
                                     allowed_missing_percentage=10.0)
    binned_spectrums_training = spectrum_binner.fit_transform(spectrums_training)
    binned_spectrums_val = spectrum_binner.transform(spectrums_val)

    # Select unique Inchikeys
    training_inchikeys = np.unique([s.get("inchikey")[:14] for s in spectrums_training])

    same_prob_bins = list(zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1, 10)))
    dimension = len(spectrum_binner.known_bins)
    training_generator = DataGeneratorAllInchikeys(
        binned_spectrums_training, training_inchikeys, tanimoto_df, dim=dimension,
        same_prob_bins=same_prob_bins, num_turns=2, augment_noise_max=10, augment_noise_intensity=0.01)

    validation_inchikeys = np.unique([s.get("inchikey")[:14] for s in spectrums_val])
    validation_generator = DataGeneratorAllInchikeys(
        binned_spectrums_val, validation_inchikeys, tanimoto_df, dim=dimension, same_prob_bins=same_prob_bins,
        num_turns=10, augment_removal_max=0, augment_removal_intensity=0, augment_intensity=0, augment_noise_max=0, use_fixed_set=True)
    model = SiameseModel(spectrum_binner, base_dims=(500, 500), embedding_dim=200,
                         dropout_rate=0.2)
    print(model.summary())

    # Save best model and include early stopping
    model.compile(loss='mse', optimizer=Adam(lr=0.01), metrics=["mae", tf.keras.metrics.RootMeanSquaredError()])
    checkpointer_model_file_name = os.path.join(output_folder, "ms2deepscore_checkpoint_model.hdf5")
    checkpointer = ModelCheckpoint(filepath=checkpointer_model_file_name, monitor='val_loss', mode="min", verbose=1, save_best_only=True)
    earlystopper_scoring_net = EarlyStopping(monitor='val_loss', mode="min", patience=10, verbose=1)

    history = model.model.fit(training_generator, validation_data=validation_generator,
                              epochs=150, verbose=1, callbacks=[earlystopper_scoring_net, checkpointer])

    # Save history
    filename = os.path.join(output_folder, '_training_history.pickle')
    with open(filename, 'wb') as f:
        pickle.dump(history.history, f)
    model_file_name = os.path.join(output_folder, "final_ms2deepscore_model.hdf5")
    model.save(model_file_name)


def save_model_with_spectrum_binner(filename: Union[str, Path]):
    """Saves the MS2Deepscore model with spectrum_binner information"""
    with h5py.File(filename, mode='r') as f:
        keras_model = keras.models.load_model(f)
    spectrum_binner = SpectrumBinner(10000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5,
                                     allowed_missing_percentage=100.0)
    training_spectra = load_pickled_file("C:/Users/jonge094/PycharmProjects/PhD_MS2Query/ms2query/data/hot_topics_metabolomics/training_spectra.pickle")
    spectrum_binner.fit_transform(training_spectra)
    SiameseModel(spectrum_binner, keras_model=keras_model).save("C:/Users/jonge094/PycharmProjects/PhD_MS2Query/ms2query/data/hot_topics_metabolomics/ms2deepscore_model_with_spectrumbinner.hdf5")


if __name__ == "__main__":
    path_root = os.path.dirname(os.getcwd())
    path_files_folder = os.path.join(path_root, "../../data/hot_topics_metabolomics/")
    training_spectra = load_pickled_file(os.path.join(path_files_folder, "training_spectra.pickle"))
    validation_spectra = load_pickled_file(os.path.join(path_files_folder, "validation_spectra.pickle"))

    tanimoto_score_df = load_pickled_file(os.path.join(path_root, "../../data/libraries_and_models/gnps_15_12_2021/in_between_files/GNPS_15_12_2021_pos_tanimoto_scores.pickle"))
    train_ms2deepscore_model(training_spectra, validation_spectra, tanimoto_score_df, os.path.join(path_files_folder, "ms2deepscore_model.hdf5"))
    save_model_with_spectrum_binner(os.path.join(path_files_folder, "ms2deepscore_model.hdf5"))