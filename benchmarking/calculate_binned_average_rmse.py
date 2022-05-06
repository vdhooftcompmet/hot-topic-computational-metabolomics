from typing import List
import numpy as np
import pandas as pd
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
from matchms import Spectrum


def tanimoto_dependent_losses(scores, scores_ref, ref_score_bins):
    """Compute errors (RMSE and MSE) for different bins of the reference scores (scores_ref).

    Parameters
    ----------

    scores
        Scores that should be evaluated
    scores_ref
        Reference scores (= ground truth).
    ref_score_bins
        Bins for the refernce score to evaluate the performance of scores.
    """
    rmses = []
    ref_scores_bins_inclusive = ref_score_bins.copy()
    ref_scores_bins_inclusive[0] = -np.inf
    ref_scores_bins_inclusive[-1] = np.inf
    for i in range(len(ref_scores_bins_inclusive) - 1):
        low = ref_scores_bins_inclusive[i]
        high = ref_scores_bins_inclusive[i + 1]
        idx = np.where((scores_ref >= low) & (scores_ref < high) & (~np.eye(scores_ref.shape[0], dtype=bool)))
        rmses.append(np.sqrt(np.square(scores_ref[idx] - scores[idx]).mean()))
    return rmses


def do_ms2ds_predictions(test_spectra, ms2ds_model):
    model = load_model(ms2ds_model)
    similarity_measure = MS2DeepScore(model)
    # Calculate scores and get matchms.Scores object
    scores = similarity_measure.matrix(test_spectra, test_spectra, is_symmetric=True)
    return scores


def select_predictions_for_test_spectra(tanimoto_df: pd.DataFrame,
                                        test_spectra: List[Spectrum]) -> np.ndarray:
    """Select the predictions for test_spectra from df with correct predictions

    tanimoto_df:
        Dataframe with as index and columns Inchikeys of 14 letters
    test_spectra:
        list of test spectra
    """
    inchikey_idx_test = np.zeros(len(test_spectra))
    for i, spec in enumerate(test_spectra):
        inchikey_idx_test[i] = np.where(tanimoto_df.index.values == spec.get("inchikey")[:14])[0]

    inchikey_idx_test = inchikey_idx_test.astype("int")
    scores_ref = tanimoto_df.values[np.ix_(inchikey_idx_test[:], inchikey_idx_test[:])].copy()
    return scores_ref


def calculate_binned_average_rmse(testing_spectra, tanimoto_score_df, ms2ds_model_file):
    # plot_parent_mass_distribution(testing_spectra)
    correct_scores = select_predictions_for_test_spectra(tanimoto_score_df, testing_spectra)

    predicted_scores = do_ms2ds_predictions(testing_spectra, ms2ds_model_file)
    rmses = tanimoto_dependent_losses(predicted_scores, correct_scores,
                                                                 np.linspace(0, 1.0, 11))
    binned_average_rmse = sum(rmses)/len(rmses)
    return binned_average_rmse
