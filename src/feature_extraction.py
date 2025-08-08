import numpy as np
from scipy.stats import skew, kurtosis

def extract_advanced_features(windows):
    """
    Извлекает статистические признаки из каждого окна.

    Args:
        windows (np.ndarray): Массив окон формы (N_окна, длина_окна, число_сенсоров)

    Returns:
        np.ndarray: Массив признаков формы (N_окна, число_признаков)
    """
    if len(windows) == 0:
        return np.array([])

    mean_features = np.mean(windows, axis=1)
    median_features = np.median(windows, axis=1)
    std_features = np.std(windows, axis=1)
    min_features = np.min(windows, axis=1)
    max_features = np.max(windows, axis=1)
    range_features = max_features - min_features

    q1_features = np.quantile(windows, 0.25, axis=1)
    q3_features = np.quantile(windows, 0.75, axis=1)
    iqr_features = q3_features - q1_features

    diff_features = np.mean(np.abs(np.diff(windows, axis=1)), axis=1)

    skew_features = skew(windows, axis=1)
    kurtosis_features = kurtosis(windows, axis=1)

    features = np.hstack([
        mean_features,
        median_features,
        std_features,
        min_features,
        max_features,
        range_features,
        q1_features,
        q3_features,
        iqr_features,
        diff_features,
        skew_features,
        kurtosis_features
    ])

    if features.ndim > 2:
        return features.reshape(features.shape[0], -1)
    return features
