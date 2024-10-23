import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd

from scipy.stats import *


def extract_features(time_series, debug_info=False):
    features = []
    i = 0
    for t in time_series.itertuples():
        dates = t[1]
        values = t[2]

        first_date = dates[0]
        last_date = dates[-1]
        mean = np.mean(values)
        median = np.median(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        skewness = skew(values)
        kurt = kurtosis(values)

        length = len(values)
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)

        values = pd.Series(values)
        values.dropna(inplace=True)

        if len(values) == 0:
            slope = 0
            intercept = 0
        else:
            poly = Polynomial.fit(range(len(values)), values, deg=1)
            slope = poly.coef[1]
            intercept = poly.coef[0]

        if len(values) == 0:
            total_amplitude = 0
            dominant_frequency = 0
        else:
            fft = np.fft.fft(values)
            freq = np.fft.fftfreq(len(values))
            fft_amplitudes = np.abs(fft)
            dominant_frequency_index = np.argmax(fft_amplitudes[1:]) + 1

            dominant_frequency = np.abs(freq[dominant_frequency_index])
            total_amplitude = np.sum(fft_amplitudes)

        features.append([mean, median, std, min_val, max_val, skewness, kurt, length, q25, q75,
                         first_date.month, first_date.year, last_date.month, last_date.year,
                         slope, intercept, total_amplitude, dominant_frequency])
        i += 1

        if i % 5000 == 0 and debug_info:
            print(i, end=' ')

    return pd.DataFrame(features)

    # Решения сверху и снизу работают примерно одинаково:

    # time_series['mean'] = time_series['values'].apply(lambda x: np.mean(x))
    # time_series['std'] = time_series['values'].apply(lambda x: np.std(x))
    # time_series['min_val'] = time_series['values'].apply(lambda x: np.min(x))
    # time_series['max_val'] = time_series['values'].apply(lambda x: np.max(x))
    # time_series['skewness'] = time_series['values'].apply(lambda x: skew(x))
    # time_series['kurtosis'] = time_series['values'].apply(lambda x: kurtosis(x))
    # time_series['length'] = time_series['values'].apply(lambda x: len(x))
    # time_series['q25'] = time_series['values'].apply(lambda x: np.percentile(x, 25))
    # time_series['q75'] = time_series['values'].apply(lambda x: np.percentile(x, 75))
    # time_series['first_date_year'] = time_series['dates'].apply(lambda x: x[0].year)
    # time_series['first_date_month'] = time_series['dates'].apply(lambda x: x[0].month)
    # time_series['last_date_year'] = time_series['dates'].apply(lambda x: x[-1].year)
    # time_series['last_date_month'] = time_series['dates'].apply(lambda x: x[-1].month)
    #
    # time_series.drop(['dates', 'values'], axis=1)
    #
    # return time_series
