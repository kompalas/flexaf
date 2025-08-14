import pandas as pd
import argparse
import os


def cleanup_feature_results(results_file):
    """Read the results file and removes identical rows, cleans column titles.
       The filtered results are stored in the same file"""
    assert os.path.exists(results_file) and os.path.isfile(results_file), 'File not found'
    assert results_file.endswith('.csv'), 'Unsupported file format. Please provide a CSV file.'

    df = pd.read_csv(results_file)
    # rename the columns
    df = df.rename(columns={'feature': 'FeatureName',
                            'num_samples': 'NumSamples',
                            'input_precision': 'Precision',
                            'area': 'Area(um2)',
                            'power': 'Power(mW)',
                            'delay': 'Delay(ns)'})
    # remove rows with identical FeatureName, NumSamples and InputPrecision
    df = df.drop_duplicates(subset=['FeatureName', 'NumSamples', 'InputPrecision'])

    # store the filtered results in the same file
    df.to_csv(results_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--results-file', type=str, required=True)
    # args = parser.parse_args()
    results_file = '/home/balaskas/pestress/hw_eval/features/results.csv'
    cleanup_feature_results(results_file)


if __name__ == '__main__':
    main()
