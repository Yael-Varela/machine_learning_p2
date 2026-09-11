'''
dataset.py

Builds the feature dataframe from the REHAB .npy files. Each of the 16
exercises has two files (sensor group 1 + hand sensor), each with 6
channels, combined into 12 channels per exercise. For each sample (a
repetition of an exercise), 6 features are computed per channel: min, max,
range, std, mean, median.
'''

import re

import numpy as np
import pandas as pd

CHANNELS_SENSOR1 = ['pitch1', 'yaw1', 'roll1', 'pitch2', 'yaw2', 'roll2']
CHANNELS_SENSOR2 = ['f1', 'f2', 'f3', 'f4', 'f5', 'pitch3']
ALL_CHANNELS = CHANNELS_SENSOR1 + CHANNELS_SENSOR2

FILENAME_PATTERN = re.compile(r'^(\d{3})_(\d)\.npy$')


def scan_files(folder):
    '''
    Groups the .npy files found in folder by movement_id, indexing each
    movement by its sensor_id (1 or 2).
    '''
    movements = {}
    for f in sorted(folder.glob('*.npy')):
        matching = FILENAME_PATTERN.match(f.name)
        if not matching:
            print(f'there is a problem with this file {f.name}')
            continue
        move_id = matching.group(1)
        sensor_id = int(matching.group(2))
        movements.setdefault(move_id, {})[sensor_id] = f
    return movements


def calculate_features(sample):
    '''
    Computes 6 features that summarize a single channel signal in a few
    representative values.
    '''
    sample_min = float(np.min(sample))
    sample_max = float(np.max(sample))
    sample_range = sample_max - sample_min
    sample_std = float(np.std(sample))
    sample_mean = float(np.mean(sample))
    sample_median = float(np.median(sample))
    return {
        'min': sample_min, 'max': sample_max, 'range': sample_range,
        'std': sample_std, 'mean': sample_mean, 'median': sample_median,
    }


def build_dataframe(movements_dictionary):
    '''
    Receives a dictionary containing each of the 16 movements. Each movement
    consists of two files, corresponding to two sensors, each with 6
    channels, which are concatenated to yield 12 channels per movement.
    Iterates over each movement (first loop), each of its samples (second
    loop), and each of the 12 channels (third loop), computing features for
    each channel.
    '''
    rows = []
    for mov_id in movements_dictionary:
        sensors = movements_dictionary[mov_id]

        # exercise 14 is missing one of its two files, so it is skipped
        if 1 not in sensors or 2 not in sensors:
            continue

        first_batch_sensors = np.load(sensors[1])
        hand_sensor = np.load(sensors[2])
        combined_channels = np.concatenate([first_batch_sensors, hand_sensor], axis=-1)

        samples = combined_channels.shape[0]
        for sample_id in range(samples):
            row = {
                'movement_id': int(mov_id),
                'sample_id': f'{mov_id}_{sample_id}',
            }
            for channel_id, channel_name in enumerate(ALL_CHANNELS):
                feats = calculate_features(combined_channels[sample_id, :, channel_id])
                for feat_name, feat_value in feats.items():
                    row[f'{channel_name}_{feat_name}'] = feat_value
            rows.append(row)

    return pd.DataFrame(rows)


ID_COLUMNS = ['movement_id', 'sample_id']
def dataframe_to_arrays(dataframe):
    '''
    Separates the dataframe into features (X) and the label (y).
    '''
    feature_columns = [c for c in dataframe.columns if c not in ID_COLUMNS]
    X = dataframe[feature_columns].to_numpy(dtype=float)
    y = dataframe['movement_id'].to_numpy(dtype=int)
    return X, y