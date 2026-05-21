"""Stratified train/val splitting utilities shared across dataset pipelines."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def filter_single_subgroups(data: pd.DataFrame, strata: List[str]) -> pd.DataFrame:
    """Filter out rows that have a single observation in the stratas."""

    # Perform a value count over the stratas
    val_counts = data[strata].value_counts()

    # Create a dataframe with stratas that have only one observation
    single_stratas = val_counts[val_counts == 1].reset_index()

    if single_stratas.empty:
        # If no single stratas are found, return the original data
        return data

    # Pop the count column to be able to iterate over the columns of interest to
    # create the filter
    single_stratas.pop("count")

    logical_and_list = []
    for _, row in single_stratas.iterrows():
        # Create filter that accounts for data point that are not in the strata and append
        # it to the list of filters to be applied
        not_banned_observations = ~np.logical_and.reduce(
            [data[col] == value for col, value in row.items()]
        )
        logical_and_list.append(not_banned_observations)

    # Bound all the filters with a logical and
    logical_and_filter = np.logical_and.reduce(logical_and_list)

    return data[logical_and_filter]


def split_data(
    data: pd.DataFrame,
    train_size: Optional[int] = 0.8,
    strata: Optional[List[str]] = [],
    random_state: Optional[int] = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation and test sets."""

    # Filter out single stratas
    filtered_data = filter_single_subgroups(data, strata) if strata else data

    # Compute number of strata
    n_strata = filtered_data[strata].value_counts().shape[0]

    # Compute size of test set
    n_test = int(np.floor(len(filtered_data) * (1 - train_size)))

    if n_test < n_strata:
        logging.info(
            "Not enough data to create a test set with at least one observation per strata."
            " Will recompute train_size so that test set has same number of observations"
            " as the number of stratas."
        )
        train_size = 1 - n_strata / len(filtered_data)

    # Split data into train and test
    train, test = train_test_split(
        filtered_data,
        train_size=train_size,
        random_state=random_state,
        stratify=filtered_data[strata] if strata else None,
    )

    return train, test
