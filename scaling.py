import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    PowerTransformer
)


def load_data(input_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found.")
    return pd.read_csv(input_file)


def get_scaling_columns(df):
    exclude_cols = ["day_of_week"]
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    scaling_cols = [col for col in numeric_cols if col not in exclude_cols]
    return scaling_cols


def standard_scale(df, cols):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[cols])
    return pd.DataFrame(
        scaled,
        columns=[f"{col}_standard" for col in cols],
        index=df.index
    )


def minmax_scale(df, cols):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[cols])
    return pd.DataFrame(
        scaled,
        columns=[f"{col}_minmax" for col in cols],
        index=df.index
    )


def robust_scale(df, cols):
    scaler = RobustScaler()
    scaled = scaler.fit_transform(df[cols])
    return pd.DataFrame(
        scaled,
        columns=[f"{col}_robust" for col in cols],
        index=df.index
    )


def maxabs_scale(df, cols):
    scaler = MaxAbsScaler()
    scaled = scaler.fit_transform(df[cols])
    return pd.DataFrame(
        scaled,
        columns=[f"{col}_maxabs" for col in cols],
        index=df.index
    )


def yeojohnson_scale(df, cols):
    scaler = PowerTransformer(method="yeo-johnson")
    scaled = scaler.fit_transform(df[cols])
    return pd.DataFrame(
        scaled,
        columns=[f"{col}_yeojohnson" for col in cols],
        index=df.index
    )


def combine_scaled_results(df, standard_df, minmax_df, robust_df, maxabs_df, yeojohnson_df):
    return pd.concat(
        [df, standard_df, minmax_df, robust_df, maxabs_df, yeojohnson_df],
        axis=1
    )


def save_file(df, output_file):
    df.to_csv(output_file, index=False)


def create_scaled_weather_file(input_file, output_file="scaled_weather_data.csv"):
    df = load_data(input_file)

    scaling_cols = get_scaling_columns(df)

    if not scaling_cols:
        print("No numeric columns available for scaling.")
        return

    standard_df = standard_scale(df, scaling_cols)
    minmax_df = minmax_scale(df, scaling_cols)
    robust_df = robust_scale(df, scaling_cols)
    maxabs_df = maxabs_scale(df, scaling_cols)
    yeojohnson_df = yeojohnson_scale(df, scaling_cols)

    final_df = combine_scaled_results(
        df,
        standard_df,
        minmax_df,
        robust_df,
        maxabs_df,
        yeojohnson_df
    )

    save_file(final_df, output_file)
    print(f"New scaled file created: {output_file}")


if __name__ == "__main__":
    create_scaled_weather_file("processed_weather_data.csv")