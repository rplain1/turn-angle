import polars as pl
import pymc as pm
import numpy as np

df = pl.concat(
    [
        pl.read_parquet("stg_data/tracking_angle_receivers.parquet"),
        pl.read_parquet("stg_data/tracking_angle_rushers.parquet"),
    ],
    how="diagonal",
)

df = df.with_columns(adj_y_change_abs=pl.col("adj_y_change").abs())
df = df.with_columns(pl.col("bc_id").cast(pl.Utf8).cast(pl.Categorical), pl.col("bc_position").cast(pl.Categorical))

df.shape


def get_pymc_data(df):
    N = df.shape[0]
    y = df["turn_angle"].to_numpy()

    mu_features = [
        "prev_angle",
        "adj_bc_x",
        "adj_bc_y",
        "adj_bc_x_from_first_down",
        "n_left_bc_defense",
        "n_front_bc_defense",
        "n_left_bc_offense",
        "n_front_bc_offense",
        "adj_x",
        "adj_y",
        "adj_x_change",
        "adj_y_change_abs",
        "dist_to_bc",
        "def_s",
        "angle_with_bc",
    ]

    kappa_features = ["bc_s", "bc_a", "bc_cum_dis", "bc_type", "bc_position", "bc_id"]

    Xc = df[mu_features].select([(pl.col(col) - pl.col(col).mean()).alias(col) for col in df[mu_features].columns])
    QR = np.linalg.qr(Xc.to_numpy())
    Xc_kappa = df[kappa_features].select(
        [(pl.col(col) - pl.col(col).mean()).alias(col) for col in df[kappa_features].columns]
    )

    return {
        "bc_id_idx": df["bc_id"].to_physical().cast(pl.Int64).to_numpy(),
        "bc_id_cat": df["bc_id"].cat.get_categories(),
        "bc_position_idx": df["bc_position"].to_physical().cast(pl.Int64).to_numpy(),
        "bc_position_cat": df["bc_position"].cat.get_categories(),
    }


get_pymc_data(df)
