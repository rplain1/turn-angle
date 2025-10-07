import polars as pl
import pymc as pm
import numpy as np
import arviz as az


def get_dataset():
    return (
        pl.concat(
            [
                pl.read_parquet("stg_data/tracking_angle_receivers.parquet"),
                pl.read_parquet("stg_data/tracking_angle_rushers.parquet"),
            ],
        )
        # target variable
        .filter(pl.col("turn_angle").is_not_null())
        # nflId integer to string
        .with_columns(bc_id=pl.col("bc_id").cast(pl.Utf8))
        # copied from original but not used?
        .with_columns(adj_y_change_abs=pl.col("adj_y_change").abs())
    )


df = get_dataset()


def create_enum_col(df: pl.DataFrame, col: str):
    enum_col = pl.Enum(df[col].unique())

    return df.with_columns(pl.col(col).cast(enum_col))


for x in ["bc_type", "bc_id", "bc_position"]:
    df = create_enum_col(df, x)


### ------------ data ------------


def make_pymc_data(df):
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

    #### ------------ transformed data ------------

    # center X features columns
    # QR decomposition
    # multiply and divide QR by sqrt(N - 1)?
    Xc = df[mu_features].select([(pl.col(col) - pl.col(col).mean()).alias(col) for col in df[mu_features].columns])
    QR = np.linalg.qr(Xc.to_numpy())
    assert np.allclose(Xc, QR.Q.dot(QR.R)), "QR decomposition failed"

    XQ = QR.Q * np.sqrt(N - 1)
    XR = QR.R / np.sqrt(N - 1)
    XR_inv = np.linalg.inv(XR)
    # same for kappa design matrix
    # only this time we have categorical and index variables
    # players are indexed
    # type and position are categorical
    kappa_cat_features = [x for x in df.select(kappa_features).columns if df[x].dtype == pl.Enum and x != "bc_id"]

    X_kappa = pl.concat(
        [
            df.select(kappa_features).drop(kappa_cat_features + ["bc_id"]),
            df.select(kappa_features).select(kappa_cat_features).to_dummies(drop_first=True),
        ],
        how="horizontal",
    )

    Xc_kappa = X_kappa.select([(pl.col(col) - pl.col(col).mean()).alias(col) for col in X_kappa.columns]).to_numpy()

    player_idx = (
        df.select(["bc_id", "bc_position"])
        .unique(subset=["bc_id"])
        .sort("bc_id")["bc_id"]
        .to_physical()
        .cast(pl.Int64)
        .to_numpy()
    )

    # Jby_1
    player_to_position = (
        df.select(["bc_id", "bc_position"])
        .unique(subset=["bc_id"])
        .sort("bc_id")["bc_position"]
        .to_physical()
        .cast(pl.Int64)
        .to_numpy()
    )

    return {
        "bc_id": df["bc_id"].to_physical().cast(pl.Int64).to_numpy(),
        "bc_id_cat": df["bc_id"].cat.get_categories(),
        "bc_position": df["bc_position"].to_physical().cast(pl.Int64).to_numpy(),
        "bc_position_cat": df["bc_position"].cat.get_categories(),
        "player_to_position": player_to_position,
        "y": df["turn_angle"].to_numpy(),
        "XQ": XQ,
        "XR_inv": XR_inv,
        "X_kappa_centered": Xc_kappa,
    }


pm_data = make_pymc_data(df)
with pm.Model(coords={"player_ids": pm_data["bc_id_cat"], "player_positions": pm_data["bc_position_cat"]}) as model:
    ### -------- DATA --------------
    bc_id = pm.Data("bc_id", pm_data["bc_id"])
    bc_position = pm.Data("bc_position", pm_data["bc_position"])
    player_to_position = pm.Data("player_to_position", pm_data["player_to_position"])
    X_kappa_centered = pm.Data("X_kappa_centered", pm_data["X_kappa_centered"])
    XQ = pm.Data("XQ", pm_data["XQ"])

    _y = pm.Data("_y", pm_data["y"])

    ### -------- PARAMETERS --------------
    Intercept_mu = pm.StudentT("Intercept_mu", nu=1, mu=0, sigma=1)
    Intercept_kappa = pm.Normal("Intercept_kappa", mu=5.0, sigma=0.8)

    betaQ = pm.Normal("betaQ", mu=0.0, sigma=1, shape=XQ.shape[1])
    beta_kappa = pm.Normal("beta_kappa", mu=0, sigma=1, shape=X_kappa_centered.shape[1])

    ### -------- TRANSFORMED PARAMETERS --------------
    z_player = pm.Normal("z_player", mu=0, sigma=1, dims="player_ids")
    sigma_position = pm.HalfNormal("sigma_position", sigma=1, dims="player_positions")
    player_effect_raw = z_player * sigma_position[player_to_position]

    player_effect = player_effect_raw[bc_id]

    mu = Intercept_mu + pm.math.sum(XQ * betaQ, axis=1)
    kappa = Intercept_kappa + pm.math.sum(X_kappa_centered * beta_kappa, axis=1)

    mu_link = 2 * pm.math.arctan(mu)
    kappa_link = pm.math.log1pexp(kappa)

    ### -------- LIKELIHOOD --------------

    y_rep = pm.VonMises("y_rep", mu=mu_link, kappa=kappa_link, observed=_y)


print("mu shape:", mu.shape.eval())
print("kappa shape:", kappa.shape.eval())

# with model:
#     prior = pm.sample_prior_predictive()
# az.plot_ppc(prior, group="prior")

with model:
    idata = pm.sample(chains=4, cores=4, init="adapt_diag", target_accept=0.9)


# az.plot_trace(idata)
# idata["posterior"].data_vars

# df_summary = az.summary(idata.posterior)

# with model:
#     ppd = pm.sample_posterior_predictive(idata)
# az.plot_ppc(ppd, group="posterior", num_pp_samples=100)


# df_summary

# tmp = (
#     pl.from_pandas(idata.posterior["z_player"].sel(chain=0).to_dataframe().reset_index())
#     .select(pl.col("z_player_dim_0").alias("bc_id"), "z_player")
#     .join(
#         df.select(["bc_id", "displayName", "bc_position"]).with_columns(bc_id=pl.col("bc_id").to_physical()).unique(),
#         on="bc_id",
#         how="left",
#     )
#     .group_by("bc_id", "displayName", "bc_position")
#     .agg(pl.col("z_player").mean())
#     .pivot("bc_position", values="z_player")
# )

# tmp
