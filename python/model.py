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
        # pl.read_parquet("scripts/tracking_angle_full.parquet")
        # target variable
        .filter(pl.col("turn_angle").is_not_null())
        # nflId integer to string
        .with_columns(bc_id=pl.col("bc_id").cast(pl.Utf8))
        # copied from original but not used?
        .with_columns(adj_y_change_abs=pl.col("adj_y_change").abs())
    )


df = get_dataset()
# ten_games = df["gameId"].unique().slice(1, 10).to_list()
# df = df.filter(pl.col("gameId").is_in(ten_games))


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
with pm.Model(
    coords={
        "player_ids": pm_data["bc_id_cat"],
        "player_positions": pm_data["bc_position_cat"],
    }
) as model:
    ### -------- DATA --------------
    bc_id = pm.Data("bc_id", pm_data["bc_id"])
    bc_position = pm.Data("bc_position", pm_data["bc_position"])
    player_to_position = pm.Data("player_to_position", pm_data["player_to_position"])
    X_kappa_centered = pm.Data("X_kappa_centered", pm_data["X_kappa_centered"])
    XQ = pm.Data("XQ", pm_data["XQ"])
    XR_inv = pm.Data("XR_inv", pm_data["XR_inv"])

    _y = pm.Data("_y", pm_data["y"])

    ### -------- PARAMETERS --------------
    Intercept_mu = pm.StudentT("Intercept_mu", nu=1, mu=0, sigma=1)
    Intercept_kappa = pm.Normal("Intercept_kappa", mu=5, sigma=0.8)

    betaQ = pm.Normal("betaQ", mu=0.0, sigma=1, shape=XQ.shape[1])
    beta_kappa = pm.Normal("beta_kappa", mu=0, sigma=1, shape=X_kappa_centered.shape[1])

    ### -------- TRANSFORMED PARAMETERS --------------
    z_player = pm.Normal("z_player", mu=0, sigma=1, dims="player_ids")
    sigma_position = pm.HalfStudentT("sigma_position", nu=3, sigma=2.5, dims="player_positions")

    player_effect = z_player[bc_id] * sigma_position[bc_position]  # per-player

    mu = Intercept_mu + pm.math.sum(XQ * betaQ, axis=1)

    kappa_raw = Intercept_kappa + pm.math.sum(X_kappa_centered * beta_kappa, axis=1) + player_effect
    kappa = pm.math.clip(kappa_raw, -5, 10)

    mu_link = 2 * pm.math.arctan(mu)
    kappa_link = pm.Deterministic("kappa_link", pm.math.exp(kappa))

    ### -------- LIKELIHOOD --------------

    y_rep = pm.VonMises("y_rep", mu=mu_link, kappa=kappa_link, observed=_y, shape=df.shape[0])


with model:
    idata = pm.sample(chains=4, cores=4, tune=2500, draws=1000, init="adapt_diag", target_accept=0.95)

# az.plot_trace(idata)
az.plot_density(idata.posterior.sigma_position)
df_summary = az.summary(idata.posterior[["Intercept_mu", "Intercept_kappa", "betaQ", "beta_kappa", "sigma_position"]])
df_summary
with model:
    ppd = pm.sample_posterior_predictive(idata)
az.plot_ppc(ppd, group="posterior", num_pp_samples=100)


plays_by_nflid = df.group_by("gameId", "playId", "bc_id", "bc_position").len().group_by("bc_id", "bc_position").len()

# df_summary

DATA_DIR = "~/git-repos/nfl-bdb-2025/raw-data/"
players = pl.read_csv(DATA_DIR + "players.csv", null_values=["NA", "N/A"], schema_overrides={"nflId": pl.Int64})
tmp = (
    pl.from_pandas(idata.posterior["z_player"].sel(chain=0).to_dataframe().reset_index())
    .select(pl.col("player_ids").alias("bc_id"), "z_player")
    .join(
        df.select(["bc_id", "bc_position"]).with_columns(bc_id=pl.col("bc_id").cast(pl.Utf8)).unique(),
        on="bc_id",
        how="left",
    )
    .join(
        players.with_columns(bc_id=pl.col("nflId").cast(pl.Utf8)).select(["bc_id", "displayName"]),
        on="bc_id",
        how="left",
    )
)

wr_filter = (
    plays_by_nflid.filter(pl.col("len") >= 15)
    .filter(pl.col("bc_position") == "WR")
    .get_column("bc_id")
    .cast(pl.Utf8)
    .to_list()
)


player_mapping = dict(zip(players["nflId"].cast(pl.Utf8), players["displayName"]))
wr_means = idata.posterior["z_player"].sel(player_ids=wr_filter).mean(("chain", "draw"))
sorted_means = wr_means.sortby(wr_means)
extreme_ids = list(sorted_means.player_ids.values[:10]) + list(sorted_means.player_ids.values[-10:])

subset = idata.posterior.z_player.sel(player_ids=extreme_ids).sortby(wr_means)

subset_named = subset.assign_coords(player_ids=[player_mapping.get(pid, pid) for pid in subset.player_ids.values])
az.plot_forest(subset_named, combined=True)


tmp
tmp.filter(pl.col("WR").is_not_null()).filter(pl.col("bc_id").is_in(wr_filter)).sort("WR", descending=True)

idata.to_netcdf("models/pymc3.netcdf")
