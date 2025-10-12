import polars as pl
from datetime import datetime, timezone
import numpy as np

DATA_DIR = "~/git-repos/nfl-bdb-2025/raw-data/"

tracking = (
    pl.scan_parquet("data/tracking.parquet")
    .filter(pl.col("frameType") != "BEFORE_SNAP")
    # .filter(pl.col("time") <= datetime(2022, 9, 17, tzinfo=timezone.utc))
    .with_columns(
        pl.col("gameId").cast(pl.Int64),
        pl.col("playId").cast(pl.Int64),
        pl.col("frameId").cast(pl.Int64),
        pl.col("nflId").cast(pl.Int64),
    )
    .collect()
)

tracking = tracking.with_columns_seq(
    x=pl.when(pl.col("playDirection") == "left").then(120 - pl.col("x")).otherwise(pl.col("x")),
    y=pl.when(pl.col("playDirection") == "left").then((160 / 3) - pl.col("y")).otherwise(pl.col("y")),
    dir=pl.when(pl.col("playDirection") == "left").then(pl.col("dir") + 180).otherwise(pl.col("dir")),
    o=pl.when(pl.col("playDirection") == "left").then(pl.col("o") + 180).otherwise(pl.col("o")),
).with_columns(
    dir=pl.when(pl.col("dir") > 360).then(pl.col("dir") - 360).otherwise(pl.col("dir")),
    o=pl.when(pl.col("o") > 360).then(pl.col("o") - 360).otherwise(pl.col("o")),
)


plays = (
    pl.read_csv(DATA_DIR + "plays.csv", null_values=["NA"])
    .with_columns(
        yards_from_endzone=pl.when(
            (pl.col("possessionTeam") == pl.col("yardlineSide")) | (pl.col("yardlineNumber") == 50)
        )
        .then(pl.col("yardlineNumber"))
        .otherwise(100 - pl.col("yardlineNumber"))
    )
    .with_columns(adj_x_first_down=pl.col("yards_from_endzone") - pl.col("yardsToGo"))
)

players = pl.read_csv(DATA_DIR + "players.csv", null_values=["NA", "N/A"])

play_rushers = (
    pl.read_csv(DATA_DIR + "player_play.csv", null_values=["NA", "N/A"])
    .filter(pl.col("hadRushAttempt") == 1)
    .join(players.select(["nflId", "position"]), on="nflId")
    .filter(pl.col("position") == "RB")
    .select(["gameId", "playId", "nflId", "teamAbbr"])
    .rename({"nflId": "bc_id", "teamAbbr": "bc_club"})
)

"""
Takes the tracking data and joins in the rushers. Calculates the frame of handoff
and the what play the frame ends.
"""
tracking_all = (
    tracking.join(play_rushers, on=["gameId", "playId"])
    .with_columns(
        frame_handoff=(pl.col("frameId").filter(pl.col("event") == "handoff").min().over(["gameId", "playId"])) - 6,
        frame_end=(
            pl.col("frameId")
            .filter(pl.col("event").is_in(["out_of_bounds", "tackle", "touchdown"]))
            .min()
            .over(["gameId", "playId"])
        ),
    )
    .filter((pl.col("frame_end").is_not_null()) & (pl.col("frame_handoff").is_not_null()))
    .filter(pl.col("frameId") >= pl.col("frame_handoff"))
    .filter(pl.col("frameId") <= pl.col("frame_end"))
)

"""
Create a sequence of frames for each play from the calculated `frame_handoff` to
`frame_end`. Uses every 3rd frame.
"""
frames = (
    tracking_all.select(["gameId", "playId", "frame_handoff", "frame_end"])
    .unique()
    .with_columns(frameId=pl.int_ranges(pl.col("frame_handoff"), pl.col("frame_end") + pl.lit(1)))
    .explode("frameId")
    .with_columns(frameId_corrected=pl.col("frameId") - pl.col("frame_handoff"))
    .filter(pl.col("frameId_corrected").mod(3) == 0)
)

tracking_all = tracking_all.join(frames, on=["gameId", "playId", "frameId", "frame_handoff", "frame_end"], how="inner")

plays_filtered = (
    tracking_all.select(["gameId", "playId", "frameId"])
    .unique()
    .group_by(["gameId", "playId"])
    .len(name="n")
    .filter(pl.col("n") > 5)
    .drop("n")
)

tracking_all = tracking_all.join(plays_filtered, on=["gameId", "playId"])

# get the bc related dataframe
tracking_bc = (
    tracking_all.filter(pl.col("nflId") == pl.col("bc_id"))
    .select(["gameId", "playId", "frameId", "bc_id", "bc_club", "x", "y", "s", "a", "dis", "o", "dir"])
    .rename({"x": "bc_x", "y": "bc_y", "dir": "bc_dir", "s": "bc_s", "a": "bc_a", "o": "bc_o", "dis": "bc_dis"})
    .with_columns(
        adj_bc_x=110 - pl.col("bc_x"),
        # this will center the y values... didn't think about that
        adj_bc_y=pl.col("bc_y") - (160 / 6),
    )
    .join(plays.select(["gameId", "playId", "adj_x_first_down"]), on=["gameId", "playId"], how="left")
    .with_columns(
        adj_bc_x_from_first_down=pl.col("adj_bc_x") - pl.col("adj_x_first_down"),
        bc_position=pl.lit("RB"),
        bc_type=pl.lit("rusher"),
    )
)

# Get the nearest defender and calculate the angle to the player
tracking_def = (
    tracking_all.filter(pl.col("club") != pl.col("bc_club"))
    .filter(pl.col("displayName") != "football")
    .join(
        tracking_bc.select(["gameId", "playId", "frameId", "bc_x", "bc_y", "adj_bc_x", "adj_bc_y"]),
        on=["gameId", "playId", "frameId"],
    )
    .with_columns(
        dist_to_bc=(((pl.col("x") - pl.col("bc_x")).pow(2)).sqrt()) + (((pl.col("y") - pl.col("bc_y")).pow(2)).sqrt())
    )
    .with_columns(
        play_dist_bc_rank=pl.col("dist_to_bc")
        .cum_count()
        .over(["gameId", "playId", "frameId"], order_by=["dist_to_bc"])
    )
    .filter(pl.col(["play_dist_bc_rank"]) == 1)
    .select(
        "gameId",
        "playId",
        "frameId",
        "playDirection",
        "nflId",
        "displayName",
        "dist_to_bc",
        pl.col("x").alias("def_x"),
        pl.col("y").alias("def_y"),
        pl.col("s").alias("def_s"),
        "bc_x",
        "bc_y",
        "adj_bc_x",
        "adj_bc_y",
    )
    .with_columns(adj_x=110 - pl.col("def_x"), adj_y=pl.col("def_y") - (160 / 6))
    .with_columns(
        adj_x_change=pl.col("adj_bc_x") - pl.col("adj_x"),
        adj_y_change=pl.col("adj_bc_y") - pl.col("adj_y"),
    )
    .with_columns(angle_with_bc=pl.arctan2("adj_y_change", pl.col("adj_x_change").neg()))
    .drop(["bc_x", "bc_y", "adj_bc_x", "adj_bc_y"])
)

# gets a wide dataframe of number of players left of and in front of ball carrier.
tracking_count_features = (
    tracking_all.filter(pl.col("displayName") != "football")
    .filter(pl.col("nflId") != pl.col("bc_id"))
    .with_columns(
        side=pl.when(pl.col("club") == pl.col("bc_club")).then(pl.lit("offense")).otherwise(pl.lit("defense"))
    )
    .join(
        tracking_bc.select(["gameId", "playId", "frameId", "bc_x", "bc_y"]),
        on=["gameId", "playId", "frameId"],
        how="left",
    )
    .group_by(["gameId", "playId", "frameId", "side"])
    .agg(n_left_bc=(pl.col("y") > pl.col("bc_y")).sum(), n_front_bc=(pl.col("x") > pl.col("bc_x")).sum())
    .pivot("side", values=["n_left_bc", "n_front_bc"])
    .select(
        [
            "gameId",
            "playId",
            "frameId",
            "n_left_bc_offense",
            "n_left_bc_defense",
            "n_front_bc_offense",
            "n_front_bc_defense",
        ]
    )
)

tracking_bc.join(tracking_def, on=["gameId", "playId", "frameId"], how="left").join(
    tracking_count_features, on=["gameId", "playId", "frameId"]
).group_by(["gameId", "playId", "bc_id"])

# calculating the turn angle for the rushers
tracking_angle_rushers = (
    tracking_bc.join(tracking_def, on=["gameId", "playId", "frameId"], how="left")
    .join(tracking_count_features, on=["gameId", "playId", "frameId"], how="left")
    .sort(["gameId", "playId", "bc_id", "frameId"])
    .with_columns(
        # TODO: understand how turn angle is calculated.
        # looks like it is the arctan2 of where the ball carrier is going relative to now (x, y)
        # munis the arctan2 of wher it has been (x, y).
        # Looks to use 3 frames total?
        turn_angle=(
            pl.arctan2(pl.col("bc_y").shift(-1) - pl.col("bc_y"), pl.col("bc_x").shift(-1) - pl.col("bc_x"))
            - pl.arctan2(pl.col("bc_y") - pl.col("bc_y").shift(1), pl.col("bc_x") - pl.col("bc_x").shift(1))
        ).over(["gameId", "playId", "bc_id"])
    )
    .with_columns(
        turn_angle=pl.when(pl.col("turn_angle") >= pl.lit(np.pi))
        .then(pl.col("turn_angle") - 2 * np.pi)
        .when(pl.col("turn_angle") <= pl.lit(-np.pi))
        .then(2 * np.pi + pl.col("turn_angle"))
        .otherwise(pl.col("turn_angle"))
    )
    .with_columns(
        turn_angle=pl.when((pl.col("turn_angle") > 3.14) | (pl.col("turn_angle") < -3.14))
        .then(0)
        .otherwise(pl.col("turn_angle"))
    )
    .with_columns(prev_angle=pl.col("turn_angle").shift(1).over(["gameId", "playId", "bc_id"]))
    .filter(pl.col("turn_angle").is_not_null())
    .filter(pl.col("prev_angle").is_not_null())
)

# get the cumulative distance traveled by ball carrier
tracking_cum_dis = (
    tracking.join(
        tracking_angle_rushers.select(["gameId", "playId", pl.col("bc_id").alias("nflId")]).unique(),
        on=["gameId", "playId", "nflId"],
        how="inner",
    )
    .filter(pl.col("frameType") != "SNAP")
    .with_columns(bc_cum_dis=pl.col("dis").cum_sum().over(["gameId", "playId", "nflId"], order_by="frameId"))
    .select(["gameId", "playId", pl.col("nflId").alias("bc_id"), "frameId", "bc_cum_dis"])
)

tracking_angle_rushers = tracking_angle_rushers.join(
    tracking_cum_dis, on=["gameId", "playId", "frameId", "bc_id"], how="left"
)

tracking_angle_rushers.write_parquet("stg_data/tracking_angle_rushers.parquet")
