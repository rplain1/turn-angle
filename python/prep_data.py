import polars as pl
from datetime import datetime, timezone

DATA_DIR = "~/git-repos/nfl-bdb-2025/raw-data/"

tracking = (
    pl.scan_parquet("data/tracking.parquet")
    .filter(pl.col("frameType") != "BEFORE_SNAP")
    .filter(pl.col("time") <= datetime(2022, 9, 17, tzinfo=timezone.utc))
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


tracking_all = (
    tracking.with_columns(pl.col("gameId").cast(pl.Int128), pl.col("playId").cast(pl.Int64))
    .join(play_rushers, on=["gameId", "playId"])
    .with_columns(
        frame_handoff=(pl.col("frameId").filter(pl.col("event") == "handoff").min().over(["gameId", "playId"])),
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
