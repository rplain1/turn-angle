library(tidyverse)
# tracking <- read_csv(glue::glue(data_dir, "tracking_week_1.csv")) |>
#   bind_rows(read_csv(glue::glue(data_dir, "tracking_week_2.csv"))) |>
#   bind_rows(read_csv(glue::glue(data_dir, "tracking_week_3.csv"))) |>
#   bind_rows(read_csv(glue::glue(data_dir, "tracking_week_4.csv"))) |>
#   bind_rows(read_csv(glue::glue(data_dir, "tracking_week_5.csv"))) |>
#   bind_rows(read_csv(glue::glue(data_dir, "tracking_week_6.csv"))) |>
#   bind_rows(read_csv(glue::glue(data_dir, "tracking_week_7.csv"))) |>
#   bind_rows(read_csv(glue::glue(data_dir, "tracking_week_8.csv"))) |>
#   bind_rows(read_csv(glue::glue(data_dir, "tracking_week_9.csv")))

# arrow::write_parquet(tracking, "data/tracking.parquet")

data_dir <- '~/git-repos/nfl-bdb-2025/raw-data/'

games <- read_csv(glue::glue(data_dir, "games.csv"))
plays <- read_csv(glue::glue(data_dir, "plays.csv"))
players <- read_csv(glue::glue(data_dir, "players.csv"))
player_play <- read_csv(glue::glue(data_dir, "player_play.csv"))

# tracking <- arrow::read_parquet("data/tracking.parquet") |>
#   filter(frameType != "BEFORE_SNAP")

tracking <- duckplyr::read_parquet_duckdb('data/tracking.parquet') |>
  dplyr::filter(
    frameType != 'BEFORE_SNAP',
    # filter a subset of data
    time <= !!as.POSIXct('2022-09-17 00:00:00', tz = "UTC")
  ) |>
  collect()

tracking <- tracking |>
  mutate(
    x = ifelse(playDirection == "left", 120 - x, x),
    y = ifelse(playDirection == "left", 160 / 3 - y, y),
    dir = ifelse(playDirection == "left", dir + 180, dir),
    dir = ifelse(dir > 360, dir - 360, dir),
    o = ifelse(playDirection == "left", o + 180, o),
    o = ifelse(o > 360, o - 360, o)
  )

plays <- plays |>
  mutate(
    yards_from_endzone = ifelse(
      (possessionTeam != yardlineSide) |
        (yardlineNumber == 50),
      yardlineNumber,
      100 - yardlineNumber
    ),
    adj_x_first_down = yards_from_endzone - yardsToGo
  )

plays_pass_receivers <- player_play |>
  filter(hadPassReception == 1) |>
  left_join(select(players, nflId, position)) |>
  filter(position %in% c("WR", "TE", "RB")) |>
  select(gameId, playId, receiver_id = nflId, receiver_club = teamAbbr)

tracking_all <- tracking |>
  inner_join(plays_pass_receivers)

tracking_all <- tracking_all |>
  group_by(gameId, playId) |>
  mutate(
    frame_caught = frameId[which(event == "pass_outcome_caught")][1] - 6,
    frame_end = frameId[which(
      event %in% c("out_of_bounds", "tackle", "touchdown")
    )][1]
  ) |>
  ungroup() |>
  filter(!is.na(frame_caught), !is.na(frame_end)) |>
  filter(frameId >= frame_caught & frameId <= frame_end)

frames <- tracking_all |>
  distinct(gameId, playId, frame_caught, frame_end) |>
  rowwise() |>
  mutate(frameId = list(seq(frame_caught, frame_end, 1))) |>
  unnest_longer(frameId) |>
  mutate(frameId_corrected = frameId - frame_caught) |>
  select(gameId, playId, frameId, frame_caught, frame_end, frameId_corrected) |>
  group_by(gameId, playId) |>
  filter(frameId_corrected %% 3 == 0) |>
  ungroup()

tracking_all <- tracking_all |>
  inner_join(frames)

plays_filtered <- tracking_all |>
  distinct(gameId, playId, frameId) |>
  count(gameId, playId) |>
  filter(n > 5) |>
  select(gameId, playId)

tracking_all <- tracking_all |>
  inner_join(plays_filtered)

# bc features
tracking_rec <- tracking_all |>
  filter(nflId == receiver_id) |>
  select(
    gameId,
    playId,
    frameId,
    bc_id = receiver_id,
    bc_club = receiver_club,
    bc_x = x,
    bc_y = y,
    bc_s = s,
    bc_a = a,
    bc_dis = dis,
    bc_o = o,
    bc_dir = dir
  ) |>
  mutate(adj_bc_x = 110 - bc_x, adj_bc_y = bc_y - (160 / 6)) |>
  left_join(select(plays, gameId, playId, adj_x_first_down)) |>
  mutate(adj_bc_x_from_first_down = adj_bc_x - adj_x_first_down) |>
  left_join(select(players, bc_id = nflId, bc_position = position)) |>
  mutate(bc_type = "receiver_after_catch")

# nearest def features
tracking_def <- tracking_all |>
  filter(club != receiver_club, displayName != "football") |>
  left_join(
    select(
      tracking_rec,
      gameId,
      playId,
      frameId,
      bc_x,
      bc_y,
      adj_bc_x,
      adj_bc_y
    ),
    by = c("gameId", "playId", "frameId")
  ) |>
  mutate(dist_to_bc = sqrt((x - bc_x)^2 + (y - bc_y)^2)) |>
  group_by(gameId, playId, frameId) |>
  arrange(dist_to_bc) |>
  mutate(player_dist_bc_rank = row_number()) |>
  ungroup() |>
  filter(player_dist_bc_rank == 1) |>
  select(
    gameId,
    playId,
    frameId,
    playDirection,
    nflId,
    displayName,
    dist_to_bc,
    def_x = x,
    def_y = y,
    def_s = s,
    bc_x,
    bc_y,
    adj_bc_x,
    adj_bc_y
  ) |>
  mutate(
    adj_x = 110 - def_x,
    adj_y = def_y - (160 / 6),
    adj_x_change = adj_bc_x - adj_x,
    adj_y_change = adj_bc_y - adj_y,
    angle_with_bc = atan2(adj_y_change, -adj_x_change)
  ) |>
  select(-bc_x, -bc_y, -adj_bc_x, -adj_bc_y)

# counts features
tracking_counts_features <- tracking_all |>
  filter(displayName != "football", nflId != receiver_id) |>
  mutate(side = ifelse(club == receiver_club, "offense", "defense")) |>
  left_join(select(tracking_rec, gameId, playId, frameId, bc_x, bc_y)) |>
  group_by(gameId, playId, frameId, side) |>
  summarize(n_left_bc = sum(y > bc_y), n_front_bc = sum(x > bc_x)) |>
  pivot_wider(names_from = side, values_from = c(n_left_bc, n_front_bc)) |>
  ungroup()


tracking_angle_receivers <- tracking_rec |>
  left_join(tracking_def, by = join_by(gameId, playId, frameId)) |>
  left_join(tracking_counts_features, by = join_by(gameId, playId, frameId)) |>
  group_by(gameId, playId, bc_id) |>
  arrange(frameId, .by_group = TRUE) |>
  mutate(
    turn_angle = atan2(lead(bc_y) - bc_y, lead(bc_x) - bc_x) -
      atan2(bc_y - lag(bc_y), bc_x - lag(bc_x)),
    turn_angle = ifelse(
      turn_angle >= pi,
      turn_angle - 2 * pi,
      ifelse(turn_angle <= -pi, 2 * pi + turn_angle, turn_angle)
    ),
    turn_angle = ifelse(turn_angle > 3.14 | turn_angle < -3.14, 0, turn_angle),
    prev_angle = lag(turn_angle)
  ) |>
  ungroup() |>
  filter(!is.na(turn_angle), !is.na(prev_angle))

tracking_cum_dis <- tracking |>
  inner_join(distinct(
    tracking_angle_receivers,
    gameId,
    playId,
    nflId = bc_id
  )) |>
  filter(frameType != "SNAP") |>
  group_by(gameId, playId, nflId) |>
  mutate(bc_cum_dis = cumsum(dis)) |>
  ungroup() |>
  select(gameId, playId, bc_id = nflId, frameId, bc_cum_dis)

tracking_angle_receivers <- tracking_angle_receivers |>
  left_join(tracking_cum_dis)

tracking_angle_receivers |>
  write_rds("scripts/tracking_angle_receivers.rds", compress = "gz")
