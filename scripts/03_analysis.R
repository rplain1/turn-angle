library(tidyverse)
library(brms)
library(tidybayes)
data_dir <- '~/git-repos/nfl-bdb-2025/raw-data/'

players <- read_csv(glue::glue(data_dir, "players.csv"))


# to prep the data,
# run 01a_prep_rusher_data.R and 01b_prep_receiver_data.R

tracking_angle_full <- read_rds("scripts/tracking_angle_receivers.rds") |>
  bind_rows(read_rds("scripts/tracking_angle_rushers.rds"))

tracking_angle_full <- tracking_angle_full |>
  mutate(adj_y_change_abs = abs(adj_y_change))

# model outupt saved after running 02_model.R
angle_brms <- read_rds("scripts/angle_brms.rds")


# model coefficients (concentration level)
angle_brms |>
  summary() |>
  pluck("fixed") |>
  rownames_to_column(var = "term") |>
  filter(str_detect(term, "kappa_"))

# random effect variance
angle_brms |>
  summary() |>
  pluck("random") |>
  pluck("bc_id")

# also plot these posterior distributions
angle_brms |>
  as_tibble() |>
  select(contains("sd_")) |>
  pivot_longer(everything(), names_to = "position") |>
  mutate(
    position = str_remove(position, "sd_bc_id__kappa_Intercept:bc_position")
  ) |>
  filter(position != 'prior_sd_bc_id') |>
  ggplot(aes(value)) +
  geom_histogram(aes(fill = position), alpha = 0.5, bins = 60) +
  ggthemes::scale_fill_colorblind() +
  facet_wrap(~position, nrow = 3)


# ball carrier ratings

# wide receivers
wr_filtered <- tracking_angle_full |>
  distinct(gameId, playId, bc_id, bc_position, bc_type) |>
  filter(bc_position == "WR") |>
  count(bc_id) |>
  filter(n >= 15) |>
  pull(bc_id)

wr_posterior <- angle_brms |>
  spread_draws(r_bc_id__kappa[nflId, term]) |>
  left_join(players) |>
  filter(nflId %in% wr_filtered) |>
  group_by(nflId, displayName, position) |>
  summarize(posterior_mean = mean(r_bc_id__kappa)) |>
  ungroup() |>
  arrange(posterior_mean)

wr_top <- wr_posterior |> slice_head(n = 10) |> pull(nflId)
wr_bottom <- wr_posterior |> slice_tail(n = 10) |> pull(nflId)

angle_brms |>
  spread_draws(r_bc_id__kappa[nflId, term]) |>
  ungroup() |>
  left_join(players) |>
  filter(nflId %in% c(wr_top, wr_bottom)) |>
  mutate(
    displayName = factor(displayName, levels = wr_posterior$displayName)
  ) |>
  ggplot(aes(x = r_bc_id__kappa, y = displayName)) +
  ggridges::geom_density_ridges(rel_min_height = 0.05, alpha = 0.3)


# tight ends

te_filtered <- tracking_angle_full |>
  distinct(gameId, playId, bc_id, bc_position, bc_type) |>
  filter(bc_position == "TE") |>
  count(bc_id) |>
  filter(n >= 10) |>
  pull(bc_id)

te_posterior <- angle_brms |>
  spread_draws(r_bc_id__kappa[nflId, term]) |>
  left_join(players) |>
  filter(nflId %in% te_filtered) |>
  group_by(nflId, displayName, position) |>
  summarize(posterior_mean = mean(r_bc_id__kappa)) |>
  ungroup() |>
  arrange(posterior_mean)

te_top <- te_posterior |> slice_head(n = 10) |> pull(nflId)
te_bottom <- te_posterior |> slice_tail(n = 10) |> pull(nflId)

angle_brms |>
  spread_draws(r_bc_id__kappa[nflId, term]) |>
  ungroup() |>
  left_join(players) |>
  filter(nflId %in% c(te_top, te_bottom)) |>
  mutate(
    displayName = factor(displayName, levels = te_posterior$displayName)
  ) |>
  ggplot(aes(x = r_bc_id__kappa, y = displayName)) +
  ggridges::geom_density_ridges(rel_min_height = 0.05, alpha = 0.3)

# running backs

rb_filtered <- tracking_angle_full |>
  distinct(gameId, playId, bc_id, bc_position, bc_type) |>
  filter(bc_position == "RB") |>
  count(bc_id) |>
  filter(n >= 25) |>
  pull(bc_id)

rb_posterior <- angle_brms |>
  spread_draws(r_bc_id__kappa[nflId, term]) |>
  left_join(players) |>
  filter(nflId %in% rb_filtered) |>
  group_by(nflId, displayName, position) |>
  summarize(posterior_mean = mean(r_bc_id__kappa)) |>
  ungroup() |>
  arrange(posterior_mean)

rb_top <- rb_posterior |> slice_head(n = 10) |> pull(nflId)
rb_bottom <- rb_posterior |> slice_tail(n = 10) |> pull(nflId)

angle_brms |>
  spread_draws(r_bc_id__kappa[nflId, term]) |>
  ungroup() |>
  left_join(players) |>
  filter(nflId %in% c(rb_top, rb_bottom)) |>
  mutate(
    displayName = factor(displayName, levels = rb_posterior$displayName)
  ) |>
  ggplot(aes(x = r_bc_id__kappa, y = displayName)) +
  ggridges::geom_density_ridges(rel_min_height = 0.05, alpha = 0.3)

# comparison with 40-yard dash time

library(nflreadr)

player_id_conversion <- load_ff_playerids() |>
  select(pfr_id, gsis_id) |>
  left_join(select(load_players(), gsis_id, nflId = nfl_id))

angle_brms |>
  spread_draws(r_bc_id__kappa[nflId, term]) |>
  left_join(players) |>
  filter(nflId %in% c(rb_filtered, wr_filtered, te_filtered)) |>
  group_by(nflId, displayName, position) |>
  summarize(posterior_mean = mean(r_bc_id__kappa)) |>
  ungroup() |>
  left_join(player_id_conversion |> mutate(nflId = as.numeric(nflId))) |>
  left_join(select(load_combine(), pfr_id, ht:shuttle)) |>
  filter(nflId != 54042) |> # hmmm
  ggplot(aes(posterior_mean, forty)) +
  geom_point(aes(color = position), size = 3, alpha = 0.6) +
  # ggrepel::geom_text_repel(aes(label = displayName)) +
  scale_color_manual(values = c("#009E73", "#1E88E5", "darkorange")) +
  expand_limits(x = 0.4, y = c(4.2, 5))
