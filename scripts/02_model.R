library(tidyverse)

# to prep the data,
# run 01a_prep_rusher_data.R and 01b_prep_receiver_data.R

tracking_angle_full <- read_rds("scripts/tracking_angle_receivers.rds") |>
  bind_rows(read_rds("scripts/tracking_angle_rushers.rds"))

tracking_angle_full |>
  arrow::write_parquet('scripts/tracking_angle_full.parquet')

tracking_angle_full <- tracking_angle_full |>
  mutate(adj_y_change_abs = abs(adj_y_change))

library(brms)

angle_brms <- brm(
  bf(
    turn_angle ~
      prev_angle +
        adj_bc_x +
        adj_bc_y +
        adj_bc_x_from_first_down +
        n_left_bc_defense +
        n_front_bc_defense +
        n_left_bc_offense +
        n_front_bc_offense +
        adj_x +
        adj_y +
        adj_x_change +
        adj_y_change_abs +
        dist_to_bc +
        def_s +
        angle_with_bc,

    kappa ~
      bc_s +
        bc_a +
        bc_cum_dis +
        bc_type +
        bc_position +
        (1 | gr(bc_id, by = bc_position)),

    decomp = "QR"
  ),
  family = von_mises(),
  chains = 4,
  iter = 3500,
  warmup = 1500,
  seed = 60660,
  cores = 4,
  init = "0",
  control = list(adapt_delta = 0.9),
  backend = "cmdstanr",
  data = tracking_angle_full,
)

write_rds(angle_brms, "scripts/angle_brms.rds", compress = "gz")
