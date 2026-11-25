# Model card: ride-hailing pricing model (version 0.2.0)

## What it does

Given an area of New York City, a day type and an hour, it quotes what a change in the platform's
price would do to trips, revenue, margin, driver pay, wait and the platform's rider share
(`POST /quote`), and gives the market-clearing and the smoothed price multiplier for today's
supply and demand (`POST /multiplier`).

## Data

NYC Taxi and Limousine Commission high-volume for-hire trip records, Uber and Lyft only,
December 2023 to April 2024 and September 2024 to June 2025 (301.6 million trips). Profiles and
calibration use February 8 to June 30, 2025.

## Parts and their evidence

| Part | Source | Evidence |
|---|---|---|
| Demand elasticity, -0.4 (range -0.2 to -0.6) | Published estimate for Uber (Cohen et al., 2016) | Not estimated here |
| Wait elasticity to demand per driver, 0.21 | Within zone and hour of week, 2025 | Descriptive |
| Rider share response to relative price and wait | Static choice model, congestion-zone days | Association, not causal |
| Daily inertia of the rider share, 0.61 | Dynamic choice model | Does not forecast better than the static model after the Lyft credit |

## Intended use

Exploring pricing scenarios and planning experiments. It is not a production price engine:
driver supply is held fixed in quotes, and the demand elasticity is borrowed.

## Known limits

- Fares in the source exclude credits and promotions; Uber's promotions are unknown.
- The wait is pickup minus request and includes boarding.
- Since 2025 part of Uber's requests go to yellow cabs and leave this source.
- Speeds of driver adaptation in the simulator are scenarios; the data have no driver identifiers.

## Monitoring

`python -m mdp.cli drift` computes the population stability index of trip distance, minutes, fare
per mile and wait each month against September to December 2024. Every month through June 2025 is
below 0.05 (stable). The monthly quality report flags changes in how the source is filled.

## Retraining

When a feature's PSI passes 0.25, when the quality report shows a change in coverage, or each
quarter: run `aggregate`, `analyse`, `export` and rebuild the Docker image.
