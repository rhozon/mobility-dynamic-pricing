# Dynamic pricing for ride-hailing

How riders and drivers adapt when prices move, on 301.6 million Uber and Lyft trips in New York
City (December 2023 to April 2024 and September 2024 to June 2025), from the public trip records
of the NYC Taxi and Limousine Commission.

Case study page, with an interactive simulator:
<https://rhozon.github.io/portfolio/mobility-dynamic-pricing.html>

## What is in it

| Part | Module | What it answers |
|---|---|---|
| Aggregation and data quality | `mdp/aggregate.py` | One DuckDB pass per monthly file: zone-hour-app aggregates, a quality row, a 0.5% trip sample |
| Fare model | `mdp/fare.py` | How much of the fare is the trip (distance, time) and how much is the market (dynamic price index by area and hour) |
| Demand forecast | `mdp/forecast.py` | Trips by zone and hour, one week ahead, rolling origin, against two seasonal baselines |
| Rider dynamics | `mdp/dynamics.py` | Static choice, logit dynamic with inertia or replicator-like drift? Lyft's January 2025 congestion-fee credit as a natural price shock, with a placebo year |
| Evolutionary simulator | `mdp/simulate.py` | Riders and drivers across six areas, day to day, under a fixed price, a market-clearing surge and a smoothed surge |
| Switchback planning | `mdp/experiment.py` | Minimum detectable effect on margin by block length and test duration |

## Results (from `results/`)

| | |
|---|---|
| Fare model, June 2025 holdout, mean absolute % error | base 23.4%, XGBoost 18.3% |
| Demand forecast a week ahead, WAPE, April to June 2025 | XGBoost 13.1%, mean of last 4 weeks 13.0%, same hour last week 16.1% |
| Daily inertia of the Lyft share (logit dynamic) | 0.61 (95% CI 0.58 to 0.64), half-life about 1.4 days |
| Credit effect, difference in differences (log-odds) | 0.021 (-0.002 to 0.045); placebo year 0.029 (0.013 to 0.045) |
| Share forecast 8 Feb to 30 Apr 2025, RMSE | static 2.80 points, dynamic 2.83 points |
| Onset of price cycles (driver imitation speed) | 0.75 with market-clearing surge, 1.75 with strong smoothing |

The page discusses each result and its limits.

## Run it

Python 3.11 or later.

```bash
pip install -e ".[dev]"
pytest -v
python -m mdp.cli download     # monthly files into MDP_RAW (about 7 GB)
python -m mdp.cli aggregate    # aggregates into MDP_DERIVED
python -m mdp.cli analyse      # tables in results/
python -m mdp.cli export site/data.json
```

`MDP_RAW` and `MDP_DERIVED` set where the raw files and the aggregates go; both stay out of the
repository because of their size.

## Sources

- NYC Taxi and Limousine Commission, High Volume For-Hire Vehicle trip records and taxi zones.
- Lyft, terms of the NYC congestion fee credit (5 to 31 January 2025).
- Cohen, Hahn, Hall, Levitt and Metcalfe (2016), *Using Big Data to Estimate Consumer Surplus:
  The Case of Uber*, NBER Working Paper 22627 (demand elasticity used in the simulator).
- Sandholm (2010), *Population Games and Evolutionary Dynamics*, MIT Press.
- Bojinov, Simchi-Levi and Zhao (2023), *Design and Analysis of Switchback Experiments*,
  Management Science.

## License

MIT. Rodrigo Hermont Ozon.
