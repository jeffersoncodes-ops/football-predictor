# Poisson Football Model ⚽

A simple, transparent Poisson-based model to analyze football matches and detect value bets using public data from [football-data.co.uk](https://www.football-data.co.uk).

No black boxes, no paid APIs, no crypto nonsense. Just math and data.

## How it works

1. **Download** historical match results from 5 European leagues
2. **Calculate** each team's attack/defense strength (home & away)
3. **Predict** expected goals using the Poisson distribution
4. **Generate** probabilities for 1X2, Over/Under, and exact scorelines
5. **Compare** with bookmaker odds to find value bets (+EV)
6. **Size** bets using the Kelly Criterion

## Quick start

```bash
pip install pandas numpy pyarrow
python poisson_model.py "Barcelona vs Real Madrid"
```

### Examples

```bash
# Analyze a match
python poisson_model.py "PSG vs Marseille"
python poisson_model.py "Bayern vs Dortmund"
python poisson_model.py "Inter vs Juventus"
python poisson_model.py "Liverpool vs Arsenal"

# List available teams per league
python poisson_model.py --teams

# Force fresh data download
python poisson_model.py "Milan vs Napoli" --refresh
```

### Sample output

```
=================================================================
  ANALISIS: Paris SG vs Marseille (Ligue 1)
=================================================================

  Expected Goals:
    Paris SG (local):  2.324
    Marseille (visit):  1.334

  Probabilidades:
    Paris SG: 59.5%  (cuota justa: 1.68)
    Empate:    19.4%  (cuota justa: 5.16)
    Marseille: 21.0%  (cuota justa: 4.75)
    Over 2.5:  70.7%

  Marcadores mas probables:
    2-1   -> 9.3%
    1-1   -> 8.0%
    3-1   -> 7.2%
    2-0   -> 7.0%
    2-2   -> 6.2%
```

## Supported leagues

| Code | League | Source |
|------|--------|--------|
| E0   | Premier League | football-data.co.uk |
| SP1  | LaLiga | football-data.co.uk |
| I1   | Serie A | football-data.co.uk |
| D1   | Bundesliga | football-data.co.uk |
| F1   | Ligue 1 | football-data.co.uk |

## The math

The model uses a **double Poisson** approach:

```
xG_home = Attack_Home × Defense_Away × League_avg_home_goals
xG_away = Attack_Away × Defense_Home × League_avg_away_goals

P(k goals) = (λ^k × e^(-λ)) / k!
```

For value betting, it compares the model's "fair odds" against actual bookmaker prices. When the model says a team has a 45% chance (fair odds 2.22) and a bookmaker offers 2.50, that's a +EV opportunity.

Staking uses fractional Kelly Criterion (25%) to manage risk.

## Limitations

- Basic Poisson under-predicts draws (0-0, 1-1 are more common than the model thinks)
- Doesn't account for injuries, weather, or lineup changes
- Uses goals scored, not expected goals (xG would be more predictive)
- Assumes home/away goal independence (Dixon-Coles would fix this)

## Roadmap

- [ ] Dixon-Coles adjustment for draw bias
- [ ] xG data from Understat/FBref
- [ ] Automated backtesting against historical results
- [ ] Live odds comparison via API
- [ ] Web UI

## License

MIT
