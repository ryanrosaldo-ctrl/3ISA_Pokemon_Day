# Pokemon Day II - 3ISA Engine System

## Section: 3ISA | Regions: Hoenn, Sinnoh, Galar

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open: http://localhost:8501

## Systems Included
1. **Team Engine** — Generates Gym Leader defending teams (native-region only)
2. **Challenger Selection Engine** — Generates challenger counter lineups
3. **Battle Prediction Engine** — Predicts winner, records ground truth

## Files
- `app.py` — Main Streamlit application
- `pokemon_data.json` — Pre-cached PokéAPI data (Hoenn/Sinnoh/Galar, 293 Pokemon)
- `pokemon_day.db` — SQLite database (auto-created on first run)
- `requirements.txt` — Python dependencies
- `run.sh` — Quick start script

## Data Source
All Pokemon data sourced from **PokéAPI** (https://pokeapi.co) and pre-cached locally
for reliable offline use on Pokemon Day.

## Battle Restrictions Enforced
- No Legendary / Mythical Pokemon
- No Paradox Pokemon
- Region-native filtering (Pikachu = Kanto, not usable here)

## Database Tables
- `team_outputs` — Generated gym leader teams
- `challenger_outputs` — Generated challenger lineups
- `predictions` — Pre-battle predictions
- `ground_truth` — Actual battle results
- `audit_log` — All system actions with timestamps
