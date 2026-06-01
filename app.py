import streamlit as st
import pandas as pd
import numpy as np
import json
import sqlite3
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Pokémon Day II — 3ISA Engine",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

SECTION = "3ISA"
ALLOWED_REGIONS = ["Hoenn", "Sinnoh", "Galar"]

TYPE_CHART = {
    "Normal":    {"Rock":-1,"Ghost":-2,"Steel":-1},
    "Fire":      {"Fire":-1,"Water":-1,"Rock":-1,"Dragon":-1,"Grass":1,"Ice":1,"Bug":1,"Steel":1},
    "Water":     {"Water":-1,"Grass":-1,"Dragon":-1,"Fire":1,"Ground":1,"Rock":1},
    "Electric":  {"Electric":-1,"Grass":-1,"Dragon":-1,"Ground":-2,"Flying":1,"Water":1},
    "Grass":     {"Fire":-1,"Grass":-1,"Poison":-1,"Flying":-1,"Bug":-1,"Dragon":-1,"Steel":-1,"Water":1,"Ground":1,"Rock":1},
    "Ice":       {"Water":-1,"Ice":-1,"Steel":-1,"Fire":-1,"Grass":1,"Ground":1,"Flying":1,"Dragon":1},
    "Fighting":  {"Poison":-1,"Bug":-1,"Psychic":-1,"Flying":-1,"Fairy":-1,"Ghost":-2,"Normal":1,"Ice":1,"Rock":1,"Dark":1,"Steel":1},
    "Poison":    {"Poison":-1,"Ground":-1,"Rock":-1,"Ghost":-1,"Steel":-2,"Grass":1,"Fairy":1},
    "Ground":    {"Grass":-1,"Bug":-1,"Flying":-2,"Electric":1,"Fire":1,"Poison":1,"Rock":1,"Steel":1},
    "Flying":    {"Electric":-1,"Rock":-1,"Steel":-1,"Grass":1,"Fighting":1,"Bug":1},
    "Psychic":   {"Psychic":-1,"Steel":-1,"Dark":-2,"Fighting":1,"Poison":1},
    "Bug":       {"Fire":-1,"Fighting":-1,"Flying":-1,"Ghost":-1,"Steel":-1,"Fairy":-1,"Grass":1,"Psychic":1,"Dark":1},
    "Rock":      {"Fighting":-1,"Ground":-1,"Steel":-1,"Fire":1,"Ice":1,"Flying":1,"Bug":1},
    "Ghost":     {"Normal":-2,"Dark":-1,"Psychic":1,"Ghost":1},
    "Dragon":    {"Steel":-1,"Fairy":-2,"Dragon":1},
    "Dark":      {"Fighting":-1,"Dark":-1,"Fairy":-1,"Psychic":1,"Ghost":1},
    "Steel":     {"Steel":-1,"Fire":-1,"Water":-1,"Electric":-1,"Ice":1,"Rock":1,"Fairy":1},
    "Fairy":     {"Fire":-1,"Poison":-1,"Steel":-1,"Fighting":1,"Dragon":1,"Dark":1},
}

TYPE_COLORS = {
    "Normal": "#a8a77a", "Fire": "#ee8130", "Water": "#6390f0", "Electric": "#f7d02c",
    "Grass": "#7ac74c", "Ice": "#96d9d6", "Fighting": "#c22e28", "Poison": "#a33ea1",
    "Ground": "#e2bf65", "Flying": "#a98ff3", "Psychic": "#f95587", "Bug": "#a6b91a",
    "Rock": "#b6a136", "Ghost": "#735797", "Dragon": "#6f35fc", "Dark": "#705746",
    "Steel": "#b7b7ce", "Fairy": "#d685ad"
}

def get_effectiveness(atk_type, def_types):
    """Returns multiplier: 2=super, 0.5=not very, 0=immune"""
    mult = 1.0
    for dt in def_types:
        if dt:
            v = TYPE_CHART.get(atk_type, {}).get(dt, 0)
            if v == 1:   mult *= 2
            elif v == -1: mult *= 0.5
            elif v == -2: mult = 0
    return mult

def type_advantage_score(attacker, defender_types):
    scores = []
    for atype in [attacker["type_1"], attacker.get("type_2","")]:
        if atype:
            scores.append(get_effectiveness(atype, defender_types))
    return max(scores) if scores else 1.0

# ─────────────────────────────────────────
#  DATABASE
# ─────────────────────────────────────────
DB_PATH = "pokemon_day.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS team_outputs (
        team_output_id INTEGER PRIMARY KEY AUTOINCREMENT,
        section TEXT, gym_leader TEXT, region TEXT,
        type_specialization TEXT, generated_team TEXT,
        model_used TEXT, metric_used TEXT, timestamp TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS challenger_outputs (
        challenger_output_id INTEGER PRIMARY KEY AUTOINCREMENT,
        section TEXT, target_gym_leader TEXT, challenger_region TEXT,
        gym_leader_team TEXT, recommended_team TEXT,
        model_used TEXT, counter_score TEXT, timestamp TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        match_id TEXT PRIMARY KEY, gym_leader TEXT, challenger TEXT,
        gym_leader_region TEXT, challenger_region TEXT,
        gym_leader_team TEXT, challenger_team TEXT,
        predicted_winner TEXT, confidence_score REAL,
        prediction_reason TEXT, timestamp_before TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS ground_truth (
        match_id TEXT PRIMARY KEY, actual_winner TEXT,
        correct_prediction INTEGER, final_score TEXT,
        number_of_turns INTEGER, replay_link TEXT,
        screenshot_link TEXT, timestamp_after TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS audit_log (
        audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_or_operator TEXT, action_done TEXT,
        affected_record TEXT, old_value TEXT,
        new_value TEXT, timestamp TEXT
    )""")
    conn.commit()
    conn.close()

def log_audit(action, record, old_val="", new_val=""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO audit_log (user_or_operator,action_done,affected_record,old_value,new_value,timestamp) VALUES (?,?,?,?,?,?)",
                 ("3ISA Operator", action, record, old_val, new_val, datetime.now().isoformat()))
    conn.commit()
    conn.close()

# ─────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────
@st.cache_data
def load_pokemon():
    with open("pokemon_data.json") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["type_2"] = df["type_2"].fillna("")
    df["total"] = df[["hp","attack","defense","special_attack","special_defense","speed"]].sum(axis=1)
    return df

def preload_session_state_from_db():
    if "db_preloaded" in st.session_state:
        return
    
    df_all = load_pokemon()
    
    # 1. Load last gym team
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""SELECT gym_leader, region, type_specialization, generated_team 
                     FROM team_outputs 
                     ORDER BY timestamp DESC LIMIT 1""")
        row = c.fetchone()
        if row:
            gym_leader, region, type_spec, gen_team_str = row
            try:
                names = json.loads(gen_team_str)
                gym_rows = []
                for name in names:
                    match = df_all[df_all["name"].str.lower() == name.lower()]
                    if len(match) > 0:
                        gym_rows.append(match.iloc[0])
                if gym_rows:
                    st.session_state["last_gym_team"] = pd.DataFrame(gym_rows)
                    st.session_state["last_gym_leader"] = gym_leader
                    st.session_state["last_gym_region"] = region
                    st.session_state["last_gym_type"] = type_spec
            except Exception:
                pass
        
        # 2. Load last challenger team
        c.execute("""SELECT target_gym_leader, challenger_region, recommended_team 
                     FROM challenger_outputs 
                     ORDER BY timestamp DESC LIMIT 1""")
        row_chal = c.fetchone()
        if row_chal:
            gym_leader_name, chal_region, rec_team_str = row_chal
            try:
                names = json.loads(rec_team_str)
                chal_rows = []
                for name in names:
                    match = df_all[df_all["name"].str.lower() == name.lower()]
                    if len(match) > 0:
                        chal_rows.append(match.iloc[0])
                if chal_rows:
                    st.session_state["last_chal_team"] = pd.DataFrame(chal_rows)
                    st.session_state["last_chal_name"] = "Challenger"
                    st.session_state["last_chal_region"] = chal_region
            except Exception:
                pass
        conn.close()
    except Exception:
        pass
        
    st.session_state["db_preloaded"] = True

def apply_battle_restrictions(df):
    """Remove all restricted Pokemon (Legendary, Mythical, Paradox)."""
    filtered = df[df["is_restricted"] == False].copy()
    removed = len(df) - len(filtered)
    return filtered, removed

def validate_team(team_df, region):
    """Check team for restriction and region violations. Returns list of issues."""
    issues = []
    for _, row in team_df.iterrows():
        if row.get("is_restricted", False):
            issues.append(f"⛔ {row['name']} is RESTRICTED (Legendary/Mythical/Paradox)")
        if row.get("native_region","") != region:
            issues.append(f"⛔ {row['name']} is NOT native to {region} (native: {row.get('native_region','')})")
    return issues

# ─────────────────────────────────────────
#  TEAM ENGINE
# ─────────────────────────────────────────
def run_team_engine(region, type_spec, model="KNN + Rule-Based Scoring"):
    df = load_pokemon()
    # Apply battle restriction filter first
    df, _ = apply_battle_restrictions(df)
    pool = df[df["native_region"] == region].copy()

    # Type filter: at least one type matches specialization
    typed = pool[(pool["type_1"] == type_spec) | (pool["type_2"] == type_spec)].copy()

    if len(typed) < 6:
        # Fill with same-region pokemon if not enough typed ones
        others = pool[~pool["id"].isin(typed["id"])].copy()
        typed = pd.concat([typed, others]).head(20)

    # Score each pokemon
    typed["role_score"] = 0.0
    typed["role"] = "Balanced"

    if model == "KNN + Rule-Based Scoring":
        # Fit a KNN model on all pokemon to predict roles based on stats
        all_df = load_pokemon()
        roles = []
        for _, r in all_df.iterrows():
            if r["speed"] >= 90: roles.append("Sweeper")
            elif r["hp"] >= 90 and r["defense"] >= 90: roles.append("Tank")
            elif r["special_attack"] >= 100: roles.append("Special Attacker")
            elif r["attack"] >= 100: roles.append("Physical Attacker")
            elif r["special_defense"] >= 90: roles.append("Support")
            else: roles.append("Balanced")
        all_df["temp_role"] = roles
        
        X = all_df[["hp", "attack", "defense", "special_attack", "special_defense", "speed"]]
        le = LabelEncoder()
        y = le.fit_transform(all_df["temp_role"])
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X, y)
        
        # Predict roles using KNN
        X_pool = typed[["hp", "attack", "defense", "special_attack", "special_defense", "speed"]]
        pred_y = knn.predict(X_pool)
        typed["role"] = le.inverse_transform(pred_y)
        
    elif model == "Random Forest Scoring":
        # Fit a Random Forest to predict if a pokemon is "Competitive" (total >= 450)
        all_df = load_pokemon()
        all_df["is_comp"] = (all_df["total"] >= 450).astype(int)
        X = all_df[["hp", "attack", "defense", "special_attack", "special_defense", "speed"]]
        y = all_df["is_comp"]
        
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X, y)
        
        # Predict probability of being competitive
        X_pool = typed[["hp", "attack", "defense", "special_attack", "special_defense", "speed"]]
        probs = rf.predict_proba(X_pool)[:, 1]
        typed["model_score"] = probs * 40
        
        # Assign traditional roles
        for idx, row in typed.iterrows():
            if row["speed"] >= 90: typed.at[idx, "role"] = "Sweeper"
            elif row["hp"] >= 90 and row["defense"] >= 90: typed.at[idx, "role"] = "Tank"
            elif row["special_attack"] >= 100: typed.at[idx, "role"] = "Special Attacker"
            elif row["attack"] >= 100: typed.at[idx, "role"] = "Physical Attacker"
            elif row["special_defense"] >= 90: typed.at[idx, "role"] = "Support"
            else: typed.at[idx, "role"] = "Balanced"
            
    else: # Rule-Based Only
        # Assign roles directly using simple rules
        for idx, row in typed.iterrows():
            if row["speed"] >= 90: typed.at[idx, "role"] = "Sweeper"
            elif row["hp"] >= 90 and row["defense"] >= 90: typed.at[idx, "role"] = "Tank"
            elif row["special_attack"] >= 100: typed.at[idx, "role"] = "Special Attacker"
            elif row["attack"] >= 100: typed.at[idx, "role"] = "Physical Attacker"
            elif row["special_defense"] >= 90: typed.at[idx, "role"] = "Support"
            else: typed.at[idx, "role"] = "Balanced"

    for idx, row in typed.iterrows():
        s = 0
        if row["type_1"] == type_spec or row["type_2"] == type_spec:
            s += 30
        
        if model == "Random Forest Scoring":
            s += row["model_score"]
        else:
            s += row["total"] * 0.05
            
        typed.at[idx, "role_score"] = s

    typed = typed.sort_values("role_score", ascending=False)

    # Pick 6 with role diversity
    team = []
    used_roles = []
    for _, row in typed.iterrows():
        if len(team) >= 6:
            break
        role = row["role"]
        if used_roles.count(role) < 2:
            team.append(row)
            used_roles.append(role)

    if len(team) < 6:
        remaining = typed[~typed["id"].isin([p["id"] for p in team])]
        for _, row in remaining.iterrows():
            if len(team) >= 6:
                break
            team.append(row)

    team_df = pd.DataFrame(team[:6])

    # Build explanation
    reasons = []
    for _, row in team_df.iterrows():
        r = f"Native {region} {row['type_1']}"
        if row["type_2"]: r += f"/{row['type_2']}"
        r += f"-type | {row['role']} | BST {row['total']}"
        reasons.append(r)
    team_df["reason"] = reasons

    return team_df[["name","native_region","type_1","type_2","role","hp","attack","defense",
                     "special_attack","special_defense","speed","total","reason"]]

# ─────────────────────────────────────────
#  CHALLENGER SELECTION ENGINE
# ─────────────────────────────────────────
def run_challenger_engine(gym_team_df, challenger_region, model="Counter Scoring + KNN"):
    df = load_pokemon()
    # Apply battle restriction filter first
    df, _ = apply_battle_restrictions(df)
    pool = df[df["native_region"] == challenger_region].copy()

    # Pre-train KNN role classifier if using KNN model
    knn_trained = False
    if "KNN" in model:
        try:
            all_df = load_pokemon()
            roles = []
            for _, r in all_df.iterrows():
                if r["speed"] >= 90: roles.append("Sweeper")
                elif r["hp"] >= 90 and r["defense"] >= 90: roles.append("Tank")
                elif r["special_attack"] >= 100: roles.append("Special Attacker")
                elif r["attack"] >= 100: roles.append("Physical Attacker")
                elif r["special_defense"] >= 90: roles.append("Support")
                else: roles.append("Balanced")
            all_df["temp_role"] = roles
            X = all_df[["hp", "attack", "defense", "special_attack", "special_defense", "speed"]]
            le = LabelEncoder()
            y = le.fit_transform(all_df["temp_role"])
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X, y)
            knn_trained = True
        except Exception:
            pass

    gym_types = []
    for _, row in gym_team_df.iterrows():
        if row["type_1"]: gym_types.append(row["type_1"])
        if row["type_2"]: gym_types.append(row["type_2"])

    # Score each challenger pokemon vs the gym team
    scores = []
    for _, poke in pool.iterrows():
        score = 0
        adv_details = []

        if model != "Stat-Based Ranking":
            # Apply Type Matchup Calculations
            for _, opp in gym_team_df.iterrows():
                opp_types = [t for t in [opp["type_1"], opp.get("type_2","")] if t]
                eff = type_advantage_score(poke, opp_types)
                score += eff * 10

                if eff >= 2:
                    adv_details.append(f"Super vs {opp['name']}")
                elif eff == 0:
                    adv_details.append(f"Immune vs {opp['name']}")

        if model != "Type Advantage Scoring":
            # Apply Stat-Based Calculations
            avg_gym_def = gym_team_df["defense"].mean()
            avg_gym_spdef = gym_team_df["special_defense"].mean()
            avg_gym_spd = gym_team_df["speed"].mean()
            avg_gym_hp = gym_team_df["hp"].mean()

            if model == "Stat-Based Ranking":
                # Pure stat comparison score
                stat_score = 0
                if poke["hp"] > avg_gym_hp:
                    stat_score += 5
                    adv_details.append("Bulkier")
                if poke["attack"] > avg_gym_def:
                    stat_score += 8
                    adv_details.append("ATK vs DEF")
                if poke["special_attack"] > avg_gym_spdef:
                    stat_score += 8
                    adv_details.append("SPA vs SPD")
                if poke["speed"] > avg_gym_spd:
                    stat_score += 10
                    adv_details.append("Outspeeds")
                score = stat_score
            else:
                # Combined bonus
                if poke["attack"] > avg_gym_def * 1.2:
                    score += 5
                    adv_details.append("High ATK")
                if poke["special_attack"] > avg_gym_spdef * 1.2:
                    score += 5
                    adv_details.append("High SPA")
                if poke["speed"] >= 90:
                    score += 3
                    adv_details.append("Fast")

        # Role assignment
        if knn_trained:
            X_poke = pd.DataFrame([poke[["hp", "attack", "defense", "special_attack", "special_defense", "speed"]]])
            pred_y = knn.predict(X_poke)
            role = le.inverse_transform(pred_y)[0]
        else:
            if poke["speed"] >= 90: role = "Sweeper"
            elif poke["hp"] >= 90 and poke["defense"] >= 90: role = "Tank"
            elif poke["special_attack"] >= 100: role = "Special Attacker"
            elif poke["attack"] >= 100: role = "Physical Attacker"
            elif poke["special_defense"] >= 90: role = "Support"
            else: role = "Balanced"

        scores.append({
            "id": poke["id"],
            "name": poke["name"],
            "native_region": poke["native_region"],
            "type_1": poke["type_1"],
            "type_2": poke["type_2"],
            "role": role,
            "hp": poke["hp"],
            "attack": poke["attack"],
            "defense": poke["defense"],
            "special_attack": poke["special_attack"],
            "special_defense": poke["special_defense"],
            "speed": poke["speed"],
            "total": poke["total"],
            "counter_score": round(score, 2),
            "advantage_details": ", ".join(adv_details[:3]) if adv_details else "Type neutral"
        })

    scored_df = pd.DataFrame(scores).sort_values("counter_score", ascending=False)

    # Pick 6 with role diversity
    team = []
    used_roles = []
    for _, row in scored_df.iterrows():
        if len(team) >= 6: break
        role = row["role"]
        if used_roles.count(role) < 2:
            team.append(row)
            used_roles.append(role)

    if len(team) < 6:
        remaining = scored_df[~scored_df["id"].isin([p["id"] for p in team])]
        for _, row in remaining.iterrows():
            if len(team) >= 6: break
            team.append(row)

    result = pd.DataFrame(team[:6])
    return result[["name","native_region","type_1","type_2","role","hp","attack","defense",
                    "special_attack","special_defense","speed","total","counter_score","advantage_details"]]

# ─────────────────────────────────────────
#  BATTLE PREDICTION ENGINE
# ─────────────────────────────────────────
def compute_team_score(team_df):
    """Aggregate team score for prediction."""
    if len(team_df) == 0:
        return 0
    score = 0
    score += team_df["total"].mean() * 0.4
    score += team_df["speed"].mean() * 0.15
    score += team_df["attack"].mean() * 0.1
    score += team_df["special_attack"].mean() * 0.1
    score += team_df["defense"].mean() * 0.1
    score += team_df["hp"].mean() * 0.15
    return score

def compute_type_coverage_score(attacker_df, defender_df):
    """How well does attacker cover defender's types?"""
    total_eff = 0
    count = 0
    for _, atk in attacker_df.iterrows():
        for _, dfd in defender_df.iterrows():
            def_types = [t for t in [dfd["type_1"], dfd.get("type_2","")] if t]
            eff = type_advantage_score(atk, def_types)
            total_eff += eff
            count += 1
    return (total_eff / count) if count else 1.0

def predict_battle(gym_team_df, chal_team_df, gym_leader_name, challenger_name):
    gym_score = compute_team_score(gym_team_df)
    chal_score = compute_team_score(chal_team_df)

    gym_coverage = compute_type_coverage_score(gym_team_df, chal_team_df)
    chal_coverage = compute_type_coverage_score(chal_team_df, gym_team_df)

    gym_final = gym_score * gym_coverage
    chal_final = chal_score * chal_coverage

    total = gym_final + chal_final
    if total == 0:
        gym_prob = 0.5
        chal_prob = 0.5
    else:
        gym_prob = gym_final / total
        chal_prob = chal_final / total

    if gym_prob > chal_prob:
        predicted_winner = gym_leader_name
        confidence = round(min(gym_prob, 0.95), 2)
    elif chal_prob > gym_prob:
        predicted_winner = challenger_name
        confidence = round(min(chal_prob, 0.95), 2)
    else:
        predicted_winner = "Tie"
        confidence = 0.50

    reasons = []
    if gym_score > chal_score:
        reasons.append(f"Gym team BST advantage (+{gym_score-chal_score:.0f})")
    else:
        reasons.append(f"Challenger BST advantage (+{chal_score-gym_score:.0f})")
    if gym_coverage > chal_coverage:
        reasons.append(f"Gym better type coverage ({gym_coverage:.2f}x avg effectiveness)")
    else:
        reasons.append(f"Challenger better type coverage ({chal_coverage:.2f}x avg effectiveness)")
    speed_adv = gym_team_df["speed"].mean() - chal_team_df["speed"].mean()
    if abs(speed_adv) > 5:
        if speed_adv > 0:
            reasons.append(f"Gym team faster (avg SPD +{speed_adv:.0f})")
        else:
            reasons.append(f"Challenger faster (avg SPD +{abs(speed_adv):.0f})")

    return {
        "predicted_winner": predicted_winner,
        "confidence_score": confidence,
        "prediction_reason": " | ".join(reasons),
        "gym_prob": gym_prob,
        "chal_prob": chal_prob
    }

# ─────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────
def get_prediction_metrics():
    conn = sqlite3.connect(DB_PATH)
    preds = pd.read_sql("SELECT * FROM predictions", conn)
    truth = pd.read_sql("SELECT * FROM ground_truth", conn)
    conn.close()
    if len(preds) == 0 or len(truth) == 0:
        return None, None, None
    merged = preds.merge(truth, on="match_id", how="inner")
    if len(merged) == 0:
        return None, None, None
    y_true = merged["actual_winner"]
    y_pred = merged["predicted_winner"]
    acc = accuracy_score(y_true, y_pred)
    labels = sorted(y_true.unique().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return acc, cm, labels, merged

# ─────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800;900&family=Rajdhani:wght@600;700&display=swap');

    /* Main billboard/header styling */
    .main-header {
        background: linear-gradient(135deg, #101530 0%, #1c234a 100%);
        border: 2px solid #3b4cca;
        border-top: 5px solid #FF1C1C; /* PokeBall Red top border */
        padding: 2rem; border-radius: 16px; margin-bottom: 2rem;
        text-align: center; color: white !important;
        box-shadow: 0 8px 30px rgba(59, 76, 202, 0.35), inset 0 0 15px rgba(255, 203, 5, 0.15);
    }
    .main-header h1 { 
        font-family: 'Orbitron', sans-serif !important;
        font-size: 2.3rem; margin: 0; font-weight: 900; letter-spacing: 2px;
        color: #FFCB05 !important;
        text-shadow: 0px 4px 0px #3b4cca, 0 0 10px rgba(255, 203, 5, 0.3);
    }
    .main-header p  { 
        font-family: 'Rajdhani', sans-serif !important;
        margin: 0.5rem 0 0; font-size: 1.1rem; opacity: 0.9; 
        font-weight: 700; letter-spacing: 1px; text-transform: uppercase; 
        color: #a5b4fc !important;
    }

    /* Engine Card - Glassmorphism style with Red PokeBall Border */
    .engine-card {
        background: #151b3d !important;
        border: 1px solid rgba(59, 76, 202, 0.5) !important;
        border-top: 4px solid #FF1C1C !important;
        border-radius: 12px !important; padding: 1.5rem !important; margin-bottom: 1.2rem !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
    }
    .engine-card h3 { 
        font-family: 'Orbitron', sans-serif !important;
        color: #FFCB05 !important; margin: 0 0 0.5rem; font-weight: 700; 
    }

    /* Dashboard Metrics Cards with Gold/Yellow PokeBall Border */
    .metric-card {
        background: #111638 !important;
        border: 1px solid #3b4cca !important;
        border-top: 4px solid #FFCB05 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important; text-align: center !important; color: white !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #FFCB05 !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4), 0 0 12px rgba(255, 203, 5, 0.3) !important;
    }
    .metric-card .value { 
        font-family: 'Orbitron', sans-serif !important;
        font-size: 2.2rem !important; font-weight: 900; color: #FFCB05 !important; 
        text-shadow: 0 0 8px rgba(255, 203, 5, 0.3);
    }
    .metric-card .label { 
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 0.9rem !important; opacity: 0.9; letter-spacing: 1px; 
        text-transform: uppercase; color: #94a3b8 !important; margin-top: 5px; 
    }

    /* Tables */
    .pokemon-table th { 
        background: #1c234a !important; color: white !important; 
        font-family: 'Orbitron', sans-serif !important;
    }

    /* Region-specific dynamic badges */
    .badge {
        display: inline-block; padding: 4px 10px; border-radius: 16px;
        font-size: 0.75rem; font-weight: 700; margin: 2px;
        text-transform: uppercase; letter-spacing: 0.5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .badge-hoenn  { background: linear-gradient(135deg, #1b4d3e, #2d6a2d); color: #90ee90; border: 1px solid #52b788; }
    .badge-sinnoh { background: linear-gradient(135deg, #3d348b, #4a2d6a); color: #dda0dd; border: 1px solid #dda0dd; }
    .badge-galar  { background: linear-gradient(135deg, #780000, #6a2d2d); color: #f08080; border: 1px solid #c1121f; }

    /* Match winner display */
    .winner-box {
        background: radial-gradient(circle at center, rgba(34, 197, 94, 0.15) 0%, rgba(20, 83, 45, 0.3) 100%);
        border: 2px dashed #22c55e; border-radius: 12px;
        padding: 1.5rem; text-align: center;
        box-shadow: 0 0 15px rgba(34, 197, 94, 0.2);
    }
    .winner-box h2 { 
        font-family: 'Orbitron', sans-serif !important;
        color: #22c55e; margin: 0; font-size: 2rem; 
        text-shadow: 0 0 8px rgba(34, 197, 94, 0.3);
    }

    /* Streamlit Action Button Overrides */
    div.stButton > button { 
        background: linear-gradient(135deg, #3b4cca 0%, #1d2c5e 100%) !important;
        color: #ffffff !important;
        border: 1px solid #FFCB05 !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        box-shadow: 0 4px 15px rgba(59, 76, 202, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #FFCB05 0%, #d8ac04 100%) !important;
        color: #1d2c5e !important;
        border-color: #3b4cca !important;
        box-shadow: 0 6px 20px rgba(255, 203, 5, 0.5) !important;
        transform: scale(1.01) !important;
    }

    /* Deep cyber-arena background glow */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 30%, #151b3d 0%, #070913 100%) !important;
        background-attachment: fixed !important;
    }
    
    /* Dynamic type glow card styles */
    .pokemon-glow-card {
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    }
    .pokemon-glow-card:hover {
        transform: translateY(-5px) scale(1.02) !important;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.5), 0 0 20px var(--glow-color, rgba(255, 203, 5, 0.4)) !important;
        background: rgba(26, 32, 74, 0.85) !important;
    }

    /* Laser Line Divider */
    .laser-line {
        height: 2px;
        background: linear-gradient(to right, rgba(59,76,202,0) 0%, #FFCB05 50%, rgba(59,76,202,0) 100%);
        margin: 20px 0;
        box-shadow: 0 0 8px #FFCB05;
        animation: pulse-laser 2s infinite alternate;
    }
    @keyframes pulse-laser {
        0% { opacity: 0.6; box-shadow: 0 0 4px #FFCB05; }
        100% { opacity: 1.0; box-shadow: 0 0 12px #FFCB05; }
    }

    /* 18 Pokémon type badges */
    .badge-type-normal   { background: linear-gradient(135deg, #a8a77a, #c6c6a7) !important; color: #ffffff !important; border: 1px solid #a8a77a !important; }
    .badge-type-fire     { background: linear-gradient(135deg, #ee8130, #f5ac78) !important; color: #ffffff !important; border: 1px solid #ee8130 !important; }
    .badge-type-water    { background: linear-gradient(135deg, #6390f0, #9db7f5) !important; color: #ffffff !important; border: 1px solid #6390f0 !important; }
    .badge-type-electric { background: linear-gradient(135deg, #f7d02c, #fae078) !important; color: #1d2c5e !important; border: 1px solid #f7d02c !important; }
    .badge-type-grass    { background: linear-gradient(135deg, #7ac74c, #a7db8d) !important; color: #ffffff !important; border: 1px solid #7ac74c !important; }
    .badge-type-ice      { background: linear-gradient(135deg, #96d9d6, #bce6e6) !important; color: #1d2c5e !important; border: 1px solid #96d9d6 !important; }
    .badge-type-fighting { background: linear-gradient(135deg, #c22e28, #d67873) !important; color: #ffffff !important; border: 1px solid #c22e28 !important; }
    .badge-type-poison   { background: linear-gradient(135deg, #a33ea1, #c183c1) !important; color: #ffffff !important; border: 1px solid #a33ea1 !important; }
    .badge-type-ground   { background: linear-gradient(135deg, #e2bf65, #ebd69d) !important; color: #1d2c5e !important; border: 1px solid #e2bf65 !important; }
    .badge-type-flying   { background: linear-gradient(135deg, #a98ff3, #c6b7f5) !important; color: #ffffff !important; border: 1px solid #a98ff3 !important; }
    .badge-type-psychic  { background: linear-gradient(135deg, #f95587, #fa92b2) !important; color: #ffffff !important; border: 1px solid #f95587 !important; }
    .badge-type-bug      { background: linear-gradient(135deg, #a6b91a, #c1d15a) !important; color: #ffffff !important; border: 1px solid #a6b91a !important; }
    .badge-type-rock     { background: linear-gradient(135deg, #b6a136, #d1c17d) !important; color: #ffffff !important; border: 1px solid #b6a136 !important; }
    .badge-type-ghost    { background: linear-gradient(135deg, #735797, #a292bc) !important; color: #ffffff !important; border: 1px solid #735797 !important; }
    .badge-type-dragon   { background: linear-gradient(135deg, #6f35fc, #a17ffd) !important; color: #ffffff !important; border: 1px solid #6f35fc !important; }
    .badge-type-dark     { background: linear-gradient(135deg, #705746, #907c6f) !important; color: #ffffff !important; border: 1px solid #705746 !important; }
    .badge-type-steel    { background: linear-gradient(135deg, #b7b7ce, #d1d1e0) !important; color: #1d2c5e !important; border: 1px solid #b7b7ce !important; }
    .badge-type-fairy    { background: linear-gradient(135deg, #d685ad, #e3b5cd) !important; color: #ffffff !important; border: 1px solid #d685ad !important; }

    /* Enforce light text globally inside the app container, except for inputs and selectbox options */
    [data-testid="stAppViewContainer"] p, 
    [data-testid="stAppViewContainer"] span, 
    [data-testid="stAppViewContainer"] li, 
    [data-testid="stAppViewContainer"] ul, 
    [data-testid="stAppViewContainer"] ol {
        color: #cbd5e1 !important;
    }

    [data-testid="stAppViewContainer"] h1, 
    [data-testid="stAppViewContainer"] h2, 
    [data-testid="stAppViewContainer"] h3 {
        color: #ffffff !important;
    }
    
    [data-testid="stAppViewContainer"] h4, 
    [data-testid="stAppViewContainer"] h5, 
    [data-testid="stAppViewContainer"] h6 {
        color: #FFCB05 !important;
        font-family: 'Orbitron', sans-serif !important;
    }

    /* Enforce light text for labels and selectbox headers */
    [data-testid="stWidgetLabel"] p, .stWidgetForm label {
        color: #FFCB05 !important;
        font-weight: 700 !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.15rem !important;
        letter-spacing: 0.5px !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }
    
    /* Tabs styling to make selected and unselected headers bright and readable */
    button[data-baseweb="tab"] p {
        color: #a5b4fc !important;
        font-weight: 700 !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.1rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #FFCB05 !important;
        text-shadow: 0 0 5px rgba(255, 203, 5, 0.4) !important;
    }
    
    /* Accordion / Expander header text styling */
    [data-baseweb="accordion"] p, [data-baseweb="accordion"] span {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Checkbox & Radio labels text readability */
    [data-testid="stCheckbox"] label span, [data-testid="stRadio"] label span {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Enforce light color for notification/alert text */
    [data-testid="stNotification"] p {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* Enforce dark text inside the light sidebar for legibility */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] li, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #111638 !important;
        text-shadow: none !important;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #ffcb05 !important;
        text-shadow: 2px 2px #3b4cca !important;
        font-family: 'Orbitron', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  INIT
# ─────────────────────────────────────────
init_db()
preload_session_state_from_db()

# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 15px;">
        <svg viewBox="0 0 100 100" width="70" height="70" style="filter: drop-shadow(0px 0px 8px rgba(255, 203, 5, 0.6));">
            <circle cx="50" cy="50" r="45" fill="white" stroke="#1d2c5e" stroke-width="6"/>
            <path d="M 5,50 A 45,45 0 0,1 95,50 Z" fill="#FF1C1C" stroke="#1d2c5e" stroke-width="6"/>
            <line x1="5" y1="50" x2="95" y2="50" stroke="#1d2c5e" stroke-width="8"/>
            <circle cx="50" cy="50" r="16" fill="white" stroke="#1d2c5e" stroke-width="6"/>
            <circle cx="50" cy="50" r="6" fill="#1d2c5e" stroke-width="0"/>
        </svg>
    </div>
    <h1>POKÉMON DAY II — 3ISA ENGINE</h1>
    <p>Team Engine · Challenger Selection Engine · Battle Prediction Engine</p>
    <p style="opacity:0.65;font-size:0.85rem;margin-top:6px;letter-spacing:0.5px;">Sections: Hoenn · Sinnoh · Galar | Data Source: PokéAPI (Cached)</p>
    <div style="background: rgba(255, 203, 5, 0.15); border: 1px solid #FFCB05; border-radius: 20px; display: inline-block; padding: 6px 18px; margin-top: 10px; font-family: 'Orbitron', sans-serif; font-size: 0.9rem; color: #ffcb05; font-weight: 700; letter-spacing: 1px; box-shadow: 0 0 10px rgba(255, 203, 5, 0.2);">🛡️ OFFICIAL HOST: AHDADDEE GYM 🛡️</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  SIDEBAR NAV
# ─────────────────────────────────────────
with st.sidebar:
    if os.path.exists("pokemon_logo.png"):
        st.image("pokemon_logo.png", width="stretch")
    else:
        st.markdown("""
        <div style="text-align:center; margin-bottom:15px; margin-top:10px;">
            <svg viewBox="0 0 100 100" width="55" height="55" style="filter: drop-shadow(0px 0px 6px rgba(255, 203, 5, 0.5));">
                <circle cx="50" cy="50" r="45" fill="white" stroke="#1d2c5e" stroke-width="6"/>
                <path d="M 5,50 A 45,45 0 0,1 95,50 Z" fill="#FF1C1C" stroke="#1d2c5e" stroke-width="6"/>
                <line x1="5" y1="50" x2="95" y2="50" stroke="#1d2c5e" stroke-width="8"/>
                <circle cx="50" cy="50" r="16" fill="white" stroke="#1d2c5e" stroke-width="6"/>
                <circle cx="50" cy="50" r="6" fill="#1d2c5e" stroke-width="0"/>
            </svg>
            <h2 style='color:#ffcb05;text-shadow: 2px 2px #3b4cca;font-family:sans-serif;margin-top:10px;margin-bottom:0px;font-size:1.5rem;'>POKÉMON DAY II</h2>
            <p style='font-size:0.8rem;color:#a5b4fc;margin-top:2px;font-weight:700;text-transform:uppercase;'>3ISA System · Ahdaddee Gym</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('<div class="laser-line"></div>', unsafe_allow_html=True)
    page = st.radio("Navigation", [
        "🏟️ Team Engine",
        "⚔️ Challenger Selection",
        "🔮 Battle Prediction",
        "📊 Analytics & Logs",
        "🗄️ Database Viewer"
    ])
    st.markdown('<div class="laser-line"></div>', unsafe_allow_html=True)
    df_all = load_pokemon()
    st.markdown(f"**Pokémon Loaded:** {len(df_all)}")
    for r in ALLOWED_REGIONS:
        cnt = len(df_all[df_all["native_region"]==r])
        st.markdown(f"- {r}: **{cnt}**")
    st.markdown('<hr style="border:0;height:2px;background:linear-gradient(to right,rgba(59,76,202,0),#FFCB05,#3b4cca,rgba(59,76,202,0));margin:12px 0;">', unsafe_allow_html=True)
    st.markdown("**Section:** 3ISA")
    st.markdown("**Regions:** Hoenn, Sinnoh, Galar")
    st.markdown("**Data Source:** PokéAPI (Pre-Cached)")

# ─────────────────────────────────────────
#  PAGE 1: TEAM ENGINE
# ─────────────────────────────────────────
if page == "🏟️ Team Engine":
    st.header("🏟️ Team Engine — Gym Leader Team Generator")
    st.markdown("Generates a 6-Pokémon Gym Leader defending team using only **native-region** Pokémon.")

    col1, col2, col3 = st.columns(3)
    with col1:
        region = st.selectbox("Gym Leader Region", ALLOWED_REGIONS)
    with col2:
        all_types = sorted(load_pokemon()["type_1"].unique().tolist())
        type_spec = st.selectbox("Type Specialization", all_types)
    with col3:
        gym_leader_name = st.text_input("Gym Leader Name", value="Ahdaddee Gym", placeholder="e.g. Ahdaddee Gym")

    model_choice = st.selectbox("Model / Logic", ["KNN + Rule-Based Scoring", "Random Forest Scoring", "Rule-Based Only"])

    if st.button("⚡ Generate Gym Leader Team", type="primary"):
        with st.spinner("Generating team..."):
            team = run_team_engine(region, type_spec, model_choice)

        if len(team) == 0:
            st.error("Not enough Pokémon found for this region/type combination.")
        else:
            st.success(f"✅ Generated {len(team)} Pokémon for **{gym_leader_name or 'Gym Leader'}** ({region} / {type_spec})")

            # Validation check
            issues = validate_team(team, region)
            if issues:
                for issue in issues:
                    st.error(issue)
            else:
                st.success("✅ Validation passed — all Pokémon are native to region and unrestricted.")

            # Restriction filter info
            df_raw = load_pokemon()
            _, removed = apply_battle_restrictions(df_raw[df_raw["native_region"]==region])
            if removed > 0:
                st.info(f"🔒 **Restriction filter applied:** {removed} restricted Pokémon were excluded from the {region} pool.")

            # Display team as Pokémon cards
            st.subheader("🏟️ Team Grid View")
            cols = st.columns(3)
            for idx, row in team.reset_index(drop=True).iterrows():
                col = cols[idx % 3]
                primary_type = row["type_1"]
                type_color = TYPE_COLORS.get(primary_type, "#3b4cca")
                t2_html = f'<span class="badge badge-type-{row["type_2"].lower()}" style="margin-left: 5px;">{row["type_2"]}</span>' if row["type_2"] else ''
                col.markdown(f"""
                <div class="pokemon-glow-card" style="background: rgba(21, 27, 61, 0.7); border: 1px solid {type_color}40; border-top: 5px solid {type_color}; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.3), 0 0 10px {type_color}15; text-align: center; --glow-color: {type_color};">
                    <h4 style="color: #FFCB05; margin: 0; font-family: 'Orbitron', sans-serif; font-size: 1.3rem;">{row['name']}</h4>
                    <p style="margin: 5px 0 10px; font-size: 0.8rem; color: #a5b4fc; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">{row['role']}</p>
                    <div style="margin-bottom: 10px;">
                        <span class="badge badge-{row['native_region'].lower()}">{row['native_region']}</span>
                        <span class="badge badge-type-{row['type_1'].lower()}">{row['type_1']}</span>{t2_html}
                    </div>
                    <div style="font-size: 0.85rem; text-align: left; background: rgba(8, 10, 24, 0.4); padding: 8px; border-radius: 8px; color: #cbd5e1; border: 1px solid rgba(59, 76, 202, 0.2);">
                        <div style="display: flex; justify-content: space-between;"><span>HP: <b>{row['hp']}</b></span><span>ATK: <b>{row['attack']}</b></span></div>
                        <div style="display: flex; justify-content: space-between;"><span>DEF: <b>{row['defense']}</b></span><span>SPA: <b>{row['special_attack']}</b></span></div>
                        <div style="display: flex; justify-content: space-between;"><span>SPD: <b>{row['special_defense']}</b></span><span>SPE: <b>{row['speed']}</b></span></div>
                        <div style="border-top: 1px solid rgba(255,255,255,0.1); margin-top: 5px; padding-top: 5px; font-weight: bold; color: #FFCB05; display: flex; justify-content: space-between;">
                            <span>BST:</span><span>{row['total']}</span>
                        </div>
                    </div>
                    <p style="font-size: 0.75rem; color: #94a3b8; font-style: italic; margin-top: 8px; margin-bottom: 0px; text-align: left; line-height: 1.2;">{row['reason']}</p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("📊 View Raw Data Table"):
                display_cols = ["name","native_region","type_1","type_2","role","hp","attack","defense","special_attack","special_defense","speed","total","reason"]
                st.dataframe(team[display_cols].reset_index(drop=True), use_container_width=True)

            # Radar chart
            st.subheader("📊 Team Stat Profile")
            fig = go.Figure()
            stat_cols = ["hp","attack","defense","special_attack","special_defense","speed"]
            for _, row in team.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row[s] for s in stat_cols],
                    theta=["HP","ATK","DEF","SPA","SPD","SPE"],
                    fill='toself', name=row["name"], opacity=0.6
                ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,200])),
                              showlegend=True, height=450,
                              paper_bgcolor="#0d1117", font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)

            # Type distribution
            types_list = team["type_1"].tolist() + [t for t in team["type_2"].tolist() if t]
            type_counts = pd.Series(types_list).value_counts().reset_index()
            type_counts.columns = ["Type","Count"]
            fig2 = px.bar(type_counts, x="Type", y="Count", title="Type Distribution in Team",
                          color="Count", color_continuous_scale="blues")
            fig2.update_layout(paper_bgcolor="#0d1117", font=dict(color="white"), plot_bgcolor="#0d1117")
            st.plotly_chart(fig2, use_container_width=True)

            # Showdown export
            st.subheader("📋 Pokémon Showdown Format")
            showdown_text = ""
            for _, row in team.iterrows():
                showdown_text += f"{row['name']}\n"
                if row["type_2"]:
                    showdown_text += f"- Type: {row['type_1']}/{row['type_2']}\n"
                else:
                    showdown_text += f"- Type: {row['type_1']}\n"
                showdown_text += f"EVs: 252 HP / 252 Atk / 4 Spe\n\n"
            st.code(showdown_text, language="text")

            # CSV export
            st.subheader("⬇️ Download Team CSV")
            team_csv_cols = ["name", "type_1", "type_2", "native_region", "hp", "attack", "defense", "special_attack", "special_defense", "speed"]
            team_csv = team[team_csv_cols].to_csv(index=False)
            st.download_button("⬇️ Download Team CSV", team_csv, f"team_{gym_leader_name or 'gym'}.csv", "text/csv")

            # Save to DB
            conn = sqlite3.connect(DB_PATH)
            conn.execute("""INSERT INTO team_outputs
                (section,gym_leader,region,type_specialization,generated_team,model_used,metric_used,timestamp)
                VALUES (?,?,?,?,?,?,?,?)""",
                (SECTION, gym_leader_name or "Unknown", region, type_spec,
                 json.dumps(team["name"].tolist()), model_choice, "Role Balance Score",
                 datetime.now().isoformat()))
            conn.commit()
            conn.close()
            log_audit("GENERATE_TEAM", f"{gym_leader_name}_{region}_{type_spec}", "", json.dumps(team["name"].tolist()))

            # Store in session for challenger engine
            st.session_state["last_gym_team"] = team
            st.session_state["last_gym_leader"] = gym_leader_name or "Gym Leader"
            st.session_state["last_gym_region"] = region
            st.session_state["last_gym_type"] = type_spec
            st.info("✅ Team saved to database. Head to **Challenger Selection** to generate a counter team.")

# ─────────────────────────────────────────
#  PAGE 2: CHALLENGER SELECTION ENGINE
# ─────────────────────────────────────────
elif page == "⚔️ Challenger Selection":
    st.header("⚔️ Challenger Selection Engine")
    st.markdown("Generates the best 6-Pokémon challenger lineup to counter a Gym Leader's team.")
    st.info(f"**3ISA Allowed Challenger Regions:** Hoenn, Sinnoh, Galar")

    col1, col2 = st.columns(2)
    with col1:
        challenger_name = st.text_input("Challenger Name / Group", value=st.session_state.get("last_chal_name",""), placeholder="e.g. Group A")
        chal_reg_val = st.session_state.get("last_chal_region", ALLOWED_REGIONS[0])
        chal_reg_idx = ALLOWED_REGIONS.index(chal_reg_val) if chal_reg_val in ALLOWED_REGIONS else 0
        challenger_region = st.selectbox("Challenger Region (3ISA)", ALLOWED_REGIONS, index=chal_reg_idx)
    with col2:
        gym_leader_name = st.text_input("Target Gym Leader Name", value=st.session_state.get("last_gym_leader","Ahdaddee Gym"), placeholder="e.g. Ahdaddee Gym")

    st.subheader("Gym Leader Team Input")
    use_previous = False
    if "last_gym_team" in st.session_state:
        use_previous = st.checkbox(f"Use last generated team ({st.session_state.get('last_gym_leader','?')} — {st.session_state.get('last_gym_region','?')} / {st.session_state.get('last_gym_type','?')})", value=True)

    if use_previous and "last_gym_team" in st.session_state:
        gym_team = st.session_state["last_gym_team"]
        st.dataframe(gym_team[["name","native_region","type_1","type_2","total"]].reset_index(drop=True), use_container_width=True)
    else:
        default_gym_input = ""
        if "last_gym_team" in st.session_state:
            default_gym_input = "\n".join(st.session_state["last_gym_team"]["name"].tolist())
        st.markdown("**Enter Gym Leader Pokémon (one per line):**")
        gym_input = st.text_area("Gym Team (names only)", value=default_gym_input, placeholder="Flygon\nSalamence\nGarchomp\nAltaria\nVibrava\nDragonite", height=150)
        df_all = load_pokemon()
        gym_rows = []
        if gym_input:
            for name in gym_input.strip().split("\n"):
                match = df_all[df_all["name"].str.lower() == name.strip().lower()]
                if len(match) > 0:
                    gym_rows.append(match.iloc[0])
                else:
                    st.warning(f"'{name}' not found in database")
        gym_team = pd.DataFrame(gym_rows) if gym_rows else pd.DataFrame()

    model_choice = st.selectbox("Model / Logic", ["Counter Scoring + KNN", "Type Advantage Scoring", "Stat-Based Ranking"])

    if st.button("⚡ Generate Challenger Lineup", type="primary"):
        if len(gym_team) == 0:
            st.error("Please provide a Gym Leader team first.")
        else:
            with st.spinner("Calculating best counters..."):
                result = run_challenger_engine(gym_team, challenger_region, model_choice)

            st.success(f"✅ Generated challenger lineup for **{challenger_name or 'Challenger'}** ({challenger_region})")

            # Validation check
            issues = validate_team(result, challenger_region)
            if issues:
                for issue in issues:
                    st.error(issue)
            else:
                st.success("✅ Validation passed — all recommended Pokémon are native to challenger region and unrestricted.")

            # Side by side comparison
            col_g, col_c = st.columns(2)
            with col_g:
                st.subheader("🔴 Gym Leader Team")
                st.dataframe(gym_team[["name","type_1","type_2","total"]].reset_index(drop=True), use_container_width=True)
            with col_c:
                st.subheader("🔵 Recommended Challenger Team")
                st.dataframe(result[["name","native_region","type_1","type_2","role","counter_score","advantage_details"]].reset_index(drop=True), use_container_width=True)

            # Display challenger team as cards
            st.subheader("⚔️ Recommended Counter Lineup")
            cols = st.columns(3)
            for idx, row in result.reset_index(drop=True).iterrows():
                col = cols[idx % 3]
                primary_type = row["type_1"]
                type_color = TYPE_COLORS.get(primary_type, "#3b4cca")
                t2_html = f'<span class="badge badge-type-{row["type_2"].lower()}" style="margin-left: 5px;">{row["type_2"]}</span>' if row["type_2"] else ''
                col.markdown(f"""
                <div class="pokemon-glow-card" style="background: rgba(21, 27, 61, 0.75); border: 1px solid {type_color}40; border-top: 5px solid {type_color}; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.3), 0 0 10px {type_color}15; text-align: center; --glow-color: {type_color};">
                    <h4 style="color: #FFCB05; margin: 0; font-family: 'Orbitron', sans-serif; font-size: 1.3rem;">{row['name']}</h4>
                    <p style="margin: 5px 0 10px; font-size: 0.8rem; color: #a5b4fc; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">{row['role']}</p>
                    <div style="margin-bottom: 10px;">
                        <span class="badge badge-{row['native_region'].lower()}">{row['native_region']}</span>
                        <span class="badge badge-type-{row['type_1'].lower()}">{row['type_1']}</span>{t2_html}
                    </div>
                    <div style="font-size: 0.85rem; text-align: left; background: rgba(8, 10, 24, 0.4); padding: 8px; border-radius: 8px; color: #cbd5e1; border: 1px solid rgba(59, 76, 202, 0.2); margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between;"><span>HP: <b>{row['hp']}</b></span><span>ATK: <b>{row['attack']}</b></span></div>
                        <div style="display: flex; justify-content: space-between;"><span>DEF: <b>{row['defense']}</b></span><span>SPA: <b>{row['special_attack']}</b></span></div>
                        <div style="display: flex; justify-content: space-between;"><span>SPD: <b>{row['special_defense']}</b></span><span>SPE: <b>{row['speed']}</b></span></div>
                        <div style="border-top: 1px solid rgba(255,255,255,0.1); margin-top: 5px; padding-top: 5px; font-weight: bold; color: #FFCB05; display: flex; justify-content: space-between;">
                            <span>BST:</span><span>{row['total']}</span>
                        </div>
                    </div>
                    <div style="background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); padding: 5px; border-radius: 6px; margin-bottom: 5px;">
                        <span style="font-size: 0.8rem; color: #22c55e; font-weight: bold;">Counter Score: {row['counter_score']}</span>
                    </div>
                    <p style="font-size: 0.75rem; color: #94a3b8; font-style: italic; margin-top: 5px; margin-bottom: 0px; text-align: left; line-height: 1.2;"><b>Advantage:</b> {row['advantage_details']}</p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("📊 View Raw Data Table"):
                st.dataframe(result.reset_index(drop=True), use_container_width=True)

            # Counter score bar chart
            fig = px.bar(result, x="name", y="counter_score",
                         color="counter_score", color_continuous_scale="greens",
                         title="Counter Score per Pokémon", labels={"name":"Pokémon","counter_score":"Counter Score"})
            fig.update_layout(paper_bgcolor="#0d1117", font=dict(color="white"), plot_bgcolor="#0d1117")
            st.plotly_chart(fig, use_container_width=True)

            # Showdown export
            st.subheader("📋 Pokémon Showdown Format")
            showdown_text = ""
            for _, row in result.iterrows():
                showdown_text += f"{row['name']}\n"
                showdown_text += f"EVs: 252 Atk / 4 Def / 252 Spe\n\n"
            st.code(showdown_text, language="text")

            # CSV export
            st.subheader("⬇️ Download Challenger Team CSV")
            chal_csv_cols = ["name", "type_1", "type_2", "native_region", "hp", "attack", "defense", "special_attack", "special_defense", "speed"]
            chal_csv = result[chal_csv_cols].to_csv(index=False)
            st.download_button("⬇️ Download Challenger Team CSV", chal_csv, f"challenger_{challenger_name or 'team'}.csv", "text/csv")

            # Save
            conn = sqlite3.connect(DB_PATH)
            conn.execute("""INSERT INTO challenger_outputs
                (section,target_gym_leader,challenger_region,gym_leader_team,recommended_team,model_used,counter_score,timestamp)
                VALUES (?,?,?,?,?,?,?,?)""",
                (SECTION, gym_leader_name or "Unknown", challenger_region,
                 json.dumps(gym_team["name"].tolist() if "name" in gym_team.columns else []),
                 json.dumps(result["name"].tolist()),
                 model_choice, json.dumps(result["counter_score"].tolist()),
                 datetime.now().isoformat()))
            conn.commit()
            conn.close()
            log_audit("GENERATE_CHALLENGER", f"{challenger_name}_{challenger_region}", "", json.dumps(result["name"].tolist()))

            st.session_state["last_chal_team"] = result
            st.session_state["last_chal_name"] = challenger_name or "Challenger"
            st.session_state["last_chal_region"] = challenger_region
            st.info("✅ Saved to database. Head to **Battle Prediction** to record a prediction.")

# ─────────────────────────────────────────
#  PAGE 3: BATTLE PREDICTION ENGINE
# ─────────────────────────────────────────
elif page == "🔮 Battle Prediction":
    st.header("🔮 Battle Prediction Engine")
    st.markdown("Predict the winner **before** the battle starts, then record the actual result.")

    tab1, tab2 = st.tabs(["📝 Record Prediction", "✅ Record Result (Ground Truth)"])

    with tab1:
        st.subheader("Pre-Battle Prediction")
        st.warning("⚠️ Prediction MUST be recorded BEFORE the battle starts!")

        col1, col2 = st.columns(2)
        with col1:
            match_id = st.text_input("Match ID", value=f"MATCH-{datetime.now().strftime('%m%d-%H%M')}")
            gym_leader_pred = st.text_input("Gym Leader Name", value=st.session_state.get("last_gym_leader","Ahdaddee Gym"))
            gym_reg_val = st.session_state.get("last_gym_region", ALLOWED_REGIONS[0])
            gym_reg_idx = ALLOWED_REGIONS.index(gym_reg_val) if gym_reg_val in ALLOWED_REGIONS else 0
            gym_region_pred = st.selectbox("Gym Leader Region", ALLOWED_REGIONS, index=gym_reg_idx, key="gym_reg_pred")
            gym_type_pred = st.text_input("Gym Leader Type", value=st.session_state.get("last_gym_type",""))
        with col2:
            challenger_pred = st.text_input("Challenger Name", value=st.session_state.get("last_chal_name",""))
            chal_reg_val = st.session_state.get("last_chal_region", ALLOWED_REGIONS[0])
            chal_reg_idx = ALLOWED_REGIONS.index(chal_reg_val) if chal_reg_val in ALLOWED_REGIONS else 0
            chal_region_pred = st.selectbox("Challenger Region", ALLOWED_REGIONS, index=chal_reg_idx, key="chal_reg_pred")

        # Team inputs
        col_g, col_c = st.columns(2)
        df_all = load_pokemon()

        with col_g:
            st.subheader("Gym Leader Lineup")
            use_last_gym = False
            if "last_gym_team" in st.session_state:
                use_last_gym = st.checkbox("Use last generated gym team", value=True, key="use_gym_pred")
            if use_last_gym and "last_gym_team" in st.session_state:
                gym_team_pred = st.session_state["last_gym_team"]
                st.dataframe(gym_team_pred[["name","type_1","type_2","total"]].reset_index(drop=True), use_container_width=True)
            else:
                default_gym_pred = ""
                if "last_gym_team" in st.session_state:
                    default_gym_pred = "\n".join(st.session_state["last_gym_team"]["name"].tolist())
                gym_input2 = st.text_area("Gym Team", value=default_gym_pred, placeholder="Flygon\nSalamence", key="gym_input_pred")
                gym_rows2 = []
                if gym_input2:
                    for name in gym_input2.strip().split("\n"):
                        match = df_all[df_all["name"].str.lower() == name.strip().lower()]
                        if len(match) > 0: gym_rows2.append(match.iloc[0])
                gym_team_pred = pd.DataFrame(gym_rows2) if gym_rows2 else pd.DataFrame()

        with col_c:
            st.subheader("Challenger Lineup")
            use_last_chal = False
            if "last_chal_team" in st.session_state:
                use_last_chal = st.checkbox("Use last generated challenger team", value=True, key="use_chal_pred")
            if use_last_chal and "last_chal_team" in st.session_state:
                chal_team_pred = st.session_state["last_chal_team"]
                st.dataframe(chal_team_pred[["name","type_1","type_2","total"]].reset_index(drop=True), use_container_width=True)
            else:
                default_chal_pred = ""
                if "last_chal_team" in st.session_state:
                    default_chal_pred = "\n".join(st.session_state["last_chal_team"]["name"].tolist())
                chal_input2 = st.text_area("Challenger Team", value=default_chal_pred, placeholder="Garchomp\nLucario", key="chal_input_pred")
                chal_rows2 = []
                if chal_input2:
                    for name in chal_input2.strip().split("\n"):
                        match = df_all[df_all["name"].str.lower() == name.strip().lower()]
                        if len(match) > 0: chal_rows2.append(match.iloc[0])
                chal_team_pred = pd.DataFrame(chal_rows2) if chal_rows2 else pd.DataFrame()

        if st.button("🔮 Generate Prediction", type="primary"):
            if len(gym_team_pred) == 0 or len(chal_team_pred) == 0:
                st.error("Both teams must be set before predicting.")
            else:
                pred = predict_battle(gym_team_pred, chal_team_pred, gym_leader_pred, challenger_pred)

                # Display result
                winner_color = "#4caf50" if pred["predicted_winner"] == challenger_pred else "#f44336"
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #101530 0%, #1c234a 100%); border: 2px solid {winner_color}; border-top: 5px solid {winner_color}; border-radius: 16px; padding: 2rem; text-align: center; margin: 1.5rem 0; box-shadow: 0 8px 30px rgba(0,0,0,0.4);">
                    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 15px;">
                        <svg viewBox="0 0 100 100" width="50" height="50" style="filter: drop-shadow(0px 0px 8px {winner_color});">
                            <circle cx="50" cy="50" r="45" fill="white" stroke="#1d2c5e" stroke-width="6"/>
                            <path d="M 5,50 A 45,45 0 0,1 95,50 Z" fill="#FF1C1C" stroke="#1d2c5e" stroke-width="6"/>
                            <line x1="5" y1="50" x2="95" y2="50" stroke="#1d2c5e" stroke-width="8"/>
                            <circle cx="50" cy="50" r="16" fill="white" stroke="#1d2c5e" stroke-width="6"/>
                            <circle cx="50" cy="50" r="6" fill="#1d2c5e" stroke-width="0"/>
                        </svg>
                    </div>
                    <p style="color:#a5b4fc; font-family: 'Rajdhani', sans-serif; font-size: 1.2rem; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; margin: 5px 0;">Predicted Winner</p>
                    <h2 style="color: {winner_color}; font-family: 'Orbitron', sans-serif; font-size: 2.5rem; font-weight: 900; margin: 10px 0; text-shadow: 0 0 15px rgba(255,203,5,0.2);">🏆 {pred['predicted_winner']}</h2>
                    <div style="display: inline-block; background: rgba(255,255,255,0.05); padding: 6px 20px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.1); margin-top: 10px;">
                        <span style="color: #FFCB05; font-family: 'Orbitron', sans-serif; font-size: 1.2rem; font-weight: bold; letter-spacing: 1px;">CONFIDENCE: {pred['confidence_score']*100:.0f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Win probability bar
                fig = go.Figure(go.Bar(
                    x=[pred["gym_prob"]*100, pred["chal_prob"]*100],
                    y=[gym_leader_pred, challenger_pred],
                    orientation="h",
                    marker_color=["#f44336","#4caf50"]
                ))
                fig.update_layout(title="Win Probability",
                                  paper_bgcolor="#0d1117", font=dict(color="white"),
                                  plot_bgcolor="#0d1117", height=200,
                                  xaxis_title="Win Probability (%)")
                st.plotly_chart(fig, use_container_width=True)

                st.info(f"**Prediction Reason:** {pred['prediction_reason']}")

                # Save prediction
                conn = sqlite3.connect(DB_PATH)
                try:
                    conn.execute("""INSERT INTO predictions
                        (match_id,gym_leader,challenger,gym_leader_region,challenger_region,
                         gym_leader_team,challenger_team,predicted_winner,confidence_score,
                         prediction_reason,timestamp_before)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (match_id, gym_leader_pred, challenger_pred, gym_region_pred,
                         chal_region_pred,
                         json.dumps(gym_team_pred["name"].tolist() if "name" in gym_team_pred.columns else []),
                         json.dumps(chal_team_pred["name"].tolist() if "name" in chal_team_pred.columns else []),
                         pred["predicted_winner"], pred["confidence_score"],
                         pred["prediction_reason"], datetime.now().isoformat()))
                    conn.commit()
                    st.success(f"✅ Prediction saved! Match ID: **{match_id}**")
                    log_audit("RECORD_PREDICTION", match_id, "", pred["predicted_winner"])
                except sqlite3.IntegrityError:
                    st.warning(f"Match ID {match_id} already exists. Use a different Match ID.")
                conn.close()

    with tab2:
        st.subheader("Post-Battle Ground Truth Recording")
        st.info("Record the actual result AFTER the battle ends.")

        conn = sqlite3.connect(DB_PATH)
        preds_df = pd.read_sql("SELECT match_id, gym_leader, challenger, predicted_winner FROM predictions", conn)
        conn.close()

        if len(preds_df) == 0:
            st.warning("No predictions recorded yet.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                match_ids = preds_df["match_id"].tolist()
                sel_match = st.selectbox("Select Match ID", match_ids)
                match_row = preds_df[preds_df["match_id"] == sel_match].iloc[0]
                st.info(f"**Predicted:** {match_row['predicted_winner']}")

                actual_winner = st.selectbox("Actual Winner",
                    [match_row["gym_leader"], match_row["challenger"], "Draw"])
                final_score = st.text_input("Final Score", placeholder="2-0")
                num_turns = st.number_input("Number of Turns", min_value=0, value=0)
            with col2:
                replay_link = st.text_input("Pokémon Showdown Replay Link", placeholder="https://replay.pokemonshowdown.com/...")
                screenshot_link = st.text_input("Screenshot / Photo Link", placeholder="https://...")

            if st.button("✅ Save Ground Truth", type="primary"):
                correct = 1 if actual_winner == match_row["predicted_winner"] else 0
                conn = sqlite3.connect(DB_PATH)
                try:
                    conn.execute("""INSERT INTO ground_truth
                        (match_id,actual_winner,correct_prediction,final_score,
                         number_of_turns,replay_link,screenshot_link,timestamp_after)
                        VALUES (?,?,?,?,?,?,?,?)""",
                        (sel_match, actual_winner, correct, final_score,
                         num_turns if num_turns > 0 else None,
                         replay_link, screenshot_link, datetime.now().isoformat()))
                    conn.commit()
                    if correct:
                        st.success(f"✅ Ground truth saved! Prediction was **CORRECT** 🎉")
                    else:
                        st.warning(f"Ground truth saved. Prediction was **INCORRECT** (predicted {match_row['predicted_winner']}, actual {actual_winner})")
                    log_audit("RECORD_GROUND_TRUTH", sel_match, match_row["predicted_winner"], actual_winner)
                except sqlite3.IntegrityError:
                    st.warning("Ground truth for this match already exists.")
                conn.close()

# ─────────────────────────────────────────
#  PAGE 4: ANALYTICS
# ─────────────────────────────────────────
elif page == "📊 Analytics & Logs":
    st.header("📊 Analytics, Metrics & Logs")

    # Summary metrics
    conn = sqlite3.connect(DB_PATH)
    n_teams = pd.read_sql("SELECT COUNT(*) as c FROM team_outputs", conn).iloc[0]["c"]
    n_challengers = pd.read_sql("SELECT COUNT(*) as c FROM challenger_outputs", conn).iloc[0]["c"]
    n_preds = pd.read_sql("SELECT COUNT(*) as c FROM predictions", conn).iloc[0]["c"]
    n_truth = pd.read_sql("SELECT COUNT(*) as c FROM ground_truth", conn).iloc[0]["c"]
    conn.close()

    col1, col2, col3, col4 = st.columns(4)
    for col, label, val, icon in zip(
        [col1,col2,col3,col4],
        ["Teams Generated","Challenger Lineups","Predictions","Results Recorded"],
        [n_teams, n_challengers, n_preds, n_truth],
        ["🏟️","⚔️","🔮","✅"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem">{icon}</div>
                <div class="value">{val}</div>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Prediction metrics
    st.subheader("🎯 Battle Prediction Metrics")
    result = get_prediction_metrics()
    if result is None or result[0] is None:
        st.info("No completed battles yet. Record predictions and results to see metrics.")
    else:
        acc, cm, labels, merged = result

        col1, col2, col3, col4 = st.columns(4)
        y_true = merged["actual_winner"]
        y_pred = merged["predicted_winner"]
        with col1:
            st.metric("Accuracy", f"{acc*100:.1f}%")
        with col2:
            prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            st.metric("Precision", f"{prec*100:.1f}%")
        with col3:
            rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            st.metric("Recall", f"{rec*100:.1f}%")
        with col4:
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            st.metric("F1-Score", f"{f1*100:.1f}%")

        # Confusion matrix
        st.subheader("Confusion Matrix")
        fig = px.imshow(cm, x=labels, y=labels,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        color_continuous_scale="blues",
                        title="Confusion Matrix — Battle Predictions")
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

        # Per-match results
        st.subheader("Match History")
        display = merged[["match_id","gym_leader","challenger","predicted_winner","actual_winner","correct_prediction","final_score","replay_link"]]
        display = display.copy()
        display["correct_prediction"] = display["correct_prediction"].map({1:"✅ Correct", 0:"❌ Wrong"})
        st.dataframe(display.reset_index(drop=True), use_container_width=True)

    # Pokemon data stats
    st.markdown("---")
    st.subheader("📈 Pokémon Data Overview")
    df = load_pokemon()
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="total", color="native_region",
                           title="BST Distribution by Region", nbins=20,
                           color_discrete_map={"Hoenn":"#4caf50","Sinnoh":"#9c27b0","Galar":"#f44336"})
        fig.update_layout(paper_bgcolor="#0d1117", font=dict(color="white"), plot_bgcolor="#0d1117")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        type_dist = df["type_1"].value_counts().reset_index()
        type_dist.columns = ["Type","Count"]
        fig2 = px.bar(type_dist, x="Type", y="Count", title="Type 1 Distribution",
                      color="Count", color_continuous_scale="viridis")
        fig2.update_layout(paper_bgcolor="#0d1117", font=dict(color="white"), plot_bgcolor="#0d1117")
        st.plotly_chart(fig2, use_container_width=True)

    # Audit log
    st.markdown("---")
    st.subheader("🔍 Audit Trail")
    conn = sqlite3.connect(DB_PATH)
    audit = pd.read_sql("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 50", conn)
    conn.close()
    if len(audit) > 0:
        st.dataframe(audit.drop("audit_id", axis=1).reset_index(drop=True), use_container_width=True)
    else:
        st.info("No audit records yet.")

# ─────────────────────────────────────────
#  PAGE 5: DATABASE VIEWER
# ─────────────────────────────────────────
elif page == "🗄️ Database Viewer":
    st.header("🗄️ Database Viewer")

    conn = sqlite3.connect(DB_PATH)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏟️ Team Outputs",
        "⚔️ Challenger Outputs",
        "🔮 Predictions",
        "✅ Ground Truth",
        "🐾 Pokémon Data"
    ])

    with tab1:
        df = pd.read_sql("SELECT * FROM team_outputs ORDER BY timestamp DESC", conn)
        st.dataframe(df, use_container_width=True)
        if len(df) > 0:
            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download Team Outputs", csv, "team_outputs.csv", "text/csv")

    with tab2:
        df = pd.read_sql("SELECT * FROM challenger_outputs ORDER BY timestamp DESC", conn)
        st.dataframe(df, use_container_width=True)
        if len(df) > 0:
            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download Challenger Outputs", csv, "challenger_outputs.csv", "text/csv")

    with tab3:
        df = pd.read_sql("SELECT * FROM predictions ORDER BY timestamp_before DESC", conn)
        st.dataframe(df, use_container_width=True)
        if len(df) > 0:
            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")

    with tab4:
        df = pd.read_sql("SELECT * FROM ground_truth ORDER BY timestamp_after DESC", conn)
        st.dataframe(df, use_container_width=True)
        if len(df) > 0:
            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download Ground Truth", csv, "ground_truth.csv", "text/csv")

    with tab5:
        df_poke = load_pokemon()
        region_filter = st.multiselect("Filter by Region", ALLOWED_REGIONS, default=ALLOWED_REGIONS)
        type_filter = st.multiselect("Filter by Type", sorted(df_poke["type_1"].unique()), default=[])
        filtered = df_poke[df_poke["native_region"].isin(region_filter)]
        if type_filter:
            filtered = filtered[(filtered["type_1"].isin(type_filter)) | (filtered["type_2"].isin(type_filter))]
        st.write(f"Showing {len(filtered)} Pokémon")
        st.dataframe(filtered.reset_index(drop=True), use_container_width=True)
        export_cols = ['name', 'type_1', 'type_2', 'native_region', 'generation', 'hp', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']
        csv = filtered[export_cols].to_csv(index=False)
        st.download_button("⬇️ Download Filtered Data", csv, "pokemon_filtered.csv", "text/csv")

    conn.close()
