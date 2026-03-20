#!/usr/bin/env python3
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from unidecode import unidecode

try:
    from google import genai
except Exception:
    genai = None

DATA_DIR = "data"
TODAY_JSON = os.path.join(DATA_DIR, "today.json")
TODAY_MAIN_JSON = os.path.join(DATA_DIR, "today_main.json")
TOMORROW_JSON = os.path.join(DATA_DIR, "tomorrow.json")
MATCH_MAP_JSON = os.path.join(DATA_DIR, "match_map.json")
HEALTH_JSON = os.path.join(DATA_DIR, "health.json")

FS_KEY = os.getenv("FOOTYSTATS_KEY", "").strip()
SM_KEY = os.getenv("SPORTMONKS_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
REQUEST_TIMEOUT = 25


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clean_name(name: str) -> str:
    n = unidecode(str(name or "")).lower()
    n = re.sub(r"[\W_]+", "", n)
    for token in [
        "footballclub", "futebolclube", "clubdefutbol", "clubdeportivo",
        "women", "ladies", "reserves", "reserve", "ii", "iii", "u21", "u23",
        "fc", "cf", "ac", "afc", "sc", "sk", "if", "fk", "bk", "nk", "cd",
        "de", "la", "the"
    ]:
        n = n.replace(token, "")
    return n


def ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        shorter = min(len(a), len(b))
        longer = max(len(a), len(b))
        return shorter / max(longer, 1)
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / max(len(a), len(b), 1)


def iso_date(offset_days: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=offset_days)).strftime("%Y-%m-%d")


def country_from_image_path(path: str) -> str:
    if not path:
        return ""
    path = path.split("/")[-1]
    if "-" not in path:
        return ""
    country = path.split("-")[0]
    return country.replace("_", " ").strip().title()


def fetch_json(url: str, timeout: int = REQUEST_TIMEOUT) -> Dict[str, Any]:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def init_vertex_client():
    if genai is None:
        return None
    gac = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not gac or not os.path.exists(gac):
        return None
    try:
        with open(gac, "r", encoding="utf-8") as f:
            info = json.load(f)
        return genai.Client(vertexai=True, project=info["project_id"], location="us-central1")
    except Exception as e:
        log(f"⚠️ Gemini istemcisi başlatılamadı: {e}")
        return None


def ai_comment_prematch(client, match: Dict[str, Any]) -> str:
    if client is None:
        return ""

    payload = {
        "home": match.get("home_name"),
        "away": match.get("away_name"),
        "league": match.get("competition_name"),
        "xg_home": safe_float(match.get("team_a_xg_prematch")),
        "xg_away": safe_float(match.get("team_b_xg_prematch")),
        "home_ppg": safe_float(match.get("home_ppg")),
        "away_ppg": safe_float(match.get("away_ppg")),
        "btts": safe_float(match.get("btts_potential")),
        "over25": safe_float(match.get("o25_potential")),
        "first_half_goal": safe_float(match.get("o05HT_potential")),
        "odds_1": safe_float(match.get("odds_ft_1")),
        "odds_x": safe_float(match.get("odds_ft_x")),
        "odds_2": safe_float(match.get("odds_ft_2")),
        "odds_over25": safe_float(match.get("odds_ft_over25")),
        "odds_btts_yes": safe_float(match.get("odds_btts_yes")),
    }

    prompt = "\n".join([
        "Sen kısa ve teknik bahis analiz motorusun.",
        "Kurallar:",
        "- Övgü, hitap, emoji, tiyatro, gereksiz giriş kullanma.",
        "- En fazla 70 kelime yaz.",
        "- Veri zayıfsa tam olarak 'AI yorumu henüz yok.' yaz.",
        "- Sadece Türkçe yaz.",
        "- 3 satır üret:",
        "1) DURUM:",
        "2) NEDEN:",
        "3) SONUÇ:",
        "",
        json.dumps(payload, ensure_ascii=False)
    ])

    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = (getattr(response, "text", "") or "").strip()
        if not text:
            return ""
        if len(text) > 480:
            text = text[:480].rsplit(" ", 1)[0]
        return text
    except Exception as e:
        log(f"⚠️ Prematch AI hatası ({match.get('home_name')} - {match.get('away_name')}): {e}")
        return ""


def extract_sm_basic(sm: Dict[str, Any]) -> Dict[str, Any]:
    league = sm.get("league") or {}
    venue = sm.get("venue") or {}
    referees = sm.get("referees") or []
    weather = sm.get("weatherReport") or {}
    participants = sm.get("participants") or []

    def find_participants():
        home = None
        away = None
        for p in participants:
            loc = (
                p.get("meta", {}).get("location")
                or p.get("location")
                or p.get("participant", {}).get("meta", {}).get("location")
                or ""
            ).lower()
            if loc == "home":
                home = p
            elif loc == "away":
                away = p
        if home is None and participants:
            home = participants[0]
        if away is None and len(participants) > 1:
            away = participants[1]
        return home, away

    home_p, away_p = find_participants()

    return {
        "sportmonks_id": sm.get("id"),
        "competition_name": league.get("name") or sm.get("name") or "",
        "competition_id": league.get("id") or sm.get("league_id") or 0,
        "venue_name": venue.get("name") or "",
        "venue_city": venue.get("city_name") or venue.get("city") or "",
        "referee_name": referees[0].get("name") if referees else "",
        "weather": weather,
        "home_sm_name": (home_p or {}).get("name", ""),
        "away_sm_name": (away_p or {}).get("name", ""),
        "home_sm_logo": (home_p or {}).get("image_path") or (home_p or {}).get("image"),
        "away_sm_logo": (away_p or {}).get("image_path") or (away_p or {}).get("image"),
        "state_name": (sm.get("state") or {}).get("developer_name", ""),
        "starting_at_timestamp": safe_int(sm.get("starting_at_timestamp"), 0),
    }


def build_flat_match(fs: Dict[str, Any], sm: Optional[Dict[str, Any]], old: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    old = old or {}
    sm_info = extract_sm_basic(sm or {})
    match_id = str(fs.get("id", ""))

    home_name = fs.get("home_name") or old.get("home_name") or sm_info.get("home_sm_name") or "Ev Sahibi"
    away_name = fs.get("away_name") or old.get("away_name") or sm_info.get("away_sm_name") or "Deplasman"

    status_old = old.get("status") or "incomplete"
    elapsed_old = old.get("elapsed") or 0
    home_goals_old = safe_int(old.get("homeGoalCount"), 0)
    away_goals_old = safe_int(old.get("awayGoalCount"), 0)

    home_image = fs.get("home_image") or old.get("home_image") or sm_info.get("home_sm_logo") or ""
    away_image = fs.get("away_image") or old.get("away_image") or sm_info.get("away_sm_logo") or ""

    data = {
        "id": match_id,
        "source_ids": {
            "footystats": match_id,
            "sportmonks": str(sm_info.get("sportmonks_id") or old.get("source_ids", {}).get("sportmonks", "")),
        },
        "home_name": home_name,
        "away_name": away_name,
        "competition_name": fs.get("competition_name") or sm_info.get("competition_name") or old.get("competition_name") or "",
        "competition_id": safe_int(fs.get("competition_id") or sm_info.get("competition_id") or old.get("competition_id"), 0),
        "date_unix": safe_int(fs.get("date_unix") or sm_info.get("starting_at_timestamp") or old.get("date_unix"), 0),
        "home_url": fs.get("home_url") or old.get("home_url") or "",
        "away_url": fs.get("away_url") or old.get("away_url") or "",
        "match_url": fs.get("match_url") or old.get("match_url") or "",
        "home_image": home_image,
        "away_image": away_image,
        "home_country": country_from_image_path(home_image),
        "away_country": country_from_image_path(away_image),
        "venue_name": sm_info.get("venue_name") or old.get("venue_name") or fs.get("stadium_name") or "",
        "venue_city": sm_info.get("venue_city") or old.get("venue_city") or fs.get("stadium_location") or "",
        "stadium_name": sm_info.get("venue_name") or old.get("stadium_name") or fs.get("stadium_name") or fs.get("venue_name") or "",
        "stadium_location": sm_info.get("venue_city") or old.get("stadium_location") or fs.get("stadium_location") or fs.get("venue_city") or "",
        "referee_name": sm_info.get("referee_name") or old.get("referee_name") or "",
        "weather": sm_info.get("weather") or old.get("weather") or {},
        # flat compatibility for frontend
        "home_ppg": safe_float(fs.get("home_ppg") or fs.get("team_a_ppg") or old.get("home_ppg") or old.get("team_a_ppg")),
        "away_ppg": safe_float(fs.get("away_ppg") or fs.get("team_b_ppg") or old.get("away_ppg") or old.get("team_b_ppg")),
        "team_a_ppg": fs.get("team_a_ppg") or fs.get("home_ppg") or old.get("team_a_ppg") or old.get("home_ppg") or "0",
        "team_b_ppg": fs.get("team_b_ppg") or fs.get("away_ppg") or old.get("team_b_ppg") or old.get("away_ppg") or "0",
        "pre_match_home_ppg": safe_float(fs.get("pre_match_home_ppg") or old.get("pre_match_home_ppg")),
        "pre_match_away_ppg": safe_float(fs.get("pre_match_away_ppg") or old.get("pre_match_away_ppg")),
        "pre_match_teamA_overall_ppg": safe_float(fs.get("pre_match_teamA_overall_ppg") or old.get("pre_match_teamA_overall_ppg")),
        "pre_match_teamB_overall_ppg": safe_float(fs.get("pre_match_teamB_overall_ppg") or old.get("pre_match_teamB_overall_ppg")),
        "team_a_xg_prematch": safe_float(fs.get("team_a_xg_prematch") or old.get("team_a_xg_prematch")),
        "team_b_xg_prematch": safe_float(fs.get("team_b_xg_prematch") or old.get("team_b_xg_prematch")),
        "total_xg_prematch": safe_float(fs.get("total_xg_prematch") or old.get("total_xg_prematch")),
        "btts_potential": safe_float(fs.get("btts_potential") or old.get("btts_potential")),
        "o25_potential": safe_float(fs.get("o25_potential") or old.get("o25_potential")),
        "o15_potential": safe_float(fs.get("o15_potential") or old.get("o15_potential")),
        "o05_potential": safe_float(fs.get("o05_potential") or old.get("o05_potential")),
        "o05HT_potential": safe_float(fs.get("o05HT_potential") or old.get("o05HT_potential")),
        "o15HT_potential": safe_float(fs.get("o15HT_potential") or old.get("o15HT_potential")),
        "o05_2H_potential": safe_float(fs.get("o05_2H_potential") or old.get("o05_2H_potential")),
        "o15_2H_potential": safe_float(fs.get("o15_2H_potential") or old.get("o15_2H_potential")),
        "corners_potential": safe_float(fs.get("corners_potential") or old.get("corners_potential")),
        "cards_potential": safe_float(fs.get("cards_potential") or old.get("cards_potential")),
        "avg_potential": safe_float(fs.get("avg_potential") or old.get("avg_potential")),
        "u25_potential": safe_float(fs.get("u25_potential") or old.get("u25_potential")),
        "u15_potential": safe_float(fs.get("u15_potential") or old.get("u15_potential")),
        "corners_o85_potential": safe_float(fs.get("corners_o85_potential") or old.get("corners_o85_potential")),
        "corners_o95_potential": safe_float(fs.get("corners_o95_potential") or old.get("corners_o95_potential")),
        "corners_o105_potential": safe_float(fs.get("corners_o105_potential") or old.get("corners_o105_potential")),
        "matches_completed_minimum": safe_int(fs.get("matches_completed_minimum") or old.get("matches_completed_minimum")),
        "home_form": fs.get("home_form") or old.get("home_form") or "",
        "away_form": fs.get("away_form") or old.get("away_form") or "",
        "odds_ft_1": safe_float(fs.get("odds_ft_1") or old.get("odds_ft_1")),
        "odds_ft_x": safe_float(fs.get("odds_ft_x") or old.get("odds_ft_x")),
        "odds_ft_2": safe_float(fs.get("odds_ft_2") or old.get("odds_ft_2")),
        "odds_ft_over15": safe_float(fs.get("odds_ft_over15") or old.get("odds_ft_over15")),
        "odds_ft_over25": safe_float(fs.get("odds_ft_over25") or old.get("odds_ft_over25")),
        "odds_btts_yes": safe_float(fs.get("odds_btts_yes") or old.get("odds_btts_yes")),
        "odds_1st_half_over05": safe_float(fs.get("odds_1st_half_over05") or old.get("odds_1st_half_over05")),
        "odds_doublechance_1x": safe_float(fs.get("odds_doublechance_1x") or old.get("odds_doublechance_1x")),
        "odds_doublechance_x2": safe_float(fs.get("odds_doublechance_x2") or old.get("odds_doublechance_x2")),
        "odds_team_a_cs_yes": safe_float(fs.get("odds_team_a_cs_yes") or old.get("odds_team_a_cs_yes")),
        "odds_team_b_cs_yes": safe_float(fs.get("odds_team_b_cs_yes") or old.get("odds_team_b_cs_yes")),
        "status": status_old,
        "elapsed": elapsed_old,
        "homeGoalCount": home_goals_old,
        "awayGoalCount": away_goals_old,
        "totalGoalCount": home_goals_old + away_goals_old,
        "live_stats_summary": old.get("live_stats_summary", ""),
        "pressure_score": safe_float(old.get("pressure_score")),
        "boss_ai_decision": old.get("boss_ai_decision", ""),
        "ai_comment": old.get("ai_comment", ""),
        "prematch_comment": old.get("prematch_comment", ""),
        "live_comment": old.get("live_comment", ""),
    }

    data["prematch"] = {
        "home_ppg": data["home_ppg"],
        "away_ppg": data["away_ppg"],
        "pre_match_home_ppg": data["pre_match_home_ppg"],
        "pre_match_away_ppg": data["pre_match_away_ppg"],
        "pre_match_teamA_overall_ppg": data["pre_match_teamA_overall_ppg"],
        "pre_match_teamB_overall_ppg": data["pre_match_teamB_overall_ppg"],
        "team_a_xg_prematch": data["team_a_xg_prematch"],
        "team_b_xg_prematch": data["team_b_xg_prematch"],
        "total_xg_prematch": data["total_xg_prematch"],
        "btts_potential": data["btts_potential"],
        "o25_potential": data["o25_potential"],
        "o15_potential": data["o15_potential"],
        "o05_potential": data["o05_potential"],
        "o05HT_potential": data["o05HT_potential"],
        "o15HT_potential": data["o15HT_potential"],
        "o05_2H_potential": data["o05_2H_potential"],
        "o15_2H_potential": data["o15_2H_potential"],
        "corners_potential": data["corners_potential"],
        "cards_potential": data["cards_potential"],
        "avg_potential": data["avg_potential"],
        "u25_potential": data["u25_potential"],
        "u15_potential": data["u15_potential"],
        "corners_o85_potential": data["corners_o85_potential"],
        "corners_o95_potential": data["corners_o95_potential"],
        "corners_o105_potential": data["corners_o105_potential"],
        "matches_completed_minimum": data["matches_completed_minimum"],
        "home_form": data["home_form"],
        "away_form": data["away_form"],
    }
    data["odds"] = {
        "ft_1": data["odds_ft_1"],
        "ft_x": data["odds_ft_x"],
        "ft_2": data["odds_ft_2"],
        "ft_over15": data["odds_ft_over15"],
        "ft_over25": data["odds_ft_over25"],
        "btts_yes": data["odds_btts_yes"],
        "ht_over05": data["odds_1st_half_over05"],
        "doublechance_1x": data["odds_doublechance_1x"],
        "doublechance_x2": data["odds_doublechance_x2"],
        "team_a_cs_yes": data["odds_team_a_cs_yes"],
        "team_b_cs_yes": data["odds_team_b_cs_yes"],
    }
    data["live"] = {
        "status": data["status"],
        "elapsed": data["elapsed"],
        "homeGoalCount": data["homeGoalCount"],
        "awayGoalCount": data["awayGoalCount"],
        "totalGoalCount": data["totalGoalCount"],
        "stats_summary": data["live_stats_summary"],
        "pressure_score": data["pressure_score"],
    }
    data["ai"] = {
        "prematch_comment": data["prematch_comment"],
        "live_comment": data["live_comment"],
        "active_comment": data["boss_ai_decision"],
    }
    return data


def build_sm_tomorrow_match(sm: Dict[str, Any], old: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    old = old or {}
    info = extract_sm_basic(sm)
    home_name = info.get("home_sm_name") or old.get("home_name") or "Ev Sahibi"
    away_name = info.get("away_sm_name") or old.get("away_name") or "Deplasman"
    home_image = info.get("home_sm_logo") or old.get("home_image") or ""
    away_image = info.get("away_sm_logo") or old.get("away_image") or ""
    sportmonks_id = str(info.get("sportmonks_id") or "")

    data = {
        "id": old.get("id") or f"sm-{sportmonks_id}",
        "source_ids": {"footystats": old.get("source_ids", {}).get("footystats", ""), "sportmonks": sportmonks_id},
        "home_name": home_name,
        "away_name": away_name,
        "competition_name": info.get("competition_name") or old.get("competition_name") or "",
        "competition_id": safe_int(info.get("competition_id") or old.get("competition_id"), 0),
        "date_unix": info.get("starting_at_timestamp") or old.get("date_unix") or 0,
        "home_url": old.get("home_url") or "",
        "away_url": old.get("away_url") or "",
        "match_url": old.get("match_url") or "",
        "home_image": home_image,
        "away_image": away_image,
        "home_country": country_from_image_path(home_image),
        "away_country": country_from_image_path(away_image),
        "venue_name": info.get("venue_name") or old.get("venue_name") or "",
        "venue_city": info.get("venue_city") or old.get("venue_city") or "",
        "stadium_name": info.get("venue_name") or old.get("stadium_name") or old.get("venue_name") or "",
        "stadium_location": info.get("venue_city") or old.get("stadium_location") or old.get("venue_city") or "",
        "referee_name": info.get("referee_name") or old.get("referee_name") or "",
        "weather": info.get("weather") or old.get("weather") or {},
        "home_ppg": safe_float(old.get("home_ppg")),
        "away_ppg": safe_float(old.get("away_ppg")),
        "team_a_ppg": old.get("team_a_ppg", "0"),
        "team_b_ppg": old.get("team_b_ppg", "0"),
        "pre_match_home_ppg": safe_float(old.get("pre_match_home_ppg")),
        "pre_match_away_ppg": safe_float(old.get("pre_match_away_ppg")),
        "pre_match_teamA_overall_ppg": safe_float(old.get("pre_match_teamA_overall_ppg")),
        "pre_match_teamB_overall_ppg": safe_float(old.get("pre_match_teamB_overall_ppg")),
        "team_a_xg_prematch": safe_float(old.get("team_a_xg_prematch")),
        "team_b_xg_prematch": safe_float(old.get("team_b_xg_prematch")),
        "total_xg_prematch": safe_float(old.get("total_xg_prematch")),
        "btts_potential": safe_float(old.get("btts_potential")),
        "o25_potential": safe_float(old.get("o25_potential")),
        "o15_potential": safe_float(old.get("o15_potential")),
        "o05_potential": safe_float(old.get("o05_potential")),
        "o05HT_potential": safe_float(old.get("o05HT_potential")),
        "o15HT_potential": safe_float(old.get("o15HT_potential")),
        "o05_2H_potential": safe_float(old.get("o05_2H_potential")),
        "o15_2H_potential": safe_float(old.get("o15_2H_potential")),
        "corners_potential": safe_float(old.get("corners_potential")),
        "cards_potential": safe_float(old.get("cards_potential")),
        "avg_potential": safe_float(old.get("avg_potential")),
        "u25_potential": safe_float(old.get("u25_potential")),
        "u15_potential": safe_float(old.get("u15_potential")),
        "corners_o85_potential": safe_float(old.get("corners_o85_potential")),
        "corners_o95_potential": safe_float(old.get("corners_o95_potential")),
        "corners_o105_potential": safe_float(old.get("corners_o105_potential")),
        "matches_completed_minimum": safe_int(old.get("matches_completed_minimum")),
        "home_form": old.get("home_form", ""),
        "away_form": old.get("away_form", ""),
        "odds_ft_1": safe_float(old.get("odds_ft_1")),
        "odds_ft_x": safe_float(old.get("odds_ft_x")),
        "odds_ft_2": safe_float(old.get("odds_ft_2")),
        "odds_ft_over15": safe_float(old.get("odds_ft_over15")),
        "odds_ft_over25": safe_float(old.get("odds_ft_over25")),
        "odds_btts_yes": safe_float(old.get("odds_btts_yes")),
        "odds_1st_half_over05": safe_float(old.get("odds_1st_half_over05")),
        "odds_doublechance_1x": safe_float(old.get("odds_doublechance_1x")),
        "odds_doublechance_x2": safe_float(old.get("odds_doublechance_x2")),
        "odds_team_a_cs_yes": safe_float(old.get("odds_team_a_cs_yes")),
        "odds_team_b_cs_yes": safe_float(old.get("odds_team_b_cs_yes")),
        "status": "incomplete",
        "elapsed": 0,
        "homeGoalCount": 0,
        "awayGoalCount": 0,
        "totalGoalCount": 0,
        "live_stats_summary": "",
        "pressure_score": 0,
        "boss_ai_decision": old.get("boss_ai_decision", ""),
        "ai_comment": old.get("ai_comment", ""),
        "prematch_comment": old.get("prematch_comment", ""),
        "live_comment": "",
    }

    data["prematch"] = {
        "home_ppg": data["home_ppg"],
        "away_ppg": data["away_ppg"],
        "pre_match_home_ppg": data["pre_match_home_ppg"],
        "pre_match_away_ppg": data["pre_match_away_ppg"],
        "pre_match_teamA_overall_ppg": data["pre_match_teamA_overall_ppg"],
        "pre_match_teamB_overall_ppg": data["pre_match_teamB_overall_ppg"],
        "team_a_xg_prematch": data["team_a_xg_prematch"],
        "team_b_xg_prematch": data["team_b_xg_prematch"],
        "total_xg_prematch": data["total_xg_prematch"],
        "btts_potential": data["btts_potential"],
        "o25_potential": data["o25_potential"],
        "o15_potential": data["o15_potential"],
        "o05_potential": data["o05_potential"],
        "o05HT_potential": data["o05HT_potential"],
        "o15HT_potential": data["o15HT_potential"],
        "o05_2H_potential": data["o05_2H_potential"],
        "o15_2H_potential": data["o15_2H_potential"],
        "corners_potential": data["corners_potential"],
        "cards_potential": data["cards_potential"],
        "avg_potential": data["avg_potential"],
        "u25_potential": data["u25_potential"],
        "u15_potential": data["u15_potential"],
        "corners_o85_potential": data["corners_o85_potential"],
        "corners_o95_potential": data["corners_o95_potential"],
        "corners_o105_potential": data["corners_o105_potential"],
        "matches_completed_minimum": data["matches_completed_minimum"],
        "home_form": data["home_form"],
        "away_form": data["away_form"],
    }
    data["odds"] = {
        "ft_1": data["odds_ft_1"],
        "ft_x": data["odds_ft_x"],
        "ft_2": data["odds_ft_2"],
        "ft_over15": data["odds_ft_over15"],
        "ft_over25": data["odds_ft_over25"],
        "btts_yes": data["odds_btts_yes"],
        "ht_over05": data["odds_1st_half_over05"],
        "doublechance_1x": data["odds_doublechance_1x"],
        "doublechance_x2": data["odds_doublechance_x2"],
        "team_a_cs_yes": data["odds_team_a_cs_yes"],
        "team_b_cs_yes": data["odds_team_b_cs_yes"],
    }
    data["live"] = {
        "status": data["status"],
        "elapsed": data["elapsed"],
        "homeGoalCount": data["homeGoalCount"],
        "awayGoalCount": data["awayGoalCount"],
        "totalGoalCount": data["totalGoalCount"],
        "stats_summary": data["live_stats_summary"],
        "pressure_score": data["pressure_score"],
    }
    data["ai"] = {
        "prematch_comment": data["prematch_comment"],
        "live_comment": data["live_comment"],
        "active_comment": data["boss_ai_decision"],
    }
    return data


def match_sm_to_fs(fs_match: Dict[str, Any], sm_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    h = clean_name(fs_match.get("home_name", ""))
    a = clean_name(fs_match.get("away_name", ""))
    best = None
    best_score = 0.0

    for sm in sm_rows:
        info = extract_sm_basic(sm)
        sh = clean_name(info.get("home_sm_name", ""))
        sa = clean_name(info.get("away_sm_name", ""))
        score = (ratio(h, sh) + ratio(a, sa)) / 2.0
        if score > best_score:
            best = sm
            best_score = score

    return best if best_score >= 0.68 else None


def main() -> None:
    ensure_data_dir()
    started = time.time()
    health = {
        "runner": "fetch",
        "started_at": datetime.utcnow().isoformat() + "Z",
        "footystats_matches": 0,
        "sportmonks_today": 0,
        "sportmonks_tomorrow": 0,
        "matched_today": 0,
        "prematch_ai_written": 0,
        "errors": [],
    }

    if not FS_KEY:
        raise RuntimeError("FOOTYSTATS_KEY eksik")

    existing_today = load_json(TODAY_JSON, {}).get("data", [])
    existing_today_by_id = {str(x.get("id")): x for x in existing_today if x.get("id")}
    existing_tomorrow = load_json(TOMORROW_JSON, {}).get("data", [])
    existing_tomorrow_by_sm = {
        str((x.get("source_ids") or {}).get("sportmonks") or ""): x for x in existing_tomorrow
    }

    fs_url = f"https://api.football-data-api.com/todays-matches?key={FS_KEY}&include=stats,odds"
    fs_json = fetch_json(fs_url)
    fs_rows = fs_json.get("data", [])
    health["footystats_matches"] = len(fs_rows)

    save_json(TODAY_MAIN_JSON, {"data": fs_rows})

    sm_today_rows: List[Dict[str, Any]] = []
    sm_tomorrow_rows: List[Dict[str, Any]] = []

    if SM_KEY:
        try:
            sm_today_url = (
                f"https://api.sportmonks.com/v3/football/fixtures/date/{iso_date(0)}"
                f"?api_token={SM_KEY}&include=participants;league;venue;referees;weatherReport;state;scores"
            )
            sm_today_json = fetch_json(sm_today_url)
            sm_today_rows = sm_today_json.get("data", [])
            health["sportmonks_today"] = len(sm_today_rows)
        except Exception as e:
            health["errors"].append(f"sportmonks_today: {e}")

        try:
            sm_tomorrow_url = (
                f"https://api.sportmonks.com/v3/football/fixtures/date/{iso_date(1)}"
                f"?api_token={SM_KEY}&include=participants;league;venue;referees;weatherReport;state;scores"
            )
            sm_tomorrow_json = fetch_json(sm_tomorrow_url)
            sm_tomorrow_rows = sm_tomorrow_json.get("data", [])
            health["sportmonks_tomorrow"] = len(sm_tomorrow_rows)
        except Exception as e:
            health["errors"].append(f"sportmonks_tomorrow: {e}")

    vertex = init_vertex_client()

    output_today: List[Dict[str, Any]] = []
    match_map = load_json(MATCH_MAP_JSON, {"sportmonks_to_footystats": {}})
    sm_to_fs = match_map.get("sportmonks_to_footystats", {})

    for fs in fs_rows:
        fs_id = str(fs.get("id", ""))
        old = existing_today_by_id.get(fs_id)
        matched_sm = None

        for sm in sm_today_rows:
            if str(sm.get("id")) and sm_to_fs.get(str(sm.get("id"))) == fs_id:
                matched_sm = sm
                break

        if matched_sm is None and sm_today_rows:
            matched_sm = match_sm_to_fs(fs, sm_today_rows)
            if matched_sm is not None and matched_sm.get("id"):
                sm_to_fs[str(matched_sm["id"])] = fs_id

        if matched_sm is not None:
            health["matched_today"] += 1

        row = build_flat_match(fs, matched_sm, old)

        if not row.get("prematch_comment") and row.get("team_a_xg_prematch", 0) > 0:
            comment = ai_comment_prematch(vertex, row)
            if comment:
                row["prematch_comment"] = comment
                row["ai"]["prematch_comment"] = comment
                if not row.get("boss_ai_decision"):
                    row["boss_ai_decision"] = comment
                    row["ai_comment"] = comment
                health["prematch_ai_written"] += 1

        output_today.append(row)

    output_tomorrow: List[Dict[str, Any]] = []
    for sm in sm_tomorrow_rows:
        sm_id = str(sm.get("id") or "")
        old = existing_tomorrow_by_sm.get(sm_id)
        row = build_sm_tomorrow_match(sm, old)
        if not row.get("prematch_comment") and vertex is not None:
            comment = ai_comment_prematch(vertex, row)
            if comment:
                row["prematch_comment"] = comment
                row["boss_ai_decision"] = comment
                row["ai_comment"] = comment
                row["ai"]["prematch_comment"] = comment
                row["ai"]["active_comment"] = comment
        output_tomorrow.append(row)

    save_json(TODAY_JSON, {"data": output_today})
    save_json(TOMORROW_JSON, {"data": output_tomorrow})
    save_json(MATCH_MAP_JSON, {"sportmonks_to_footystats": sm_to_fs})

    health["finished_at"] = datetime.utcnow().isoformat() + "Z"
    health["duration_sec"] = round(time.time() - started, 2)
    health["today_written"] = len(output_today)
    health["tomorrow_written"] = len(output_tomorrow)
    save_json(HEALTH_JSON, health)

    log(f"✅ today.json yazıldı: {len(output_today)}")
    log(f"✅ tomorrow.json yazıldı: {len(output_tomorrow)}")
    log(f"✅ prematch AI yorum sayısı: {health['prematch_ai_written']}")
    log(f"✅ Sportmonks eşleşen maç: {health['matched_today']}")


if __name__ == "__main__":
    main()
