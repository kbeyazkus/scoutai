#!/usr/bin/env python3
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from unidecode import unidecode

try:
    from google import genai
except Exception:
    genai = None

DATA_DIR = "data"
TODAY_JSON = os.path.join(DATA_DIR, "today.json")
LIVE_JSON = os.path.join(DATA_DIR, "live.json")
MATCH_MAP_JSON = os.path.join(DATA_DIR, "match_map.json")
HEALTH_JSON = os.path.join(DATA_DIR, "health.json")

SM_KEY = os.getenv("SPORTMONKS_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
REQUEST_TIMEOUT = 25

LIVE_STATES = {
    "INPLAY_1ST_HALF", "INPLAY_2ND_HALF", "HT", "HALF_TIME",
    "INPLAY_ET", "INPLAY_ET_2", "BREAK", "PEN_LIVE",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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


def fetch_json(url: str) -> Dict[str, Any]:
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


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


def get_side_participants(parts: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    home = None
    away = None
    for p in parts:
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
    if home is None and parts:
        home = parts[0]
    if away is None and len(parts) > 1:
        away = parts[1]
    return home or {}, away or {}


def current_score(scores: List[Dict[str, Any]]) -> Tuple[int, int]:
    home = 0
    away = 0
    for s in scores or []:
        desc = (s.get("description") or "").upper()
        participant = (s.get("score") or {}).get("participant")
        goals = safe_int((s.get("score") or {}).get("goals"), 0)
        if desc == "CURRENT":
            if participant == "home":
                home = goals
            elif participant == "away":
                away = goals
    return home, away


def extract_minute(live_row: Dict[str, Any]) -> int:
    for key in ["minute", "timer", "time"]:
        v = live_row.get(key)
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, dict):
            for sub in ["minute", "minutes"]:
                if sub in v:
                    return safe_int(v[sub], 0)
    return 0


def extract_stats(live_row: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    stats = live_row.get("statistics") or []
    out = {"home": {}, "away": {}}

    for st in stats:
        raw_name = (
            st.get("type", {}).get("developer_name")
            or st.get("type", {}).get("name")
            or st.get("name")
            or ""
        )
        key = clean_name(raw_name)
        if not key:
            continue

        side = (
            st.get("participant", {}).get("meta", {}).get("location")
            or st.get("location")
            or ""
        ).lower()
        if side not in ("home", "away"):
            participant = st.get("participant") or {}
            side = (participant.get("location") or "").lower()
        if side not in ("home", "away"):
            continue

        value = st.get("data")
        if isinstance(value, dict):
            value = value.get("value") or value.get("count") or value.get("all") or 0
        val = safe_float(value, 0)
        out[side][key] = val

    return out


def pressure_score(stats: Dict[str, Dict[str, float]]) -> float:
    h = stats.get("home", {})
    a = stats.get("away", {})

    def stat(d: Dict[str, float], *keys: str) -> float:
        for k in keys:
            if k in d:
                return safe_float(d[k], 0)
        return 0.0

    home_pressure = (
        stat(h, "shotsontarget", "shotontarget") * 2.5
        + stat(h, "shots", "totalshots") * 1.0
        + stat(h, "corners", "cornerkicks") * 1.1
        + stat(h, "dangerousattacks", "dangerousattack") * 0.08
        + stat(h, "attacks", "attacksrecorded") * 0.03
        + stat(h, "possessionpercentage", "ballpossession", "possession") * 0.04
    )
    away_pressure = (
        stat(a, "shotsontarget", "shotontarget") * 2.5
        + stat(a, "shots", "totalshots") * 1.0
        + stat(a, "corners", "cornerkicks") * 1.1
        + stat(a, "dangerousattacks", "dangerousattack") * 0.08
        + stat(a, "attacks", "attacksrecorded") * 0.03
        + stat(a, "possessionpercentage", "ballpossession", "possession") * 0.04
    )
    return round(home_pressure - away_pressure, 2)


def summarize_stats(stats: Dict[str, Dict[str, float]]) -> str:
    h = stats.get("home", {})
    a = stats.get("away", {})

    def stat(d: Dict[str, float], *keys: str) -> float:
        for k in keys:
            if k in d:
                return d[k]
        return 0.0

    return (
        f"Şut isabet {stat(h,'shotsontarget','shotontarget')}-{stat(a,'shotsontarget','shotontarget')}, "
        f"Şut {stat(h,'shots','totalshots')}-{stat(a,'shots','totalshots')}, "
        f"Korner {stat(h,'corners','cornerkicks')}-{stat(a,'corners','cornerkicks')}, "
        f"Topla oynama {stat(h,'possessionpercentage','ballpossession','possession')}-{stat(a,'possessionpercentage','ballpossession','possession')}"
    )


def ai_comment_live(client, match: Dict[str, Any]) -> str:
    if client is None:
        return ""

    minute = safe_int(match.get("elapsed"), 0)
    score_h = safe_int(match.get("homeGoalCount"), 0)
    score_a = safe_int(match.get("awayGoalCount"), 0)
    pressure = safe_float(match.get("pressure_score"), 0)
    payload = {
        "home": match.get("home_name"),
        "away": match.get("away_name"),
        "league": match.get("competition_name"),
        "minute": minute,
        "score_home": score_h,
        "score_away": score_a,
        "pressure_score": pressure,
        "live_stats_summary": match.get("live_stats_summary", ""),
        "prematch_xg_home": safe_float(match.get("team_a_xg_prematch")),
        "prematch_xg_away": safe_float(match.get("team_b_xg_prematch")),
        "home_ppg": safe_float(match.get("home_ppg") or match.get("team_a_ppg")),
        "away_ppg": safe_float(match.get("away_ppg") or match.get("team_b_ppg")),
        "btts": safe_float(match.get("btts_potential")),
        "over25": safe_float(match.get("o25_potential")),
        "odds_1": safe_float(match.get("odds_ft_1")),
        "odds_x": safe_float(match.get("odds_ft_x")),
        "odds_2": safe_float(match.get("odds_ft_2")),
    }

    prompt = "\n".join([
        "Sen kısa ve teknik canlı bahis analiz motorusun.",
        "Kurallar:",
        "- En fazla 85 kelime yaz.",
        "- Övgü, hitap, komutan, boss, tiyatro, emoji kullanma.",
        "- 3 satır yaz:",
        "1) DURUM:",
        "2) NEDEN:",
        "3) SONUÇ:",
        "- Veri yetersizse tam olarak 'AI yorumu henüz yok.' yaz.",
        "- Dakika 65+ ise canlı baskıyı prematch veriden daha önemli say.",
        "",
        json.dumps(payload, ensure_ascii=False)
    ])

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        text = (getattr(response, "text", "") or "").strip()
        if not text:
            return ""
        if len(text) > 520:
            text = text[:520].rsplit(" ", 1)[0]
        return text
    except Exception as e:
        log(f"⚠️ Live AI hatası ({match.get('home_name')} - {match.get('away_name')}): {e}")
        return ""


def find_match(today_rows: List[Dict[str, Any]], fixture: Dict[str, Any], match_map: Dict[str, str]) -> Optional[Dict[str, Any]]:
    fixture_id = str(fixture.get("id") or "")
    if fixture_id and fixture_id in match_map:
        fs_id = match_map[fixture_id]
        for row in today_rows:
            if str(row.get("id")) == fs_id:
                return row

    parts = fixture.get("participants") or []
    home_p, away_p = get_side_participants(parts)
    h = clean_name(home_p.get("name", ""))
    a = clean_name(away_p.get("name", ""))
    best = None
    best_score = 0.0
    for row in today_rows:
        rh = clean_name(row.get("home_name", ""))
        ra = clean_name(row.get("away_name", ""))
        score = (ratio(h, rh) + ratio(a, ra)) / 2.0
        if score > best_score:
            best = row
            best_score = score
    if best is not None and best_score >= 0.68:
        if fixture_id:
            match_map[fixture_id] = str(best.get("id"))
        return best
    return None


def fetch_live_rows() -> List[Dict[str, Any]]:
    include_sets = [
        "participants;scores;state;statistics;events",
        "participants;scores;state;statistics",
        "participants;scores;state",
    ]

    last_err = None
    for include in include_sets:
        try:
            url = (
                "https://api.sportmonks.com/v3/football/livescores"
                f"?api_token={SM_KEY}&include={include}"
            )
            data = fetch_json(url).get("data", [])
            log(f"✅ Sportmonks livescores OK | include={include} | rows={len(data)}")
            return data
        except Exception as e:
            log(f"⚠️ Include başarısız: {include} | {e}")
            last_err = e

    raise RuntimeError(f"Sportmonks livescores alınamadı: {last_err}")


def main() -> None:
    if not SM_KEY:
        raise RuntimeError("SPORTMONKS_KEY eksik")

    today_data = load_json(TODAY_JSON, {}).get("data", [])
    match_map_full = load_json(MATCH_MAP_JSON, {"sportmonks_to_footystats": {}})
    match_map = match_map_full.get("sportmonks_to_footystats", {})
    health = load_json(HEALTH_JSON, {})
    health.update({
        "live_runner": "live_radar",
        "live_started_at": datetime.utcnow().isoformat() + "Z",
        "live_fixtures_seen": 0,
        "live_fixtures_matched": 0,
        "live_ai_written": 0,
        "live_errors": [],
    })

    client = init_vertex_client()

    sm_rows = fetch_live_rows()
    health["live_fixtures_seen"] = len(sm_rows)

    live_json_rows = []

    for fixture in sm_rows:
        state_name = (fixture.get("state") or {}).get("developer_name", "")
        if state_name and state_name not in LIVE_STATES and "INPLAY" not in state_name and state_name != "HT":
            continue

        parts = fixture.get("participants") or []
        if len(parts) < 2:
            continue

        row = find_match(today_data, fixture, match_map)
        if row is None:
            continue

        health["live_fixtures_matched"] += 1

        home_score, away_score = current_score(fixture.get("scores") or [])
        minute = extract_minute(fixture)
        stats = extract_stats(fixture)
        summary = summarize_stats(stats)
        pscore = pressure_score(stats)

        row["source_ids"] = row.get("source_ids") or {}
        row["source_ids"]["sportmonks"] = str(fixture.get("id") or "")
        row["status"] = "live" if state_name != "HT" else "paused"
        row["elapsed"] = "HY" if state_name == "HT" else minute
        row["homeGoalCount"] = home_score
        row["awayGoalCount"] = away_score
        row["totalGoalCount"] = home_score + away_score
        row["live_stats_summary"] = summary
        row["pressure_score"] = pscore
        row["live"] = row.get("live") or {}
        row["live"].update({
            "status": row["status"],
            "elapsed": row["elapsed"],
            "homeGoalCount": home_score,
            "awayGoalCount": away_score,
            "totalGoalCount": home_score + away_score,
            "stats_summary": summary,
            "pressure_score": pscore,
            "events": fixture.get("events") or [],
            "inplayOdds": {},
        })

        live_comment = ai_comment_live(client, row)
        if live_comment:
            row["live_comment"] = live_comment
            row["boss_ai_decision"] = live_comment
            row["ai_comment"] = live_comment
            row["ai"] = row.get("ai") or {}
            row["ai"]["live_comment"] = live_comment
            row["ai"]["active_comment"] = live_comment
            health["live_ai_written"] += 1

        live_json_rows.append({
            "id": fixture.get("id"),
            "minute": minute,
            "state": state_name,
            "homeTeam": {"name": row.get("home_name")},
            "awayTeam": {"name": row.get("away_name")},
            "score": {"fullTime": {"home": home_score, "away": away_score}},
        })

    save_json(TODAY_JSON, {"data": today_data})
    save_json(MATCH_MAP_JSON, {"sportmonks_to_footystats": match_map})
    save_json(LIVE_JSON, {"matches": live_json_rows, "resultSet": {"count": len(live_json_rows)}})

    health["live_finished_at"] = datetime.utcnow().isoformat() + "Z"
    save_json(HEALTH_JSON, health)

    log(f"✅ Live eşleşen maç: {health['live_fixtures_matched']}")
    log(f"✅ Live AI yorum: {health['live_ai_written']}")
    log(f"✅ live.json maç sayısı: {len(live_json_rows)}")


if __name__ == "__main__":
    main()
