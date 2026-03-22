#!/usr/bin/env python3
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from unidecode import unidecode

try:
    from google import genai
except Exception:
    genai = None

DATA_DIR = 'data'
FOOTYSTATS_TODAY_JSON = os.path.join(DATA_DIR, 'footystats_today.json')
FOOTYSTATS_LIVE_JSON = os.path.join(DATA_DIR, 'footystats_live.json')
SPORTMONKS_LIVE_JSON = os.path.join(DATA_DIR, 'sportmonks_live.json')
BUNDLE_JSON = os.path.join(DATA_DIR, 'sportmonks_bundle.json')
HEALTH_JSON = os.path.join(DATA_DIR, 'health.json')

SM_KEY = os.getenv('SPORTMONKS_KEY', '').strip()
GEMINI_MODEL = (os.getenv('GEMINI_MODEL') or 'gemini-2.0-flash').strip()
REQUEST_TIMEOUT = 25
LIVE_INCLUDE = 'participants;scores;periods;events;league.country;round;state'
DETAIL_INCLUDE = 'participants;league.country;venue;state;scores;periods;events.type;events.period;events.player;statistics.type;lineups.player;lineups.type;lineups.details.type;metadata.type;coaches;sidelined.sideline.player;sidelined.sideline.type;weatherReport'
# FIX 6: Raised from 120 to 300 sec — prevents rate limit with 10+ live matches
DETAIL_TTL_SEC = 300

def log(msg: str):
    print(msg, flush=True)

def load_json(path: str, default: Any):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, data: Any):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def safe_int(v: Any, default: int = 0) -> int:
    try:
        if v in (None, ''): return default
        return int(float(v))
    except Exception: return default

def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v in (None, ''): return default
        return float(v)
    except Exception: return default

def fetch_json(url: str) -> Dict[str, Any]:
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"API Hatasi: {e}")
        return {}

def init_vertex_client():
    if genai is None:
        return None
    gac = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '').strip()
    if gac and os.path.exists(gac):
        try:
            with open(gac, 'r', encoding='utf-8') as f:
                info = json.load(f)
            project = info.get('project_id')
            if project:
                return genai.Client(vertexai=True, project=project, location='us-central1')
        except Exception as e:
            log(f'Credentials read error: {e}')
    project = os.getenv('GCP_PROJECT_ID', '').strip()
    location = os.getenv('GCP_LOCATION', 'us-central1').strip()
    if not project:
        return None
    try:
        return genai.Client(vertexai=True, project=project, location=location)
    except Exception as e:
        log(f'Vertex AI init error: {e}')
        return None

def clean_name(name: str) -> str:
    n = unidecode(str(name or '')).lower()
    n = re.sub(r'[\W_]+', '', n)
    for token in ['footballclub', 'futebolclube', 'clubdefutbol', 'clubdeportivo', 'women', 'ladies', 'reserves', 'reserve', 'ii', 'iii', 'u21', 'u23', 'fc', 'cf', 'ac', 'afc', 'sc', 'sk', 'if', 'fk', 'bk', 'nk', 'cd', 'de', 'la', 'the']:
        n = n.replace(token, '')
    return n

def ratio(a: str, b: str) -> float:
    if not a or not b: return 0.0
    if a == b: return 1.0
    if a in b or b in a:
        shorter = min(len(a), len(b))
        longer = max(len(a), len(b))
        return shorter / max(longer, 1)
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / max(len(a), len(b), 1)

def get_side_participants(parts: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    home, away = {}, {}
    for p in parts:
        loc = str((p.get('meta') or {}).get('location') or '').lower()
        if loc == 'home': home = p
        elif loc == 'away': away = p
    if not home and parts: home = parts[0]
    if not away and len(parts) > 1: away = parts[1]
    return home, away

def current_score(fixture: Dict[str, Any]) -> Tuple[int, int]:
    scores = fixture.get('scores') or []
    home, away = None, None
    for s in scores:
        if str(s.get('description') or '').upper() == 'CURRENT':
            participant = (s.get('score') or {}).get('participant')
            goals = safe_int((s.get('score') or {}).get('goals'), 0)
            if participant == 'home': home = goals
            elif participant == 'away': away = goals
    if home is not None and away is not None:
        return home, away
    latest = None
    latest_ord = -1
    for ev in fixture.get('events') or []:
        result = ev.get('result')
        total = safe_int(ev.get('minute'), 0) * 100 + safe_int(ev.get('extra_minute'), 0)
        if result and total >= latest_ord:
            latest_ord = total
            latest = result
    if latest:
        m = re.match(r'^\s*(\d+)\s*-\s*(\d+)\s*$', str(latest))
        if m: return int(m.group(1)), int(m.group(2))
    return 0, 0

def extract_minute(fixture: Dict[str, Any]) -> int:
    state = fixture.get('state') or {}
    minute = safe_int(state.get('minute'), 0)
    if minute: return minute
    best = 0
    for p in fixture.get('periods') or []:
        best = max(best, safe_int(p.get('minutes'), 0), safe_int(p.get('minute'), 0))
    for ev in fixture.get('events') or []:
        total = safe_int(ev.get('minute'), 0) + (1 if safe_int(ev.get('extra_minute'), 0) > 0 else 0)
        best = max(best, total)
    return best

def summarize_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    out = {'over25': 0.0, 'btts': 0.0}
    for p in predictions or []:
        dev = str((p.get('type') or {}).get('developer_name') or '').upper()
        vals = p.get('predictions') or {}
        if dev == 'OVER_UNDER_2_5_PROBABILITY':
            out['over25'] = safe_float(vals.get('yes'))
        elif dev == 'BTTS_PROBABILITY':
            out['btts'] = safe_float(vals.get('yes'))
    return out

def heur_live_comment(row: Dict[str, Any], detail: Dict[str, Any]) -> str:
    h = safe_int(row.get('homeGoalCount'), 0)
    a = safe_int(row.get('awayGoalCount'), 0)
    minute = safe_int(row.get('elapsed'), 0)
    pressure = safe_float(row.get('pressure_score'), 0)
    preds = summarize_predictions(detail.get('predictions') or [])
    shots_home = shots_away = 0
    for st in detail.get('statistics') or []:
        dev = str((st.get('type') or {}).get('developer_name') or '').upper()
        if dev == 'SHOTS_ON_TARGET':
            if st.get('location') == 'home': shots_home = safe_int((st.get('data') or {}).get('value'))
            elif st.get('location') == 'away': shots_away = safe_int((st.get('data') or {}).get('value'))
            
    if minute <= 0 and not pressure and not shots_home and not shots_away:
        return f"STATUS: Match is live, score {h}-{a}.\nREASON: Limited live data available.\nCONCLUSION: No Bet / Skip for now."
    if h == a:
        if pressure >= 8 or shots_home >= shots_away + 2:
            return "STATUS: Score level but home team applying pressure.\nREASON: Pressure gap and shot advantage on home side.\nCONCLUSION: Home goal or Over side worth monitoring."
        if pressure <= -8 or shots_away >= shots_home + 2:
            return "STATUS: Score level but away team applying pressure.\nREASON: Pressure gap and shot advantage on away side.\nCONCLUSION: Away goal or Over side worth monitoring."
        if preds.get('over25') >= 60 or preds.get('btts') >= 60:
            return "STATUS: Score level.\nREASON: Model maintains high-scoring match tendency.\nCONCLUSION: Over 2.5 or Both Teams Score worth monitoring."
        return "STATUS: Match evenly contested.\nREASON: No clear pressure dominance established.\nCONCLUSION: No Bet / Skip for now."
        
    leader = row.get('home_name') if h > a else row.get('away_name')
    trailer = row.get('away_name') if h > a else row.get('home_name')
    if abs(pressure) >= 8:
        side = 'home' if pressure > 0 else 'away'
        return f"STATUS: {leader} leads but pressure on {side} side.\nREASON: Tempo and pressure data indicate match still open.\nCONCLUSION: Additional goal probability warrants live monitoring."
    return f"STATUS: {leader} holds score advantage.\nREASON: Match at minute {minute}, {trailer} produced no clear response.\nCONCLUSION: Leading team scenario maintained at this stage."

def ai_comment_live(client, row: Dict[str, Any], detail: Dict[str, Any]) -> str:
    heuristic = heur_live_comment(row, detail)
    if client is None:
        return heuristic

    minute = safe_int(row.get('elapsed'), 0)

    # FIX 2: Only call AI for matches 65+ minutes — saves API quota
    if minute < 65:
        return heuristic

    payload = {
        'home': row.get('home_name'),
        'away': row.get('away_name'),
        'minute': minute,
        'score_home': safe_int(row.get('homeGoalCount'), 0),
        'score_away': safe_int(row.get('awayGoalCount'), 0),
        'pressure': safe_float(row.get('pressure_score'), 0),
        'shots': [s for s in detail.get('statistics', []) if (s.get('type') or {}).get('developer_name') in ('SHOTS_TOTAL', 'SHOTS_ON_TARGET')],
        'prediction': detail.get('predictions', [])[:4],
    }
    prompt = '\n'.join([
        'You are a technical live betting analysis engine that cross-checks data.',
        'Cross-check all metrics (pressure, shots, xG, PPG). If data conflicts flag as RISKY.',
        'Recommend only ONE lowest-risk bet from: Home Win, Away Win, Over 2.5, Both Teams Score, First Half Over 0.5, Asian Handicap, Home Team +1.5 Over Goals, Away Team +1.5 Over Goals.',
        'If confidence is low output exactly: "No Bet / Skip".',
        'Rules: Max 80 words. 3 sections only: STATUS, REASON, CONCLUSION. Be sharp and technical.',
        '- Live match 65+ min: weight live pressure over prematch stats.',
        '- Never use emojis, flattery or drama.',
        json.dumps(payload, ensure_ascii=False)
    ])

    # FIX 1: Retry with exponential backoff on 429
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            text = (getattr(response, 'text', '') or '').strip()
            return text[:520].rsplit(' ', 1)[0] if len(text) > 520 else (text or heuristic)
        except Exception as e:
            err = str(e)
            if '429' in err or 'RESOURCE_EXHAUSTED' in err:
                wait = 10 * (2 ** attempt)  # 10s, 20s, 40s
                log(f'⚠️  429 rate limit — waiting {wait}s (attempt {attempt+1}/{max_retries})')
                time.sleep(wait)
            else:
                log(f'Live AI error ({row.get("home_name")} - {row.get("away_name")}): {e}')
                return heuristic
    log(f'⚠️  AI skipped after {max_retries} retries — using heuristic')
    return heuristic

def fetch_live_rows() -> List[Dict[str, Any]]:
    url = f'https://api.sportmonks.com/v3/football/livescores/inplay?api_token={SM_KEY}&include={LIVE_INCLUDE}'
    return fetch_json(url).get('data', []) or []

def fetch_fixture_detail(fid: int) -> Dict[str, Any]:
    if not fid: return {}
    url = f'https://api.sportmonks.com/v3/football/fixtures/{fid}?api_token={SM_KEY}&include={DETAIL_INCLUDE}'
    return fetch_json(url).get('data', {}) or {}

def get_cached_detail(bundle: Dict[str, Any], fid: str) -> Dict[str, Any]:
    fixture_cache = (bundle.get('fixtures') or {}).get(fid) or {}
    fetched_at = fixture_cache.get('fetched_at') or ''
    if fetched_at:
        try:
            ts = datetime.fromisoformat(fetched_at.replace('Z', ''))
            if (datetime.utcnow() - ts).total_seconds() < DETAIL_TTL_SEC:
                return fixture_cache.get('detail') or {}
        except Exception: pass
    return {}

def build_sm_row_from_live(fixture: Dict[str, Any]) -> Dict[str, Any]:
    parts = fixture.get('participants') or []
    home, away = get_side_participants(parts)
    league = fixture.get('league') or {}
    h, a = current_score(fixture)
    minute = extract_minute(fixture)
    return {
        'id': f"sm-{fixture.get('id')}",
        'source_ids': {'footystats': '', 'sportmonks': str(fixture.get('id') or '')},
        'home_name': home.get('name') or 'Home',
        'away_name': away.get('name') or 'Away',
        'competition_name': league.get('name') or fixture.get('name') or 'Live Match',
        'competition_id': safe_int(league.get('id'), 0),
        'league_country': (league.get('country') or {}).get('name', ''),
        'league_country_image': (league.get('country') or {}).get('image_path', ''),
        'date_unix': safe_int(fixture.get('starting_at_timestamp'), 0),
        'home_image': home.get('image_path', ''),
        'away_image': away.get('image_path', ''),
        # FIX 8: derive country from image path so flag emojis work in frontend
        'home_country': (league.get('country') or {}).get('name', '') or home.get('image_path', '').split('/')[-1].split('-')[0].replace('_',' ').strip().title(),
        'away_country': (league.get('country') or {}).get('name', '') or away.get('image_path', '').split('/')[-1].split('-')[0].replace('_',' ').strip().title(),
        'stadium_name': '',
        'stadium_location': '',
        'referee_name': '',
        'weather': {},
        'home_ppg': 0.0,
        'away_ppg': 0.0,
        'team_a_ppg': '0',
        'team_b_ppg': '0',
        'pre_match_home_ppg': 0.0,
        'pre_match_away_ppg': 0.0,
        'team_a_xg_prematch': 0.0,
        'team_b_xg_prematch': 0.0,
        'btts_potential': 0.0,
        'o25_potential': 0.0,
        'o05HT_potential': 0.0,
        'pressure_score': 0.0,
        'odds_ft_1': 0.0,
        'odds_ft_x': 0.0,
        'odds_ft_2': 0.0,
        'odds_ft_over25': 0.0,
        'odds_btts_yes': 0.0,
        'odds_1st_half_over05': 0.0,
        'homeGoalCount': h,
        'awayGoalCount': a,
        'status': 'live',
        'elapsed': minute,
        'boss_ai_decision': '',
        'prematch_comment': '',
        'live_comment': '',
    }

def main():
    if not SM_KEY: return
    client = init_vertex_client()
    bundle = load_json(BUNDLE_JSON, {'fixtures': {}})
    health = load_json(HEALTH_JSON, {})

    live_rows = fetch_live_rows()
    sm_live_out = []

    health.update({
        'live_runner': 'live_radar_optimized',
        'live_started_at': datetime.utcnow().isoformat() + 'Z',
        'live_fixtures_seen': len(live_rows),
        'live_ai_written': 0,
        'live_errors': [],
    })

    for fixture in live_rows:
        fid = str(fixture.get('id') or '')
        row = build_sm_row_from_live(fixture)

        detail = get_cached_detail(bundle, fid)
        if not detail:
            detail = fetch_fixture_detail(safe_int(fid))
            if detail:
                bundle.setdefault('fixtures', {})[fid] = {
                    'detail': detail,
                    'fetched_at': datetime.utcnow().isoformat() + 'Z',
                }

        if detail:
            state = detail.get('state') or {}
            row['elapsed'] = safe_int(state.get('minute'), row.get('elapsed', 0))
            vals = {'home': 0.0, 'away': 0.0}
            for p in (detail.get('pressure') or [])[-12:]:
                loc = str((p.get('participant') or {}).get('meta', {}).get('location') or p.get('location') or '').lower()
                if loc == 'home': vals['home'] += safe_float(p.get('value') or p.get('amount'))
                elif loc == 'away': vals['away'] += safe_float(p.get('value') or p.get('amount'))
            row['pressure_score'] = round(vals['home'] - vals['away'], 2)
            row['lineups'] = detail.get('lineups') or []
            row['events'] = detail.get('events') or []
            row['statistics'] = detail.get('statistics') or []

        if row.get('elapsed', 0) > 0:
            live_comment = ai_comment_live(client, row, detail or {})
        else:
            live_comment = heur_live_comment(row, detail or {})

        if live_comment:
            row['live_comment'] = live_comment
            row['boss_ai_decision'] = live_comment
            health['live_ai_written'] += 1

        sm_live_out.append(row)

    fs_today = load_json(FOOTYSTATS_TODAY_JSON, {}).get('data', [])
    fs_live = [x for x in fs_today if str(x.get('status', '')).lower() in ('live', 'inplay', 'ht') or safe_int(x.get('elapsed'), 0) > 0]

    save_json(FOOTYSTATS_LIVE_JSON, {'data': fs_live, 'updated_at': datetime.utcnow().isoformat() + 'Z'})
    save_json(SPORTMONKS_LIVE_JSON, {'data': sm_live_out, 'updated_at': datetime.utcnow().isoformat() + 'Z'})
    save_json(BUNDLE_JSON, bundle)
    health['live_finished_at'] = datetime.utcnow().isoformat() + 'Z'
    save_json(HEALTH_JSON, health)

if __name__ == '__main__':
    main()
