#!/usr/bin/env python3
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import requests
from unidecode import unidecode

try:
    from google import genai
except Exception:
    genai = None

DATA_DIR = 'data'
FOOTYSTATS_TODAY_JSON = os.path.join(DATA_DIR, 'footystats_today.json')
FOOTYSTATS_TOMORROW_JSON = os.path.join(DATA_DIR, 'footystats_tomorrow.json')
FOOTYSTATS_LIVE_JSON = os.path.join(DATA_DIR, 'footystats_live.json')
SPORTMONKS_TODAY_JSON = os.path.join(DATA_DIR, 'sportmonks_today.json')
SPORTMONKS_TOMORROW_JSON = os.path.join(DATA_DIR, 'sportmonks_tomorrow.json')
SPORTMONKS_LIVE_JSON = os.path.join(DATA_DIR, 'sportmonks_live.json')
SOURCE_TABS_JSON = os.path.join(DATA_DIR, 'source_tabs.json')
HEALTH_JSON = os.path.join(DATA_DIR, 'health.json')

FS_KEY = os.getenv('FOOTYSTATS_KEY', '').strip()
SM_KEY = os.getenv('SPORTMONKS_KEY', '').strip()
GEMINI_MODEL = (os.getenv('GEMINI_MODEL') or 'gemini-3-flash').strip()
REQUEST_TIMEOUT = 25
SM_BASE_INCLUDE = 'participants;league.country;venue;referees;weatherReport;state;scores;periods;round;predictions'

def log(msg: str): print(msg, flush=True)

def ensure_dir(): os.makedirs(DATA_DIR, exist_ok=True)

def safe_float(v: Any, default: float = 0.0) -> float:
    try: return float(v) if v not in (None, '') else default
    except Exception: return default

def safe_int(v: Any, default: int = 0) -> int:
    try: return int(float(v)) if v not in (None, '') else default
    except Exception: return default

def load_json(path: str, default: Any):
    try:
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception: return default

def save_json(path: str, data: Any):
    with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False)

def fetch_json(url: str) -> Dict[str, Any]:
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f'⚠️ API Hatası: {e}')
        return {}

def fetch_all_pages(url: str) -> list:
    all_data, page = [], 1
    while True:
        try:
            r = requests.get(f"{url}&page={page}", timeout=REQUEST_TIMEOUT)
            res = r.json()
            all_data.extend(res.get('data', []))
            if not res.get('pagination', {}).get('has_more'): break
            page += 1
        except Exception as e:
            log(f'⚠️ Sayfalama Hatası: {e}')
            break
    return all_data

def init_vertex_client():
    if genai is None: return None
    project = os.getenv('GCP_PROJECT_ID')
    location = os.getenv('GCP_LOCATION', 'us-central1')
    if not project: return None
    try:
        return genai.Client(vertexai=True, project=project, location=location)
    except Exception as e:
        log(f'⚠️ Gemini başlatılamadı: {e}')
        return None

def iso_date(offset: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=offset)).strftime('%Y-%m-%d')

def country_from_image_path(path: str) -> str:
    if not path: return ''
    last = path.split('/')[-1]
    if '-' not in last: return ''
    return last.split('-')[0].replace('_', ' ').strip().title()

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
    home, away = 0, 0
    for s in scores:
        if str(s.get('description') or '').upper() != 'CURRENT': continue
        participant = (s.get('score') or {}).get('participant')
        goals = safe_int((s.get('score') or {}).get('goals'), 0)
        if participant == 'home': home = goals
        elif participant == 'away': away = goals
    return home, away

def extract_sm_basic(sm: Dict[str, Any]) -> Dict[str, Any]:
    parts = sm.get('participants') or []
    home, away = get_side_participants(parts)
    league = sm.get('league') or {}
    venue = sm.get('venue') or {}
    refs = sm.get('referees') or []
    ref = refs[0] if refs else {}
    weather = sm.get('weatherreport') or sm.get('weatherReport') or {}
    state = sm.get('state') or {}
    h, a = current_score(sm)
    return {
        'sportmonks_id': safe_int(sm.get('id'), 0),
        'home_name': home.get('name', ''),
        'away_name': away.get('name', ''),
        'home_image': home.get('image_path', ''),
        'away_image': away.get('image_path', ''),
        'competition_name': league.get('name', '') or sm.get('name', ''),
        'competition_id': safe_int(league.get('id'), 0),
        'league_country': (league.get('country') or {}).get('name', ''),
        'league_country_image': (league.get('country') or {}).get('image_path', ''),
        'date_unix': safe_int(sm.get('starting_at_timestamp'), 0),
        'stadium_name': venue.get('name', ''),
        'stadium_location': venue.get('city_name', ''),
        'referee_name': ref.get('name') or ref.get('common_name') or '',
        'weather': weather,
        'status': state.get('state') or 'incomplete',
        'elapsed': safe_int(state.get('minute'), 0),
        'homeGoalCount': h,
        'awayGoalCount': a,
    }

def ai_comment_prematch(client, match: Dict[str, Any]) -> str:
    if client is None: return "AI yorumu henüz yok."
    payload = {
        'home': match.get('home_name'),
        'away': match.get('away_name'),
        'league': match.get('competition_name'),
        'xg_home': safe_float(match.get('team_a_xg_prematch')),
        'xg_away': safe_float(match.get('team_b_xg_prematch')),
        'home_ppg': safe_float(match.get('home_ppg')),
        'away_ppg': safe_float(match.get('away_ppg')),
        'btts': safe_float(match.get('btts_potential')),
        'over25': safe_float(match.get('o25_potential')),
        'odds': {'1': safe_float(match.get('odds_ft_1')), 'x': safe_float(match.get('odds_ft_x')), '2': safe_float(match.get('odds_ft_2'))}
    }
    prompt = '\n'.join([
        'Sen çapraz veri sorgulayan teknik bahis motorusun.',
        'Metrikleri (xG, PPG, Oranlar) çarpıştır. Zıtlık varsa riskli kabul et ve ele.',
        'Sadece şu türlerden en risksiz TEK kuponu öner veya riskliyse "Oynama / Pas" de: MS1/MS2, Üst 2.5, KG Var, İY Üst 0.5, Asya Handikap, Takım 1.5 Üst.',
        'Kurallar: Max 80 kelime. 3 bölüm yaz: DURUM, NEDEN, SONUÇ. Sert ve teknik ol.',
        json.dumps(payload, ensure_ascii=False)
    ])
    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (getattr(response, 'text', '') or '').strip()
    except Exception as e:
        log(f'⚠️ AI hatası: {e}')
        return "AI yorumu henüz yok."

def normalize_fs_row(fs: Dict[str, Any], client=None) -> Dict[str, Any]:
    row = {
        'id': str(fs.get('id', '')),
        'source_ids': {'footystats': str(fs.get('id', '')), 'sportmonks': ''},
        'home_name': fs.get('home_name') or 'Ev Sahibi',
        'away_name': fs.get('away_name') or 'Deplasman',
        'competition_name': fs.get('competition_name') or '',
        'competition_id': safe_int(fs.get('competition_id'), 0),
        'league_country': '',
        'league_country_image': '',
        'date_unix': safe_int(fs.get('date_unix'), 0),
        'home_image': fs.get('home_image') or '',
        'away_image': fs.get('away_image') or '',
        'home_country': country_from_image_path(fs.get('home_image') or ''),
        'away_country': country_from_image_path(fs.get('away_image') or ''),
        'home_ppg': safe_float(fs.get('home_ppg') or fs.get('team_a_ppg')),
        'away_ppg': safe_float(fs.get('away_ppg') or fs.get('team_b_ppg')),
        'team_a_xg_prematch': safe_float(fs.get('team_a_xg_prematch')),
        'team_b_xg_prematch': safe_float(fs.get('team_b_xg_prematch')),
        'btts_potential': safe_float(fs.get('btts_potential')),
        'o25_potential': safe_float(fs.get('o25_potential')),
        'o05HT_potential': safe_float(fs.get('o05HT_potential')),
        'odds_ft_1': safe_float(fs.get('odds_ft_1')),
        'odds_ft_x': safe_float(fs.get('odds_ft_x')),
        'odds_ft_2': safe_float(fs.get('odds_ft_2')),
        'homeGoalCount': safe_int(fs.get('homeGoalCount'), 0),
        'awayGoalCount': safe_int(fs.get('awayGoalCount'), 0),
        'status': fs.get('status') or 'incomplete',
        'elapsed': safe_int(fs.get('elapsed'), 0),
        'boss_ai_decision': '',
    }
    row['boss_ai_decision'] = ai_comment_prematch(client, row)
    return row

def normalize_sm_row(sm: Dict[str, Any], client=None) -> Dict[str, Any]:
    base = extract_sm_basic(sm)
    predictions = sm.get('predictions', {})
    row = {
        'id': f"sm-{base['sportmonks_id']}",
        'source_ids': {'footystats': '', 'sportmonks': str(base['sportmonks_id'])},
        'home_name': base['home_name'],
        'away_name': base['away_name'],
        'competition_name': base['competition_name'],
        'competition_id': base['competition_id'],
        'league_country': base['league_country'],
        'league_country_image': base['league_country_image'],
        'date_unix': base['date_unix'],
        'home_image': base['home_image'],
        'away_image': base['away_image'],
        'home_ppg': safe_float(predictions.get('home_ppg')),
        'away_ppg': safe_float(predictions.get('away_ppg')),
        'team_a_xg_prematch': 0.0,
        'team_b_xg_prematch': 0.0,
        'btts_potential': safe_float(predictions.get('btts_probability')),
        'o25_potential': safe_float(predictions.get('over_25_probability')),
        'o05HT_potential': 0.0,
        'odds_ft_1': 0.0,
        'odds_ft_x': 0.0,
        'odds_ft_2': 0.0,
        'homeGoalCount': base['homeGoalCount'],
        'awayGoalCount': base['awayGoalCount'],
        'status': base['status'],
        'elapsed': base['elapsed'],
        'boss_ai_decision': '',
    }
    row['boss_ai_decision'] = ai_comment_prematch(client, row)
    return row

def main():
    ensure_dir()
    client = init_vertex_client()
    today_str = iso_date(0)
    tomorrow_str = iso_date(1)

    fs_today = [normalize_fs_row(x, client) for x in fetch_footystats_for_date(today_str)]
    fs_tomorrow = [normalize_fs_row(x, client) for x in fetch_footystats_for_date(tomorrow_str)]
    sm_today = [normalize_sm_row(x, client) for x in fetch_sportmonks_for_date(today_str)]
    sm_tomorrow = [normalize_sm_row(x, client) for x in fetch_sportmonks_for_date(tomorrow_str)]

    save_json(FOOTYSTATS_TODAY_JSON, {'data': fs_today, 'updated_at': datetime.utcnow().isoformat() + 'Z'})
    save_json(FOOTYSTATS_TOMORROW_JSON, {'data': fs_tomorrow, 'updated_at': datetime.utcnow().isoformat() + 'Z'})
    save_json(SPORTMONKS_TODAY_JSON, {'data': sm_today, 'updated_at': datetime.utcnow().isoformat() + 'Z'})
    save_json(SPORTMONKS_TOMORROW_JSON, {'data': sm_tomorrow, 'updated_at': datetime.utcnow().isoformat() + 'Z'})
    save_json(SOURCE_TABS_JSON, {'tabs': [{'key': 'footystats', 'label': 'FootyStats'}, {'key': 'sportmonks', 'label': 'Sportmonks'}], 'updated_at': datetime.utcnow().isoformat() + 'Z'})

if __name__ == '__main__':
    main()
