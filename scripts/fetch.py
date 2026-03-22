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

DATA_DIR = 'data'
FOOTYSTATS_TODAY_JSON     = os.path.join(DATA_DIR, 'footystats_today.json')
FOOTYSTATS_TOMORROW_JSON  = os.path.join(DATA_DIR, 'footystats_tomorrow.json')
SPORTMONKS_TODAY_JSON     = os.path.join(DATA_DIR, 'sportmonks_today.json')
SPORTMONKS_TOMORROW_JSON  = os.path.join(DATA_DIR, 'sportmonks_tomorrow.json')
# FIX 1: Bundle must exist before live_radar runs
BUNDLE_JSON               = os.path.join(DATA_DIR, 'sportmonks_bundle.json')
MATCH_MAP_JSON            = os.path.join(DATA_DIR, 'match_map.json')
SOURCE_TABS_JSON          = os.path.join(DATA_DIR, 'source_tabs.json')
HEALTH_JSON               = os.path.join(DATA_DIR, 'health.json')

FS_KEY       = os.getenv('FOOTYSTATS_KEY', '').strip()
SM_KEY       = os.getenv('SPORTMONKS_KEY', '').strip()
# FIX 4: Doğru model adı
GEMINI_MODEL = (os.getenv('GEMINI_MODEL') or 'gemini-2.5-flash').strip()
REQUEST_TIMEOUT  = 25
SM_BASE_INCLUDE  = 'participants;league.country;venue;referees;weatherReport;state;scores;periods;round;predictions.type;statistics.type;odds.market'


# ─── Yardımcı fonksiyonlar ────────────────────────────────────────────────────

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
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

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
            sep = '&' if '?' in url else '?'
            r = requests.get(f"{url}{sep}page={page}", timeout=REQUEST_TIMEOUT)
            res = r.json()
            all_data.extend(res.get('data', []))
            if not res.get('pagination', {}).get('has_more'):
                break
            page += 1
        except Exception as e:
            log(f'⚠️ Sayfalama Hatası: {e}')
            break
    return all_data

def iso_date(offset: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=offset)).strftime('%Y-%m-%d')

def country_from_image_path(path: str) -> str:
    if not path: return ''
    last = path.split('/')[-1]
    if '-' not in last: return ''
    return last.split('-')[0].replace('_', ' ').strip().title()


# ─── FIX 5: Vertex / Gemini istemcisi — credentials dosyasını önce dene ──────

def init_vertex_client():
    if genai is None: return None
    # Önce credentials dosyasından project_id oku (eski yöntem)
    gac = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '').strip()
    if gac and os.path.exists(gac):
        try:
            with open(gac, 'r', encoding='utf-8') as f:
                info = json.load(f)
            project = info.get('project_id')
            if project:
                return genai.Client(vertexai=True, project=project, location='global')
        except Exception as e:
            log(f'⚠️ Credentials dosyasından proje okunamadı: {e}')
    # Fallback: env variable
    project  = os.getenv('GCP_PROJECT_ID', '').strip()
    location = os.getenv('GCP_LOCATION', 'global').strip()
    if not project: return None
    try:
        return genai.Client(vertexai=True, project=project, location=location)
    except Exception as e:
        log(f'⚠️ Gemini başlatılamadı: {e}')
        return None


# ─── İsim temizleme & eşleştirme (FIX 6 için) ────────────────────────────────

def clean_name(name: str) -> str:
    n = unidecode(str(name or '')).lower()
    n = re.sub(r'[\W_]+', '', n)
    for t in ['footballclub','futebolclube','clubdefutbol','clubdeportivo',
              'women','ladies','reserves','reserve','ii','iii','u21','u23',
              'fc','cf','ac','afc','sc','sk','if','fk','bk','nk','cd','de','la','the']:
        n = n.replace(t, '')
    return n

def name_ratio(a: str, b: str) -> float:
    if not a or not b: return 0.0
    if a == b: return 1.0
    if a in b or b in a:
        return min(len(a), len(b)) / max(len(a), len(b), 1)
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / max(len(a), len(b), 1)

# FIX 6: Sportmonks ↔ FootyStats eşleştirme
def match_sm_to_fs(sm_row: Dict[str, Any],
                   fs_rows: List[Dict[str, Any]],
                   match_map: Dict[str, str]) -> Optional[Dict[str, Any]]:
    sm_id = str(sm_row.get('id', ''))
    # Önce kayıtlı mapping'e bak
    if sm_id and sm_id in match_map:
        fs_id = match_map[sm_id]
        for fs in fs_rows:
            if str(fs.get('id')) == fs_id:
                return fs
    # Fuzzy eşleştir
    sm_h = clean_name(sm_row.get('home_name', ''))
    sm_a = clean_name(sm_row.get('away_name', ''))
    best, best_score = None, 0.0
    for fs in fs_rows:
        score = (name_ratio(sm_h, clean_name(fs.get('home_name', ''))) +
                 name_ratio(sm_a, clean_name(fs.get('away_name', '')))) / 2.0
        if score > best_score:
            best, best_score = fs, score
    # FIX 5: Raised threshold to 0.75 to prevent wrong team score overlay
    if best is not None and best_score >= 0.75:
        if sm_id:
            match_map[sm_id] = str(best.get('id', ''))
        return best
    return None


# ─── Sportmonks yardımcıları ─────────────────────────────────────────────────

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
    parts   = sm.get('participants') or []
    home_p, away_p = get_side_participants(parts)
    league  = sm.get('league') or {}
    venue   = sm.get('venue') or {}
    refs    = sm.get('referees') or []
    ref     = refs[0] if refs else {}
    weather = sm.get('weatherreport') or sm.get('weatherReport') or {}
    state   = sm.get('state') or {}
    h, a    = current_score(sm)
    return {
        'sportmonks_id':       safe_int(sm.get('id'), 0),
        'home_name':           home_p.get('name', ''),
        'away_name':           away_p.get('name', ''),
        'home_image':          home_p.get('image_path', ''),
        'away_image':          away_p.get('image_path', ''),
        'competition_name':    league.get('name', '') or sm.get('name', ''),
        'competition_id':      safe_int(league.get('id'), 0),
        'league_country':      (league.get('country') or {}).get('name', ''),
        'league_country_image':(league.get('country') or {}).get('image_path', ''),
        'date_unix':           safe_int(sm.get('starting_at_timestamp'), 0),
        'stadium_name':        venue.get('name', ''),
        'stadium_location':    venue.get('city_name', ''),
        'referee_name':        ref.get('name') or ref.get('common_name') or '',
        'weather':             weather,
        'status':              state.get('state') or 'incomplete',
        'elapsed':             safe_int(state.get('minute'), 0),
        'homeGoalCount':       h,
        'awayGoalCount':       a,
    }


# ─── AI yorum ────────────────────────────────────────────────────────────────

def ai_comment_prematch(client, match: Dict[str, Any]) -> str:
    # NOTE: fetch.py does NOT call AI — client is always None here.
    # AI comments are written by ai_comment.py (rich bundle payload).
    # This stub exists only for compatibility — do not add AI calls here.
    return ''


# ─── FIX 1: FootyStats veri çekme fonksiyonları ──────────────────────────────

def fetch_footystats_for_date(date_str: str) -> List[Dict[str, Any]]:
    if not FS_KEY:
        log('⚠️ FOOTYSTATS_KEY missing, skipping.')
        return []
    today_str = iso_date(0)
    # FIX 2: todays-matches only works for today — use date-specific endpoint for tomorrow
    if date_str == today_str:
        url = f'https://api.football-data-api.com/todays-matches?key={FS_KEY}&include=stats,odds'
    else:
        url = f'https://api.football-data-api.com/matches-by-date?key={FS_KEY}&include=stats,odds&date={date_str}'
    result = fetch_json(url)
    rows = result.get('data', [])
    log(f'✅ FootyStats {date_str}: {len(rows)} matches')
    return rows


# ─── FIX 1: Sportmonks veri çekme fonksiyonları ──────────────────────────────

def fetch_sportmonks_for_date(date_str: str) -> List[Dict[str, Any]]:
    if not SM_KEY:
        log('⚠️ SPORTMONKS_KEY eksik, atlanıyor.')
        return []
    url = (
        f'https://api.sportmonks.com/v3/football/fixtures/date/{date_str}'
        f'?api_token={SM_KEY}&include={SM_BASE_INCLUDE}'
    )
    rows = fetch_all_pages(url)
    log(f'✅ Sportmonks {date_str}: {len(rows)} maç')
    return rows


# ─── Normalize ───────────────────────────────────────────────────────────────

def normalize_fs_row(fs: Dict[str, Any], client=None,
                     sm_match: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    sm_base = extract_sm_basic(sm_match) if sm_match else {}
    row = {
        'id':               str(fs.get('id', '')),
        'source_ids':       {'footystats': str(fs.get('id', '')),
                             'sportmonks': str(sm_base.get('sportmonks_id', ''))},
        'home_name':        fs.get('home_name') or sm_base.get('home_name') or 'Home',
        'away_name':        fs.get('away_name') or sm_base.get('away_name') or 'Away',
        'competition_name': fs.get('competition_name') or sm_base.get('competition_name') or '',
        'competition_id':   safe_int(fs.get('competition_id'), 0),
        'league_country':   sm_base.get('league_country', ''),
        'league_country_image': sm_base.get('league_country_image', ''),
        'date_unix':        safe_int(fs.get('date_unix'), 0),
        'home_image':       fs.get('home_image') or sm_base.get('home_image') or '',
        'away_image':       fs.get('away_image') or sm_base.get('away_image') or '',
        'home_country':     country_from_image_path(fs.get('home_image') or ''),
        'away_country':     country_from_image_path(fs.get('away_image') or ''),
        'venue_name':       sm_base.get('stadium_name', '') or fs.get('stadium_name', ''),
        'venue_city':       sm_base.get('stadium_location', '') or fs.get('stadium_location', ''),
        'referee_name':     sm_base.get('referee_name', ''),
        'weather':          sm_base.get('weather', {}),
        'home_ppg':         safe_float(fs.get('home_ppg') or fs.get('team_a_ppg')),
        'away_ppg':         safe_float(fs.get('away_ppg') or fs.get('team_b_ppg')),
        'team_a_xg_prematch': safe_float(fs.get('team_a_xg_prematch')),
        'team_b_xg_prematch': safe_float(fs.get('team_b_xg_prematch')),
        'btts_potential':   safe_float(fs.get('btts_potential')),
        'o25_potential':    safe_float(fs.get('o25_potential')),
        'o05HT_potential':  safe_float(fs.get('o05HT_potential')),
        'odds_ft_1':        safe_float(fs.get('odds_ft_1')),
        'odds_ft_x':        safe_float(fs.get('odds_ft_x')),
        'odds_ft_2':        safe_float(fs.get('odds_ft_2')),
        'homeGoalCount':    safe_int(fs.get('homeGoalCount'), 0),
        'awayGoalCount':    safe_int(fs.get('awayGoalCount'), 0),
        'status':           fs.get('status') or 'incomplete',
        'elapsed':          safe_int(fs.get('elapsed'), 0),
        # AI yorum live_radar tarafından doldurulacak
        'boss_ai_decision': '',
        'prematch_comment': '',
        'ai_comment':       '',
    }
    return row

def normalize_sm_row(sm: Dict[str, Any], client=None) -> Dict[str, Any]:
    base = extract_sm_basic(sm)

    # --- Predictions ---
    raw_predictions = sm.get('predictions') or []
    btts_potential = o25_potential = home_ppg = away_ppg = o05HT_potential = 0.0
    xg_home = xg_away = 0.0

    pred_list = raw_predictions if isinstance(raw_predictions, list) else []
    for p in pred_list:
        if not isinstance(p, dict): continue
        dev  = str((p.get('type') or {}).get('developer_name') or '').upper()
        vals = p.get('predictions') or {}
        if not isinstance(vals, dict): continue
        if dev == 'BTTS_PROBABILITY':
            btts_potential = safe_float(vals.get('yes'))
        elif 'OVER_UNDER_2_5' in dev and 'HALF' not in dev:
            o25_potential  = safe_float(vals.get('yes'))
        elif 'OVER_UNDER_0_5' in dev and ('HALF' in dev or '1ST' in dev):
            o05HT_potential = safe_float(vals.get('yes'))
        elif dev in ('HOME_WIN_PROBABILITY', 'WINNING_ODDS'):
            home_ppg = safe_float(vals.get('yes') or vals.get('home'))
        elif dev == 'AWAY_WIN_PROBABILITY':
            away_ppg = safe_float(vals.get('yes') or vals.get('away'))
        elif 'XG' in dev or 'EXPECTED_GOALS' in dev:
            loc = str((p.get('participant') or {}).get('meta', {}).get('location') or '').lower()
            if loc == 'home': xg_home = safe_float(vals.get('value') or vals.get('yes'))
            elif loc == 'away': xg_away = safe_float(vals.get('value') or vals.get('yes'))

    if isinstance(raw_predictions, dict):
        btts_potential  = safe_float(raw_predictions.get('btts_probability'))
        o25_potential   = safe_float(raw_predictions.get('over_25_probability'))
        home_ppg        = safe_float(raw_predictions.get('home_ppg'))
        away_ppg        = safe_float(raw_predictions.get('away_ppg'))

    # --- Statistics (xG from stats) ---
    for st in (sm.get('statistics') or []):
        dev = str((st.get('type') or {}).get('developer_name') or '').upper()
        loc = str(st.get('location') or '').lower()
        val = safe_float((st.get('data') or {}).get('value'))
        if 'XG' in dev or 'EXPECTED_GOALS' in dev:
            if loc == 'home' and not xg_home: xg_home = val
            elif loc == 'away' and not xg_away: xg_away = val
        elif 'PPG' in dev or 'POINTS_PER_GAME' in dev:
            if loc == 'home' and not home_ppg: home_ppg = val
            elif loc == 'away' and not away_ppg: away_ppg = val

    # --- Odds ---
    odds_ft_1 = odds_ft_x = odds_ft_2 = odds_ft_over25 = odds_btts_yes = 0.0
    for o in (sm.get('odds') or []):
        market = str((o.get('market') or {}).get('developer_name') or (o.get('market') or {}).get('name') or '').upper()
        label  = str(o.get('label') or o.get('name') or '').strip()
        val    = safe_float(o.get('value') or o.get('odds'))
        if not val: continue
        if '3WAY' in market or 'MATCH_WINNER' in market or '1X2' in market:
            if label == '1' and not odds_ft_1:   odds_ft_1 = val
            elif label == 'X' and not odds_ft_x: odds_ft_x = val
            elif label == '2' and not odds_ft_2: odds_ft_2 = val
        elif 'OVER_UNDER' in market and '2.5' in label:
            if 'OVER' in label.upper() and not odds_ft_over25: odds_ft_over25 = val
        elif 'BTTS' in market or 'BOTH_TEAMS' in market:
            if 'YES' in label.upper() and not odds_btts_yes: odds_btts_yes = val

    row = {
        'id':               f"sm-{base['sportmonks_id']}",
        'source_ids':       {'footystats': '', 'sportmonks': str(base['sportmonks_id'])},
        'home_name':        base['home_name'],
        'away_name':        base['away_name'],
        'competition_name': base['competition_name'],
        'competition_id':   base['competition_id'],
        'league_country':   base['league_country'],
        'league_country_image': base['league_country_image'],
        'date_unix':        base['date_unix'],
        'home_image':       base['home_image'],
        'away_image':       base['away_image'],
        'home_country':     country_from_image_path(base['home_image']),
        'away_country':     country_from_image_path(base['away_image']),
        'venue_name':       base['stadium_name'],
        'venue_city':       base['stadium_location'],
        'referee_name':     base['referee_name'],
        'weather':          base['weather'],
        'home_ppg':         home_ppg,
        'away_ppg':         away_ppg,
        'pre_match_home_ppg': home_ppg,
        'pre_match_away_ppg': away_ppg,
        'team_a_xg_prematch': xg_home,
        'team_b_xg_prematch': xg_away,
        'btts_potential':   btts_potential,
        'o25_potential':    o25_potential,
        'o05HT_potential':  o05HT_potential,
        'odds_ft_1':        odds_ft_1,
        'odds_ft_x':        odds_ft_x,
        'odds_ft_2':        odds_ft_2,
        'odds_ft_over25':   odds_ft_over25,
        'odds_btts_yes':    odds_btts_yes,
        'homeGoalCount':    base['homeGoalCount'],
        'awayGoalCount':    base['awayGoalCount'],
        'status':           base['status'],
        'elapsed':          base['elapsed'],
        'boss_ai_decision': '',
        'prematch_comment': '',
        'ai_comment':       '',
    }
    return row


# ─── Ana fonksiyon ───────────────────────────────────────────────────────────

def main():
    ensure_dir()
    started = time.time()
    # AI yorumlar live_radar.py tarafından yapılıyor — fetch hızlı kalır
    client = None

    today_str    = iso_date(0)
    tomorrow_str = iso_date(1)

    # ── FootyStats verileri ──────────────────────────────────────────────────
    fs_today_raw    = fetch_footystats_for_date(today_str)
    fs_tomorrow_raw = fetch_footystats_for_date(tomorrow_str)

    # ── Sportmonks verileri ──────────────────────────────────────────────────
    sm_today_raw    = fetch_sportmonks_for_date(today_str)
    sm_tomorrow_raw = fetch_sportmonks_for_date(tomorrow_str)

    # FIX 4: Prune match_map — only keep SM IDs seen today/tomorrow to prevent bloat
    match_map_full = load_json(MATCH_MAP_JSON, {'sportmonks_to_footystats': {}})
    match_map = match_map_full.get('sportmonks_to_footystats', {})
    active_sm_ids = {str(s.get('id', '')) for s in sm_today_raw + sm_tomorrow_raw if s.get('id')}
    match_map = {k: v for k, v in match_map.items() if k in active_sm_ids}

    # ── FIX 6: SM → FS eşleştirmeli FootyStats normalize ───────────────────
    fs_today: List[Dict] = []
    for fs in fs_today_raw:
        sm_match = match_sm_to_fs(
            {'id': '', 'home_name': fs.get('home_name', ''), 'away_name': fs.get('away_name', '')},
            [extract_sm_basic(s) | {'id': s.get('id')} for s in sm_today_raw],
            match_map,
        )
        # Eşleşen SM satırını bul
        sm_full = None
        if sm_match:
            sm_id = str(sm_match.get('sportmonks_id') or sm_match.get('id', ''))
            sm_full = next((s for s in sm_today_raw if str(s.get('id', '')) == sm_id), None)
        fs_today.append(normalize_fs_row(fs, client, sm_full))

    fs_tomorrow: List[Dict] = []
    for fs in fs_tomorrow_raw:
        sm_match = match_sm_to_fs(
            {'id': '', 'home_name': fs.get('home_name', ''), 'away_name': fs.get('away_name', '')},
            [extract_sm_basic(s) | {'id': s.get('id')} for s in sm_tomorrow_raw],
            match_map,
        )
        sm_full = None
        if sm_match:
            sm_id = str(sm_match.get('sportmonks_id') or sm_match.get('id', ''))
            sm_full = next((s for s in sm_tomorrow_raw if str(s.get('id', '')) == sm_id), None)
        fs_tomorrow.append(normalize_fs_row(fs, client, sm_full))

    # ── Sadece SM'de olan (FS'de eşleşmeyen) maçlar ─────────────────────────
    matched_sm_ids = set(match_map.keys())
    sm_today    = [normalize_sm_row(s, client) for s in sm_today_raw
                   if str(s.get('id', '')) not in matched_sm_ids]
    sm_tomorrow = [normalize_sm_row(s, client) for s in sm_tomorrow_raw
                   if str(s.get('id', '')) not in matched_sm_ids]

    # ── Kaydet ──────────────────────────────────────────────────────────────
    now_str = datetime.utcnow().isoformat() + 'Z'
    save_json(FOOTYSTATS_TODAY_JSON,    {'data': fs_today,    'updated_at': now_str})
    save_json(FOOTYSTATS_TOMORROW_JSON, {'data': fs_tomorrow, 'updated_at': now_str})
    save_json(SPORTMONKS_TODAY_JSON,    {'data': sm_today,    'updated_at': now_str})
    save_json(SPORTMONKS_TOMORROW_JSON, {'data': sm_tomorrow, 'updated_at': now_str})
    save_json(MATCH_MAP_JSON,           {'sportmonks_to_footystats': match_map})
    # FIX 1: Initialize bundle if it doesn't exist — live_radar depends on it
    if not os.path.exists(BUNDLE_JSON):
        save_json(BUNDLE_JSON, {'fixtures': {}, 'updated_at': now_str})
    save_json(SOURCE_TABS_JSON, {
        'tabs': [
            {'key': 'footystats', 'label': 'FootyStats'},
            {'key': 'sportmonks', 'label': 'Sportmonks'},
        ],
        'updated_at': now_str,
    })

    health = load_json(HEALTH_JSON, {})
    health.update({
        'fetch_runner':         'fetch',
        'fetch_started_at':     now_str,
        'fetch_duration_sec':   round(time.time() - started, 2),
        'fs_today_count':       len(fs_today),
        'fs_tomorrow_count':    len(fs_tomorrow),
        'sm_today_count':       len(sm_today),
        'sm_tomorrow_count':    len(sm_tomorrow),
        'match_map_size':       len(match_map),
    })
    save_json(HEALTH_JSON, health)

    log(f'✅ footystats_today.json    : {len(fs_today)} maç')
    log(f'✅ footystats_tomorrow.json : {len(fs_tomorrow)} maç')
    log(f'✅ sportmonks_today.json    : {len(sm_today)} maç')
    log(f'✅ sportmonks_tomorrow.json : {len(sm_tomorrow)} maç')
    log(f'✅ match_map boyutu         : {len(match_map)}')
    log(f'✅ Süre                     : {health["fetch_duration_sec"]}s')


if __name__ == '__main__':
    main()
