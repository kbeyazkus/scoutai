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
MATCH_MAP_JSON            = os.path.join(DATA_DIR, 'match_map.json')
SOURCE_TABS_JSON          = os.path.join(DATA_DIR, 'source_tabs.json')
HEALTH_JSON               = os.path.join(DATA_DIR, 'health.json')

FS_KEY       = os.getenv('FOOTYSTATS_KEY', '').strip()
SM_KEY       = os.getenv('SPORTMONKS_KEY', '').strip()
# FIX 4: Doğru model adı
GEMINI_MODEL = (os.getenv('GEMINI_MODEL') or 'gemini-2.5-flash').strip()
REQUEST_TIMEOUT  = 25
SM_BASE_INCLUDE  = 'participants;league.country;venue;referees;weatherReport;state;scores;periods;round;predictions'


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
                return genai.Client(vertexai=True, project=project, location='us-central1')
        except Exception as e:
            log(f'⚠️ Credentials dosyasından proje okunamadı: {e}')
    # Fallback: env variable
    project  = os.getenv('GCP_PROJECT_ID', '').strip()
    location = os.getenv('GCP_LOCATION', 'us-central1').strip()
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
    if best is not None and best_score >= 0.68:
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
    if client is None: return ''
    payload = {
        'home':     match.get('home_name'),
        'away':     match.get('away_name'),
        'league':   match.get('competition_name'),
        'xg_home':  safe_float(match.get('team_a_xg_prematch')),
        'xg_away':  safe_float(match.get('team_b_xg_prematch')),
        'home_ppg': safe_float(match.get('home_ppg')),
        'away_ppg': safe_float(match.get('away_ppg')),
        'btts':     safe_float(match.get('btts_potential')),
        'over25':   safe_float(match.get('o25_potential')),
        'odds':     {
            '1': safe_float(match.get('odds_ft_1')),
            'x': safe_float(match.get('odds_ft_x')),
            '2': safe_float(match.get('odds_ft_2')),
        },
    }
    prompt = '\n'.join([
        'Sen çapraz veri sorgulayan teknik bahis motorusun.',
        'Metrikleri (xG, PPG, Oranlar) çarpıştır. Zıtlık varsa riskli kabul et.',
        'Sadece şu türlerden en risksiz TEK kuponu öner veya riskliyse "Oynama / Pas" de:',
        'MS1/MS2, Üst 2.5, KG Var, İY Üst 0.5, Asya Handikap, Takım 1.5 Üst.',
        'Kurallar: Max 80 kelime. 3 bölüm: DURUM, NEDEN, SONUÇ. Sert ve teknik ol.',
        json.dumps(payload, ensure_ascii=False),
    ])
    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = (getattr(response, 'text', '') or '').strip()
        return text if text else ''
    except Exception as e:
        log(f'⚠️ AI hatası: {e}')
        return ''


# ─── FIX 1: FootyStats veri çekme fonksiyonları ──────────────────────────────

def fetch_footystats_for_date(date_str: str) -> List[Dict[str, Any]]:
    if not FS_KEY:
        log('⚠️ FOOTYSTATS_KEY eksik, atlanıyor.')
        return []
    url = (
        f'https://api.football-data-api.com/todays-matches'
        f'?key={FS_KEY}&include=stats,odds&date={date_str}'
    )
    result = fetch_json(url)
    rows = result.get('data', [])
    log(f'✅ FootyStats {date_str}: {len(rows)} maç')
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
        'home_name':        fs.get('home_name') or sm_base.get('home_name') or 'Ev Sahibi',
        'away_name':        fs.get('away_name') or sm_base.get('away_name') or 'Deplasman',
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
        'boss_ai_decision': '',
    }
    row['boss_ai_decision'] = ai_comment_prematch(client, row)
    return row

def normalize_sm_row(sm: Dict[str, Any], client=None) -> Dict[str, Any]:
    base = extract_sm_basic(sm)
    predictions = sm.get('predictions') or {}
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
        'home_ppg':         safe_float(predictions.get('home_ppg')),
        'away_ppg':         safe_float(predictions.get('away_ppg')),
        'team_a_xg_prematch': 0.0,
        'team_b_xg_prematch': 0.0,
        'btts_potential':   safe_float(predictions.get('btts_probability')),
        'o25_potential':    safe_float(predictions.get('over_25_probability')),
        'o05HT_potential':  0.0,
        'odds_ft_1':        0.0,
        'odds_ft_x':        0.0,
        'odds_ft_2':        0.0,
        'homeGoalCount':    base['homeGoalCount'],
        'awayGoalCount':    base['awayGoalCount'],
        'status':           base['status'],
        'elapsed':          base['elapsed'],
        'boss_ai_decision': '',
    }
    row['boss_ai_decision'] = ai_comment_prematch(client, row)
    return row


# ─── Ana fonksiyon ───────────────────────────────────────────────────────────

def main():
    ensure_dir()
    started = time.time()
    client = init_vertex_client()

    today_str    = iso_date(0)
    tomorrow_str = iso_date(1)

    # FIX 6: match_map yükle
    match_map_full = load_json(MATCH_MAP_JSON, {'sportmonks_to_footystats': {}})
    match_map = match_map_full.get('sportmonks_to_footystats', {})

    # ── FootyStats verileri ──────────────────────────────────────────────────
    fs_today_raw    = fetch_footystats_for_date(today_str)
    fs_tomorrow_raw = fetch_footystats_for_date(tomorrow_str)

    # ── Sportmonks verileri ──────────────────────────────────────────────────
    sm_today_raw    = fetch_sportmonks_for_date(today_str)
    sm_tomorrow_raw = fetch_sportmonks_for_date(tomorrow_str)

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
