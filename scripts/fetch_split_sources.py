#!/usr/bin/env python3
import json, os, re, time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from unidecode import unidecode

try:
    from google import genai
except Exception:
    genai = None

DATA_DIR = 'data'
FS_TODAY_JSON = os.path.join(DATA_DIR, 'footystats_today.json')
FS_TOMORROW_JSON = os.path.join(DATA_DIR, 'footystats_tomorrow.json')
SM_TODAY_JSON = os.path.join(DATA_DIR, 'sportmonks_today.json')
SM_TOMORROW_JSON = os.path.join(DATA_DIR, 'sportmonks_tomorrow.json')
FS_BUNDLE_JSON = os.path.join(DATA_DIR, 'footystats_bundle.json')
SM_BUNDLE_JSON = os.path.join(DATA_DIR, 'sportmonks_bundle.json')
HEALTH_JSON = os.path.join(DATA_DIR, 'health.json')
SOURCE_TABS_JSON = os.path.join(DATA_DIR, 'source_tabs.json')

FS_KEY = os.getenv('FOOTYSTATS_KEY', '').strip()
SM_KEY = os.getenv('SPORTMONKS_KEY', '').strip()
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash').strip()
REQUEST_TIMEOUT = 25

FIXTURE_BASIC_INCLUDE = 'participants;league.country;venue;referees;weatherReport;state;scores;round'
DETAIL_BASIC_INCLUDE = 'participants;league.country;venue;state;scores;periods'
DETAIL_BLOCKS = [
    'events.type;events.period;events.player',
    'statistics.type;predictions.type',
    'lineups.player;lineups.type;lineups.details.type;metadata.type;coaches',
    'sidelined.sideline.player;sidelined.sideline.type;weatherReport',
]
ODDS_FILTERS = 'markets:1;bookmakers:2'
ODDS_INCLUDE = 'fixtures.odds.market;fixtures.odds.bookmaker;fixtures.participants;league.country'
STANDINGS_INCLUDE = 'participant;rule.type;details.type;form;stage;league;group'
LIVE_STANDINGS_INCLUDE = 'stage;league;details.type;participant'
H2H_INCLUDE = 'participants;league;scores;state;venue;events'


def log(msg: str):
    print(msg, flush=True)


def ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iso_date(offset: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=offset)).strftime('%Y-%m-%d')


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v in (None, ''):
            return default
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default: int = 0) -> int:
    try:
        if v in (None, ''):
            return default
        return int(float(v))
    except Exception:
        return default


def load_json(path: str, default: Any):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, data: Any):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def fetch_json(url: str, timeout: int = REQUEST_TIMEOUT, retries: int = 3) -> Dict[str, Any]:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 403:
                log(f'🔒 Erişim yok (plan kısıtı): {url}')
                return {}
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            log(f'⚠️ fetch_json retry {attempt+1}/{retries} ({wait}s): {e}')
            time.sleep(wait)
    return {}


def init_vertex_client():
    if genai is None:
        return None
    gac = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '').strip()
    if not gac or not os.path.exists(gac):
        return None
    try:
        with open(gac, 'r', encoding='utf-8') as f:
            info = json.load(f)
        return genai.Client(vertexai=True, project=info['project_id'], location='us-central1')
    except Exception as e:
        log(f'⚠️ Gemini başlatılamadı: {e}')
        return None


def clean_name(name: str) -> str:
    n = unidecode(str(name or '')).lower()
    n = re.sub(r'[\W_]+', ' ', n).strip()
    tokens_to_remove = {'footballclub', 'futebolclube', 'clubdefutbol', 'clubdeportivo', 'women', 'ladies', 'reserves', 'reserve', 'ii', 'iii', 'u21', 'u23', 'fc', 'cf', 'ac', 'afc', 'sc', 'sk', 'if', 'fk', 'bk', 'nk', 'cd', 'de', 'la', 'the'}
    words = [w for w in n.split() if w not in tokens_to_remove]
    return ''.join(words)


def merge_value(a: Any, b: Any) -> Any:
    if b in (None, '', [], {}):
        return a
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = merge_value(out.get(k), v) if k in out else v
        return out
    return b


def get_side_participants(parts: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    home, away = {}, {}
    for p in parts:
        loc = str((p.get('meta') or {}).get('location') or '').lower()
        if loc == 'home':
            home = p
        elif loc == 'away':
            away = p
    if not home and parts:
        home = parts[0]
    if not away and len(parts) > 1:
        away = parts[1]
    return home, away


def country_from_image_path(path: str) -> str:
    if not path:
        return ''
    last = path.split('/')[-1]
    if '-' not in last:
        return ''
    return last.split('-')[0].replace('_', ' ').strip().title()


def normalize_status(v: Any) -> str:
    s = str(v or '').strip().lower()
    if not s:
        return 'incomplete'
    if s in ('not started', 'ns', 'scheduled', 'pre-match', 'prematch'):
        return 'incomplete'
    return s


def summarize_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    out = {'home': 0.0, 'away': 0.0, 'draw': 0.0, 'over25': 0.0, 'btts': 0.0, 'firsthalf': 0.0}
    for p in predictions or []:
        dev = str((p.get('type') or {}).get('developer_name') or '').upper()
        vals = p.get('predictions') or {}
        if dev == 'FULLTIME_RESULT_PROBABILITY':
            out['home'] = safe_float(vals.get('home'))
            out['away'] = safe_float(vals.get('away'))
            out['draw'] = safe_float(vals.get('draw'))
        elif dev == 'OVER_UNDER_2_5_PROBABILITY':
            out['over25'] = safe_float(vals.get('yes'))
        elif dev == 'BTTS_PROBABILITY':
            out['btts'] = safe_float(vals.get('yes'))
        elif dev in ('FIRST_HALF_RESULT_PROBABILITY', 'OVER_UNDER_0_5_1ST_HALF_PROBABILITY', 'OVER_UNDER_1ST_HALF_0_5_PROBABILITY'):
            out['firsthalf'] = max(out['firsthalf'], safe_float(vals.get('yes')), safe_float(vals.get('home')), safe_float(vals.get('away')))
    return out


def odds_value(odds: List[Dict[str, Any]], market_dev: str, label_match: str) -> float:
    for o in odds or []:
        market = str((o.get('market') or {}).get('developer_name') or '').upper()
        if market != market_dev:
            continue
        label = str(o.get('label') or o.get('name') or '').lower()
        if label_match.lower() in label:
            return safe_float(o.get('value') or o.get('odds'))
    return 0.0


def heur_ai_prematch(row: Dict[str, Any], detail: Optional[Dict[str, Any]] = None, odds: Optional[List[Dict[str, Any]]] = None) -> str:
    detail = detail or {}
    odds = odds or []
    preds = summarize_predictions(detail.get('predictions') or [])

    hppg = safe_float(row.get('home_ppg'))
    appg = safe_float(row.get('away_ppg'))
    hxg = safe_float(row.get('team_a_xg_prematch'))
    axg = safe_float(row.get('team_b_xg_prematch'))
    btts = safe_float(row.get('btts_potential')) or preds.get('btts', 0.0)
    o25 = safe_float(row.get('o25_potential')) or preds.get('over25', 0.0)
    iy = safe_float(row.get('o05HT_potential')) or preds.get('firsthalf', 0.0)
    oh = safe_float(row.get('odds_ft_1')) or odds_value(odds, 'FULLTIME_RESULT', '1')
    oa = safe_float(row.get('odds_ft_2')) or odds_value(odds, 'FULLTIME_RESULT', '2')

    home_edge = (hppg - appg) + (hxg - axg) + ((preds.get('home', 0) - preds.get('away', 0)) / 100.0)
    away_edge = (appg - hppg) + (axg - hxg) + ((preds.get('away', 0) - preds.get('home', 0)) / 100.0)

    if o25 >= 62 or btts >= 60:
        sonuc = 'Üst 2.5' if o25 >= btts else 'KG Var'
        return f"DURUM: Maç gollü profile yakın.\nNEDEN: Üst/KG sinyali güçlü.\nSONUÇ: {sonuc} tarafı önde."
    if home_edge > 0.28:
        return f"DURUM: {row.get('home_name')} tarafı önde.\nNEDEN: PPG/xG verisi ev sahibini destekliyor.\nSONUÇ: MS1 veya 1X önde."
    if away_edge > 0.28:
        return f"DURUM: {row.get('away_name')} tarafı önde.\nNEDEN: PPG/xG verisi deplasmanı destekliyor.\nSONUÇ: MS2 veya X2 önde."
    if iy >= 68:
        return "DURUM: İlk yarıda tempo bekleniyor.\nNEDEN: İlk yarı gol potansiyeli yüksek.\nSONUÇ: İY 0.5 üst izlenebilir."
    if oh and oa:
        if oh < oa:
            return f"DURUM: {row.get('home_name')} oran avantajına sahip.\nNEDEN: Piyasa ev sahibini önde fiyatlıyor.\nSONUÇ: 1X daha güvenli."
        if oa < oh:
            return f"DURUM: {row.get('away_name')} oran avantajına sahip.\nNEDEN: Piyasa deplasmanı önde fiyatlıyor.\nSONUÇ: X2 daha güvenli."
    return 'AI yorumu henüz yok.'


def ai_comment_prematch(client, match: Dict[str, Any], detail: Optional[Dict[str, Any]] = None, odds: Optional[List[Dict[str, Any]]] = None) -> str:
    heuristic = heur_ai_prematch(match, detail, odds)
    if client is None:
        return heuristic
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
        'odds1': safe_float(match.get('odds_ft_1')),
        'odds2': safe_float(match.get('odds_ft_2')),
    }
    prompt = '\n'.join([
        'Sen kısa ve teknik prematch bahis analiz motorusun.',
        'En fazla 85 kelime. 3 satır yaz: DURUM, NEDEN, SONUÇ. Veri yetersizse tam olarak "AI yorumu henüz yok." yaz.',
        json.dumps(payload, ensure_ascii=False)
    ])
    try:
        time.sleep(1.5)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = (getattr(response, 'text', '') or '').strip()
        return text or heuristic
    except Exception as e:
        if '429' in str(e) or 'quota' in str(e).lower() or 'rate' in str(e).lower():
            log(f'⚠️ Gemini rate limit, heuristic kullanılıyor: {e}')
        else:
            log(f'⚠️ Prematch AI hatası: {e}')
        return heuristic


def current_score_from_fs(fs: Dict[str, Any]) -> Tuple[int, int]:
    home = safe_int(fs.get('homeGoalCount') or fs.get('team_a_goals') or fs.get('home_goals'), 0)
    away = safe_int(fs.get('awayGoalCount') or fs.get('team_b_goals') or fs.get('away_goals'), 0)
    return home, away


def build_fs_row(fs: Dict[str, Any], old: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    old = old or {}
    home_image = fs.get('home_image') or old.get('home_image') or ''
    away_image = fs.get('away_image') or old.get('away_image') or ''
    h, a = current_score_from_fs(fs)
    status = normalize_status(fs.get('status') or old.get('status'))
    elapsed = safe_int(fs.get('elapsed') or fs.get('minute') or old.get('elapsed'), 0)
    row = {
        'id': str(fs.get('id') or old.get('id') or ''),
        'source_ids': {'footystats': str(fs.get('id') or old.get('id') or ''), 'sportmonks': ''},
        'source': 'footystats',
        'home_name': fs.get('home_name') or old.get('home_name') or 'Ev Sahibi',
        'away_name': fs.get('away_name') or old.get('away_name') or 'Deplasman',
        'competition_name': fs.get('competition_name') or old.get('competition_name') or '',
        'competition_id': safe_int(fs.get('competition_id') or old.get('competition_id'), 0),
        'league_country': fs.get('league_country') or old.get('league_country') or '',
        'league_country_image': fs.get('league_country_image') or old.get('league_country_image') or '',
        'date_unix': safe_int(fs.get('date_unix') or old.get('date_unix'), 0),
        'home_image': home_image,
        'away_image': away_image,
        'home_country': old.get('home_country') or country_from_image_path(home_image),
        'away_country': old.get('away_country') or country_from_image_path(away_image),
        'stadium_name': fs.get('stadium_name') or old.get('stadium_name') or '',
        'stadium_location': fs.get('stadium_location') or old.get('stadium_location') or '',
        'referee_name': fs.get('referee_name') or old.get('referee_name') or '',
        'weather': fs.get('weather') or old.get('weather') or {},
        'home_ppg': safe_float(fs.get('home_ppg') or fs.get('team_a_ppg') or old.get('home_ppg')),
        'away_ppg': safe_float(fs.get('away_ppg') or fs.get('team_b_ppg') or old.get('away_ppg')),
        'team_a_ppg': fs.get('team_a_ppg') or old.get('team_a_ppg') or str(safe_float(fs.get('home_ppg'))),
        'team_b_ppg': fs.get('team_b_ppg') or old.get('team_b_ppg') or str(safe_float(fs.get('away_ppg'))),
        'pre_match_home_ppg': safe_float(fs.get('pre_match_home_ppg') or old.get('pre_match_home_ppg')),
        'pre_match_away_ppg': safe_float(fs.get('pre_match_away_ppg') or old.get('pre_match_away_ppg')),
        'team_a_xg_prematch': safe_float(fs.get('team_a_xg_prematch') or old.get('team_a_xg_prematch')),
        'team_b_xg_prematch': safe_float(fs.get('team_b_xg_prematch') or old.get('team_b_xg_prematch')),
        'btts_potential': safe_float(fs.get('btts_potential') or old.get('btts_potential')),
        'o25_potential': safe_float(fs.get('o25_potential') or old.get('o25_potential')),
        'o05HT_potential': safe_float(fs.get('o05HT_potential') or old.get('o05HT_potential')),
        'pressure_score': safe_float(fs.get('pressure_score') or old.get('pressure_score')),
        'odds_ft_1': safe_float(fs.get('odds_ft_1') or old.get('odds_ft_1')),
        'odds_ft_x': safe_float(fs.get('odds_ft_x') or old.get('odds_ft_x')),
        'odds_ft_2': safe_float(fs.get('odds_ft_2') or old.get('odds_ft_2')),
        'odds_ft_over25': safe_float(fs.get('odds_ft_over25') or old.get('odds_ft_over25')),
        'odds_btts_yes': safe_float(fs.get('odds_btts_yes') or old.get('odds_btts_yes')),
        'odds_1st_half_over05': safe_float(fs.get('odds_1st_half_over05') or old.get('odds_1st_half_over05')),
        'homeGoalCount': h,
        'awayGoalCount': a,
        'status': status,
        'elapsed': elapsed,
        'boss_ai_decision': old.get('boss_ai_decision') or '',
        'prematch_comment': old.get('prematch_comment') or '',
        'live_comment': old.get('live_comment') or '',
    }
    return row


def extract_sm_basic(sm: Dict[str, Any]) -> Dict[str, Any]:
    if not sm:
        return {}
    parts = sm.get('participants') or []
    home, away = get_side_participants(parts)
    league = sm.get('league') or {}
    venue = sm.get('venue') or {}
    refs = sm.get('referees') or []
    ref = refs[0] if refs else {}
    weather = sm.get('weatherreport') or sm.get('weatherReport') or {}
    return {
        'sportmonks_id': safe_int(sm.get('id'), 0),
        'home_sm_name': home.get('name', ''),
        'away_sm_name': away.get('name', ''),
        'home_sm_logo': home.get('image_path', ''),
        'away_sm_logo': away.get('image_path', ''),
        'competition_name': league.get('name', ''),
        'competition_id': safe_int(league.get('id'), 0),
        'league_country': (league.get('country') or {}).get('name', ''),
        'league_country_image': (league.get('country') or {}).get('image_path', ''),
        'venue_name': venue.get('name', ''),
        'venue_city': venue.get('city_name', ''),
        'referee_name': ref.get('name') or ref.get('common_name') or '',
        'starting_at_timestamp': safe_int(sm.get('starting_at_timestamp'), 0),
        'round_id': safe_int((sm.get('round') or {}).get('id') or sm.get('round_id'), 0),
        'season_id': safe_int(sm.get('season_id'), 0),
        'weather': weather,
        'state': sm.get('state') or {},
    }


def build_sm_row(sm: Dict[str, Any], old: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    old = old or {}
    smi = extract_sm_basic(sm)
    state = smi.get('state') or {}
    return {
        'id': old.get('id') or f"sm-{smi.get('sportmonks_id')}",
        'source_ids': {'footystats': '', 'sportmonks': str(smi.get('sportmonks_id'))},
        'source': 'sportmonks',
        'home_name': smi.get('home_sm_name') or old.get('home_name') or 'Ev Sahibi',
        'away_name': smi.get('away_sm_name') or old.get('away_name') or 'Deplasman',
        'competition_name': smi.get('competition_name') or old.get('competition_name') or '',
        'competition_id': smi.get('competition_id') or old.get('competition_id') or 0,
        'league_country': smi.get('league_country') or old.get('league_country') or '',
        'league_country_image': smi.get('league_country_image') or old.get('league_country_image') or '',
        'date_unix': smi.get('starting_at_timestamp') or old.get('date_unix') or 0,
        'home_image': smi.get('home_sm_logo') or old.get('home_image') or '',
        'away_image': smi.get('away_sm_logo') or old.get('away_image') or '',
        'home_country': country_from_image_path(smi.get('home_sm_logo') or '') or old.get('home_country') or '',
        'away_country': country_from_image_path(smi.get('away_sm_logo') or '') or old.get('away_country') or '',
        'stadium_name': smi.get('venue_name') or old.get('stadium_name') or '',
        'stadium_location': smi.get('venue_city') or old.get('stadium_location') or '',
        'referee_name': smi.get('referee_name') or old.get('referee_name') or '',
        'weather': smi.get('weather') or old.get('weather') or {},
        'home_ppg': safe_float(old.get('home_ppg')),
        'away_ppg': safe_float(old.get('away_ppg')),
        'team_a_ppg': old.get('team_a_ppg', '0'),
        'team_b_ppg': old.get('team_b_ppg', '0'),
        'pre_match_home_ppg': safe_float(old.get('pre_match_home_ppg')),
        'pre_match_away_ppg': safe_float(old.get('pre_match_away_ppg')),
        'team_a_xg_prematch': safe_float(old.get('team_a_xg_prematch')),
        'team_b_xg_prematch': safe_float(old.get('team_b_xg_prematch')),
        'btts_potential': safe_float(old.get('btts_potential')),
        'o25_potential': safe_float(old.get('o25_potential')),
        'o05HT_potential': safe_float(old.get('o05HT_potential')),
        'pressure_score': safe_float(old.get('pressure_score')),
        'odds_ft_1': safe_float(old.get('odds_ft_1')),
        'odds_ft_x': safe_float(old.get('odds_ft_x')),
        'odds_ft_2': safe_float(old.get('odds_ft_2')),
        'odds_ft_over25': safe_float(old.get('odds_ft_over25')),
        'odds_btts_yes': safe_float(old.get('odds_btts_yes')),
        'odds_1st_half_over05': safe_float(old.get('odds_1st_half_over05')),
        'homeGoalCount': safe_int(old.get('homeGoalCount'), 0),
        'awayGoalCount': safe_int(old.get('awayGoalCount'), 0),
        'status': normalize_status((state or {}).get('state') or old.get('status') or 'incomplete'),
        'elapsed': safe_int((state or {}).get('minute') or old.get('elapsed'), 0),
        'boss_ai_decision': old.get('boss_ai_decision') or '',
        'prematch_comment': old.get('prematch_comment') or '',
        'live_comment': old.get('live_comment') or '',
        'lineups': old.get('lineups') or [],
        'events': old.get('events') or [],
        'statistics': old.get('statistics') or [],
    }


def fetch_round_odds(round_id: int, cache: Dict[int, Any]) -> Dict[str, Any]:
    if not SM_KEY or not round_id:
        return {}
    if round_id in cache:
        return cache[round_id]
    url = f"https://api.sportmonks.com/v3/football/rounds/{round_id}?api_token={SM_KEY}&include={ODDS_INCLUDE}&filters={ODDS_FILTERS}"
    try:
        data = fetch_json(url).get('data', {})
    except Exception as e:
        log(f'⚠️ Odds alınamadı round={round_id}: {e}')
        data = {}
    cache[round_id] = data
    return data


def odds_for_fixture(round_data: Dict[str, Any], fixture_id: int) -> List[Dict[str, Any]]:
    for f in round_data.get('fixtures') or []:
        if safe_int(f.get('id')) == fixture_id:
            return f.get('odds') or []
    return []


def fetch_standings(season_id: int, cache: Dict[int, Any]) -> Any:
    if not SM_KEY or not season_id:
        return []
    if season_id in cache:
        return cache[season_id]
    url = f"https://api.sportmonks.com/v3/football/standings/seasons/{season_id}?api_token={SM_KEY}&include={STANDINGS_INCLUDE}"
    try:
        data = fetch_json(url).get('data', []) or []
    except Exception as e:
        log(f'⚠️ Standings alınamadı season={season_id}: {e}')
        data = []
    cache[season_id] = data
    return data


def fetch_live_standings(round_id: int, cache: Dict[int, Any]) -> Any:
    if not SM_KEY or not round_id:
        return []
    if round_id in cache:
        return cache[round_id]
    url = f"https://api.sportmonks.com/v3/football/standings/rounds/{round_id}?api_token={SM_KEY}&include={LIVE_STANDINGS_INCLUDE}"
    try:
        data = fetch_json(url).get('data', []) or []
    except Exception as e:
        log(f'⚠️ Live standings alınamadı round={round_id}: {e}')
        data = []
    cache[round_id] = data
    return data


def fetch_h2h(home_id: int, away_id: int, cache: Dict[str, Any]) -> Any:
    if not SM_KEY or not home_id or not away_id:
        return []
    key = f'{home_id}:{away_id}'
    if key in cache:
        return cache[key]
    url = f"https://api.sportmonks.com/v3/football/fixtures/head-to-head/{home_id}/{away_id}?api_token={SM_KEY}&include={H2H_INCLUDE}"
    try:
        data = fetch_json(url).get('data', []) or []
    except Exception as e:
        log(f'⚠️ H2H alınamadı {key}: {e}')
        data = []
    cache[key] = data
    return data


def fetch_fixture_detail(fid: int) -> Dict[str, Any]:
    if not SM_KEY or not fid:
        return {}
    merged = {}
    for include in [DETAIL_BASIC_INCLUDE] + DETAIL_BLOCKS:
        url = f"https://api.sportmonks.com/v3/football/fixtures/{fid}?api_token={SM_KEY}&include={include}"
        try:
            data = fetch_json(url).get('data', {}) or {}
            merged = merge_value(merged, data)
        except Exception as e:
            log(f'⚠️ Fixture detail blok alınamadı fixture={fid} include={include}: {e}')
    return merged


def enrich_sm_row(row: Dict[str, Any], bundle_fixture: Dict[str, Any]):
    detail = bundle_fixture.get('detail') or {}
    parts = detail.get('participants') or []
    home, away = get_side_participants(parts)
    if home.get('image_path') and not row.get('home_image'):
        row['home_image'] = home.get('image_path')
    if away.get('image_path') and not row.get('away_image'):
        row['away_image'] = away.get('image_path')
    row['home_country'] = row.get('home_country') or country_from_image_path(row.get('home_image'))
    row['away_country'] = row.get('away_country') or country_from_image_path(row.get('away_image'))

    state = detail.get('state') or {}
    row['elapsed'] = safe_int(state.get('minute'), row.get('elapsed', 0))
    row['status'] = normalize_status(state.get('state') or row.get('status'))

    row['lineups'] = detail.get('lineups') or row.get('lineups') or []
    row['events'] = detail.get('events') or row.get('events') or []
    row['statistics'] = detail.get('statistics') or row.get('statistics') or []

    odds = bundle_fixture.get('odds') or []
    if not row.get('odds_ft_1'):
        row['odds_ft_1'] = odds_value(odds, 'FULLTIME_RESULT', '1')
    if not row.get('odds_ft_x'):
        row['odds_ft_x'] = odds_value(odds, 'FULLTIME_RESULT', 'x')
    if not row.get('odds_ft_2'):
        row['odds_ft_2'] = odds_value(odds, 'FULLTIME_RESULT', '2')
    if not row.get('odds_ft_over25'):
        row['odds_ft_over25'] = odds_value(odds, 'GOALS_OVER_UNDER', 'over 2.5')
    if not row.get('odds_btts_yes'):
        row['odds_btts_yes'] = odds_value(odds, 'BOTH_TEAMS_TO_SCORE', 'yes')

    preds = summarize_predictions(detail.get('predictions') or [])
    if not row.get('o25_potential') and preds.get('over25'):
        row['o25_potential'] = preds['over25']
    if not row.get('btts_potential') and preds.get('btts'):
        row['btts_potential'] = preds['btts']
    if not row.get('o05HT_potential') and preds.get('firsthalf'):
        row['o05HT_potential'] = preds['firsthalf']

    if not row.get('prematch_comment') or row.get('prematch_comment') == 'AI yorumu henüz yok.':
        row['prematch_comment'] = heur_ai_prematch(row, detail, odds)
    if row.get('prematch_comment'):
        row['boss_ai_decision'] = row.get('prematch_comment')


def fetch_footystats_for_date(date_str: str) -> List[Dict[str, Any]]:
    if not FS_KEY:
        return []
    url = f'https://api.football-data-api.com/todays-matches?key={FS_KEY}&date={date_str}&include=stats,odds'
    try:
        return fetch_json(url).get('data', []) or []
    except Exception as e:
        log(f'⚠️ FootyStats alınamadı date={date_str}: {e}')
        return []


def save_source_tabs():
    save_json(SOURCE_TABS_JSON, {
        'updated_at': now_iso(),
        'sources': [
            {
                'id': 'footystats',
                'label': 'FootyStats',
                'today_file': 'footystats_today.json',
                'tomorrow_file': 'footystats_tomorrow.json',
                'live_file': 'footystats_live.json',
            },
            {
                'id': 'sportmonks',
                'label': 'Sportmonks',
                'today_file': 'sportmonks_today.json',
                'tomorrow_file': 'sportmonks_tomorrow.json',
                'live_file': 'sportmonks_live.json',
            },
        ],
    })


def main():
    ensure_dir()
    started = time.time()
    health = load_json(HEALTH_JSON, {})
    health.update({
        'split_runner': 'fetch_split_sources',
        'split_started_at': now_iso(),
        'footystats_today_count': 0,
        'footystats_tomorrow_count': 0,
        'sportmonks_today_count': 0,
        'sportmonks_tomorrow_count': 0,
        'sportmonks_bundle_details_written': 0,
        'footystats_ai_written': 0,
        'sportmonks_ai_written': 0,
        'split_errors': [],
    })

    client = init_vertex_client()

    fs_old_today = {str(x.get('id')): x for x in load_json(FS_TODAY_JSON, {}).get('data', []) if x.get('id')}
    fs_old_tomorrow = {str(x.get('id')): x for x in load_json(FS_TOMORROW_JSON, {}).get('data', []) if x.get('id')}
    sm_old_today = {str((x.get('source_ids') or {}).get('sportmonks') or x.get('id')): x for x in load_json(SM_TODAY_JSON, {}).get('data', [])}
    sm_old_tomorrow = {str((x.get('source_ids') or {}).get('sportmonks') or x.get('id')): x for x in load_json(SM_TOMORROW_JSON, {}).get('data', [])}
    fs_bundle = load_json(FS_BUNDLE_JSON, {'generated_at': now_iso(), 'fixtures': {}})
    sm_bundle = load_json(SM_BUNDLE_JSON, {'generated_at': now_iso(), 'fixtures': {}})

    fs_today_raw = fetch_footystats_for_date(iso_date(0))
    fs_tomorrow_raw = fetch_footystats_for_date(iso_date(1))
    health['footystats_today_count'] = len(fs_today_raw)
    health['footystats_tomorrow_count'] = len(fs_tomorrow_raw)

    fs_today, fs_tomorrow = [], []
    for raw, old_map, out in ((fs_today_raw, fs_old_today, fs_today), (fs_tomorrow_raw, fs_old_tomorrow, fs_tomorrow)):
        for fs in raw:
            old = old_map.get(str(fs.get('id')))
            row = build_fs_row(fs, old)
            
            old_comment = (old or {}).get('prematch_comment', '')
            if old_comment and old_comment != 'AI yorumu henüz yok.':
                row['prematch_comment'] = old_comment
                row['boss_ai_decision'] = (old or {}).get('boss_ai_decision') or old_comment
            elif not row.get('prematch_comment') or row.get('prematch_comment') == 'AI yorumu henüz yok.':
                comment = ai_comment_prematch(client, row, {}, [])
                if comment:
                    row['prematch_comment'] = comment
                    row['boss_ai_decision'] = comment
                    health['footystats_ai_written'] += 1
            
            out.append(row)
            fs_bundle.setdefault('fixtures', {})[str(row.get('id'))] = {'raw': fs, 'fetched_at': now_iso()}

    sm_today_raw, sm_tomorrow_raw = [], []
    if SM_KEY:
        try:
            sm_today_raw = fetch_json(f'https://api.sportmonks.com/v3/football/fixtures/date/{iso_date(0)}?api_token={SM_KEY}&include={FIXTURE_BASIC_INCLUDE}').get('data', [])
        except Exception as e:
            health['split_errors'].append(f'sportmonks_today: {e}')
        try:
            sm_tomorrow_raw = fetch_json(f'https://api.sportmonks.com/v3/football/fixtures/date/{iso_date(1)}?api_token={SM_KEY}&include={FIXTURE_BASIC_INCLUDE}').get('data', [])
        except Exception as e:
            health['split_errors'].append(f'sportmonks_tomorrow: {e}')
    health['sportmonks_today_count'] = len(sm_today_raw)
    health['sportmonks_tomorrow_count'] = len(sm_tomorrow_raw)

    sm_today = [build_sm_row(sm, sm_old_today.get(str(sm.get('id')))) for sm in sm_today_raw]
    sm_tomorrow = [build_sm_row(sm, sm_old_tomorrow.get(str(sm.get('id')))) for sm in sm_tomorrow_raw]

    odds_cache, standings_cache, live_standings_cache, h2h_cache = {}, {}, {}, {}
    for row in sm_today + sm_tomorrow:
        sid = safe_int((row.get('source_ids') or {}).get('sportmonks'), 0)
        if not sid:
            continue
        cached = sm_bundle.get('fixtures', {}).get(str(sid)) or {}
        detail = cached.get('detail') or fetch_fixture_detail(sid)
        if not detail:
            continue
        round_id = safe_int((detail.get('round') or {}).get('id') or detail.get('round_id'), 0)
        season_id = safe_int(detail.get('season_id'), 0)
        participants = detail.get('participants') or []
        home, away = get_side_participants(participants)
        odds = cached.get('odds') or (odds_for_fixture(fetch_round_odds(round_id, odds_cache), sid) if round_id else [])
        standings = cached.get('standings') or (fetch_standings(season_id, standings_cache) if season_id else [])
        live_standings = cached.get('live_standings') or (fetch_live_standings(round_id, live_standings_cache) if round_id else [])
        h2h = cached.get('h2h') or (fetch_h2h(safe_int(home.get('id'), 0), safe_int(away.get('id'), 0), h2h_cache) if home and away else [])
        
        sm_bundle.setdefault('fixtures', {})[str(sid)] = {
            'fetched_at': now_iso(),
            'detail': detail,
            'odds': odds,
            'standings': standings,
            'live_standings': live_standings,
            'h2h': h2h,
        }
        enrich_sm_row(row, sm_bundle['fixtures'][str(sid)])
        
        old_sm_comment = (sm_old_today.get(str(sid)) or sm_old_tomorrow.get(str(sid)) or {}).get('prematch_comment', '')
        if old_sm_comment and old_sm_comment != 'AI yorumu henüz yok.':
            row['prematch_comment'] = old_sm_comment
            row['boss_ai_decision'] = old_sm_comment
        elif not row.get('prematch_comment') or row.get('prematch_comment') == 'AI yorumu henüz yok.':
            row['prematch_comment'] = ai_comment_prematch(client, row, detail, odds)
            if row.get('prematch_comment'):
                row['boss_ai_decision'] = row.get('boss_ai_decision') or row['prematch_comment']
                health['sportmonks_ai_written'] += 1
                
        health['sportmonks_bundle_details_written'] += 1

    cutoff = time.time() - 48 * 3600
    for bundle_obj in (fs_bundle, sm_bundle):
        old_count = len(bundle_obj.get('fixtures', {}))
        bundle_obj['fixtures'] = {
            k: v for k, v in bundle_obj.get('fixtures', {}).items()
            if safe_int(
                (v.get('detail') or {}).get('starting_at_timestamp') or
                (v.get('raw') or {}).get('date_unix') or
                cutoff + 1
            ) > cutoff
        }
        removed = old_count - len(bundle_obj['fixtures'])
        if removed:
            log(f'🗑️ Bundle temizlendi: {removed} eski fixture silindi')

    save_json(FS_TODAY_JSON, {'source': 'footystats', 'day': 'today', 'updated_at': now_iso(), 'data': fs_today})
    save_json(FS_TOMORROW_JSON, {'source': 'footystats', 'day': 'tomorrow', 'updated_at': now_iso(), 'data': fs_tomorrow})
    save_json(SM_TODAY_JSON, {'source': 'sportmonks', 'day': 'today', 'updated_at': now_iso(), 'data': sm_today})
    save_json(SM_TOMORROW_JSON, {'source': 'sportmonks', 'day': 'tomorrow', 'updated_at': now_iso(), 'data': sm_tomorrow})
    save_json(FS_BUNDLE_JSON, fs_bundle)
    save_json(SM_BUNDLE_JSON, sm_bundle)
    save_source_tabs()

    health['split_finished_at'] = now_iso()
    health['split_duration_sec'] = round(time.time() - started, 2)
    save_json(HEALTH_JSON, health)
    log(f'✅ footystats_today.json {len(fs_today)}')
    log(f'✅ footystats_tomorrow.json {len(fs_tomorrow)}')
    log(f'✅ sportmonks_today.json {len(sm_today)}')
    log(f'✅ sportmonks_tomorrow.json {len(sm_tomorrow)}')

if __name__ == '__main__':
    main()
