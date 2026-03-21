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
SOURCE_TABS_JSON = os.path.join(DATA_DIR, 'source_tabs.json')
HEALTH_JSON = os.path.join(DATA_DIR, 'health.json')

FS_KEY = os.getenv('FOOTYSTATS_KEY', '').strip()
SM_KEY = os.getenv('SPORTMONKS_KEY', '').strip()
GEMINI_MODEL = (os.getenv('GEMINI_MODEL') or 'gemini-2.5-flash').strip()
REQUEST_TIMEOUT = 25
SM_BASE_INCLUDE = 'participants;league.country;venue;referees;weatherReport;state;scores;periods;round'
PER_PAGE = 50


def log(msg: str):
    print(msg, flush=True)


def ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


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


def fetch_json(url: str) -> Dict[str, Any]:
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_all_pages(url: str) -> List[Dict[str, Any]]:
    page = 1
    out: List[Dict[str, Any]] = []
    while True:
        sep = '&' if '?' in url else '?'
        data = fetch_json(f'{url}{sep}page={page}&per_page={PER_PAGE}')
        out.extend(data.get('data', []) or [])
        pagination = data.get('pagination') or data.get('meta', {}).get('pagination') or {}
        has_more = bool(pagination.get('has_more'))
        current_page = safe_int(pagination.get('current_page') or page, page)
        total_pages = safe_int(pagination.get('total_pages') or current_page, current_page)
        if not has_more and current_page >= total_pages:
            break
        page += 1
        if page > 100:
            break
    return out


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


def iso_date(offset: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=offset)).strftime('%Y-%m-%d')


def country_from_image_path(path: str) -> str:
    if not path:
        return ''
    last = path.split('/')[-1]
    if '-' not in last:
        return ''
    return last.split('-')[0].replace('_', ' ').strip().title()


def clean_name(name: str) -> str:
    n = unidecode(str(name or '')).lower()
    n = re.sub(r'[\W_]+', '', n)
    for token in ['footballclub', 'futebolclube', 'clubdefutbol', 'clubdeportivo', 'women', 'ladies', 'reserves', 'reserve', 'ii', 'iii', 'u21', 'u23', 'fc', 'cf', 'ac', 'afc', 'sc', 'sk', 'if', 'fk', 'bk', 'nk', 'cd', 'de', 'la', 'the']:
        n = n.replace(token, '')
    return n


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


def current_score(fixture: Dict[str, Any]) -> Tuple[int, int]:
    scores = fixture.get('scores') or []
    home, away = 0, 0
    for s in scores:
        if str(s.get('description') or '').upper() != 'CURRENT':
            continue
        participant = (s.get('score') or {}).get('participant')
        goals = safe_int((s.get('score') or {}).get('goals'), 0)
        if participant == 'home':
            home = goals
        elif participant == 'away':
            away = goals
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


def heur_ai_prematch(row: Dict[str, Any]) -> str:
    hppg = safe_float(row.get('home_ppg'))
    appg = safe_float(row.get('away_ppg'))
    hxg = safe_float(row.get('team_a_xg_prematch'))
    axg = safe_float(row.get('team_b_xg_prematch'))
    btts = safe_float(row.get('btts_potential'))
    o25 = safe_float(row.get('o25_potential'))
    iy = safe_float(row.get('o05HT_potential'))
    oh = safe_float(row.get('odds_ft_1'))
    oa = safe_float(row.get('odds_ft_2'))

    home_edge = (hppg - appg) + (hxg - axg)
    away_edge = (appg - hppg) + (axg - hxg)

    if home_edge <= 0 and away_edge <= 0 and not o25 and not btts and not iy:
        if oh and oa:
            if oh < oa:
                return f"DURUM: {row.get('home_name')} oran avantajına sahip.\nNEDEN: Piyasa ev sahibini hafif önde fiyatlıyor.\nSONUÇ: Ev sahibi kaybetmez tarafı daha güvenli duruyor."
            if oa < oh:
                return f"DURUM: {row.get('away_name')} oran avantajına sahip.\nNEDEN: Piyasa deplasman tarafını hafif önde fiyatlıyor.\nSONUÇ: Deplasman kaybetmez tarafı daha güvenli duruyor."
        return f"DURUM: {row.get('home_name')} - {row.get('away_name')} maçı için veri sınırlı.\nNEDEN: Güçlü model sinyali oluşmadı.\nSONUÇ: Şimdilik pas geçmek daha sağlıklı."

    if o25 >= 62 or btts >= 60:
        neden = []
        if o25 >= 62:
            neden.append(f"Üst 2.5 %{round(o25)}")
        if btts >= 60:
            neden.append(f"KG %{round(btts)}")
        if iy >= 68:
            neden.append(f"İY gol %{round(iy)}")
        sonuc = 'Üst 2.5' if o25 >= btts else 'KG Var'
        return f"DURUM: Maç gollü profile yakın.\nNEDEN: {', '.join(neden)} destek veriyor.\nSONUÇ: {sonuc} tarafı öncelikli."
    if home_edge > 0.28:
        return f"DURUM: {row.get('home_name')} tarafı önde.\nNEDEN: PPG/xG verisi ev sahibini destekliyor.\nSONUÇ: MS1 veya ev sahibi kaybetmez seçenekleri önde."
    if away_edge > 0.28:
        return f"DURUM: {row.get('away_name')} tarafı önde.\nNEDEN: PPG/xG verisi deplasmanı destekliyor.\nSONUÇ: MS2 veya deplasman kaybetmez seçenekleri önde."
    if iy >= 68:
        return f"DURUM: İlk yarıda tempo bekleniyor.\nNEDEN: İlk yarı gol olasılığı %{round(iy)} seviyesinde.\nSONUÇ: İY 0.5 üst tarafı izlenebilir."
    return 'AI yorumu henüz yok.'


def ai_comment_prematch(client, match: Dict[str, Any]) -> str:
    heuristic = heur_ai_prematch(match)
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
        'oddsx': safe_float(match.get('odds_ft_x')),
        'odds2': safe_float(match.get('odds_ft_2')),
    }
    prompt = '\n'.join([
        'Sen kısa ve teknik prematch bahis analiz motorusun.',
        'Kurallar:',
        '- En fazla 85 kelime.',
        '- 3 satır yaz: DURUM, NEDEN, SONUÇ.',
        '- Veri yetersizse tam olarak "AI yorumu henüz yok." yaz.',
        json.dumps(payload, ensure_ascii=False)
    ])
    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = (getattr(response, 'text', '') or '').strip()
        return text or heuristic
    except Exception as e:
        log(f'⚠️ Prematch AI hatası: {e}')
        return heuristic


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
        'stadium_name': '',
        'stadium_location': '',
        'referee_name': '',
        'weather': {},
        'home_ppg': safe_float(fs.get('home_ppg') or fs.get('team_a_ppg')),
        'away_ppg': safe_float(fs.get('away_ppg') or fs.get('team_b_ppg')),
        'team_a_ppg': fs.get('team_a_ppg') or str(safe_float(fs.get('home_ppg'))),
        'team_b_ppg': fs.get('team_b_ppg') or str(safe_float(fs.get('away_ppg'))),
        'pre_match_home_ppg': safe_float(fs.get('pre_match_home_ppg')),
        'pre_match_away_ppg': safe_float(fs.get('pre_match_away_ppg')),
        'team_a_xg_prematch': safe_float(fs.get('team_a_xg_prematch')),
        'team_b_xg_prematch': safe_float(fs.get('team_b_xg_prematch')),
        'btts_potential': safe_float(fs.get('btts_potential')),
        'o25_potential': safe_float(fs.get('o25_potential')),
        'o05HT_potential': safe_float(fs.get('o05HT_potential')),
        'pressure_score': 0.0,
        'odds_ft_1': safe_float(fs.get('odds_ft_1')),
        'odds_ft_x': safe_float(fs.get('odds_ft_x')),
        'odds_ft_2': safe_float(fs.get('odds_ft_2')),
        'odds_ft_over25': safe_float(fs.get('odds_ft_over25')),
        'odds_btts_yes': safe_float(fs.get('odds_btts_yes')),
        'odds_1st_half_over05': safe_float(fs.get('odds_1st_half_over05')),
        'homeGoalCount': safe_int(fs.get('homeGoalCount'), 0),
        'awayGoalCount': safe_int(fs.get('awayGoalCount'), 0),
        'status': fs.get('status') or 'incomplete',
        'elapsed': safe_int(fs.get('elapsed'), 0),
        'boss_ai_decision': '',
        'prematch_comment': '',
        'live_comment': '',
    }
    row['prematch_comment'] = ai_comment_prematch(client, row)
    row['boss_ai_decision'] = row['prematch_comment']
    return row


def normalize_sm_row(sm: Dict[str, Any], client=None) -> Dict[str, Any]:
    base = extract_sm_basic(sm)
    row = {
        'id': f"sm-{base['sportmonks_id']}",
        'source_ids': {'footystats': '', 'sportmonks': str(base['sportmonks_id'])},
        'home_name': base['home_name'] or 'Ev Sahibi',
        'away_name': base['away_name'] or 'Deplasman',
        'competition_name': base['competition_name'] or '',
        'competition_id': base['competition_id'],
        'league_country': base['league_country'],
        'league_country_image': base['league_country_image'],
        'date_unix': base['date_unix'],
        'home_image': base['home_image'],
        'away_image': base['away_image'],
        'home_country': country_from_image_path(base['home_image']),
        'away_country': country_from_image_path(base['away_image']),
        'stadium_name': base['stadium_name'],
        'stadium_location': base['stadium_location'],
        'referee_name': base['referee_name'],
        'weather': base['weather'],
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
        'homeGoalCount': base['homeGoalCount'],
        'awayGoalCount': base['awayGoalCount'],
        'status': base['status'],
        'elapsed': base['elapsed'],
        'boss_ai_decision': '',
        'prematch_comment': '',
        'live_comment': '',
    }
    row['prematch_comment'] = ai_comment_prematch(client, row)
    row['boss_ai_decision'] = row['prematch_comment']
    return row


def fetch_footystats_for_date(date_str: str) -> List[Dict[str, Any]]:
    if not FS_KEY:
        return []
    url = f'https://api.football-data-api.com/todays-matches?key={FS_KEY}&date={date_str}&include=stats,odds'
    return fetch_json(url).get('data', []) or []


def fetch_sportmonks_for_date(date_str: str) -> List[Dict[str, Any]]:
    if not SM_KEY:
        return []
    url = f'https://api.sportmonks.com/v3/football/fixtures/date/{date_str}?api_token={SM_KEY}&include={SM_BASE_INCLUDE}'
    return fetch_all_pages(url)


def main():
    ensure_dir()
    client = init_vertex_client()
    health = {
        'runner': 'fetch_split_sources_optimized',
        'started_at': datetime.utcnow().isoformat() + 'Z',
        'errors': [],
    }

    today_str = iso_date(0)
    tomorrow_str = iso_date(1)

    fs_today_raw = []
    fs_tomorrow_raw = []
    sm_today_raw = []
    sm_tomorrow_raw = []

    try:
        fs_today_raw = fetch_footystats_for_date(today_str)
    except Exception as e:
        health['errors'].append(f'footystats_today: {e}')
    try:
        fs_tomorrow_raw = fetch_footystats_for_date(tomorrow_str)
    except Exception as e:
        health['errors'].append(f'footystats_tomorrow: {e}')
    try:
        sm_today_raw = fetch_sportmonks_for_date(today_str)
    except Exception as e:
        health['errors'].append(f'sportmonks_today: {e}')
    try:
        sm_tomorrow_raw = fetch_sportmonks_for_date(tomorrow_str)
    except Exception as e:
        health['errors'].append(f'sportmonks_tomorrow: {e}')

    fs_today = [normalize_fs_row(x, client) for x in fs_today_raw]
    fs_tomorrow = [normalize_fs_row(x, client) for x in fs_tomorrow_raw]
    sm_today = [normalize_sm_row(x, client) for x in sm_today_raw]
    sm_tomorrow = [normalize_sm_row(x, client) for x in sm_tomorrow_raw]

    # FootyStats live: bugün feed içinden canlı filtre
    fs_live = [x for x in fs_today if str(x.get('status', '')).lower() in ('live', 'inplay', 'ht') or safe_int(x.get('elapsed'), 0) > 0]

    save_json(FOOTYSTATS_TODAY_JSON, {'data': fs_today, 'updated_at': datetime.utcnow().isoformat() + 'Z'})
    save_json(FOOTYSTATS_TOMORROW_JSON, {'data': fs_tomorrow, 'updated_at': datetime.utcnow().isoformat() + 'Z'})
    save_json(FOOTYSTATS_LIVE_JSON, {'data': fs_live, 'updated_at': datetime.utcnow().isoformat() + 'Z'})
    save_json(SPORTMONKS_TODAY_JSON, {'data': sm_today, 'updated_at': datetime.utcnow().isoformat() + 'Z'})
    save_json(SPORTMONKS_TOMORROW_JSON, {'data': sm_tomorrow, 'updated_at': datetime.utcnow().isoformat() + 'Z'})
    save_json(SOURCE_TABS_JSON, {
        'tabs': [
            {
                'key': 'footystats',
                'label': 'FootyStats',
                'today_file': 'footystats_today.json',
                'tomorrow_file': 'footystats_tomorrow.json',
                'live_file': 'footystats_live.json',
            },
            {
                'key': 'sportmonks',
                'label': 'Sportmonks',
                'today_file': 'sportmonks_today.json',
                'tomorrow_file': 'sportmonks_tomorrow.json',
                'live_file': 'sportmonks_live.json',
            },
        ],
        'updated_at': datetime.utcnow().isoformat() + 'Z'
    })

    health.update({
        'finished_at': datetime.utcnow().isoformat() + 'Z',
        'footystats_today': len(fs_today),
        'footystats_tomorrow': len(fs_tomorrow),
        'footystats_live': len(fs_live),
        'sportmonks_today': len(sm_today),
        'sportmonks_tomorrow': len(sm_tomorrow),
    })
    save_json(HEALTH_JSON, health)
    log(f'✅ footystats_today {len(fs_today)}')
    log(f'✅ footystats_tomorrow {len(fs_tomorrow)}')
    log(f'✅ footystats_live {len(fs_live)}')
    log(f'✅ sportmonks_today {len(sm_today)}')
    log(f'✅ sportmonks_tomorrow {len(sm_tomorrow)}')


if __name__ == '__main__':
    main()
