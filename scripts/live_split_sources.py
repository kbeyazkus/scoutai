#!/usr/bin/env python3
import json, os, re, time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from unidecode import unidecode

try:
    from google import genai
except Exception:
    genai = None

DATA_DIR = 'data'
FS_TODAY_JSON = os.path.join(DATA_DIR, 'footystats_today.json')
FS_LIVE_JSON = os.path.join(DATA_DIR, 'footystats_live.json')
SM_LIVE_JSON = os.path.join(DATA_DIR, 'sportmonks_live.json')
SM_TODAY_JSON = os.path.join(DATA_DIR, 'sportmonks_today.json')
FS_BUNDLE_JSON = os.path.join(DATA_DIR, 'footystats_bundle.json')
SM_BUNDLE_JSON = os.path.join(DATA_DIR, 'sportmonks_bundle.json')
HEALTH_JSON = os.path.join(DATA_DIR, 'health.json')
SOURCE_TABS_JSON = os.path.join(DATA_DIR, 'source_tabs.json')

FS_KEY = os.getenv('FOOTYSTATS_KEY', '').strip()
SM_KEY = os.getenv('SPORTMONKS_KEY', '').strip()
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash').strip()
REQUEST_TIMEOUT = 25

LIVE_INCLUDE = 'participants;scores;periods;events;league.country;round;state'
DETAIL_BASIC_INCLUDE = 'participants;league.country;venue;state;scores;periods'
DETAIL_BLOCKS = [
    'events.type;events.period;events.player',
    'statistics.type;xGFixture.type;predictions.type',
    'lineups.player;lineups.type;lineups.details.type;metadata.type;coaches',
    'sidelined.sideline.player;sidelined.sideline.type;weatherReport;comments',
    'pressure.participant;trends.type;trends.participant',
    'prematchNews.lines;postmatchNews.lines',
]


def log(msg: str):
    print(msg, flush=True)


def now_iso() -> str:
    return datetime.utcnow().isoformat() + 'Z'


def safe_int(v: Any, default: int = 0) -> int:
    try:
        if v in (None, ''):
            return default
        return int(float(v))
    except Exception:
        return default


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v in (None, ''):
            return default
        return float(v)
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


def fetch_json(url: str, retries: int = 3) -> Dict[str, Any]:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
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
    n = re.sub(r'[\W_]+', '', n)
    for token in ['footballclub', 'futebolclube', 'clubdefutbol', 'clubdeportivo', 'women', 'ladies', 'reserves', 'reserve', 'ii', 'iii', 'u21', 'u23', 'fc', 'cf', 'ac', 'afc', 'sc', 'sk', 'if', 'fk', 'bk', 'nk', 'cd', 'de', 'la', 'the']:
        n = n.replace(token, '')
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


def normalize_status(v: Any) -> str:
    s = str(v or '').strip().lower()
    if not s:
        return 'incomplete'
    if s in ('not started', 'ns', 'scheduled', 'pre-match', 'prematch'):
        return 'incomplete'
    return s


def current_score(fixture: Dict[str, Any]) -> Tuple[int, int]:
    scores = fixture.get('scores') or []
    home, away = None, None
    for s in scores:
        if str(s.get('description') or '').upper() == 'CURRENT':
            participant = s.get('score', {}).get('participant')
            goals = safe_int(s.get('score', {}).get('goals'), 0)
            if participant == 'home':
                home = goals
            elif participant == 'away':
                away = goals
    if home is not None and away is not None:
        return home, away
    latest, latest_ord = None, -1
    for ev in fixture.get('events') or []:
        result = ev.get('result')
        total = safe_int(ev.get('minute'), 0) * 100 + safe_int(ev.get('extra_minute'), 0)
        if result and total >= latest_ord:
            latest_ord = total
            latest = result
    if latest:
        m = re.match(r'^\s*(\d+)\s*-\s*(\d+)\s*$', str(latest))
        if m:
            return int(m.group(1)), int(m.group(2))
    return 0, 0


def extract_minute_sm(fixture: Dict[str, Any]) -> int:
    state = fixture.get('state') or {}
    if state.get('minute') is not None:
        return safe_int(state.get('minute'), 0)
    best = 0
    for p in fixture.get('periods') or []:
        best = max(best, safe_int(p.get('minutes'), 0), safe_int(p.get('minute'), 0))
    for ev in fixture.get('events') or []:
        total = safe_int(ev.get('minute'), 0) + (1 if safe_int(ev.get('extra_minute'), 0) > 0 else 0)
        best = max(best, total)
    if not best:
        st = safe_int(fixture.get('starting_at_timestamp'), 0)
        if st:
            elapsed = int((time.time() - st) // 60)
            if elapsed > 0:
                if elapsed <= 45:
                    best = elapsed
                elif elapsed <= 60:
                    best = 45
                elif elapsed <= 105:
                    best = min(elapsed - 15, 90)
                else:
                    best = 90
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


def heur_live_comment(row:Dict[str,Any], detail:Dict[str,Any])->str:
    h = safe_int(row.get('homeGoalCount'),0)
    a = safe_int(row.get('awayGoalCount'),0)
    minute = safe_int(row.get('elapsed'),0)
    pressure = safe_float(row.get('pressure_score'),0)
    preds = summarize_predictions(detail.get('predictions') or [])
    shots_home = shots_away = 0
    for st in detail.get('statistics') or []:
        dev = str((st.get('type') or {}).get('developer_name') or '').upper()
        if dev == 'SHOTS_ON_TARGET':
            if st.get('location') == 'home':
                shots_home = safe_int((st.get('data') or {}).get('value'))
            elif st.get('location') == 'away':
                shots_away = safe_int((st.get('data') or {}).get('value'))
    if minute <= 0 and not pressure and not shots_home and not shots_away:
        return f"DURUM: Maç canlı izleniyor.\nNEDEN: Detay veri sınırlı ama skor {h}-{a}.\nSONUÇ: Şimdilik pas daha sağlıklı."
    if h == a:
        if pressure >= 8 or shots_home >= shots_away + 2:
            return f"DURUM: Skor dengede ama ev sahibi baskı kuruyor.\nNEDEN: Baskı farkı ve isabetli şut üstünlüğü ev tarafında.\nSONUÇ: Ev yönlü gol veya üst tarafı izlenebilir."
        if pressure <= -8 or shots_away >= shots_home + 2:
            return f"DURUM: Skor dengede ama deplasman baskı kuruyor.\nNEDEN: Baskı farkı ve isabetli şut üstünlüğü deplasman tarafında.\nSONUÇ: Deplasman yönlü gol veya üst tarafı izlenebilir."
        if preds.get('over25') >= 60 or preds.get('btts') >= 60:
            return f"DURUM: Skor dengede.\nNEDEN: Modelde gollü maç eğilimi korunuyor.\nSONUÇ: Üst veya KG tarafı izlenebilir."
        return "DURUM: Maç dengede gidiyor.\nNEDEN: Net baskı üstünlüğü oluşmadı.\nSONUÇ: Şimdilik pas daha sağlıklı."
    leader = row.get('home_name') if h > a else row.get('away_name')
    trailer = row.get('away_name') if h > a else row.get('home_name')
    if abs(pressure) >= 8:
        side = 'ev sahibi' if pressure > 0 else 'deplasman'
        return f"DURUM: {leader} önde, baskı ise {side} tarafında.\nNEDEN: Oyun temposu ve baskı verisi maçın hâlâ açık olduğunu gösteriyor.\nSONUÇ: Ek gol ihtimali canlı takip için uygun."
    return f"DURUM: {leader} skor üstünlüğünü aldı.\nNEDEN: Maç {minute}. dakikada ve {trailer} tarafı net tepki üretmedi.\nSONUÇ: Şu aşamada önde olan taraf lehine senaryo korunuyor."


def ai_comment_live(client, row:Dict[str,Any], detail:Dict[str,Any])->str:
    heuristic = heur_live_comment(row, detail)
    if client is None:
        return heuristic
    payload={
        'home':row.get('home_name'),'away':row.get('away_name'),'league':row.get('competition_name'),'minute':safe_int(row.get('elapsed'),0),
        'score_home':safe_int(row.get('homeGoalCount'),0),'score_away':safe_int(row.get('awayGoalCount'),0),
        'pressure': safe_float(row.get('pressure_score'),0),
        'shots': [s for s in detail.get('statistics',[]) if (s.get('type') or {}).get('developer_name') in ('SHOTS_TOTAL','SHOTS_ON_TARGET')],
        'prediction': detail.get('predictions',[])[:4],
    }
    prompt='\n'.join([
        'Sen kısa ve teknik canlı bahis analiz motorusun.',
        'Kurallar: en fazla 85 kelime. 3 satır yaz: DURUM, NEDEN, SONUÇ. Veri yetersizse "AI yorumu henüz yok." yaz.',
        json.dumps(payload, ensure_ascii=False)
    ])
    try:
        time.sleep(1.5)
        response=client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text=(getattr(response,'text','') or '').strip()
        return text or heuristic
    except Exception as e:
        if '429' in str(e) or 'quota' in str(e).lower() or 'rate' in str(e).lower():
            log(f'⚠️ Gemini rate limit, heuristic kullanılıyor: {e}')
        else:
            log(f'⚠️ Live AI hatası: {e}')
        return heuristic


def fetch_fixture_detail(fid: int) -> Dict[str, Any]:
    merged = {}
    for include in [DETAIL_BASIC_INCLUDE] + DETAIL_BLOCKS:
        url = f'https://api.sportmonks.com/v3/football/fixtures/{fid}?api_token={SM_KEY}&include={include}'
        try:
            data = fetch_json(url).get('data', {}) or {}
            merged = merge_value(merged, data)
        except Exception as e:
            log(f'⚠️ Live detail blok alınamadı fixture={fid} include={include}: {e}')
    return merged


def fetch_sportmonks_live_rows() -> List[Dict[str, Any]]:
    if not SM_KEY:
        return []
    try:
        url = f'https://api.sportmonks.com/v3/football/livescores/inplay?api_token={SM_KEY}&include={LIVE_INCLUDE}'
        return fetch_json(url).get('data', []) or []
    except Exception as e:
        log(f'⚠️ Sportmonks live rows alınamadı: {e}')
        return []


def infer_fs_live(raw: Dict[str, Any], old_row: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    old_row = old_row or {}
    status = normalize_status(raw.get('status') or old_row.get('status'))
    elapsed = safe_int(raw.get('elapsed') or raw.get('minute') or old_row.get('elapsed'), 0)
    home = safe_int(raw.get('homeGoalCount') or raw.get('team_a_goals') or raw.get('home_goals'), 0)
    away = safe_int(raw.get('awayGoalCount') or raw.get('team_b_goals') or raw.get('away_goals'), 0)
    is_live = status in ('live', 'inplay', '1h', '2h', 'ht') or elapsed > 0
    if not is_live:
        return None
    row = {
        'id': str(raw.get('id') or old_row.get('id') or ''),
        'source_ids': {'footystats': str(raw.get('id') or old_row.get('id') or ''), 'sportmonks': ''},
        'source': 'footystats',
        'home_name': raw.get('home_name') or old_row.get('home_name') or 'Ev Sahibi',
        'away_name': raw.get('away_name') or old_row.get('away_name') or 'Deplasman',
        'competition_name': raw.get('competition_name') or old_row.get('competition_name') or '',
        'competition_id': safe_int(raw.get('competition_id') or old_row.get('competition_id'), 0),
        'league_country': raw.get('league_country') or old_row.get('league_country') or '',
        'league_country_image': raw.get('league_country_image') or old_row.get('league_country_image') or '',
        'date_unix': safe_int(raw.get('date_unix') or old_row.get('date_unix'), 0),
        'home_image': raw.get('home_image') or old_row.get('home_image') or '',
        'away_image': raw.get('away_image') or old_row.get('away_image') or '',
        'home_country': old_row.get('home_country') or '',
        'away_country': old_row.get('away_country') or '',
        'stadium_name': raw.get('stadium_name') or old_row.get('stadium_name') or '',
        'stadium_location': raw.get('stadium_location') or old_row.get('stadium_location') or '',
        'referee_name': raw.get('referee_name') or old_row.get('referee_name') or '',
        'weather': raw.get('weather') or old_row.get('weather') or {},
        'home_ppg': safe_float(raw.get('home_ppg') or raw.get('team_a_ppg') or old_row.get('home_ppg')),
        'away_ppg': safe_float(raw.get('away_ppg') or raw.get('team_b_ppg') or old_row.get('away_ppg')),
        'team_a_ppg': raw.get('team_a_ppg') or old_row.get('team_a_ppg') or '0',
        'team_b_ppg': raw.get('team_b_ppg') or old_row.get('team_b_ppg') or '0',
        'team_a_xg_prematch': safe_float(raw.get('team_a_xg_prematch') or old_row.get('team_a_xg_prematch')),
        'team_b_xg_prematch': safe_float(raw.get('team_b_xg_prematch') or old_row.get('team_b_xg_prematch')),
        'btts_potential': safe_float(raw.get('btts_potential') or old_row.get('btts_potential')),
        'o25_potential': safe_float(raw.get('o25_potential') or old_row.get('o25_potential')),
        'o05HT_potential': safe_float(raw.get('o05HT_potential') or old_row.get('o05HT_potential')),
        'pressure_score': safe_float(raw.get('pressure_score') or old_row.get('pressure_score')),
        'odds_ft_1': safe_float(raw.get('odds_ft_1') or old_row.get('odds_ft_1')),
        'odds_ft_x': safe_float(raw.get('odds_ft_x') or old_row.get('odds_ft_x')),
        'odds_ft_2': safe_float(raw.get('odds_ft_2') or old_row.get('odds_ft_2')),
        'odds_ft_over25': safe_float(raw.get('odds_ft_over25') or old_row.get('odds_ft_over25')),
        'odds_btts_yes': safe_float(raw.get('odds_btts_yes') or old_row.get('odds_btts_yes')),
        'odds_1st_half_over05': safe_float(raw.get('odds_1st_half_over05') or old_row.get('odds_1st_half_over05')),
        'homeGoalCount': home,
        'awayGoalCount': away,
        'status': status,
        'elapsed': elapsed,
        'boss_ai_decision': old_row.get('boss_ai_decision') or '',
        'prematch_comment': old_row.get('prematch_comment') or '',
        'live_comment': old_row.get('live_comment') or '',
    }
    return row


def fetch_footystats_today_raw() -> List[Dict[str, Any]]:
    if not FS_KEY:
        return []
    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    url = f'https://api.football-data-api.com/todays-matches?key={FS_KEY}&date={date_str}&include=stats,odds'
    try:
        return fetch_json(url).get('data', []) or []
    except Exception as e:
        log(f'⚠️ FootyStats live feed alınamadı: {e}')
        return []


def save_source_tabs():
    save_json(SOURCE_TABS_JSON, {
        'updated_at': now_iso(),
        'sources': [
            {'id': 'footystats', 'label': 'FootyStats', 'today_file': 'footystats_today.json', 'tomorrow_file': 'footystats_tomorrow.json', 'live_file': 'footystats_live.json'},
            {'id': 'sportmonks', 'label': 'Sportmonks', 'today_file': 'sportmonks_today.json', 'tomorrow_file': 'sportmonks_tomorrow.json', 'live_file': 'sportmonks_live.json'},
        ],
    })


def main():
    started = time.time()
    client = init_vertex_client()
    health = load_json(HEALTH_JSON, {})
    fs_bundle = load_json(FS_BUNDLE_JSON, {'fixtures': {}})
    sm_bundle = load_json(SM_BUNDLE_JSON, {'fixtures': {}})
    fs_today_existing = {str(x.get('id')): x for x in load_json(FS_TODAY_JSON, {}).get('data', []) if x.get('id')}
    sm_today = load_json(SM_TODAY_JSON, {}).get('data', [])

    # FootyStats live
    fs_today_raw = fetch_footystats_today_raw()
    fs_live_rows: List[Dict[str, Any]] = []
    for raw in fs_today_raw:
        row = infer_fs_live(raw, fs_today_existing.get(str(raw.get('id'))))
        if row is None:
            continue
        row['live_comment'] = row.get('live_comment') or ''
        row['boss_ai_decision'] = row.get('boss_ai_decision') or row.get('live_comment') or ''
        fs_live_rows.append(row)
        fs_bundle.setdefault('fixtures', {})[str(row.get('id'))] = {'raw': raw, 'fetched_at': now_iso()}

    # Sportmonks live
    sm_rows = fetch_sportmonks_live_rows()
    sm_live_rows: List[Dict[str, Any]] = []
    sm_cards: List[Dict[str, Any]] = []
    today_by_sm = {str((x.get('source_ids') or {}).get('sportmonks') or ''): x for x in sm_today if (x.get('source_ids') or {}).get('sportmonks')}

    for fixture in sm_rows:
        sid = str(fixture.get('id') or '')
        base = dict(today_by_sm.get(sid) or {})
        parts = fixture.get('participants') or []
        home, away = get_side_participants(parts)
        h, a = current_score(fixture)
        minute = extract_minute_sm(fixture)
        row = {
            **base,
            'id': base.get('id') or f'sm-{sid}',
            'source_ids': {'footystats': '', 'sportmonks': sid},
            'source': 'sportmonks',
            'home_name': home.get('name') or base.get('home_name') or 'Ev Sahibi',
            'away_name': away.get('name') or base.get('away_name') or 'Deplasman',
            'competition_name': (fixture.get('league') or {}).get('name') or base.get('competition_name') or fixture.get('name') or 'Canlı Maç',
            'competition_id': safe_int((fixture.get('league') or {}).get('id') or base.get('competition_id'), 0),
            'league_country': ((fixture.get('league') or {}).get('country') or {}).get('name') or base.get('league_country') or '',
            'league_country_image': ((fixture.get('league') or {}).get('country') or {}).get('image_path') or base.get('league_country_image') or '',
            'date_unix': safe_int(fixture.get('starting_at_timestamp') or base.get('date_unix'), 0),
            'home_image': home.get('image_path') or base.get('home_image') or '',
            'away_image': away.get('image_path') or base.get('away_image') or '',
            'home_country': base.get('home_country') or '',
            'away_country': base.get('away_country') or '',
            'stadium_name': base.get('stadium_name') or '',
            'stadium_location': base.get('stadium_location') or '',
            'referee_name': base.get('referee_name') or '',
            'weather': base.get('weather') or {},
            'home_ppg': safe_float(base.get('home_ppg')),
            'away_ppg': safe_float(base.get('away_ppg')),
            'team_a_ppg': base.get('team_a_ppg', '0'),
            'team_b_ppg': base.get('team_b_ppg', '0'),
            'team_a_xg_prematch': safe_float(base.get('team_a_xg_prematch')),
            'team_b_xg_prematch': safe_float(base.get('team_b_xg_prematch')),
            'btts_potential': safe_float(base.get('btts_potential')),
            'o25_potential': safe_float(base.get('o25_potential')),
            'o05HT_potential': safe_float(base.get('o05HT_potential')),
            'pressure_score': safe_float(base.get('pressure_score')),
            'odds_ft_1': safe_float(base.get('odds_ft_1')),
            'odds_ft_x': safe_float(base.get('odds_ft_x')),
            'odds_ft_2': safe_float(base.get('odds_ft_2')),
            'odds_ft_over25': safe_float(base.get('odds_ft_over25')),
            'odds_btts_yes': safe_float(base.get('odds_btts_yes')),
            'odds_1st_half_over05': safe_float(base.get('odds_1st_half_over05')),
            'homeGoalCount': h,
            'awayGoalCount': a,
            'status': 'live',
            'elapsed': minute,
            'boss_ai_decision': '',
            'prematch_comment': base.get('prematch_comment') or '',
            'live_comment': '',
        }

        cached = sm_bundle.get('fixtures', {}).get(sid) or {}
        detail = cached.get('detail') or fetch_fixture_detail(safe_int(sid))
        if detail:
            vals = {'home': 0.0, 'away': 0.0}
            for p in (detail.get('pressure') or [])[-12:]:
                loc = str((p.get('participant') or {}).get('meta', {}).get('location') or p.get('location') or '').lower()
                if loc == 'home':
                    vals['home'] += safe_float(p.get('value') or p.get('amount'))
                elif loc == 'away':
                    vals['away'] += safe_float(p.get('value') or p.get('amount'))
            row['pressure_score'] = round(vals['home'] - vals['away'], 2)
            row['lineups'] = detail.get('lineups') or []
            row['events'] = detail.get('events') or []
            row['statistics'] = detail.get('statistics') or []
            sm_bundle.setdefault('fixtures', {})[sid] = {**cached, 'detail': detail, 'fetched_at': now_iso()}
        
        live_comment = ai_comment_live(client, row, detail or {})
        if live_comment and live_comment != 'AI yorumu henüz yok.':
            row['live_comment'] = live_comment
            row['boss_ai_decision'] = live_comment
        sm_live_rows.append(row)
        sm_cards.append({
            'id': sid,
            'name': fixture.get('name'),
            'minute': minute,
            'homeTeam': {'name': home.get('name', ''), 'logo': home.get('image_path', '')},
            'awayTeam': {'name': away.get('name', ''), 'logo': away.get('image_path', '')},
            'score': {'fullTime': {'home': h, 'away': a}},
        })

    # Çöpçü: 48 saatten eski fixture'ları bundle'dan temizle
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

    save_json(FS_LIVE_JSON, {'source': 'footystats', 'updated_at': now_iso(), 'matches': fs_live_rows})
    save_json(SM_LIVE_JSON, {'source': 'sportmonks', 'updated_at': now_iso(), 'matches': sm_live_rows, 'cards': sm_cards})
    save_json(FS_BUNDLE_JSON, fs_bundle)
    save_json(SM_BUNDLE_JSON, sm_bundle)
    save_source_tabs()

    health.update({
        'split_live_runner': 'live_split_sources',
        'split_live_started_at': now_iso(),
        'footystats_live_count': len(fs_live_rows),
        'sportmonks_live_count': len(sm_live_rows),
        'split_live_finished_at': now_iso(),
        'split_live_duration_sec': round(time.time() - started, 2),
    })
    save_json(HEALTH_JSON, health)

if __name__ == '__main__':
    main()
