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
TODAY_JSON = os.path.join(DATA_DIR,'today.json')
TOMORROW_JSON = os.path.join(DATA_DIR,'tomorrow.json')
TODAY_MAIN_JSON = os.path.join(DATA_DIR,'today_main.json')
MATCH_MAP_JSON = os.path.join(DATA_DIR,'match_map.json')
HEALTH_JSON = os.path.join(DATA_DIR,'health.json')
BUNDLE_JSON = os.path.join(DATA_DIR,'sportmonks_bundle.json')

FS_KEY = os.getenv('FOOTYSTATS_KEY','').strip()
SM_KEY = os.getenv('SPORTMONKS_KEY','').strip()
GEMINI_MODEL = os.getenv('GEMINI_MODEL','gemini-2.5-flash').strip()
REQUEST_TIMEOUT = 25

DETAIL_INCLUDE = ';'.join([
    'participants','league.country','venue','state','scores','periods',
    'events.type','events.period','events.player',
    'statistics.type',
    'sidelined.sideline.player','sidelined.sideline.type',
    'weatherReport','comments','predictions.type',
    'pressure.participant','trends.type','trends.participant',
    'xGFixture.type','lineups.player','lineups.type','lineups.details.type',
    'metadata.type','coaches','prematchNews.lines','postmatchNews.lines'
])
FIXTURE_BASIC_INCLUDE = 'participants;league.country;venue;referees;weatherReport;state;scores;round'
ODDS_FILTERS = 'markets:1;bookmakers:2'
ODDS_INCLUDE = 'fixtures.odds.market;fixtures.odds.bookmaker;fixtures.participants;league.country'
STANDINGS_INCLUDE = 'participant;rule.type;details.type;form;stage;league;group'
LIVE_STANDINGS_INCLUDE = 'stage;league;details.type;participant'
H2H_INCLUDE = 'participants;league;scores;state;venue;events'


def log(msg:str):
    print(msg, flush=True)


def ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v in (None,''): return default
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default: int = 0) -> int:
    try:
        if v in (None,''): return default
        return int(float(v))
    except Exception:
        return default


def load_json(path:str, default:Any):
    try:
        with open(path,'r',encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path:str, data:Any):
    with open(path,'w',encoding='utf-8') as f:
        json.dump(data,f,indent=2,ensure_ascii=False)


def clean_name(name:str)->str:
    n = unidecode(str(name or '')).lower()
    n = re.sub(r'[\W_]+','',n)
    for token in ['footballclub','futebolclube','clubdefutbol','clubdeportivo','women','ladies','reserves','reserve','ii','iii','u21','u23','fc','cf','ac','afc','sc','sk','if','fk','bk','nk','cd','de','la','the']:
        n = n.replace(token,'')
    return n


def ratio(a:str,b:str)->float:
    if not a or not b: return 0.0
    if a == b: return 1.0
    if a in b or b in a:
        shorter = min(len(a),len(b)); longer = max(len(a),len(b))
        return shorter/max(longer,1)
    same = sum(1 for x,y in zip(a,b) if x==y)
    return same/max(len(a),len(b),1)


def iso_date(offset:int=0)->str:
    return (datetime.now(timezone.utc)+timedelta(days=offset)).strftime('%Y-%m-%d')


def country_from_image_path(path:str)->str:
    if not path: return ''
    last = path.split('/')[-1]
    if '-' not in last: return ''
    return last.split('-')[0].replace('_',' ').strip().title()


def fetch_json(url:str, timeout:int=REQUEST_TIMEOUT)->Dict[str,Any]:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def init_vertex_client():
    if genai is None:
        return None
    gac = os.getenv('GOOGLE_APPLICATION_CREDENTIALS','').strip()
    if not gac or not os.path.exists(gac):
        return None
    try:
        with open(gac,'r',encoding='utf-8') as f:
            info = json.load(f)
        return genai.Client(vertexai=True, project=info['project_id'], location='us-central1')
    except Exception as e:
        log(f'⚠️ Gemini başlatılamadı: {e}')
        return None


def ai_comment_prematch(client, match:Dict[str,Any])->str:
    if client is None:
        return ''
    payload = {
        'home': match.get('home_name'), 'away': match.get('away_name'), 'league': match.get('competition_name'),
        'xg_home': safe_float(match.get('team_a_xg_prematch')), 'xg_away': safe_float(match.get('team_b_xg_prematch')),
        'home_ppg': safe_float(match.get('home_ppg')), 'away_ppg': safe_float(match.get('away_ppg')),
        'btts': safe_float(match.get('btts_potential')), 'over25': safe_float(match.get('o25_potential')),
        'odds1': safe_float(match.get('odds_ft_1')), 'oddsx': safe_float(match.get('odds_ft_x')), 'odds2': safe_float(match.get('odds_ft_2')),
    }
    prompt = '\n'.join([
        'Sen kısa ve teknik prematch bahis analiz motorusun.',
        'Kurallar:',
        '- En fazla 85 kelime.',
        '- Övgü, emoji, komutan, boss kullanma.',
        '- 3 satır yaz: DURUM, NEDEN, SONUÇ.',
        '- Veri yetersizse tam olarak "AI yorumu henüz yok." yaz.',
        json.dumps(payload, ensure_ascii=False)
    ])
    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (getattr(response,'text','') or '').strip()
    except Exception as e:
        log(f'⚠️ Prematch AI hatası: {e}')
        return ''


def extract_sm_basic(sm:Dict[str,Any])->Dict[str,Any]:
    parts = sm.get('participants') or []
    home, away = {}, {}
    for p in parts:
        loc = str((p.get('meta') or {}).get('location') or '').lower()
        if loc == 'home': home = p
        elif loc == 'away': away = p
    if not home and parts: home = parts[0]
    if not away and len(parts) > 1: away = parts[1]
    league = sm.get('league') or {}
    venue = sm.get('venue') or {}
    refs = sm.get('referees') or []
    ref = refs[0] if refs else {}
    weather = sm.get('weatherreport') or sm.get('weatherReport') or {}
    return {
        'sportmonks_id': safe_int(sm.get('id'),0),
        'home_sm_name': home.get('name',''), 'away_sm_name': away.get('name',''),
        'home_sm_logo': home.get('image_path',''), 'away_sm_logo': away.get('image_path',''),
        'home_sm_country_id': home.get('country_id'), 'away_sm_country_id': away.get('country_id'),
        'competition_name': league.get('name',''), 'competition_id': safe_int(league.get('id'),0),
        'league_country': (league.get('country') or {}).get('name',''),
        'league_country_image': (league.get('country') or {}).get('image_path',''),
        'venue_name': venue.get('name',''), 'venue_city': venue.get('city_name',''),
        'referee_name': ref.get('name') or ref.get('common_name') or '',
        'starting_at_timestamp': safe_int(sm.get('starting_at_timestamp'),0),
        'round_id': safe_int((sm.get('round') or {}).get('id') or sm.get('round_id'),0),
        'season_id': safe_int(sm.get('season_id'),0),
        'weather': weather,
    }


def match_sm_to_fs(fs_match:Dict[str,Any], sm_rows:List[Dict[str,Any]])->Optional[Dict[str,Any]]:
    h=clean_name(fs_match.get('home_name','')); a=clean_name(fs_match.get('away_name',''))
    best=None; best_score=0.0
    for sm in sm_rows:
        info=extract_sm_basic(sm)
        sh=clean_name(info.get('home_sm_name','')); sa=clean_name(info.get('away_sm_name',''))
        score=(ratio(h,sh)+ratio(a,sa))/2.0
        if score>best_score:
            best=sm; best_score=score
    return best if best_score>=0.68 else None


def build_flat_match(fs:Dict[str,Any], sm:Optional[Dict[str,Any]], old:Optional[Dict[str,Any]])->Dict[str,Any]:
    old = old or {}
    smi = extract_sm_basic(sm or {})
    match_id = str(fs.get('id','') or old.get('id',''))
    home_image = fs.get('home_image') or old.get('home_image') or smi.get('home_sm_logo') or ''
    away_image = fs.get('away_image') or old.get('away_image') or smi.get('away_sm_logo') or ''
    data = {
        'id': match_id or f"sm-{smi.get('sportmonks_id','')}",
        'source_ids': {'footystats': match_id, 'sportmonks': str(smi.get('sportmonks_id') or (old.get('source_ids') or {}).get('sportmonks',''))},
        'home_name': fs.get('home_name') or old.get('home_name') or smi.get('home_sm_name') or 'Ev Sahibi',
        'away_name': fs.get('away_name') or old.get('away_name') or smi.get('away_sm_name') or 'Deplasman',
        'competition_name': fs.get('competition_name') or smi.get('competition_name') or old.get('competition_name') or '',
        'competition_id': safe_int(fs.get('competition_id') or smi.get('competition_id') or old.get('competition_id'),0),
        'league_country': smi.get('league_country') or old.get('league_country') or '',
        'league_country_image': smi.get('league_country_image') or old.get('league_country_image') or '',
        'date_unix': safe_int(fs.get('date_unix') or smi.get('starting_at_timestamp') or old.get('date_unix'),0),
        'home_image': home_image, 'away_image': away_image,
        'home_country': country_from_image_path(home_image), 'away_country': country_from_image_path(away_image),
        'stadium_name': smi.get('venue_name') or old.get('stadium_name') or '',
        'stadium_location': smi.get('venue_city') or old.get('stadium_location') or '',
        'referee_name': smi.get('referee_name') or old.get('referee_name') or '',
        'weather': smi.get('weather') or old.get('weather') or {},
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
        'pressure_score': safe_float(old.get('pressure_score')),
        'odds_ft_1': safe_float(fs.get('odds_ft_1') or old.get('odds_ft_1')),
        'odds_ft_x': safe_float(fs.get('odds_ft_x') or old.get('odds_ft_x')),
        'odds_ft_2': safe_float(fs.get('odds_ft_2') or old.get('odds_ft_2')),
        'odds_ft_over25': safe_float(fs.get('odds_ft_over25') or old.get('odds_ft_over25')),
        'odds_btts_yes': safe_float(fs.get('odds_btts_yes') or old.get('odds_btts_yes')),
        'odds_1st_half_over05': safe_float(fs.get('odds_1st_half_over05') or old.get('odds_1st_half_over05')),
        'homeGoalCount': safe_int(old.get('homeGoalCount'),0), 'awayGoalCount': safe_int(old.get('awayGoalCount'),0),
        'status': old.get('status') or 'incomplete', 'elapsed': old.get('elapsed') or 0,
        'boss_ai_decision': old.get('boss_ai_decision') or '',
        'prematch_comment': old.get('prematch_comment') or '',
        'live_comment': old.get('live_comment') or '',
    }
    return data


def build_sm_only_match(sm:Dict[str,Any], old:Optional[Dict[str,Any]]=None)->Dict[str,Any]:
    old = old or {}
    smi = extract_sm_basic(sm)
    return {
        'id': old.get('id') or f"sm-{smi.get('sportmonks_id')}",
        'source_ids': {'footystats': (old.get('source_ids') or {}).get('footystats',''), 'sportmonks': str(smi.get('sportmonks_id'))},
        'home_name': smi.get('home_sm_name') or old.get('home_name') or 'Ev Sahibi',
        'away_name': smi.get('away_sm_name') or old.get('away_name') or 'Deplasman',
        'competition_name': smi.get('competition_name') or old.get('competition_name') or '',
        'competition_id': smi.get('competition_id') or old.get('competition_id') or 0,
        'league_country': smi.get('league_country') or old.get('league_country') or '',
        'league_country_image': smi.get('league_country_image') or old.get('league_country_image') or '',
        'date_unix': smi.get('starting_at_timestamp') or old.get('date_unix') or 0,
        'home_image': smi.get('home_sm_logo') or old.get('home_image') or '',
        'away_image': smi.get('away_sm_logo') or old.get('away_image') or '',
        'home_country': country_from_image_path(smi.get('home_sm_logo') or ''), 'away_country': country_from_image_path(smi.get('away_sm_logo') or ''),
        'stadium_name': smi.get('venue_name') or old.get('stadium_name') or '',
        'stadium_location': smi.get('venue_city') or old.get('stadium_location') or '',
        'referee_name': smi.get('referee_name') or old.get('referee_name') or '',
        'weather': smi.get('weather') or old.get('weather') or {},
        'home_ppg': safe_float(old.get('home_ppg')), 'away_ppg': safe_float(old.get('away_ppg')),
        'team_a_ppg': old.get('team_a_ppg','0'), 'team_b_ppg': old.get('team_b_ppg','0'),
        'team_a_xg_prematch': safe_float(old.get('team_a_xg_prematch')), 'team_b_xg_prematch': safe_float(old.get('team_b_xg_prematch')),
        'btts_potential': safe_float(old.get('btts_potential')), 'o25_potential': safe_float(old.get('o25_potential')), 'o05HT_potential': safe_float(old.get('o05HT_potential')),
        'odds_ft_1': safe_float(old.get('odds_ft_1')), 'odds_ft_x': safe_float(old.get('odds_ft_x')), 'odds_ft_2': safe_float(old.get('odds_ft_2')),
        'odds_ft_over25': safe_float(old.get('odds_ft_over25')), 'odds_btts_yes': safe_float(old.get('odds_btts_yes')), 'odds_1st_half_over05': safe_float(old.get('odds_1st_half_over05')),
        'homeGoalCount': safe_int(old.get('homeGoalCount'),0), 'awayGoalCount': safe_int(old.get('awayGoalCount'),0),
        'status': old.get('status') or 'incomplete', 'elapsed': old.get('elapsed') or 0,
        'boss_ai_decision': old.get('boss_ai_decision') or '', 'prematch_comment': old.get('prematch_comment') or '', 'live_comment': old.get('live_comment') or '',
    }


def fetch_round_odds(round_id:int, cache:Dict[int,Any])->Dict[str,Any]:
    if not SM_KEY or not round_id: return {}
    if round_id in cache: return cache[round_id]
    url = f"https://api.sportmonks.com/v3/football/rounds/{round_id}?api_token={SM_KEY}&include={ODDS_INCLUDE}&filters={ODDS_FILTERS}"
    try:
        data = fetch_json(url).get('data',{})
    except Exception as e:
        log(f'⚠️ Odds alınamadı round={round_id}: {e}')
        data = {}
    cache[round_id] = data
    return data


def odds_for_fixture(round_data:Dict[str,Any], fixture_id:int)->List[Dict[str,Any]]:
    fixtures = round_data.get('fixtures') or []
    for f in fixtures:
        if safe_int(f.get('id')) == fixture_id:
            return f.get('odds') or []
    return []


def fetch_standings(season_id:int, cache:Dict[int,Any])->Any:
    if not SM_KEY or not season_id: return []
    if season_id in cache: return cache[season_id]
    url = f"https://api.sportmonks.com/v3/football/standings/seasons/{season_id}?api_token={SM_KEY}&include={STANDINGS_INCLUDE}"
    try:
        data = fetch_json(url).get('data',[]) or []
    except Exception as e:
        log(f'⚠️ Standings alınamadı season={season_id}: {e}')
        data = []
    cache[season_id]=data
    return data


def fetch_live_standings(round_id:int, cache:Dict[int,Any])->Any:
    if not SM_KEY or not round_id: return []
    if round_id in cache: return cache[round_id]
    url = f"https://api.sportmonks.com/v3/football/standings/rounds/{round_id}?api_token={SM_KEY}&include={LIVE_STANDINGS_INCLUDE}"
    try:
        data = fetch_json(url).get('data',[]) or []
    except Exception as e:
        log(f'⚠️ Live standings alınamadı round={round_id}: {e}')
        data = []
    cache[round_id]=data
    return data


def fetch_h2h(home_id:int, away_id:int, cache:Dict[str,Any])->Any:
    if not SM_KEY or not home_id or not away_id: return []
    key=f'{home_id}:{away_id}'
    if key in cache: return cache[key]
    url = f"https://api.sportmonks.com/v3/football/fixtures/head-to-head/{home_id}/{away_id}?api_token={SM_KEY}&include={H2H_INCLUDE}"
    try:
        data = fetch_json(url).get('data',[]) or []
    except Exception as e:
        log(f'⚠️ H2H alınamadı {key}: {e}')
        data=[]
    cache[key]=data
    return data


def fetch_fixture_detail(fid:int)->Dict[str,Any]:
    if not SM_KEY or not fid: return {}
    url = f"https://api.sportmonks.com/v3/football/fixtures/{fid}?api_token={SM_KEY}&include={DETAIL_INCLUDE}"
    try:
        return fetch_json(url).get('data',{}) or {}
    except Exception as e:
        log(f'⚠️ Fixture detail alınamadı fixture={fid}: {e}')
        return {}


def main():
    ensure_dir()
    started = time.time()
    health = {'runner':'fetch_vnext','started_at':datetime.utcnow().isoformat()+'Z','footystats_matches':0,'sportmonks_today':0,'sportmonks_tomorrow':0,'matched_today':0,'prematch_ai_written':0,'bundle_details_written':0,'errors':[]}

    existing_today = load_json(TODAY_JSON,{}).get('data',[])
    existing_today_by_id = {str(x.get('id')):x for x in existing_today if x.get('id')}
    existing_tomorrow = load_json(TOMORROW_JSON,{}).get('data',[])
    existing_tomorrow_by_sm = {str((x.get('source_ids') or {}).get('sportmonks') or ''):x for x in existing_tomorrow if (x.get('source_ids') or {}).get('sportmonks')}
    bundle_old = load_json(BUNDLE_JSON, {'fixtures':{}})
    bundle = {'generated_at': datetime.utcnow().isoformat()+'Z', 'fixtures': bundle_old.get('fixtures',{})}
    match_map = load_json(MATCH_MAP_JSON, {'sportmonks_to_footystats':{}})
    sm_to_fs = match_map.get('sportmonks_to_footystats',{})

    fs_rows=[]
    if FS_KEY:
        fs_url = f'https://api.football-data-api.com/todays-matches?key={FS_KEY}&include=stats,odds'
        fs_json = fetch_json(fs_url)
        fs_rows = fs_json.get('data',[])
        health['footystats_matches']=len(fs_rows)
        save_json(TODAY_MAIN_JSON, {'data': fs_rows})

    sm_today_rows=[]; sm_tomorrow_rows=[]
    if SM_KEY:
        try:
            sm_today_rows = fetch_json(f'https://api.sportmonks.com/v3/football/fixtures/date/{iso_date(0)}?api_token={SM_KEY}&include={FIXTURE_BASIC_INCLUDE}').get('data',[])
            health['sportmonks_today']=len(sm_today_rows)
        except Exception as e:
            health['errors'].append(f'sportmonks_today: {e}')
        try:
            sm_tomorrow_rows = fetch_json(f'https://api.sportmonks.com/v3/football/fixtures/date/{iso_date(1)}?api_token={SM_KEY}&include={FIXTURE_BASIC_INCLUDE}').get('data',[])
            health['sportmonks_tomorrow']=len(sm_tomorrow_rows)
        except Exception as e:
            health['errors'].append(f'sportmonks_tomorrow: {e}')

    vertex = init_vertex_client()
    out_today=[]; used_sm_today=set()

    for fs in fs_rows:
        fs_id=str(fs.get('id',''))
        old=existing_today_by_id.get(fs_id)
        matched_sm=None
        for sm in sm_today_rows:
            if str(sm.get('id')) and sm_to_fs.get(str(sm.get('id'))) == fs_id:
                matched_sm=sm; break
        if matched_sm is None and sm_today_rows:
            matched_sm = match_sm_to_fs(fs, sm_today_rows)
            if matched_sm is not None and matched_sm.get('id'):
                sm_to_fs[str(matched_sm['id'])]=fs_id
        if matched_sm is not None:
            used_sm_today.add(str(matched_sm.get('id')))
            health['matched_today'] += 1
        row=build_flat_match(fs, matched_sm, old)
        if not row.get('prematch_comment') and (row.get('team_a_xg_prematch') or row.get('home_ppg')):
            comment = ai_comment_prematch(vertex,row)
            if comment:
                row['prematch_comment']=comment
                if not row.get('boss_ai_decision'):
                    row['boss_ai_decision']=comment
                health['prematch_ai_written'] += 1
        out_today.append(row)

    # add unmatched sportmonks today fixtures so they still show
    for sm in sm_today_rows:
        sid=str(sm.get('id'))
        if sid in used_sm_today: continue
        out_today.append(build_sm_only_match(sm, existing_tomorrow_by_sm.get(sid)))

    out_tomorrow=[]
    for sm in sm_tomorrow_rows:
        sid=str(sm.get('id'))
        out_tomorrow.append(build_sm_only_match(sm, existing_tomorrow_by_sm.get(sid)))

    # Build bundle for sportmonks ids present in today/tomorrow
    detail_ids=[]
    for row in out_today + out_tomorrow:
        sid = safe_int((row.get('source_ids') or {}).get('sportmonks'),0)
        if sid: detail_ids.append(sid)
    detail_ids = list(dict.fromkeys(detail_ids))

    odds_cache={}; standings_cache={}; live_standings_cache={}; h2h_cache={}
    for sid in detail_ids:
        detail = fetch_fixture_detail(sid)
        if not detail:
            continue
        round_id = safe_int((detail.get('round') or {}).get('id') or detail.get('round_id'),0)
        season_id = safe_int(detail.get('season_id'),0)
        participants = detail.get('participants') or []
        home = next((p for p in participants if str((p.get('meta') or {}).get('location') or '').lower()=='home'), participants[0] if participants else {})
        away = next((p for p in participants if str((p.get('meta') or {}).get('location') or '').lower()=='away'), participants[1] if len(participants)>1 else {})
        odds = odds_for_fixture(fetch_round_odds(round_id, odds_cache), sid) if round_id else []
        standings = fetch_standings(season_id, standings_cache) if season_id else []
        live_standings = fetch_live_standings(round_id, live_standings_cache) if round_id else []
        h2h = fetch_h2h(safe_int(home.get('id'),0), safe_int(away.get('id'),0), h2h_cache) if home and away else []
        bundle['fixtures'][str(sid)] = {
            'fetched_at': datetime.utcnow().isoformat()+'Z',
            'detail': detail,
            'odds': odds,
            'standings': standings,
            'live_standings': live_standings,
            'h2h': h2h,
        }
        health['bundle_details_written'] += 1

    save_json(TODAY_JSON, {'data': out_today})
    save_json(TOMORROW_JSON, {'data': out_tomorrow})
    save_json(BUNDLE_JSON, bundle)
    save_json(MATCH_MAP_JSON, {'sportmonks_to_footystats': sm_to_fs})

    health['finished_at']=datetime.utcnow().isoformat()+'Z'
    health['duration_sec']=round(time.time()-started,2)
    save_json(HEALTH_JSON, health)
    log(f"✅ today.json {len(out_today)}")
    log(f"✅ tomorrow.json {len(out_tomorrow)}")
    log(f"✅ bundle fixtures {health['bundle_details_written']}")

if __name__ == '__main__':
    main()
