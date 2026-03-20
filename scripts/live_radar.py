#!/usr/bin/env python3
import json, os, re
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import requests
from unidecode import unidecode

try:
    from google import genai
except Exception:
    genai = None

DATA_DIR='data'
TODAY_JSON=os.path.join(DATA_DIR,'today.json')
LIVE_JSON=os.path.join(DATA_DIR,'live.json')
MATCH_MAP_JSON=os.path.join(DATA_DIR,'match_map.json')
HEALTH_JSON=os.path.join(DATA_DIR,'health.json')
BUNDLE_JSON=os.path.join(DATA_DIR,'sportmonks_bundle.json')

SM_KEY=os.getenv('SPORTMONKS_KEY','').strip()
GEMINI_MODEL=os.getenv('GEMINI_MODEL','gemini-2.5-flash').strip()
REQUEST_TIMEOUT=25

LIVE_INCLUDE='participants;scores;periods;events;league.country;round'
DETAIL_INCLUDE='participants;league.country;venue;state;scores;periods;events.type;events.period;events.player;statistics.type;sidelined.sideline.player;sidelined.sideline.type;weatherReport;comments;predictions.type;pressure.participant;trends.type;trends.participant;xGFixture.type;lineups.player;lineups.type;lineups.details.type;metadata.type;coaches;prematchNews.lines;postmatchNews.lines'


def log(msg:str):
    print(msg, flush=True)


def load_json(path:str, default:Any):
    try:
        with open(path,'r',encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path:str, data:Any):
    with open(path,'w',encoding='utf-8') as f:
        json.dump(data,f,indent=2,ensure_ascii=False)


def safe_int(v:Any, default:int=0)->int:
    try:
        if v in (None,''): return default
        return int(float(v))
    except Exception:
        return default


def safe_float(v:Any, default:float=0.0)->float:
    try:
        if v in (None,''): return default
        return float(v)
    except Exception:
        return default


def fetch_json(url:str)->Dict[str,Any]:
    r=requests.get(url,timeout=REQUEST_TIMEOUT)
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
            info=json.load(f)
        return genai.Client(vertexai=True, project=info['project_id'], location='us-central1')
    except Exception as e:
        log(f'⚠️ Gemini başlatılamadı: {e}')
        return None


def clean_name(name:str)->str:
    n=unidecode(str(name or '')).lower()
    n=re.sub(r'[\W_]+','',n)
    for token in ['footballclub','futebolclube','clubdefutbol','clubdeportivo','women','ladies','reserves','reserve','ii','iii','u21','u23','fc','cf','ac','afc','sc','sk','if','fk','bk','nk','cd','de','la','the']:
        n=n.replace(token,'')
    return n


def ratio(a:str,b:str)->float:
    if not a or not b: return 0.0
    if a==b: return 1.0
    if a in b or b in a:
        shorter=min(len(a),len(b)); longer=max(len(a),len(b))
        return shorter/max(longer,1)
    same=sum(1 for x,y in zip(a,b) if x==y)
    return same/max(len(a),len(b),1)


def get_side_participants(parts:List[Dict[str,Any]])->Tuple[Dict[str,Any],Dict[str,Any]]:
    home,away={},{}
    for p in parts:
        loc=str((p.get('meta') or {}).get('location') or '').lower()
        if loc=='home': home=p
        elif loc=='away': away=p
    if not home and parts: home=parts[0]
    if not away and len(parts)>1: away=parts[1]
    return home,away


def current_score(fixture:Dict[str,Any])->Tuple[int,int]:
    scores=fixture.get('scores') or []
    home,away=None,None
    for s in scores:
        if str(s.get('description') or '').upper() == 'CURRENT':
            participant=s.get('score',{}).get('participant')
            goals=safe_int(s.get('score',{}).get('goals'),0)
            if participant=='home': home=goals
            elif participant=='away': away=goals
    if home is not None and away is not None:
        return home,away
    latest=None; latest_ord=-1
    for ev in fixture.get('events') or []:
        result=ev.get('result')
        total=safe_int(ev.get('minute'),0)*100 + safe_int(ev.get('extra_minute'),0)
        if result and total>=latest_ord:
            latest_ord=total; latest=result
    if latest:
        m=re.match(r'^\s*(\d+)\s*-\s*(\d+)\s*$',str(latest))
        if m:
            return int(m.group(1)), int(m.group(2))
    return 0,0


def extract_minute(fixture:Dict[str,Any])->int:
    for p in fixture.get('periods') or []:
        mins=safe_int(p.get('minutes'),0)
        if mins: return mins
    best=0
    for ev in fixture.get('events') or []:
        best=max(best, safe_int(ev.get('minute'),0))
    return best


def build_unmatched_row(fixture:Dict[str,Any])->Dict[str,Any]:
    parts=fixture.get('participants') or []
    home,away=get_side_participants(parts)
    h,a=current_score(fixture)
    minute=extract_minute(fixture)
    league=fixture.get('league') or {}
    return {
        'id': f"sm_{fixture.get('id')}",
        'source_ids': {'footystats':'','sportmonks':str(fixture.get('id') or '')},
        'home_name': home.get('name') or 'Ev Sahibi',
        'away_name': away.get('name') or 'Deplasman',
        'competition_name': league.get('name') or fixture.get('name') or 'Canlı Maç',
        'competition_id': safe_int(league.get('id'),0),
        'league_country': (league.get('country') or {}).get('name',''),
        'league_country_image': (league.get('country') or {}).get('image_path',''),
        'date_unix': safe_int(fixture.get('starting_at_timestamp'),0),
        'home_image': home.get('image_path',''), 'away_image': away.get('image_path',''),
        'home_country': '', 'away_country': '',
        'home_ppg': 0.0, 'away_ppg': 0.0, 'team_a_ppg':'0', 'team_b_ppg':'0',
        'team_a_xg_prematch':0.0, 'team_b_xg_prematch':0.0, 'btts_potential':0.0, 'o25_potential':0.0, 'o05HT_potential':0.0,
        'odds_ft_1':0.0,'odds_ft_x':0.0,'odds_ft_2':0.0,'odds_ft_over25':0.0,'odds_btts_yes':0.0,'odds_1st_half_over05':0.0,
        'stadium_name':'','stadium_location':'','referee_name':'','weather':{},
        'homeGoalCount':h,'awayGoalCount':a,'status':'live','elapsed':minute,'pressure_score':0.0,
        'boss_ai_decision':'','prematch_comment':'','live_comment':'',
    }


def find_row(today_rows:List[Dict[str,Any]], fixture:Dict[str,Any], match_map:Dict[str,str])->Optional[Dict[str,Any]]:
    fid=str(fixture.get('id') or '')
    if fid and fid in match_map:
        fsid=match_map[fid]
        for row in today_rows:
            if str(row.get('id'))==str(fsid): return row
    home,away=get_side_participants(fixture.get('participants') or [])
    h=clean_name(home.get('name','')); a=clean_name(away.get('name',''))
    best=None; score_best=0.0
    for row in today_rows:
        score=(ratio(h,clean_name(row.get('home_name','')))+ratio(a,clean_name(row.get('away_name',''))))/2.0
        if score>score_best:
            best=row; score_best=score
    if best is not None and score_best>=0.68:
        if fid: match_map[fid]=str(best.get('id'))
        return best
    return None


def ai_comment_live(client, row:Dict[str,Any], detail:Dict[str,Any])->str:
    if client is None:
        return ''
    payload={
        'home':row.get('home_name'),'away':row.get('away_name'),'league':row.get('competition_name'),'minute':safe_int(row.get('elapsed'),0),
        'score_home':safe_int(row.get('homeGoalCount'),0),'score_away':safe_int(row.get('awayGoalCount'),0),
        'pressure': safe_float(row.get('pressure_score'),0),
        'shots': [s for s in detail.get('statistics',[]) if (s.get('type') or {}).get('developer_name') in ('SHOTS_TOTAL','SHOTS_ON_TARGET')],
        'prediction': detail.get('predictions',[])[:4],
    }
    prompt='\n'.join([
        'Sen kısa ve teknik canlı bahis analiz motorusun.',
        'Kurallar: en fazla 85 kelime. 3 satır yaz: DURUM, NEDEN, SONUÇ. Övgü, emoji, tiyatro yok. Veri yetersizse "AI yorumu henüz yok." yaz.',
        json.dumps(payload, ensure_ascii=False)
    ])
    try:
        response=client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (getattr(response,'text','') or '').strip()
    except Exception as e:
        log(f'⚠️ Live AI hatası: {e}')
        return ''


def fetch_live_rows()->List[Dict[str,Any]]:
    url=f'https://api.sportmonks.com/v3/football/livescores/inplay?api_token={SM_KEY}&include={LIVE_INCLUDE}'
    return fetch_json(url).get('data',[]) or []


def fetch_fixture_detail(fid:int)->Dict[str,Any]:
    url=f'https://api.sportmonks.com/v3/football/fixtures/{fid}?api_token={SM_KEY}&include={DETAIL_INCLUDE}'
    return fetch_json(url).get('data',{}) or {}


def main():
    if not SM_KEY:
        raise RuntimeError('SPORTMONKS_KEY eksik')
    today = load_json(TODAY_JSON,{}).get('data',[])
    bundle = load_json(BUNDLE_JSON, {'fixtures':{}})
    health = load_json(HEALTH_JSON,{})
    match_map = load_json(MATCH_MAP_JSON, {'sportmonks_to_footystats':{}}).get('sportmonks_to_footystats',{})
    client = init_vertex_client()

    rows = fetch_live_rows()
    live_out=[]
    health.update({'live_runner':'live_radar_vnext','live_started_at':datetime.utcnow().isoformat()+'Z','live_fixtures_seen':len(rows),'live_fixtures_matched':0,'live_unmatched_added':0,'live_ai_written':0,'live_errors':[]})
    log(f'RAW live rows count = {len(rows)}')

    for fixture in rows:
        row=find_row(today, fixture, match_map)
        if row is None:
            row=build_unmatched_row(fixture)
            today.append(row)
            health['live_unmatched_added'] += 1
        else:
            health['live_fixtures_matched'] += 1
        detail={}
        try:
            detail=fetch_fixture_detail(safe_int(fixture.get('id'),0))
        except Exception as e:
            health['live_errors'].append(str(e))
        h,a=current_score(detail or fixture)
        minute=extract_minute(detail or fixture)
        old_total=safe_int(row.get('homeGoalCount'),0)+safe_int(row.get('awayGoalCount'),0)
        new_total=h+a
        if new_total < old_total:
            h=safe_int(row.get('homeGoalCount'),0); a=safe_int(row.get('awayGoalCount'),0)
        row['status']='live'; row['elapsed']=minute or row.get('elapsed') or 0; row['homeGoalCount']=h; row['awayGoalCount']=a
        if detail.get('venue') and not row.get('stadium_name'):
            row['stadium_name']=detail['venue'].get('name',''); row['stadium_location']=detail['venue'].get('city_name','')
        if detail.get('participants'):
            home,away=get_side_participants(detail['participants'])
            row['home_image']=home.get('image_path') or row.get('home_image'); row['away_image']=away.get('image_path') or row.get('away_image')
        # pressure_score from pressure entries if present
        pressure=detail.get('pressure') or []
        if pressure:
            vals={'home':0,'away':0}
            for p in pressure[-8:]:
                loc=str((p.get('participant') or {}).get('meta',{}).get('location') or '').lower()
                vals[loc]=vals.get(loc,0)+safe_float(p.get('value') or p.get('amount'),0)
            row['pressure_score']=round(vals.get('home',0)-vals.get('away',0),2)
        bundle.setdefault('fixtures',{})[str(fixture.get('id'))] = {**bundle.get('fixtures',{}).get(str(fixture.get('id')),{}), 'detail':detail, 'fetched_at':datetime.utcnow().isoformat()+'Z'}
        live_comment=ai_comment_live(client,row,detail) if detail else ''
        if live_comment:
            row['live_comment']=live_comment; row['boss_ai_decision']=live_comment; health['live_ai_written'] += 1
        live_out.append({'id':fixture.get('id'),'minute':minute,'homeTeam':{'name':row.get('home_name')},'awayTeam':{'name':row.get('away_name')},'score':{'fullTime':{'home':h,'away':a}}})

    save_json(TODAY_JSON, {'data':today})
    save_json(LIVE_JSON, {'matches':live_out,'resultSet':{'count':len(live_out)}})
    save_json(BUNDLE_JSON, bundle)
    save_json(MATCH_MAP_JSON, {'sportmonks_to_footystats':match_map})
    health['live_finished_at']=datetime.utcnow().isoformat()+'Z'
    save_json(HEALTH_JSON, health)
    log(f"✅ Live eşleşen maç: {health['live_fixtures_matched']}")
    log(f"✅ Live eşleşmeyen ama eklenen maç: {health['live_unmatched_added']}")
    log(f"✅ live.json maç sayısı: {len(live_out)}")

if __name__=='__main__':
    main()
