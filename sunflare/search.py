"""Find solar flares (GOES events from the HEK, via sunpy) and GBM triggers for a given day."""
import datetime
import re
import urllib.request

GBM_TRIG_URL = 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/{year}/'


def _day(date):
    if isinstance(date, datetime.date):
        return datetime.datetime(date.year, date.month, date.day)
    return datetime.datetime.strptime(str(date)[:10], '%Y-%m-%d')


def goes_flares(date, min_class='C'):
    """GOES flares of one UTC day from the HEK: list of dicts sorted by start time."""
    from sunpy.net import attrs as a
    from sunpy.net.hek import HEKClient
    d0 = _day(date)
    res = HEKClient().search(a.Time(d0, d0 + datetime.timedelta(days=1)), a.hek.EventType('FL'),
                             a.hek.OBS.Observatory == 'GOES')
    order = 'ABCMX'
    out, seen = [], set()
    for r in res:
        cls = str(r['fl_goescls'] or '').strip()
        if not cls or order.find(cls[0]) < order.find(min_class[0].upper()):
            continue
        key = (str(r['event_starttime']), cls)
        if key in seen:
            continue
        seen.add(key)
        loc = ''
        try:
            if abs(float(r['hgs_x'])) < 1e-6 and abs(float(r['hgs_y'])) < 1e-6:
                raise ValueError('no position reported')      # GOES events without a position give (0, 0)
            loc = f"{'N' if r['hgs_y'] >= 0 else 'S'}{abs(r['hgs_y']):02.0f}{'W' if r['hgs_x'] >= 0 else 'E'}{abs(r['hgs_x']):02.0f}"
        except Exception:
            pass
        out.append({'start': str(r['event_starttime'])[:19], 'peak': str(r['event_peaktime'])[:19],
                    'end': str(r['event_endtime'])[:19], 'class': cls,
                    'ar': int(r['ar_noaanum']) if r['ar_noaanum'] else None, 'location': loc})
    return sorted(out, key=lambda x: x['start'])


def gbm_triggers(date):
    """GBM triggers of one UTC day: [{'name': 'bn251111415', 'utc': '2025-11-11 09:57:36'}, ...]."""
    d0 = _day(date)
    with urllib.request.urlopen(GBM_TRIG_URL.format(year=d0.year), timeout=60) as r:
        html = r.read().decode('utf-8', 'replace')
    tag = d0.strftime('%y%m%d')
    names = sorted(set(re.findall(r'bn(%s\d{3})' % tag, html)))
    out = []
    for n in names:
        frac = int(n[6:]) / 1000.0
        t = d0 + datetime.timedelta(days=frac)
        out.append({'name': 'bn' + n, 'utc': t.strftime('%Y-%m-%d %H:%M:%S'), 'approx': True})
    return out


def search(date, min_class='C'):
    out = {'date': str(date)[:10], 'flares': [], 'triggers': [], 'errors': []}
    try:
        out['flares'] = goes_flares(date, min_class)
    except Exception as e:
        out['errors'].append(f'HEK flare search failed: {type(e).__name__}: {e}')
    try:
        out['triggers'] = gbm_triggers(date)
    except Exception as e:
        out['errors'].append(f'GBM trigger list failed: {type(e).__name__}: {e}')
    return out


if __name__ == '__main__':
    import json
    import sys
    print(json.dumps(search(sys.argv[1] if len(sys.argv) > 1 else '2025-11-11', 'M'), indent=1))
