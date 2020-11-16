import json
import urllib.request
import csv
from datetime import date


def crawl(t, debug=False, length=10):
    url = "https://api.tgju.online/v1/market/indicator/summary-table-data/%s" % t
    if debug:
        url = url + '?start=0&length=%s' % length

    data = urllib.request.urlopen(url).read()
    data = data.decode('ascii')
    data = json.loads(data)
    data = data['data']

    print('extracted from %s: %s' % (t, len(data)))
    to_csv = []
    for row in data:
        is_int=True
        for i in range(0, 4):
            row[i] = row[i].replace(',', '')
            if row[i].find('.'):
                is_int=False

        for i in range(0, 4):
            row[i] = int(row[i]) if is_int else float(row[i])

        date_gr_split = [int(x) for x in row[6].split('/')]
        d1 = date(date_gr_split[0], date_gr_split[1], date_gr_split[2])
        d0 = date(2008, 1, 1)

        d = d1 - d0

        item = {
            'start': row[0],
            'min': row[1],
            'max': row[2],
            'end': row[3],
            'date_gr': row[6],
            'date_fa': row[7],
            'date': d.days
        }
        to_csv.append(item)

    keys = to_csv[0].keys()

    with open('Predictor/csv/%s.csv' % t, 'w', encoding='utf8', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(to_csv)


t_list = [
    'price_dollar_rl',
    'crypto-ethereum',
    'crypto-bitcoin',
    'geram24',
    'geram18'
]

for l in t_list:
    crawl(l, True)
