import json
import urllib.request
import csv
from datetime import datetime


# t =
# geram24,
# geram18,
# price_dollar_rl,
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
        date_gr_split = [int(x) for x in row[6].split('/')]
        date = datetime(date_gr_split[0], date_gr_split[1], date_gr_split[2])

        item = {
            'start': row[0],
            'min': row[1],
            'max': row[2],
            'end': row[3],
            'date_gr': row[6],
            'date_fa': row[7],
            'date': date
        }
        to_csv.append(item)

    keys = to_csv[0].keys()

    with open('Predictor/csv/%s.csv' % t, 'w', encoding='utf8', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(to_csv)


t_list = ['price_dollar_rl', 'crypto-ethereum', 'crypto-bitcoin', 'geram24', 'geram18']

for i in t_list:
    crawl(i, True)
