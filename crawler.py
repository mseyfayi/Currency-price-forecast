import csv
import json
import urllib.request


def crawl(t, length=None):
    url = "https://api.tgju.online/v1/market/indicator/summary-table-data/%s" % t
    if length is not None:
        url = url + '?start=0&length=%s' % length

    data = urllib.request.urlopen(url).read()
    data = data.decode('ascii')
    data = json.loads(data)
    data = data['data']

    print('extracted from %s: %s' % (t, len(data)))
    to_csv = []
    for row in data:
        is_int = True
        for i in range(0, 4):
            row[i] = row[i].replace(',', '')
            if row[i].find('.') > 0:
                is_int = False

        for i in range(0, 4):
            row[i] = int(row[i]) if is_int else float(row[i])

        item = {
            'Open': row[0],
            'Low': row[1],
            'High': row[2],
            'Close': row[3],
            'Date': row[6],
            'Date_fa': row[7],
        }
        to_csv.append(item)

    keys = to_csv[0].keys()

    with open('/home/mohamad/Predictor/csv/%s.csv' % t, 'w', encoding='utf8', newline='') as output_file:
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
crawl(t_list[0])
