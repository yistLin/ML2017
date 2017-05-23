#!/usr/local/bin/python3
import sys
import csv
import numpy as np

nb_voters = 0
vote_dics = [{} for i in range(1234)]

for file_path in sys.argv[1:]:
    print('load outputs from:', file_path)
    nb_voters += 1
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        line_cnt = 0
        for row in reader:
            tags = row[1].split(' ')
            for tag in tags:
                if tag not in vote_dics[line_cnt]:
                    vote_dics[line_cnt][tag] = 1
                else:
                    vote_dics[line_cnt][tag] += 1
            line_cnt += 1

print()
print('nb_voters = {}'.format(nb_voters))
thresh = nb_voters // 2
print('thresh = {}'.format(thresh))
predictions = []

for line_id, dic in enumerate(vote_dics):
    tags = []
    for key, val in dic.items():
        if val > thresh:
            tags.append(key)
    if len(tags) == 0:
        for key, val in dic.items():
            if val > (thresh - 1):
                tags.append(key)
        
    predictions.append(tags)

with open('output.csv', 'w') as out_f:
    outputs = ['"id","tags"\n']
    for idx, tags in enumerate(predictions):
        all_tags = ' '.join(tags)
        outputs.append('"{}","{}"\n'.format(idx, all_tags))
    out_f.write(''.join(outputs))

