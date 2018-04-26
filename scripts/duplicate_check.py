import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Look for duplicates')
parser.add_argument('classifications', help='Location of classifications csv')
args = parser.parse_args()
classifications_path = args.classifications

df = pd.read_csv(classifications_path)

seen = {}
double_saw = {}
saws = []
for row in df.iterrows():
    key = (row[1]['user_name'], row[1]['subject_ids'])
    saw = 'seen_before' in row[1]['metadata']
    saws.append(saw)
    if key not in seen and saw:
        # print('Huh, {0} is not in seen, but we saw it before'.format(key))
        pass
    elif key in seen and saw:
        pass
    elif key in seen and not saw:
        # print('Huh, {0} is in seen, but we do not realize it'.format(key))
        double_saw[key] = False
    elif key not in seen and not saw:
        seen[key] = True

saws = np.array(saws)
df['saw'] = saws

print('Printing duplicate (user_name, subject_ids) pairs that are not `not-logged-in`')
for key in double_saw:
    user_name, subject_ids = key
    if 'not-logged-in' not in user_name:
        print(df[(df['user_name'] == user_name) & (df['subject_ids'] == subject_ids)][['user_name', 'subject_ids', 'created_at', 'classification_id', 'saw']])
