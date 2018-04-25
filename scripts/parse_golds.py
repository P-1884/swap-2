"""
A basic script to create a golds csv from a set of classifications
"""
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-o', '--out', dest='out', default='golds.csv',
                    help='Specify out path for golds. Defaults to writing to ./golds.csv')
parser.add_argument('classifications', help='Location of classifications csv')

args = parser.parse_args()
classifications_path = args.classifications
out_path = args.out
# classifications_path = '/Users/cpd/Downloads/mark-lensed-features-beta-test-classifications.csv'
# out_path = '/Users/cpd/Downloads/mark-lensed-features-beta-test-golds.csv'

df = pd.read_csv(classifications_path)

# extract subject_id and gold status. Purge if not clear
subjects = []
skip_subjects = []
golds = []
unique_ids = set(df['subject_ids'])

types = ['SUB', 'DUD', 'LENS']
for row in df.iterrows():
    subject = row[1]['subject_ids']
    data = row[1]['subject_data']
    if subject in subjects:
        continue
    elif subject in skip_subjects:
        continue
    else:
        if 'SUB' in data:
            skip_subjects.append(subject)
        elif 'DUD' in data:
            subjects.append(subject)
            golds.append(0)
        elif 'LENS' in data:
            subjects.append(subject)
            golds.append(1)

golds = pd.DataFrame({'subject': subjects, 'gold': golds})
golds.to_csv(out_path, index=False)
