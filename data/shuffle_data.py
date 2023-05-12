import csv
import random

'''
Don't forget to shuffle your train set. CEA model works on batch level so it's bad if a batch only has instances of one class.
'''

def shuffle_csv(filename: str = "train_bias.csv", out_filename: str = "train_bias.csv"):
    rows = []
    with open(filename) as file_obj:
        reader = csv.reader(file_obj)
        next(reader)  # skip header
        for row in reader:
            rows.append(row)
    random.shuffle(rows)

    with open(out_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'publisher', 'text'])
        for row in rows:
            writer.writerow(row)

shuffle_csv('train.csv', 'train.csv')