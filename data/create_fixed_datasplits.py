import csv
import os
import shutil
from shuffle_data import shuffle_csv

'''
place original dataset in data/bias and create a fixed dev split
take either first or last 0.1 of the train set for dev (balanced classes)
'''

def create_fixed_splits(data_directory: str = "bias",
                        train_filename: str = "train_bias",
                        test_filename: str = "test_bias",
                        out_directory: str = "fixedsplits/bias",
                        use_head_of_train_set: bool = True,  # use first 0.1 of the train set for dev set
                        ):
    train_set_path = f"{data_directory}/{train_filename}.csv"
    test_set_path = f"{data_directory}/{test_filename}.csv"

    # store sentences in a dictionary, sort by bias labels
    train_set = {}
    new_train_set, dev_set = {}, {}
    train_set_size = 0

    with open(train_set_path) as file_obj:
        reader = csv.reader(file_obj)
        next(reader)  # skip header
        for row in reader:
            bias, publisher, text = row[0], row[1], row[2]
            if bias not in train_set:
                train_set[bias] = []
            train_set[bias].append((publisher, text))
            train_set_size += 1

    num_classes = len(list(train_set.keys()))
    dev_set_size = train_set_size * 0.1
    dev_per_class_size = int(dev_set_size / num_classes)

    for bias_label in train_set:
        if use_head_of_train_set:
            new_train_set[bias_label] = train_set[bias_label][dev_per_class_size:]
            dev_set[bias_label] = train_set[bias_label][:dev_per_class_size]
        else:
            new_train_set[bias_label] = train_set[bias_label][:len(train_set[bias_label]) - dev_per_class_size]
            dev_set[bias_label] = train_set[bias_label][len(train_set[bias_label]) - dev_per_class_size:]

    # create a directory for new train, dev, test splits
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    # write dev split
    with open(f'{out_directory}/dev.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'publisher', 'text'])
        for bias_label in dev_set:
            for sample in dev_set[bias_label]:
                writer.writerow([bias_label, sample[0], sample[1]])

    # write train split
    with open(f'{out_directory}/train.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'publisher', 'text'])
        for bias_label in new_train_set:
            for sample in new_train_set[bias_label]:
                writer.writerow([bias_label, sample[0], sample[1]])

    # copy test split
    shutil.copyfile(test_set_path, f'{out_directory}/test.csv')

    # print some info just to check
    print("NEW TRAIN SET INFO")
    for bias_label in new_train_set:
        print(f"Label: {bias_label}, Instances: {len(new_train_set[bias_label])}")

    print("\nNEW DEV SET INFO")
    for bias_label in dev_set:
        print(f"Label: {bias_label}, Instances: {len(dev_set[bias_label])}")

create_fixed_splits(data_directory='bias',
                    out_directory='fixedsplits/bias',
                    use_head_of_train_set=True,
                    )

shuffle_csv(filename='fixedsplits/bias/train.csv',
            out_filename='fixedsplits/bias/train.csv')
