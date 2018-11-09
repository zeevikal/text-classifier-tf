import os
from tqdm import tqdm

DATA_DIR = "data"


def file_len(fname):
    with open(fname, encoding="utf8") as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def csv_data_to_text(data_dirs):
    with open(r'{}/pos.txt'.format(DATA_DIR), 'w') as cs_outfile:
        with open(r'{}/neg.txt'.format(DATA_DIR), 'w') as other_outfile:
            for data_dir in data_dirs:
                for txt_file in tqdm(os.listdir(data_dir)):
                    if data_dir.__contains__('CS'):
                        if txt_file.endswith(".txt"):
                            with open(txt_file) as infile:
                                for line in infile:
                                    cs_outfile.write(line)
                    else:
                        if txt_file.endswith(".txt"):
                            with open(txt_file) as infile:
                                for line in infile:
                                    other_outfile.write(line)

    print('pos len: ', file_len(r'{}/pos.txt'.format(DATA_DIR)))
    print('neg len: ', file_len(r'{}/neg.txt'.format(DATA_DIR)))


if __name__ == '__main__':
    csv_data_to_text(DATA_DIR)
