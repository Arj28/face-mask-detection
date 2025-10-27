# prepare_dataset.py
# Optional helper: move images from raw_dataset/with_mask and raw_dataset/without_mask
# into dataset/train/... and dataset/test/... with a train/test split.
import os, shutil, random

RAW_WITH = 'raw_dataset/with_mask'
RAW_WITHOUT = 'raw_dataset/without_mask'
DEST_ROOT = 'dataset'   # will create dataset/train and dataset/test
TRAIN_RATIO = 0.8

def make_dirs():
    for cls in ['with_mask','without_mask']:
        for split in ['train','test']:
            path = os.path.join(DEST_ROOT, split, cls)
            os.makedirs(path, exist_ok=True)

def split_and_copy(src, cls_name):
    files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src,f))]
    random.shuffle(files)
    cut = int(len(files)*TRAIN_RATIO)
    train_files = files[:cut]
    test_files  = files[cut:]
    for f in train_files:
        shutil.copy(os.path.join(src,f), os.path.join(DEST_ROOT,'train',cls_name,f))
    for f in test_files:
        shutil.copy(os.path.join(src,f), os.path.join(DEST_ROOT,'test',cls_name,f))

if __name__ == '__main__':
    make_dirs()
    if os.path.exists(RAW_WITH) and os.path.exists(RAW_WITHOUT):
        split_and_copy(RAW_WITH, 'with_mask')
        split_and_copy(RAW_WITHOUT, 'without_mask')
        print('Done. Created dataset/train and dataset/test.')
    else:
        print('raw_dataset/with_mask and/or raw_dataset/without_mask not found. Place your raw images there or organize dataset/ with_mask & without_mask directly.')