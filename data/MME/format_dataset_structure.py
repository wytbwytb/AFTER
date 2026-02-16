import os
from tqdm import tqdm

os.system('rm -rf /path/to/your/workdir/AFTER/data/MME/images')
os.system('mkdir /path/to/your/workdir/AFTER/data/MME/images')

os.system('cp -r /path/to/your/workdir/AFTER/data/MME/OCR /path/to/your/workdir/AFTER/data/MME/images/')

os.system('mkdir /path/to/your/workdir/AFTER/data/MME/images/artwork')
os.system('cp /path/to/your/workdir/AFTER/data/MME/artwork/questions_answers_YN/* /path/to/your/workdir/AFTER/data/MME/images/artwork/')
with open('LaVIN/artwork.txt') as fin:
    paths = [ line.strip().split('\t', 1)[0] for line in fin ]
    paths = list(set(paths))
    for path in tqdm(paths):
        os.system(f'cp /path/to/your/workdir/AFTER/data/MME/artwork/images/toy_dataset/{path} /path/to/your/workdir/AFTER/data/MME/images/artwork/{path}')

os.system('mkdir /path/to/your/workdir/AFTER/data/MME/images/celebrity')
os.system('cp /path/to/your/workdir/AFTER/data/MME/celebrity/images/* /path/to/your/workdir/AFTER/data/MME/images/celebrity/')
os.system('cp /path/to/your/workdir/AFTER/data/MME/celebrity/questions_answers_YN/* /path/to/your/workdir/AFTER/data/MME/images/celebrity/')

os.system('cp -r /path/to/your/workdir/AFTER/data/MME/code_reasoning /path/to/your/workdir/AFTER/data/MME/images/')

os.system('cp -r /path/to/your/workdir/AFTER/data/MME/color /path/to/your/workdir/AFTER/data/MME/images/')

os.system('cp -r /path/to/your/workdir/AFTER/data/MME/commonsense_reasoning /path/to/your/workdir/AFTER/data/MME/images/')

os.system('cp -r /path/to/your/workdir/AFTER/data/MME/count /path/to/your/workdir/AFTER/data/MME/images/')

os.system('cp -r /path/to/your/workdir/AFTER/data/MME/existence /path/to/your/workdir/AFTER/data/MME/images/')

os.system('mkdir /path/to/your/workdir/AFTER/data/MME/images/landmark')
os.system('cp /path/to/your/workdir/AFTER/data/MME/landmark/images/* /path/to/your/workdir/AFTER/data/MME/images/landmark/')
os.system('cp /path/to/your/workdir/AFTER/data/MME/landmark/questions_answers_YN/* /path/to/your/workdir/AFTER/data/MME/images/landmark/')

os.system('cp -r /path/to/your/workdir/AFTER/data/MME/numerical_calculation /path/to/your/workdir/AFTER/data/MME/images/')

os.system('cp -r /path/to/your/workdir/AFTER/data/MME/position /path/to/your/workdir/AFTER/data/MME/images/')

os.system('mkdir /path/to/your/workdir/AFTER/data/MME/images/posters')
os.system('cp /path/to/your/workdir/AFTER/data/MME/posters/images/* /path/to/your/workdir/AFTER/data/MME/images/posters/')
os.system('cp /path/to/your/workdir/AFTER/data/MME/posters/questions_answers_YN/* /path/to/your/workdir/AFTER/data/MME/images/posters/')

os.system('mkdir /path/to/your/workdir/AFTER/data/MME/images/scene')
os.system('cp /path/to/your/workdir/AFTER/data/MME/scene/images/* /path/to/your/workdir/AFTER/data/MME/images/scene/')
os.system('cp /path/to/your/workdir/AFTER/data/MME/scene/questions_answers_YN/* /path/to/your/workdir/AFTER/data/MME/images/scene/')

os.system('cp -r /path/to/your/workdir/AFTER/data/MME/text_translation /path/to/your/workdir/AFTER/data/MME/images/')