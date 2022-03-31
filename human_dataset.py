"""
human_dataset.py
Create a dataset, where 1 means, that there is a human on the image and 0 means, that there isn't.
"""

import os
import random

has_human = os.listdir('./out/radar/')[1:]
no_human = os.listdir('./out/no_human/')[1:]

random.shuffle(has_human)
random.shuffle(no_human)

l1 = len(has_human)
l2 = len(no_human)
l = min(l1, l2)

with open('hashuman.csv', 'w') as f:
    f.write('file,has_human\n')
    for i in range(l):
        f.write(f'/out/radar/{has_human[i]},1\n')
        f.write(f'/out/no_human/{no_human[i]},0\n')

print('Done!')