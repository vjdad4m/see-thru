#!/usr/bin/env python3
import random
import time

N_DATA = 100

t = time.time()

for i in range(N_DATA):
    i = random.choice([True, False])
    ti = t - random.randrange(-1000, 1000) + random.random()
    if i:
        with open(f'./out/radar/{ti}.txt', 'w') as f:
            f.write('0')
    else:
        with open(f'./out/img/{ti}.txt', 'w') as f:
            f.write('0')

        with open(f'./out/pose/{ti}.txt', 'w') as f:
            f.write('0')