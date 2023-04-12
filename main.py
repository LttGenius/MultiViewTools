import time
from tqdm import tqdm
pbar = tqdm(range(100))
for i in pbar:
    time.sleep(.01)
    pbar.set_description("Processing %s" % i)

