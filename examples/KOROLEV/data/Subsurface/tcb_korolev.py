import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = "tcb_korolev_Dec2012.dat"

# Read all lines
with open(file_path, "r") as f:
    lines = f.readlines()

output_data = []
clean_rows = []

for line in lines:
    line = line.strip()
    
    # Skip non data lines
    if "#" in line or "@" in line or len(line) == 0:
        continue

    # if the end of the data is reached export to output array
    if line == "EOD":
        output_data.append(np.array(clean_rows))
        del clean_rows
        clean_rows = []
        continue
    
    parts = line.split()
    
    numeric_row = [int(parts[0][-10:]), int(parts[1][:-3]), float(parts[2])]
    clean_rows.append(numeric_row)

# make a plot
for dat in output_data:
    
    # just select the region we care about
    dat_sel = dat[dat[:, 0] == 554201000]

    # now remove everything outside of crater
    cmin = 873
    cmax = 1006
    dat_sel = dat_sel[(dat_sel[:, 1] > cmin) * (dat_sel[:, 1] < cmax)]

    trc = dat_sel[:, 1]
    depth = dat_sel[:, 2]

    # convert depth into actual depth
    depth -= 9750
    depth = (((depth)*(1e-8))*(299.792458e6 / 2) / np.sqrt(3.15)) / 1e3
    
    # now plot
    plt.plot(trc, depth, color="black", linewidth=1)

plt.gca().invert_yaxis()
plt.title("Korolev Interior for 554201000")
plt.ylabel("Milliseconds")
plt.xlabel("Trace #")
plt.savefig("../../figures/horizons.png")
plt.close()