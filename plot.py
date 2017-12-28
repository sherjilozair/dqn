import numpy as np
import sys
import re
import subprocess

def get_job_name(jobid):
    cmd = "sacct --format=\"JobName%30\" -j {}".format(jobid)
    result = subprocess.check_output(cmd, shell=True)
    return str(result).split("\\n")[2].strip()

jobids = sys.argv[2:]
exp_name = [get_job_name(jobid) for jobid in jobids]
exp_data = [np.load("{}/{}.npy".format(jobid, jobid)) for jobid in jobids]

N = int(sys.argv[1])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

exp_data = map(lambda d: np.convolve(d, np.ones((N,))/N, mode='valid'), exp_data)

plt.figure(figsize=(16, 8))

for data in exp_data:
    plt.plot(data)

plt.xlabel('#episodes')
plt.ylabel('returns')
plt.legend(exp_name, loc='best')

plotpath = "plot_{}.png".format("_".join(jobids))
print ("Saved to {}".format(plotpath))
plt.savefig(plotpath)
