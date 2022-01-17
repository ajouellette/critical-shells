import sys
import os
import numpy as np
import matplotlib.pyplot as plt


def parse_sync_point(line, keys):
    assert line.startswith("Sync-Point")
    values = []
    line = line.strip().replace(' ', '').split(',')
    for key in keys:
        for str in line:
            if str.startswith(key):
                values.append(float(str.split(':')[-1]))
                line.remove(str)
    return values


if __name__ == "__main__":

    if len(sys.argv) >= 2:
        out_file = sys.argv[1]
    else:
        out_file = "job.output"

    if not os.path.exists(out_file):
        print(f"Error: file {out_file} not found.")
        sys.exit(1)

    redshifts = []
    times = []
    time_steps = []
    keys = ["Time", "Redshift", "Systemstep"]
    output_dir = None
    time_spent = []
    time_spent_step = None
    with open(out_file, 'r') as file:
        for line in file:
            if "OutputDir" in line:
                output_dir = line.strip().split(' ')[-1]
                print(f"Found output location {output_dir}.")

            if line.startswith("Sync-Point"):
                if time_spent_step != None:
                    time_spent.append(time_spent_step)
                time_spent_step = 0
                time, redshift, time_step = parse_sync_point(line, keys)
                times.append(time)
                redshifts.append(redshift)
                time_steps.append(time_step)

            if time_spent_step != None:
                if "DOMAIN:" in line and "took in total" in line:
                    time_spent_step += float(line.split(' ')[7])
                if "PEANO:" in line and "took" in line:
                    time_spent_step += float(line.split(' ')[3])
                if "PM-PERIODIC:" in line and "took" in line:
                    time_spent_step += float(line.split(' ')[4])
                if "GRAVTREE:" in line and "construction" in line and "took" in line:
                    time_spent_step += float(line.split(' ')[5])
                if "GRAVTREE:" in line and "calculated" in line and "took" in line:
                    time_spent_step += float(line.split(' ')[8])
                if "SNAPSHOT:" in line and "done" in line and "took" in line:
                    time_spent_step += float(line.split(' ')[7])

    plt.figure()
    plt.plot(times)
    plt.xlabel("Sync point")
    plt.ylabel("scale factor")
    if np.max(times) > 100*np.min(times):
        plt.yscale('log')

    plt.figure()
    plt.plot(redshifts)
    plt.xlabel("Sync point")
    plt.ylabel("redshift")

    plt.figure()
    plt.plot(time_steps)
    plt.xlabel("Sync point")
    plt.ylabel("Time step")

    plt.figure()
    plt.plot(time_spent)
    plt.xlabel("Sync Point")
    plt.ylabel("time spent per step, s")
    if np.max(times) > 100*np.min(times):
        plt.yscale('log')

    plt.figure()
    plt.plot(np.cumsum(time_spent) / 3600)
    plt.xlabel("Sync Point")
    plt.ylabel("cumulative time spent, hrs")
    plt.show()
