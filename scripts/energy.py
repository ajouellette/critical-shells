import sys
import time
import glob
import numpy as np
from mpi4py import MPI
import h5py
from gadgetutils import snapshot
from gadgetutils import energy
from gadgetutils.utils import mean_pos_pbc, center_box_pbc


G_sim_units = 43.007  # Gravitiational constant in Mpc/h, 1e10 Msun/h, km/s


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    data_dir = sys.argv[1]
    subhalos = "subhalos" in sys.argv

    files = glob.glob(data_dir + "/snapshot_*.hdf5")
    files.sort()

    if rank == 0:
        print(f"Using data directory {data_dir}, found {len(files)} output files.")

        a_vals = []
        all_all_tu_vals = []
        avg_tu = []
        std_tu = []

    for f in files:
        comm.Barrier()

        pd = snapshot.ParticleData(f)
        if rank == 0:
            print(f"Starting {f}")
            print(f"a = {pd.a:.3f}")

        halos_fname = data_dir + f"/fof_subhalo_tab_{pd.snap_num:03}.hdf5"
        if rank == 0:
            print(f"Using group catalog {halos_fname}")
        hc = snapshot.HaloCatalog(halos_fname, load_subhalos=True)
        Nhalos = hc.n_subhalos if subhalos else hc.n_halos

        if Nhalos < 1:
            if rank == 0:
                print("No halos present.")
            continue
        if rank == 0:
            print(f"{Nhalos} halos present.")

        avg, res = divmod(int(Nhalos), int(n_ranks))
        count = np.array([avg+1 if r < res else avg for r in range(n_ranks)])
        displ = np.array([sum(count[:r]) for r in range(n_ranks)])

        if rank == 0:
            shuffle = np.arange(Nhalos, dtype=int)
            np.random.shuffle(shuffle)
        else:
            shuffle = np.zeros(Nhalos, dtype=int)

        comm.Bcast(shuffle, root=0)

        tu_vals = np.zeros(count[rank])
        time_start1 = time.perf_counter()
        for i in range(count[rank]):
            halo_ind = shuffle[displ[rank] + i]
            inds = hc.get_particles(halo_ind, subhalo=subhalos)
            pos = pd.pos[inds]
            pos_center = mean_pos_pbc(pos, pd.box_size)
            pos = pd.a * center_box_pbc(pos, pos_center, pd.box_size)
            vel = pd.vel[inds]
            vel_center = np.mean(vel, axis=0)
            vel = np.sqrt(pd.a) * (vel - vel_center)

            time_start2 = time.perf_counter()
            T = energy.calc_kinetic(vel)
            U = energy.calc_potential(pos, pd.part_mass / 1e10, G_sim_units)
            time_end2 = time.perf_counter()
            time_ms = (time_end2 - time_start2) * 1e3

            if halo_ind < 10:
                print(f"rank {rank} halo {halo_ind} {len(inds)} particles {T/U:.3f} in {time_ms:.2f} ms")
            tu_vals[i] = T/U

        time_end1 = time.perf_counter()
        time_avg = (time_end1 - time_start1) / count[rank] * 1e3
        print(f"Rank {rank} done. Average of {time_avg} ms per halo.")

        comm.Barrier()

        if rank == 0:
            all_tu_vals = np.zeros(Nhalos)
        else:
            all_tu_vals = None

        comm.Gatherv(tu_vals, [all_tu_vals, count, displ, MPI.DOUBLE], root=0)

        if rank == 0:
            print("Gathered")
            print(np.min(all_tu_vals), np.max(all_tu_vals))
            print(f"Average {np.mean(all_tu_vals)}")
            print(f"Std dev {np.std(all_tu_vals)}")

            a_vals.append(pd.a)
            all_all_tu_vals.append(all_tu_vals[np.argsort(shuffle)])
            avg_tu.append(np.mean(all_tu_vals))
            std_tu.append(np.std(all_tu_vals))

    if rank == 0:
        out_file = data_dir + f"-analysis/energies{'_subhalos' if subhalos else ''}.hdf5"
        print("Saving data to", out_file)
        with h5py.File(out_file, 'w') as f:
            f.create_dataset("ScaleFactors", data=a_vals)
            f.create_dataset("AvgRatio", data=avg_tu)
            f.create_dataset("StdRatio", data=std_tu)
            for i in range(len(a_vals)):
                f.create_dataset(f"Ratios_a{i}", data=all_all_tu_vals[i])
        print("Done.")


if __name__ == "__main__":
    main()
