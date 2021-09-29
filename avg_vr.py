import numpy as np
from mpi4py import MPI
from astropy.cosmology import FlatLambdaCDM
import pickle
import snapshot


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    particles = snapshot.ParticleData("/projects/caps/aaronjo2/dm-l256-n256-a100/snapshot_021.hdf5")

    cosmo = FlatLambdaCDM(H0=particles.Hubble0, Om0=particles.OmegaMatter)

    if rank == 0:
        with open("spherical_halos_mpi", 'rb') as f:
            data = pickle.load(f)
        all_centers = data["centers"]
        all_radii = data["radii"]
        avg, res = divmod(len(all_radii), n_ranks)
        count = np.array([avg+1 if r < res else avg for r in range(n_ranks)])
        displ = np.array([sum(count[:r]) for r in range(n_ranks)])
    else:
        all_centers = None
        all_radii = None
        count = np.zeros(n_ranks, dtype=int)
        displ = 0

    comm.Bcast(count, root=0)

    centers = np.zeros((count[rank], 3))
    radii = np.zeros(count[rank])

    comm.Scatterv([all_radii, count, displ, MPI.DOUBLE], radii, root=0)
    comm.Scatterv([all_centers, 3*count, 3*displ, MPI.DOUBLE], centers, root=0)

    avg_vr = np.zeros_like(radii)
    std_vr = np.zeros_like(radii)
    n_parts = np.zeros_like(radii)

    for i in  range(len(radii)):
        center = centers[i]
        radius = radii[i] / cosmo.h
        p_radii = np.linalg.norm(particles.pos - center, axis=1) / cosmo.h
        mask = p_radii < radius
        vel_center = np.mean(particles.vel[mask], axis=0)
        pos_cut = (particles.pos[mask] - center) / cosmo.h
        vel_cut = np.sqrt(particles.a) * (particles.vel[mask] - vel_center)
        p_radii_cut = p_radii[mask]
        unit_vectors = pos_cut / np.expand_dims(p_radii_cut, 1)
        vr_peculiar = np.multiply(vel_cut, unit_vectors).sum(1)
        hubble_flow = particles.Hubble * particles.a * p_radii_cut
        vr_physical = vr_peculiar + hubble_flow
        avg_vr[i] = np.mean(vr_physical)
        std_vr[i] = np.std(vr_physical)
        n_parts[i] = len(vr_physical)

        print(avg_vr[i], std_vr[i], n_parts[i])
        print()

    comm.Barrier()
    if rank == 0:
        print("Done")
        all_avg_vr = np.zeros_like(all_radii)
        all_std_vr = np.zeros_like(all_radii)
        all_n_parts = np.zeros_like(all_radii)
    else:
        all_avg_vr = None
        all_std_vr = None
        all_n_parts = None

    comm.Gatherv(avg_vr, [all_avg_vr, count, displ, MPI.DOUBLE], root=0)
    comm.Gatherv(std_vr, [all_std_vr, count, displ, MPI.DOUBLE], root=0)
    comm.Gatherv(n_parts, [all_n_parts, count, displ, MPI.DOUBLE], root=0)

    if rank == 0:
        stats = {"avg":all_avg_vr, "std":all_std_vr, "n":all_n_parts}
        with open("vr_stats", 'wb') as f:
            pickle.dump(stats, f)

