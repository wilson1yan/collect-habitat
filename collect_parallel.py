import os
import os.path as osp
import argparse
import multiprocessing as mp


def worker(i, args):
    output_dir = osp.join(args.output_path, f'{i}')
    cmd = f'MAGNUM_LOG=quiet GLOG_minloglevel=2 python collect.py -l {args.traj_length} -r {args.resolution} --output {output_dir} --data {args.data_path} --n_traj {args.n_traj} --n_chunks {args.n_parallel} --chunk_idx {i}'
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-d', '--output_path', type=str, required=True)
    parser.add_argument('-n', '--n_traj', type=int, default=160) # per env so total is n_traj * # envs
    parser.add_argument('-l', '--traj_length', type=int, default=300)
    parser.add_argument('-r', '--resolution', type=int, default=128)
    parser.add_argument('-p', '--n_parallel', type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    procs = [mp.Process(target=worker, args=(i, args)) for i in range(args.n_parallel)]
    [p.start() for p in procs]
    [p.join() for p in procs]
