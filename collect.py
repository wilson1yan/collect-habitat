from typing import Dict, List, Optional, Union
from tqdm import tqdm
import os
import os.path as osp
import habitat
import numpy as np
import skvideo.io
import argparse
import glob
import random

from habitat.core.simulator import ShortestPathPoint
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.datasets.utils import get_action_shortest_path
from habitat.utils.geometry_utils import quaternion_to_list


def _create_episode(
    episode_id: Union[int, str],
    scene_id: str,
    start_position: List[float],
    start_rotation: List[float],
    target_position: List[float],
    shortest_paths: Optional[List[List[ShortestPathPoint]]] = None,
    radius: Optional[float] = None,
    info: Optional[Dict[str, float]] = None,
):
    goals = [NavigationGoal(position=target_position, radius=radius)]
    return NavigationEpisode(
        episode_id=str(episode_id),
        goals=goals,
        scene_id=scene_id,
        start_position=start_position,
        start_rotation=start_rotation,
        shortest_paths=shortest_paths,
        info=info,
    )

def generate_pointnav_episode(
    sim: "HabitatSim",
    source_mode = 'sim',
    target_position = None,
    shortest_path_success_distance: float = 0.2,
    shortest_path_max_steps: int = 500,
    number_retries_per_target: int = 100,
    keep_same_floor=False,
):
    if source_mode == 'sim':
        source_position = sim.agents[0].get_state().position.tolist()
        source_rotation = quaternion_to_list(sim.agents[0].get_state().rotation)
    elif source_mode == 'random':
        source_position = sim.sample_navigable_point()
        angle = np.random.uniform(0, 2 * np.pi)
        source_rotation = [0.0, np.sin(angle / 2), 0, np.cos(angle / 2)]
    else:
        raise Exception(source_mode)

    if target_position is None:
        sampled_positions = []
        for _retry in range(number_retries_per_target):
            target_position = sim.sample_navigable_point()
            if keep_same_floor and np.abs(source_position[1] - target_position[1]) > 0.5:
                continue
            dist = sim.geodesic_distance(source_position, target_position)
            if args.min_dist <= dist <= args.max_dist and dist != np.inf:
                sampled_positions.append((target_position, sim.geodesic_distance(source_position, target_position)))
        sampled_positions.sort(key=lambda x: x[1])
        target_position, dist = random.choice(sampled_positions[-50:])

    shortest_paths = [
        get_action_shortest_path(
            sim,
            source_position=source_position,
            source_rotation=source_rotation,
            goal_position=target_position,
            success_distance=shortest_path_success_distance,
            max_episode_steps=shortest_path_max_steps,
        )
    ]

    episode = _create_episode(
        episode_id=0,
        scene_id=sim.habitat_config.SCENE,
        start_position=source_position,
        start_rotation=source_rotation,
        target_position=target_position,
        shortest_paths=shortest_paths,
        radius=shortest_path_success_distance,
        info={"geodesic_distance": None},
    )
    return episode, target_position


def gen_traj(sim, ep_len):
    keep_same_floor = random.random() < 0.75
    out, target_position = generate_pointnav_episode(sim, source_mode='random', keep_same_floor=keep_same_floor)
    start_position, start_rotation = out.start_position, out.start_rotation
    points = [start_position, target_position]
    actions = [point.action for point in out.shortest_paths[0]]
    while len(actions) < ep_len:
        if len(points) >= args.n_points:
            target_position = random.choice(points)
            out, _ = generate_pointnav_episode(sim, source_mode='sim', target_position=target_position)
        else:
            keep_same_floor = random.random() < 0.75
            out, target_position = generate_pointnav_episode(sim, source_mode='sim', keep_same_floor=keep_same_floor)
            points.append(target_position)
        actions.extend([point.action for point in out.shortest_paths[0]])
    return actions[:ep_len], (start_position, start_rotation)


def main(scenes):
    total = len(scenes) * args.n_traj
    pbar = tqdm(total=total)
    for scene in scenes:
        name = osp.basename(scene)[:-4]

        cfg = habitat.get_config()
        cfg.defrost()
        cfg.SIMULATOR.SCENE = scene
        cfg.SIMULATOR.AGENT_0.RADIUS = 0.01
        cfg.SIMULATOR.RGB_SENSOR.HEIGHT = args.resolution
        cfg.SIMULATOR.RGB_SENSOR.WIDTH = args.resolution 
        cfg.freeze()

        sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
        sim.seed(random.randint(0, 1000000000))
        i = 0
        while i < args.n_traj:
            output_path = osp.join(args.output, f'{name}_{i}')
            try:
                actions, (start_position, start_rotation) = gen_traj(sim, args.traj_length)
            except Exception:
                continue

            sim.reset()
            sim.set_agent_state(start_position, start_rotation)

            video = []
            for act in actions:
                obs = sim.step(act)
                video.append(obs['rgb'])
            video = np.stack(video, axis=0)
            actions = np.array(actions).astype(np.int32)

            skvideo.io.vwrite(output_path + '.mp4', video)
            np.savez_compressed(output_path + '.npz', actions=actions)
            i += 1
            pbar.update(1)
        sim.close()
    pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-d', '--data', type=str, required=True)
    
    parser.add_argument('-n', '--n_traj', type=int, default=160)
    parser.add_argument('--n_chunks', type=int, default=1)
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('-l', '--traj_length', type=int, default=300)
    parser.add_argument('-r', '--resolution', type=int, default=128)
    parser.add_argument('--min_dist', type=float, default=1)
    parser.add_argument('--max_dist', type=float, default=15)
    parser.add_argument('--n_points', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    paths = glob.glob(osp.join(args.data, '**', '*.glb'), recursive=True)
    paths.sort()

    paths = np.array_split(paths, args.n_chunks)[args.chunk_idx].tolist()

    main(paths)
