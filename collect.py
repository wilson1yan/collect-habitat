from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union
from tqdm import tqdm
import os
import os.path as osp
import habitat
import numpy as np
import cv2
import quaternion
import skvideo.io
import multiprocessing as mp
import argparse
import glob

from habitat.core.simulator import ShortestPathPoint
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.datasets.utils import get_action_shortest_path
from habitat.datasets.pointnav.pointnav_generator import is_compatible_episode
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
    target_rotation = None,
    is_gen_shortest_path: bool = True,
    shortest_path_success_distance: float = 0.2,
    shortest_path_max_steps: int = 500,
    closest_dist_limit: float = 1,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.1,
    number_retries_per_target: int = 10,
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
        for _retry in range(number_retries_per_target):
            target_position = sim.sample_navigable_point()

            is_compatible, dist = is_compatible_episode(
                source_position,
                target_position,
                sim,
                near_dist=closest_dist_limit,
                far_dist=furthest_dist_limit,
                geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
            )
            if is_compatible:
                break

    shortest_paths = None
    if is_gen_shortest_path:
        try:
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
        # Throws an error when it can't find a path
        except Exception:
            pass

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
    return episode


def gen_traj(sim, ep_len=500):
    out = generate_pointnav_episode(sim, source_mode='random')
    start_position, start_rotation = out.start_position, out.start_rotation
    actions = [point.action for point in out.shortest_paths[0]]
    while len(actions) < ep_len:
        out = generate_pointnav_episode(sim, source_mode='sim')
        actions.extend([point.action for point in out.shortest_paths[0]])
    return actions[:ep_len], (start_position, start_rotation)


def main(scenes):
    total = len(scenes) * args.n_traj
    n = 0
    for scene in scenes:
        name = osp.basename(scene)[:-4]

        cfg = habitat.get_config()
        cfg.defrost()
        cfg.SIMULATOR.SCENE = scene
        cfg.freeze()

        sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
        sim.seed(args.seed)
        for i in range(args.n_traj):
            output_path = osp.join(args.output, f'{name}_{i}')
            actions, (start_position, start_rotation) = gen_traj(sim)

            sim.reset()
            sim.set_agent_state(start_position, start_rotation)

            video = []
            for act in actions:
                obs = sim.step(act)['rgb']
                H, W = obs.shape[:2]
                L = min(H, W)
                ih = (H - L) // 2
                iw = (W - L) // 2
                obs = obs[ih:ih+L, iw:iw+L]
                obs = cv2.resize(obs, dsize=(args.resolution, args.resolution), interpolation=cv2.INTER_LINEAR)
                video.append(obs)
            video = np.stack(video, axis=0)
            actions = np.array(actions).astype(np.int32)

            skvideo.io.vwrite(output_path + '.mp4', video)
            np.save(output_path + '.npy', actions)
            n += 1

            print(f'completed {n}/{total}')
        sim.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_traj', type=int, default=10)
    parser.add_argument('-r', '--resolution', type=int, default=256)
    parser.add_argument('-o', '--output', type=str, default='/shared/wilson/datasets/habitat_samples')
    parser.add_argument('-s', '--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    paths = glob.glob('/shared/wilson/datasets/gibson/*.glb')
    main(paths)
