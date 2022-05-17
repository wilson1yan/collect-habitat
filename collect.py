from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union
import habitat
import numpy as np
import cv2
import quaternion
import skvideo.io

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


def main(scene):
    cfg = habitat.get_config()
    cfg.defrost()
    cfg.SIMULATOR.SCENE = scene
    cfg.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
    out = generate_pointnav_episode(sim, source_mode='sim')
    out2 = generate_pointnav_episode(sim, source_mode='sim', target_position=out.start_position, target_rotation=out.start_rotation)
    actions = [point.action for point in out.shortest_paths[0]]
    actions.extend([point.action for point in out2.shortest_paths[0]])

    sim.reset()
    sim.set_agent_state(out.start_position, out.start_rotation)

    video = []
    for act in actions:
        obs = sim.step(act)
        video.append(obs['rgb'])

    video = np.stack(video, axis=0)
    print(video.shape)

    skvideo.io.vwrite('video.mp4', video)


if __name__ == '__main__':
    main('scene_datasets/habitat-test-scenes/apartment_1.glb')
