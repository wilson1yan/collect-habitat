import habitat
from habitat.datasets.pointnav.pointnav_generator import generate_pointnav_episode
import skvideo.io
import numpy as np


def main(scene):
    cfg = habitat.get_config()
    cfg.defrost()
    cfg.SIMULATOR.SCENE = scene
    cfg.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
    print(sim)
    out = list(generate_pointnav_episode(sim, 1, is_gen_shortest_path=True))
    video = []
    for point in out[0].shortest_paths[0]:
        obs = sim.step(point.action)
        video.append(obs['rgb'])
    video = np.stack(video, axis=0)
    print(video.shape, video.dtype)
    skvideo.io.vwrite('video.mp4', video, outputdict={'-r': '4'})


if __name__ == '__main__':
    main('/home/wilson/habitat/scene_datasets/habitat-test-scenes/apartment_1.glb')
