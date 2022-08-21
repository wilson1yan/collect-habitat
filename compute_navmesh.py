import os.path as osp
from tqdm import tqdm
import glob
import habitat_sim
import multiprocessing as mp


def compute_navmesh(scene_path):
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()

    keys = ['cell_size', 'cell_height', 'agent_height', 'agent_radius', 'agent_max_climb', 'agent_max_slope', 'filter_low_hanging_obstacles', 'filter_ledge_spans', 'filter_walkable_low_height_spans', 'region_min_size', 'region_merge_size', 'edge_max_len', 'edge_max_error', 'verts_per_poly', 'detail_sample_dist', 'detail_sample_max_error']

    update = dict(agent_height=1.4, agent_radius=0.01, agent_max_climb=0.5, agent_max_slope=60)

    for k, v in update.items():
        assert hasattr(navmesh_settings, k), f'Could not find {k}!'
        setattr(navmesh_settings, k, v)


    sim_settings = {
        'scene': scene_path,
        'default_agent': 0,
        'sensor_height': 1.4,
        'width': 256,
        'height': 256,
        'enable_physics': False
    }

    def make_cfg(settings):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_id = settings["scene"]
        sim_cfg.enable_physics = settings["enable_physics"]

        # Note: all sensors must have the same resolution
        sensor_specs = []

        color_sensor_spec = habitat_sim.CameraSensorSpec()
        color_sensor_spec.uuid = "color_sensor"
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_spec.resolution = [settings["height"], settings["width"]]
        color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_spec)

        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [settings["height"], settings["width"]]
        depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(depth_sensor_spec)

        semantic_sensor_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic_sensor"
        semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
        semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(semantic_sensor_spec)

        # Here you can specify the amount of displacement in a forward action and the turn angle
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
            ),
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)
    navmesh_success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings, include_static_objects=False)

    #if navmesh_success:
    #    save_path = scene_path[:-3] + 'navmesh'
    #    sim.pathfinder.save_nav_mesh(save_path)
    return navmesh_success


if __name__ == '__main__':
    scene_paths = glob.glob(osp.join('/shared/wilson/datasets/3d_scenes', '**', '*.glb'), recursive=True)[:100]
    print(f'Found {len(scene_paths)} glb files')

    pool = mp.Pool(64)
    results = list(tqdm(pool.imap(compute_navmesh, scene_paths), total=len(scene_paths)))
    failed = [path for path, result in zip(scene_paths, results) if not result]
    print(f'Failed {len(failed)}')
    json.dump(failed, open('failed.json', 'w'))
