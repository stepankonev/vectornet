import argparse

import os
import numpy as np
import multiprocessing
from tqdm import tqdm
import numcodecs
import glob
from sklearn import preprocessing

from l5kit.data import LocalDataManager

import config
import dataset
import utils

from numba import jit

import pandas as pd

IDX = {
    "x": 0, #coord
    "y": 1, #coord
    "prev_x": 2,
    "prev_y": 3,
    "default": 4, #road_type
    "red": 5, #road_type
    "yellow": 6, #road_type
    "green": 7, #road_type
    "crosswalk": 8, #road_type
    "ego": 9,
    "other": 10,
    "timestamp": 11,
    "pindex": 12
}

__x__ = 0
__y__ = 1
__prev_x__ = 2
__prev_y__ = 3
__default__ = 4
__red__ = 5
__yellow__ = 6
__green__ = 7
__crosswalk__ = 8
__ego__ = 9
__other__ = 10
__timestamp__ = 11
__pindex__ = 12


numcodecs.blosc.set_nthreads(1)

os.environ["L5KIT_DATA_FOLDER"] = config.L5KIT_DATA_FOLDER
DST_DIR = None  # f"{config.L5KIT_DATA_FOLDER}/pre_render_320_0.3"

"""
pre_render_512_0.3:
                raster_size=[512, 512],
                pixel_size=[0.3, 0.3],
                ego_center=[0.15625, 0.5]


pre_render_224_0.5:
                raster_size=[224, 224],
                pixel_size=[0.5, 0.5],
                ego_center=[0.25, 0.5]

pre_render_h01248: default
                raster_size=[224, 224],
                pixel_size=[0.5, 0.5],
                ego_center=[0.25, 0.5],
                history_box_frames=[0, 1, 2, 4, 8],

pre_render_h50all: default
                raster_size=[224, 224],
                pixel_size=[0.5, 0.5],
                ego_center=[0.25, 0.5],
                history_box_frames=range(0, 51),
pre_render_320_0.3:
                raster_size=[320, 320],
                pixel_size=[0.3, 0.3],
                ego_center=[0.25, 0.5],

pre_render_256_0.5:
                raster_size=[256, 256],
                pixel_size=[0.5, 0.5],
                ego_center=[0.1875, 0.5],  # keep 0 pos at the center of 32x32 block for better pos encoding

pre_render_288_0.3:
                raster_size=[288, 288],
                pixel_size=[0.3, 0.3],
                ego_center=[0.1666666666666667, 0.5],

pre_render_288_0.5:
                raster_size=[288, 288],
                pixel_size=[0.5, 0.5],
                ego_center=[0.1666666666666667, 0.5],
"""


def create_lyft_dataset(dset_name, zarr_name):
    ds = dataset.LyftDataset(
        dset_name=dset_name,
        cfg_data=dict(
            raster_params=dict(
                raster_size=[224, 224],
                pixel_size=[0.5, 0.5],
                ego_center=[0.25, 0.5],
                map_type="box_semantic_fast",
                satellite_map_key="aerial_map/aerial_map.png",
                semantic_map_key="semantic_map/semantic_map.pb",
                dataset_meta_key="meta.json",
                filter_agents_threshold=0.5,
                disable_traffic_light_faces=False,
                set_origin_to_bottom=True,
            ),
            train_data_loader=dict(key=f"scenes/{zarr_name}.zarr"),
            val_data_loader=dict(key=f"scenes/{zarr_name}.zarr"),
            test_data_loader=dict(key=f"scenes/{zarr_name}.zarr"),
            model_params=dict(
                history_num_frames=20,  # used to retrive appropriate slices of history data from dataset, but only indices form history_box_frames will be rendered by box_semantic_fast rasterizer
                history_step_size=1,
                history_delta_time=0.1,
                history_box_frames=[0, 1, 2, 4, 8],
                future_num_frames=50,
                future_step_size=1,
                future_delta_time=0.1,
                step_time=0.1,
                add_agent_state=False,
            ),
        ),
    )
    return ds


def pos_ahead(agent):
    time_ahead = 2.5
    distance_ahead = 5.0

    xy = agent["centroid"]
    vel = agent["velocity"]
    return xy + vel * time_ahead


def filter_agents(history_agents, current_agent):
    current_agents = history_agents[0]
    agent_pos_ahead = pos_ahead(current_agent)

    current_agents_sorted = list(
        sorted(
            current_agents,
            key=lambda x: np.linalg.norm(agent_pos_ahead - pos_ahead(x)),
        )
    )
    return history_agents


def filter_tl_faces(history_tl_faces):
    return [
        t[t['traffic_light_face_status'][:, 2] < 0.5]
        for t in history_tl_faces
    ]


# @jit(nopython=True)
# def distil(X):
#     X[:, 0] = X[:, 0] / 256.
#     X[:, 1] = X[:, 1] / 256.
#     X[:, 2] = X[:, 2] / 256.
#     X[:, 3] = X[:, 3] / 256.
# #     df = pd.DataFrame(X)
#     prev_pid = None
#     prev_row = None
#     res = []
#     for idx, row in enumerate(X):
#         if row[__ego__] == 1 or row[__other__] == 1 or row[__crosswalk__] == 1:
#             res.append(row)
#             prev_pid = row[__pindex__]
#             continue
#         if row[__pindex__] != prev_pid:
#             if prev_row is not None and not (res[-1] == prev_row).all():
#                 res.append(prev_row)
#             pX, pY = row[0], row[1]
#             prev_pid = row[__pindex__]
#             res.append(row)
#             continue
#         dist = np.sqrt((pX - row[0])**2 + (pY - row[1])**2)
#         prev_row = row
#         if dist > 7:
#             res.append(row)
# #             res[-1][2], res[-1][3] = pX, pY
#             pX, pY = row[0], row[1]
#             prev_pid = row[__pindex__]
#     EDGES = []
#     res = np.array(res)
    
    
#     for idx, row in enumerate(res):
#         if idx == 0:
#             prev_index = row[__pindex__]
#             continue
#         if row[__pindex__] == prev_index:
#             res[idx - 1, __prev_x__] = res[idx, __x__]
#             res[idx - 1, __prev_y__] = res[idx, __y__]
#         prev_index = row[__pindex__]

#     prev_polyline_idx = -1
#     for i in range(res.shape[0]):
#         if res[i, __pindex__] != prev_polyline_idx:
#             prev_polyline_idx = res[i, __pindex__]
#             continue
#         EDGES.append([i-1, i])
#         prev_polyline_idx = res[i, __pindex__]

#     edges = np.array(EDGES)
    
#     return res, edges

@jit(nopython=True)
def distil(X):
    X[:, 0] = X[:, 0] / 256.
    X[:, 1] = X[:, 1] / 256.
    X[:, 2] = X[:, 2] / 256.
    X[:, 3] = X[:, 3] / 256.
#     df = pd.DataFrame(X)
    prev_pid = None
    prev_row = None
    res = []
    for idx in range(X.shape[0]):
        row = X[idx]
        if row[__ego__] == 1 or row[__other__] == 1 or row[__crosswalk__] == 1:
            res.append(row)
            prev_pid = row[__pindex__]
            continue
        if row[__pindex__] != prev_pid:
            if prev_row is not None and not (res[-1][__x__] == prev_row[__x__] and res[-1][__y__] == prev_row[__y__]):
                res.append(prev_row)
            pX, pY = row[0], row[1]
            prev_pid = row[__pindex__]
            res.append(row)
            continue
        dist = np.sqrt((pX - row[0])**2 + (pY - row[1])**2)
        prev_row = row
        if dist > 7:
            res.append(row)
#             res[-1][2], res[-1][3] = pX, pY
            pX, pY = row[0], row[1]
            prev_pid = row[__pindex__]
    EDGES = []
#     res = np.array(res)
    
    
#     res2 = []
    prev_index = -1
    for idx, row in enumerate(res):
        if row[__pindex__] != prev_index:
            prev_x = res[idx][__x__]
            prev_y = res[idx][__y__]
        if row[__pindex__] == prev_index:
            res[idx][__prev_x__] = prev_x#res[idx][ __x__]
            res[idx][__prev_y__] = prev_y#res[idx][ __y__]
            prev_x = res[idx][__x__]
            prev_y = res[idx][__y__]
#             res2.append(res[idx])
        prev_index = row[__pindex__]
    
#     res = res2

    prev_polyline_idx = -1
    for i in range(len(res)):
        if res[i][ __pindex__] != prev_polyline_idx:
            prev_polyline_idx = res[i][ __pindex__]
            continue
        EDGES.append([i-1, i])
        prev_polyline_idx = res[i][ __pindex__]

#     edges = np.array(EDGES)
    
    return res, EDGES


def pre_render_scenes(initial_scene, dset_name, scene_step, zarr_name, skip_frame_step, verbose=10):
    print(f"Job {initial_scene}: creating dataset...")
    with utils.timeit_context(f"Job {initial_scene} dataset creation"):
        print("!!!!", (dset_name, zarr_name))
        ds = create_lyft_dataset(dset_name, zarr_name)
    print(f"Job {initial_scene}: creating dataset... [OK]")

    nb_scenes = len(ds.zarr_dataset.scenes)
    if verbose:
        print("total scenes:", nb_scenes)

    for scene_num in tqdm(
        range(initial_scene, nb_scenes, scene_step),
        desc=f"job {initial_scene} scenes:",
        disable=not verbose,
        total=(nb_scenes - initial_scene) // scene_step,
    ):
        if verbose >= 10:
            print("processing scene", scene_num)
        from_frame, to_frame = ds.zarr_dataset.scenes["frame_index_interval"][scene_num]
#         print("GETTING ALL FRAMES")
        all_frames = ds.zarr_dataset.frames[from_frame:to_frame].copy()
#         print(all_frames)

        for frame_num in range(from_frame, to_frame):
            if (frame_num + 1) % (1 + skip_frame_step) > 0:
                continue

            dir_created = False
            state_index = frame_num - from_frame
            dst_dir = f"{DST_DIR}/{zarr_name}/{scene_num:05}/{state_index:03}/"

            agent_from, agent_to = all_frames["agent_index_interval"][state_index]

            frame_agents = ds.zarr_dataset.agents[agent_from:agent_to]
            # frame_agents_state = ds.agent_state[agent_from:agent_to]
            relative_agents_indices = np.nonzero(ds.agent_dataset.agents_mask[agent_from:agent_to])[0]

            non_masked_agents = {}
            for agent_num in relative_agents_indices:
                non_masked_agents[agent_num + agent_from] = (
                    frame_agents[agent_num],
                    None,  # frame_agents_state[agent_num],
                )

            data_by_agent = {}
            target_by_agent_id = {}

            for agent_num in relative_agents_indices:
                track_id = frame_agents[agent_num]["track_id"]
                
                
                data = ds.agent_dataset.get_frame(scene_num, state_index=state_index, track_id=track_id)
#                 print(ds)
                raster = data["image"]

                del data["image"]
                # data["image"] = img  # np.clip(img * 255.0, 0, 255).astype(np.uint8)
                data["image_semantic"] = raster["image_semantic"]
                data["image_box"] = raster["image_box"]
                data["tl_lanes_masks4"] = raster["tl_lanes_masks4"]

                # data["history_frames"] = raster["history_frames"]
                # data["history_agents"] = filter_agents(raster["history_agents"], frame_agents[agent_num])
                data["history_tl_faces"] = filter_tl_faces(raster["history_tl_faces"])

                # print(img.shape)
                # agent_state = frame_agents_state[agent_num]
                data["agent_state"] = frame_agents[agent_num]
                data["agent_id"] = agent_num + agent_from
                data["scene_id"] = scene_num
                data["frame_id"] = frame_num
                data["non_masked_frame_agents"] = non_masked_agents
#                 data["default"] = raster["default"]
#                 data["green"] = raster["green"]
#                 data["yellow"] = raster["yellow"]
#                 data["red"] = raster["red"]
#                 data["crosswalks"] = raster["crosswalks"]
#                 print(raster["X"].shape, (raster["other"][:,-1] + raster["X"][:,-1].max() + 1).shape)
#                 raster["other"][:,-1] =+ raster["X"][:,-1].max() + 100
                data["X"] = raster["X"]#np.concatenate([raster["X"], raster["other"]], axis = 0)
                le = preprocessing.LabelEncoder()
#                 data["X"][:,-1] = le.fit_transform(data["X"][:,-1])
                data["ego"] = raster["ego"]
                data["other"] = raster["other"]
#                 x[np.lexsort((I, x[:,IDX["timestamp"]]))]
                data["other"] = data["other"][np.lexsort((data["other"][:,IDX["timestamp"]], data["other"][:,IDX["pindex"]]))]
                data["other"][:,-1] = le.fit_transform(data["other"][:,-1])
#                 data["X"][:,-1] = le.fit_transform(data["X"][:,-1])
#                 print(data["other"][:,-1].max() + 1)
                data["X"][:,-1] += int(data["other"][:,-1].max()) + 1
                data["X"] = np.concatenate([data["other"], data["X"]], axis = 0)
                data["X"], data["edges"] = distil(data["X"])
                data["X"] = np.array(data["X"])
                data["edges"] = np.array(data["edges"])
                
                EDGES = []
                
#                 prev_polyline_idx = -1
#                 for i in range(data["X"].shape[0]):
#                     if data["X"][i, IDX["pindex"]] != prev_polyline_idx:
#                         prev_polyline_idx = data["X"][i, IDX["pindex"]]
#                         continue
#                     EDGES.append([i-1, i])
#                     prev_polyline_idx = data["X"][i, IDX["pindex"]]
                
#                 data["edges"] = np.array(EDGES)
                    
                
#                 data["RGB"] = raster["RGB"]
                # warning all_frame_agent is very large and will sacriface the reading speed
#                 data["all_frame_agents"] = frame_agents

                data_by_agent[track_id] = data
                target_by_agent_id[track_id] = {
                    key: data[key]
                    for key in [
                        "target_availabilities",
                        "target_positions",
                        "world_from_agent",
                        "world_to_image",
                        "raster_from_world",
                        "raster_from_agent",
                        "agent_from_world",
                    ]
                }

            for agent_num in relative_agents_indices:
                track_id = frame_agents[agent_num]["track_id"]
                fn = f"{dst_dir}/{track_id:04}.npz"
                # if os.path.exists(fn):
                # if verbose >= 1:
                #    print(f" - {fn} is already rendered")
                # continue

                if not dir_created:
                    os.makedirs(dst_dir, exist_ok=True)
                    dir_created = True

                data = data_by_agent[track_id]
                np.savez_compressed(fn, **data, target_by_agent_id=target_by_agent_id)


def pre_render_parallel(dset_name, scene_step, zarr_name, skip_frame_step, initial_scenes, num_jobs):
    # if zarr_name in ["train_uncompressed", "validate_uncompressed"] and
    if dset_name in [
        "train",
        "train_XXL",
        "val",
    ]:
        try:
            # Agent masks must be generated first
            dm = LocalDataManager(None)
            dm.require(f"scenes/{zarr_name}.zarr/agents_mask/0.5/.zarray")
        except FileNotFoundError:
            print("-- Create dataset to generate agent masks first")
            _ = create_lyft_dataset(dset_name, zarr_name)

    p = multiprocessing.Pool(num_jobs)
    res = []
    for i, initial_scene in enumerate(initial_scenes):
        res.append(
            p.apply_async(
                pre_render_scenes,
                kwds=dict(
                    initial_scene=initial_scene,
                    scene_step=scene_step,
                    zarr_name=zarr_name,
                    skip_frame_step=skip_frame_step,
                    dset_name=dset_name,
                    verbose=(i == 0),
                ),
            )
        )
        # pre_render_scenes(
        #     initial_scene=initial_scene,
        #     scene_step=scene_step,
        #     zarr_name=zarr_name,
        #     skip_frame_step=skip_frame_step,
        #     dset_name=dset_name,
        #     verbose=(i == 0),
        # )

    for r in res:
        print(".")
        r.get()
    print("Done all")

    dataset_dir = f"{DST_DIR}/{zarr_name}"
    all_files_fn = f"{dataset_dir}/filepaths_1_of_{skip_frame_step + 1}.npy"
    print(f"Generating and caching filenames in {all_files_fn}")
    all_files = list(sorted(glob.glob(f"{dataset_dir}/**/*.npz", recursive=True)))
    np.save(all_files_fn, all_files)
    print(f"Generated all npz paths and saved to {all_files_fn}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, default="render")
    parser.add_argument("--dset_name", type=str, default="train")
    parser.add_argument("--zarr_name", type=str, default=None)
    parser.add_argument("--dir_name", type=str, default="pre_render_h01248")
    # parser.add_argument('--initial_scene', type=int, default=0)
    parser.add_argument("--scene_step", type=int, default=1)
    parser.add_argument("--skip_frame_step", type=int, default=0)
    parser.add_argument("--initial_scenes", type=int, nargs="+")
    parser.add_argument("--num_jobs", type=int, default=16)

    args = parser.parse_args()
    DST_DIR = f"{config.L5KIT_DATA_FOLDER}/{args.dir_name}"
    print("Root DST DIR:", DST_DIR)
    action = args.action
    dset_name = args.dset_name

    zarr_names = {
        "train_XXL": "train_XXL",
        "train": "train",
        "val": "validate",
        "test": "test",
    }
    zarr_name = zarr_names[dset_name] if args.zarr_name is None else args.zarr_name

    if action == "render":
        pre_render_parallel(
            scene_step=args.scene_step,
            zarr_name=zarr_name,
            dset_name=dset_name,
            skip_frame_step=args.skip_frame_step,
            initial_scenes=args.initial_scenes,
            num_jobs=args.num_jobs,
        )
