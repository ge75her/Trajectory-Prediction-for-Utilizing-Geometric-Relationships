import os
import sys
import copy
import glob
import logging
import json
import math
import cv2
import numpy as np
from pathlib import Path
from beartype import beartype
from einops import rearrange
import pickle
from os.path import join
import torch
from torch.utils.data import Dataset
import ujson
from PIL import Image,ImageDraw
from torchvision import transforms

TOWNS=['Town01','Town02','Town03','Town04','Town05','Town06','Town07']
class PlanTDataset(Dataset):
    @beartype
    def __init__(self, root: str, cfg, shared_dict=None, split: str = "all") -> None:
        self.cfg = cfg
        self.cfg_train = cfg.model.training
        self.data_cache = shared_dict
        self.cnt = 0

        self.input_sequence_files = []
        self.output_sequence_files = []
        self.labels = []
        self.measurements = []
        self.town_all=[]
        with open(join(cfg.carla_map_dir, 'carla_maps.pickle'), 'rb') as handle:
            self.maps = pickle.load(handle)
            # transpose all arrays
            self.carla_maps = {}
            for town_name, lane_list in self.maps.items():
                new_lane_list = []
                for lane in lane_list:
                    new_lane = {
                        'center': lane['center'].T,
                        'left_boundary': lane['left_boundary'].T,
                        'right_boundary': lane['right_boundary'].T,
                    }
                    new_lane_list.append(new_lane)
                self.carla_maps[town_name] = new_lane_list

        with open(join(cfg.carla_map_dir, 'carla_meta.pickle'), 'rb') as handle:
            self.carla_meta=pickle.load(handle)

         
        print('finish load carla maps!')
        
        label_raw_path_all = glob.glob(root + "/**/Routes*", recursive=True)
        label_raw_path = []

        label_raw_path = self.filter_data_by_town(label_raw_path_all, split)
        print('finish raw label!')

        logging.info(f"Found {len(label_raw_path)} Route folders containing {cfg.trainset_size} datasets.")
        for sub_route in label_raw_path:

            root_files = os.listdir(sub_route)
            routes = [
                folder
                for folder in root_files
                if not os.path.isfile(os.path.join(sub_route, folder))
            ]
            for route in routes:
                route_dir = Path(f"{sub_route}/{route}")
                town_name=None
                for town in TOWNS:
                    if town in str(route_dir):
                        town_name=town
                        break
                    if town_name==None:
                        continue
                num_seq = len(os.listdir(route_dir / "boxes"))

                # ignore the first 5 and last two frames
                for seq in range(
                    5,
                    num_seq - self.cfg_train.pred_len - self.cfg_train.seq_len - 2,
                ):
                    # load input seq and pred seq jointly
                    label = []
                    measurement = []
                    for idx in range(
                        self.cfg_train.seq_len + self.cfg_train.pred_len
                    ):
                        labels_file = route_dir / "boxes" / f"{seq + idx:04d}.json"
                        measurements_file = (
                            route_dir / "measurements" / f"{seq + idx:04d}.json"
                        )
                        label.append(labels_file)
                        measurement.append(measurements_file)

                    self.labels.append(label)
                    self.measurements.append(measurement)
                    self.town_all.append(town_name)
        # There is a complex "memory leak"/performance issue when using Python objects like lists in a Dataloader that is loaded with multiprocessing, num_workers > 0
        # A summary of that ongoing discussion can be found here https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # A workaround is to store the string lists as numpy byte objects because they only have 1 refcount.
        self.labels       = np.array(self.labels      ).astype(np.string_)
        self.measurements = np.array(self.measurements).astype(np.string_)
        self.town_all = np.array(self.town_all).astype(np.string_)
        print(f"Loading {len(self.labels)} samples from {len(root)} folders")


    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.measurements)


    def __getitem__(self, index):
        """Returns the item at index idx."""

        labels = self.labels[index]
        measurements = self.measurements[index]
        town_name=self.town_all[index]
        town_name=str(town_name,'utf-8')

        sample = {
            "input": [],
            "output": [],
            "brake": [],
            "waypoints": [],
            "target_point": [],
            "light": [],
            'label_junction':[],
            'label_lane':[],
            # 'dist_junction':[]
        }
        if not self.data_cache is None and labels[0] in self.data_cache:
            sample = self.data_cache[labels[0]]
        else:
            loaded_labels = []
            loaded_measurements = []

            for i in range(self.cfg_train.seq_len + self.cfg_train.pred_len):
                measurements_i = json.load(open(measurements[i]))
                labels_i = json.load(open(labels[i]))

                loaded_labels.append(labels_i)
                loaded_measurements.append(measurements_i)

            # ego car is always the first one in label file
            waypoints = get_waypoints(loaded_measurements[self.cfg_train.seq_len - 1 :])
            waypoints = transform_waypoints(waypoints)

            # save waypoints in meters
            filtered_waypoints = []
            for id in ["1"]:
                waypoint = []
                for matrix, _ in waypoints[id][1:]:
                    waypoint.append(matrix[:2, 3])
                filtered_waypoints.append(waypoint)
            waypoints = np.array(filtered_waypoints)

            ego_waypoint = waypoints[-1]

            sample["waypoints"] = ego_waypoint

            measurement_ego = loaded_measurements[0]
            ego_loc = np.array([measurement_ego['x'], -measurement_ego['y']])
            ego_ori = -measurement_ego['theta']
            sin_heading = np.sin(ego_ori)
            cos_heading = np.cos(ego_ori)
            R_ego = np.stack([sin_heading, cos_heading, -cos_heading, sin_heading]).reshape(2, 2)



            step_x = self.carla_meta[town_name]['step_x']
            step_y = self.carla_meta[town_name]['step_y']
            idx_x = int((ego_loc[0] - self.carla_meta[town_name]['map_min_x'] + 0.5*step_x) // step_x)
            idx_y = int((ego_loc[1] - self.carla_meta[town_name]['map_min_y'] + 0.5*step_y) // step_y)
            lane_ids = []
            if (idx_x, idx_y) in self.carla_meta[town_name]['lane_grid']:
                lane_ids = self.carla_meta[town_name]['lane_grid'][idx_x, idx_y]
            # lane_list = np.array([self.carla_maps[town_name][lane_id] for lane_id in lane_ids])
            num_lanes = len(lane_ids)
            local_maps = [[town_name], [num_lanes], [lane_ids]]
            
            local_command_point = np.array(loaded_measurements[self.cfg_train.seq_len - 1]["target_point"])
            sample["target_point"] = tuple(local_command_point)
            sample["light"] = loaded_measurements[self.cfg_train.seq_len - 1][
                "light_hazard"
            ]

            if self.cfg.model.pre_training.pretraining == "forecast":
                offset = (
                    self.cfg.model.pre_training.future_timestep
                )  # target is next timestep
            elif self.cfg.model.pre_training.pretraining == "none":
                offset = 0
            else:
                print(
                    f"ERROR: pretraining {self.cfg.model.pre_training.pretraining} is not supported"
                )
                sys.exit()

            for sample_key, file in zip(
                ["input", "output"],
                [
                    (
                        loaded_measurements[self.cfg_train.seq_len - 1],
                        loaded_labels[self.cfg_train.seq_len - 1],
                    ),
                    (
                        loaded_measurements[self.cfg_train.seq_len - 1 + offset],
                        loaded_labels[self.cfg_train.seq_len - 1 + offset],
                    ),
                ],
            ):

                measurements_data = file[0]
                ego_matrix = np.array(measurements_data["ego_matrix"])
                ego_yaw = measurements_data['theta']
                if sample_key == "input":
                    ego_matrix_input = ego_matrix
                    ego_yaw_input = ego_yaw
                    
                labels_data_all = file[1]

                if self.cfg_train.input_ego:
                    labels_data = file[1]
                else:
                    labels_data = file[1][1:]  # remove ego car

                # for future timesteps transform position to ego frame of first timestep
                pos = []
                yaw = []
                for labels_data_i in labels_data:
                    p = np.array(copy.deepcopy(labels_data_i["position"])) - np.array(labels_data_all[0]["position"])
                    p = np.append(p,[1])
                    p[1] = -p[1]
                    p_global = ego_matrix @ p
                    p_t2 = np.linalg.inv(ego_matrix_input) @ p_global
                    p_t2[1] = -p_t2[1]
                    pos.append(p_t2[:2])
                    yaw.append(labels_data_i["yaw"]+ego_yaw-ego_yaw_input)
                
                data_car = [
                    [
                        1.0,  # type indicator for cars
                        float(pos[j][0]),
                        float(pos[j][1]),
                        float(yaw[j] * 180 / 3.14159265359),  # in degrees
                        float(x["speed"] * 3.6),  # in km/h
                        float(x["extent"][2]),
                        float(x["extent"][1]),
                        float(x["id"]),
                    ]
                    for j, x in enumerate(labels_data)
                    if x["class"] == "Car" and ((self.cfg_train.remove_back and float(pos[j][0]) >= 0) or not self.cfg_train.remove_back)
                ]

                if sample_key == "output":
                    # discretize box
                    if self.cfg.model.pre_training.quantize:
                        if len(data_car) > 0:
                            data_car = self.quantize_box(data_car)

                    if self.cfg.model.pre_training.pretraining == "forecast":
                        # we can only use vehicles where we have the corresponding object in the input timestep
                        # if we don't have the object in the input timestep, we remove the vehicle
                        # if we don't have the object in the output timestep we add a dummy vehicle, that is not considered for the loss
                        data_car_by_id = {}
                        i = 0
                        for ii, x in enumerate(labels_data):
                            if x["class"] == "Car" and ((self.cfg_train.remove_back and float(pos[ii][0]) >= 0) or not self.cfg_train.remove_back):
                                data_car_by_id[x["id"]] = data_car[i]
                                i += 1
                        data_car_matched = []
                        for i, x in enumerate(data_car_input):
                            input_id = x[7]
                            if input_id in data_car_by_id:
                                data_car_matched.append(data_car_by_id[input_id])
                            else:
                                # append dummy data
                                dummy_data = x
                                dummy_data[0] = 10.0  # type indicator for dummy
                                data_car_matched.append(dummy_data)

                        data_car = data_car_matched
                        assert len(data_car) == len(data_car_input)

                else:
                    data_car_input = data_car

                # remove id from data_car
                data_car = [x[:-1] for x in data_car]

                # if we use the far_node as target waypoint we need the route as input
                data_route = [
                    [
                        2.0,  # type indicator for route
                        float(x["position"][0]) - float(labels_data_all[0]["position"][0]),
                        float(x["position"][1]) - float(labels_data_all[0]["position"][1]),
                        float(x["yaw"] * 180 / 3.14159265359),  # in degrees
                        float(x["id"]),
                        float(x["extent"][2]),
                        float(x["extent"][1]),
                    ]
                    for j, x in enumerate(labels_data)
                    if x["class"] == "Route"
                    and float(x["id"]) < self.cfg_train.max_NextRouteBBs
                ]
                
                # we split route segment slonger than 10m into multiple segments
                # improves generalization
                data_route_split = []
                for route in data_route:
                    if route[6] > 10:
                        routes = split_large_BB(
                            route, len(data_route_split)
                        )
                        data_route_split.extend(routes)
                    else:
                        data_route_split.append(route)
                data_route = data_route_split[: self.cfg_train.max_NextRouteBBs]

            
                if sample_key == "output":
                    data_route = data_route[: len(data_route_input)]
                    if len(data_route) < len(data_route_input):
                        diff = len(data_route_input) - len(data_route)
                        data_route.extend([data_route[-1]] * diff)
                else:
                    data_route_input = data_route

                data_lane = [
                    [
                        3.0,  # type indicator for route
                        float(x["position"][0]) - float(labels_data_all[0]["position"][0]) ,
                        float(x["position"][1]) - float(labels_data_all[0]["position"][1]) ,
                        float(x["yaw"] * 180 / 3.14159265359),  # in degrees
                        float(x["lane_id"]),
                        0.5,
                        float(x["distance"]),
                    
                    ]
                    for j, x in enumerate(labels_data)
                    if x["class"] == "Lane"
                    
                ]
                if sample_key == "output":
                    data_lane = data_lane_input
                else:
                    data_lane_input = data_lane
                    

                if self.cfg.model.training.remove_velocity == 'input':
                    if sample_key == 'input':
                        for i in range(len(data_car)):
                            data_car[i][4] = 0.
                elif self.cfg.model.training.remove_velocity == 'None':
                    pass
                else:
                    raise NotImplementedError
                
                if self.cfg.model.training.route_only_wp == True:
                    if sample_key == 'input':
                        for i in range(len(data_route)):
                            data_route[i][3] = 0.
                            data_route[i][-2] = 0.
                            data_route[i][-1] = 0.
                elif self.cfg.model.training.route_only_wp == False:
                    pass
                else:
                    raise NotImplementedError


                assert len(data_route) == len(
                    data_route_input
                ), "Route and route input not the same length"

                assert (
                    len(data_route) <= self.cfg_train.max_NextRouteBBs
                ), "Too many routes"

                if len(data_route) == 0:
                    # quit programm
                    print("ERROR: no route found")
                    logging.error("No route found in file: {}".format(file))
                    sys.exit()

                sample[sample_key] = data_car + data_route +data_lane
                # sample[sample_key] = data_car + data_route
                #aux
                y_lane=self.if_same_lane(self.maps,ego_loc,R_ego,data_car_input,local_maps)
                y_juc=self.if_junction(self.maps,ego_loc,R_ego,data_car_input,local_maps)
                sample['label_junction']=y_juc
                sample['label_lane']=y_lane
            if not self.data_cache is None:
                self.data_cache[labels[0]] = sample

        assert len(sample["input"]) == len(
            sample["output"]
        ), "Input and output have different length"

        self.cnt+=1
        return sample

    def if_same_lane(self,maps,ego_loc,R_ego,data_car,local_maps):
        car=[]
        for vehicle in data_car:
            pos=np.array([-vehicle[2],vehicle[1]])
            actor_loc=R_ego@pos+ego_loc
            car.append(actor_loc)
        car=np.array(car)
        ego_lane=0
        ego_min_idx=0
        lane_points=[]
        vor_index=[]
        nach_index=[]
        for i,a in enumerate(car):
            a=a.reshape(1,2)
            for town, num_lanes, lane_ids_padded, last_point in zip(*local_maps, a):
                lane_ids = lane_ids_padded[:num_lanes]
                lane_list = [maps[town][i] for i in lane_ids]

                lane_c = []
                lane_i = []
                lane_s = []


                for j, lane in zip(lane_ids, lane_list):
                    s = lane['s']
                    num_points = len(s)
                    lane_center = lane['center'].T

                    lane_c.append(lane_center)
                    lane_i.append(np.full(num_points, j))
                    lane_s.append(s)
                    
                lane_c = np.concatenate(lane_c)
                lane_i = np.concatenate(lane_i)
                lane_s = np.concatenate(lane_s)

                dists = np.linalg.norm(lane_c - last_point, axis=-1)
                if i==0:
                    ego_min_idx=np.argmin(dists)
                    ego_lane=lane_i[ego_min_idx]

                min_idx = np.argmin(dists)
                lane_points.append(lane_i[min_idx])
                if lane_i[min_idx]==ego_lane:
                    if min_idx>ego_min_idx:
                        nach_index.append(i)
                    elif min_idx<ego_min_idx:
                        vor_index.append(i)
  
        same_lane=[lane_points[0]]
        # print('lane id',same_lane)
        pre=maps[town][lane_points[0]]['predecessor']
        suc=maps[town][lane_points[0]]['successor']

        vor_lane=[]
        nach_lane=[]
        for pre_idx in pre:
            while pre_idx in lane_ids and pre_idx not in same_lane:
                same_lane.append(pre_idx)
                vor_lane.append(pre_idx)
                if maps[town][pre_idx]['predecessor']:
                    pre_idx=maps[town][pre_idx]['predecessor'][0]
                else:
                    break
            # print(pre_idx)
        for i,suc_idx in enumerate(suc):
            while suc_idx in lane_ids and suc_idx not in same_lane:
                same_lane.append(suc_idx)
                nach_lane.append(suc_idx)
                if maps[town][suc_idx]['successor']:
                    suc_idx=maps[town][suc_idx]['successor'][0]   
                else:
                    break
        
        # pos_index=[]
        for i,veh in enumerate(lane_points):
            if veh in vor_lane and i not in vor_index:
                vor_index.append(i)
            elif veh in nach_lane and i not in nach_index:
                nach_index.append(i)
        y=torch.zeros(len(data_car),dtype=torch.long)
        y[nach_index]=2
        y[vor_index]=3
        y[0]=1
        return y
    
    def if_junction(self,maps,ego_loc,R_ego,data_car,local_maps):
        car=[]
        for vehicle in data_car:
            pos=np.array([-vehicle[2],vehicle[1]])
            actor_loc=R_ego@pos+ego_loc
            car.append(actor_loc)
        car=np.array(car)

        y=torch.zeros(len(data_car),dtype=torch.long)
        for i,a in enumerate(car):
            a=a.reshape(1,2)
            stack_in=[]
            stack_out=[]
            for town, num_lanes, lane_ids_padded, last_point in zip(*local_maps, a):
                lane_ids = lane_ids_padded[:num_lanes]
                lane_list = [maps[town][i] for i in lane_ids]

                lane_c = []
                lane_i = []
                lane_s = []


                for j, lane in zip(lane_ids, lane_list):
                    s = lane['s']
                    num_points = len(s)
                    lane_center = lane['center'].T

                    lane_c.append(lane_center)
                    lane_i.append(np.full(num_points, j))
                    lane_s.append(s)
                    
                lane_c = np.concatenate(lane_c)
                lane_i = np.concatenate(lane_i)
                lane_s = np.concatenate(lane_s)

                dists = np.linalg.norm(lane_c - last_point, axis=-1)

                min_idx = np.argmin(dists)
                cur_id=lane_i[min_idx]
                stack_in.append(cur_id)
                stack_out.append(cur_id)


            if maps[town][cur_id]['successor']:
                suc=maps[town][cur_id]['successor'][0]
                if len(maps[town][suc]['predecessor'])>1:
                    y[i]=1
            
                else:
                    while stack_in:
                        cur=stack_in.pop()
                        if maps[town][cur]['successor'] and maps[town][cur]['successor'][0] in lane_ids:
                            if len(maps[town][cur]['successor'])>1:
                                y[i]=1
                                break
                            else:
                                stack_in.extend(maps[town][cur]['successor'])
                        else:
                            break
            
            if y[i]==0:
                while stack_out:
                    cur=stack_out.pop()
                    if maps[town][cur]['predecessor'] and maps[town][cur]['predecessor'][0] in lane_ids:
                        if len(maps[town][cur]['predecessor'])>1:
                            y[i]=2
                            break
                        else:
                            stack_out.extend(maps[town][cur]['predecessor'])
                    else:
                        break

        return y
    

    def junction_dist(self,maps,ego_loc,R_ego,data_car,local_maps):
        car=[]
        for vehicle in data_car:
            pos=np.array([-vehicle[2],vehicle[1]])
            actor_loc=R_ego@pos+ego_loc
            car.append(actor_loc)
        car=np.array(car)

        y=torch.zeros(len(data_car),dtype=torch.float)+100
        for i,a in enumerate(car):
            a=a.reshape(1,2)
            stack_in=[]
            for town, num_lanes, lane_ids_padded, last_point in zip(*local_maps, a):
                lane_ids = lane_ids_padded[:num_lanes]
                lane_list = [maps[town][i] for i in lane_ids]

                lane_c = []
                lane_i = []
                lane_s = []


                for j, lane in zip(lane_ids, lane_list):
                    s = lane['s']
                    # length=lane['length']
                    num_points = len(s)
                    lane_center = lane['center'].T

                    lane_c.append(lane_center)
                    lane_i.append(np.full(num_points, j))
                    # lane_length.append(np.full(num_points,length))
                    lane_s.append(s)
                    
                lane_c = np.concatenate(lane_c)
                lane_i = np.concatenate(lane_i)
                lane_s = np.concatenate(lane_s)
                # lane_length=np.concatenate(lane_length)

                dists = np.linalg.norm(lane_c - last_point, axis=-1)

                min_idx = np.argmin(dists)
                cur_id=lane_i[min_idx]
                stack_in.append(cur_id)
                # cur_s=lane_s[min_idx]
                cur_length=maps[town][cur_id]['length']-lane_s[min_idx]

            if maps[town][cur_id]['successor']:
                suc=maps[town][cur_id]['successor'][0]
                if len(maps[town][suc]['predecessor'])>1:
                    y[i]=torch.tensor([cur_length],dtype=torch.float)

                else:
                    while stack_in:
                        cur=stack_in.pop()
                        if maps[town][cur]['successor'] and maps[town][cur]['successor'][0] in lane_ids:
                            if len(maps[town][cur]['successor'])>1:
                                y[i]=torch.tensor([cur_length],dtype=torch.float)
                                break
                            else:
                                stack_in.extend(maps[town][cur]['successor'])
                                cur_length+=maps[town][maps[town][cur]['successor'][0]]['length']
                        else:
                            break
                    

        return y
    def quantize_box(self, boxes):
        boxes = np.array(boxes)

        # range of xy is [-30, 30]
        # range of yaw is [-360, 0]
        # range of speed is [0, 60]
        # range of extent is [0, 30]

        # quantize xy
        boxes[:, 1] = (boxes[:, 1] + 30) / 60
        boxes[:, 2] = (boxes[:, 2] + 30) / 60

        # quantize yaw
        boxes[:, 3] = (boxes[:, 3] + 360) / 360

        # quantize speed
        boxes[:, 4] = boxes[:, 4] / 60

        # quantize extent
        boxes[:, 5] = boxes[:, 5] / 30
        boxes[:, 6] = boxes[:, 6] / 30

        boxes[:, 1:] = np.clip(boxes[:, 1:], 0, 1)

        size_pos = pow(2, self.cfg.model.pre_training.precision_pos)
        size_speed = pow(2, self.cfg.model.pre_training.precision_speed)
        size_angle = pow(2, self.cfg.model.pre_training.precision_angle)

        boxes[:, [1, 2, 5, 6]] = (boxes[:, [1, 2, 5, 6]] * (size_pos - 1)).round()
        boxes[:, 3] = (boxes[:, 3] * (size_angle - 1)).round()
        boxes[:, 4] = (boxes[:, 4] * (size_speed - 1)).round()

        return boxes.astype(np.int32).tolist()




    def filter_data_by_town(self, label_raw_path_all, split):
        # in case we want to train without T2 and T5
        label_raw_path = []
        if split == "train":
            for path in label_raw_path_all:
                if "Town02" in path or "Town05" in path:
                    continue
                label_raw_path.append(path)
        elif split == "val":
            for path in label_raw_path_all:
                if "Town02" in path or "Town05" in path:
                    label_raw_path.append(path)
        elif split == "all":
            # for path in label_raw_path_all:
            #     if "Town10HD" in path:
            #         continue
            #     label_raw_path.append(path)
            label_raw_path = label_raw_path_all
        elif split == 'test':
            for path in label_raw_path_all:
                if 'Town01' in path:
                    label_raw_path_all.append(path)
            label_raw_path_all=label_raw_path_all[:10]            
        return label_raw_path

def split_large_BB(route, start_id):
    x = route[1]
    y = route[2]
    angle = route[3] - 90
    extent_x = route[5] / 2
    extent_y = route[6] / 2

    x1 = x - extent_y * math.sin(math.radians(angle))
    y1 = y - extent_y * math.cos(math.radians(angle))

    x0 = x + extent_y * math.sin(math.radians(angle))
    y0 = y + extent_y * math.cos(math.radians(angle))

    number_of_points = (
        math.ceil(extent_y * 2 / 10) - 1
    )  # 5 is the minimum distance between two points, we want to have math.ceil(extent_y / 5) and that minus 1 points
    xs = np.linspace(
        x0, x1, number_of_points + 2
    )  # +2 because we want to have the first and last point
    ys = np.linspace(y0, y1, number_of_points + 2)

    splitted_routes = []
    for i in range(len(xs) - 1):
        route_new = route.copy()
        route_new[1] = (xs[i] + xs[i + 1]) / 2
        route_new[2] = (ys[i] + ys[i + 1]) / 2
        route_new[4] = float(start_id + i)
        route_new[5] = extent_x * 2
        route_new[6] = route[6] / (
            number_of_points + 1
        )
        splitted_routes.append(route_new)

    return splitted_routes


def get_waypoints(measurements):
    assert len(measurements) == 5
    num = 5
    waypoints = {"1": []}

    for i in range(0, num):
        waypoints["1"].append([measurements[i]["ego_matrix"], True])

    Identity = list(list(row) for row in np.eye(4))
    # padding here
    for k in waypoints.keys():
        while len(waypoints[k]) < num:
            waypoints[k].append([Identity, False])
    return waypoints


def transform_waypoints(waypoints):
    """transform waypoints to be origin at ego_matrix"""

    # TODO should transform to virtual lidar coordicate?
    T = get_vehicle_to_virtual_lidar_transform()

    for k in waypoints.keys():
        vehicle_matrix = np.array(waypoints[k][0][0])
        vehicle_matrix_inv = np.linalg.inv(vehicle_matrix)
        for i in range(1, len(waypoints[k])):
            matrix = np.array(waypoints[k][i][0])
            waypoints[k][i][0] = T @ vehicle_matrix_inv @ matrix

    return waypoints


def get_virtual_lidar_to_vehicle_transform():
    # This is a fake lidar coordinate
    T = np.eye(4)
    T[0, 3] = 1.3
    T[1, 3] = 0.0
    T[2, 3] = 2.5
    return T


def get_vehicle_to_virtual_lidar_transform():
    return np.linalg.inv(get_virtual_lidar_to_vehicle_transform())

def get_BB(pos,heading,pixel_per_meter,image_offset,extent_x,extent_y):

    R_actor = [[np.sin(heading), np.cos(heading)], [-np.cos(heading), np.sin(heading)]]
    
    l_half = 0.5*extent_y
    w_half = 0.5*extent_x
    
    bbox_vertices = np.array([[l_half, -w_half], [l_half + w_half / 2, 0], [l_half, w_half], [-l_half, w_half], [-l_half, -w_half]])
    bbox_vertices = bbox_vertices @ R_actor +pos
    bbox_vertices = bbox_vertices * pixel_per_meter + image_offset


    return bbox_vertices

def get_env_relations(carla_maps,local_maps, trajectories, pos, heading):
    sin_heading = torch.sin(heading)
    cos_heading = torch.cos(heading)
    R_ego = torch.stack([sin_heading, cos_heading, -cos_heading, sin_heading], dim=-1).view(-1, 2, 2)

    last_points = trajectories
    last_points = torch.bmm(R_ego, last_points.unsqueeze(-1)).squeeze(-1) + pos # in world coordinate system
    last_points = last_points.numpy()
    
    lane_points = []
    lane_normals = []
    
    for town, num_lanes, lane_ids_padded, last_point in zip(*local_maps, last_points):
        lane_ids = lane_ids_padded[:num_lanes]
        lane_list = [carla_maps[town][i] for i in lane_ids]

        lane_c = []
        lane_i = []
        lane_s = []
        for i, lane in zip(lane_ids, lane_list):
            s = lane['s']
            num_points = len(s)
            lane_center = lane['center'].T
            
            lane_c.append(lane_center)
            lane_i.append(np.full(num_points, i))
            lane_s.append(s)
        
        lane_c = np.concatenate(lane_c)
        lane_i = np.concatenate(lane_i)
        lane_s = np.concatenate(lane_s)

        dists = np.linalg.norm(lane_c - last_point, axis=-1)
        min_idx = np.argmin(dists)
        
        # refinement
        lane = carla_maps[town][lane_i[min_idx]]
        s = lane_s[min_idx]
        lane_s = np.linspace(max(s - 0.6, 0), min(s + 0.6, lane['length']), 10)
        lane_c = np.array([np.interp(lane_s, lane['s'], lane['center'][i]) for i in range(2)]).T
        
        dists = np.linalg.norm(lane_c - last_point, axis=-1)
        min_idx = np.argmin(dists)
        lane_point = lane_c[min_idx]
        lane_normal = np.array([np.interp(lane_s[min_idx], lane['s'], lane['normal'][i]) for i in range(2)])
        lane_normal /= np.linalg.norm(lane_normal)

        lane_points.append(lane_point)
        lane_normals.append(lane_normal)
    
    lane_points = torch.tensor(np.array(lane_points), dtype=torch.float)
    lane_normals = torch.tensor(np.array(lane_normals), dtype=torch.float)

    lane_points = torch.bmm((lane_points - pos).unsqueeze(1), R_ego).squeeze(1)
    lane_normals = torch.bmm(lane_normals.unsqueeze(1), R_ego).squeeze(1)

    return lane_points, lane_normals

def get_bev(carla_maps,measurements,local_map,data_car,town_name):
    with open(measurements[0],'r') as f:
        measurement=ujson.load(f)
    ego_loc = np.array([measurement['x'], -measurement['y']])
    ego_ori = -measurement['theta'] 
    # with open(local_maps[0],'r') as f:
    # local_map=ujson.load(f)
    R_ego = [[np.sin(ego_ori), np.cos(ego_ori)], [-np.cos(ego_ori), np.sin(ego_ori)]]

    image_size=(256,256) # bev image size
    ego_offset=(0,-32) # offset of rendered ego vehicle from image center
    pixel_per_meter=3 
    upscaling = 8 # upsampling factor for more accurate rendering
    render_size = (image_size[0] * upscaling, image_size[1] * upscaling)
    image_center = 0.5 * np.array(image_size) * upscaling
    image_offset = image_center + np.array(ego_offset) * upscaling
    pixel_per_meter = pixel_per_meter * upscaling
    line_width = int(pixel_per_meter / 2)
    img = Image.new("RGB", size=render_size, color=(0, 0, 0))
    lane_list = [carla_maps[town_name][lane_id] for lane_id in local_map]


    for lane in lane_list:
        lane_center = lane['center']
        left_boundary = lane['left_boundary']
        right_boundary = lane['right_boundary']
        
        # transform to ego coordinate system
        lane_center = (lane_center - ego_loc) @ R_ego
        left_boundary = (left_boundary - ego_loc) @ R_ego
        right_boundary = (right_boundary - ego_loc) @ R_ego
        # transform to image coordinate system
        lane_center_img = lane_center * pixel_per_meter + image_offset
        left_boundary_img =left_boundary * pixel_per_meter + image_offset
        right_boundary_img = right_boundary * pixel_per_meter + image_offset

        ImageDraw.Draw(img).line(tuple(map(tuple, lane_center_img)), width=line_width, fill='red')
        ImageDraw.Draw(img).line(tuple(map(tuple, left_boundary_img)), width=line_width, fill='yellow')
        ImageDraw.Draw(img).line(tuple(map(tuple, right_boundary_img)), width=line_width, fill='yellow')
    for ix, sequence in enumerate([data_car]):
    
        for ixx, vehicle in enumerate(sequence):
            pos=np.array([[-vehicle[2],vehicle[1]]])
            
            #yaw = vehicle[3]
            yaw=vehicle[3]*np.pi/180
            #print(yaw)
            extent_x = vehicle[5]
            extent_y = vehicle[6]


            bbox_vertics= get_BB(pos,yaw,pixel_per_meter,image_offset, extent_x, extent_y)
            
            if ix == 0:

                if ixx==0:
                    ImageDraw.Draw(img).polygon(tuple(map(tuple, bbox_vertics)), fill='green') 
                else:
                    ImageDraw.Draw(img).polygon(tuple(map(tuple, bbox_vertics)), fill='blue') 
                #else: 
                # ImageDraw.Draw(img).polygon(tuple(map(tuple, bbox_vertics)), fill='pink') 
        
    transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
    tran_img=transform(img)
    img_out=torch.fliplr(tran_img)
    return img_out

def generate_batch(data_batch):
    input_batch, output_batch,label_lane,label_juc = [], [],[],[]
    for element_id, sample in enumerate(data_batch):
        input_item = torch.tensor(sample["input"], dtype=torch.float32)
        output_item = torch.tensor(sample["output"])
 

        input_indices = torch.tensor([element_id] * len(input_item)).unsqueeze(1)
        output_indices = torch.tensor([element_id] * len(output_item)).unsqueeze(1)
        

        input_batch.append(torch.cat([input_indices, input_item], dim=1))
        output_batch.append(torch.cat([output_indices, output_item], dim=1))

        label_juc.append(sample['label_junction'])
        label_lane.append(sample['label_lane'])


    waypoints_batch = torch.tensor(np.array([sample["waypoints"] for sample in data_batch]))
    tp_batch = torch.tensor(np.array(
        [sample["target_point"] for sample in data_batch]), dtype=torch.float32
    )
    light_batch = rearrange(
        torch.tensor(np.array([sample["light"] for sample in data_batch])), "b -> b 1"
    )
 

    return input_batch, output_batch, waypoints_batch, tp_batch, light_batch, label_lane,label_juc
