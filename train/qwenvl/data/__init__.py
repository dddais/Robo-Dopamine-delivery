import re

MY_CARROT_DATASET = {
    "annotation_path": "/path/to/Robo-Dopamine/dataset/my_train_data/train_jsons/finetune_data_final.json",
    "data_path": "/path/to/workspace/Robo-Dopamine/dataset",
}

SUB1_APPROACH_GRASP = {
    "annotation_path": "/path/to/workspace/Robo-Dopamine/dataset/sub1_train_data/train_jsons/finetune_data_final.json",
    "data_path": "/path/to/workspace/Robo-Dopamine/dataset",
}

SUC_1_CARROT = {
    "annotation_path": "/path/to/workspace/Robo-Dopamine/dataset/suc_1_train_data/train_jsons/finetune_data_final.json",
    "data_path": "/path/to/workspace/Robo-Dopamine/dataset",
}

SUC_3_BOTTLE = {
    "annotation_path": "/path/to/workspace/Robo-Dopamine/dataset/suc_3_train_data/train_jsons/finetune_data_final.json",
    "data_path": "/path/to/workspace/Robo-Dopamine/dataset",
}

SUC_4_CUBE = {
    "annotation_path": "/path/to/workspace/Robo-Dopamine/dataset/suc_4_train_data/train_jsons/finetune_data_final.json",
    "data_path": "/path/to/workspace/Robo-Dopamine/dataset",
}

data_dict = {
    "my_carrot_dataset": MY_CARROT_DATASET,
    "sub1_approach_grasp": SUB1_APPROACH_GRASP,
    "suc_1_carrot": SUC_1_CARROT,
    "suc_3_bottle": SUC_3_BOTTLE,
    "suc_4_cube": SUC_4_CUBE,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
