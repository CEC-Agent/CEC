import gym.spaces as S
import numpy as np

horizons = {
    "PickPlaceCan": 500,
    "Lift": 500,
    "NutAssemblySquare": 500,
    "TwoArmTransport": 1100,
}


obs_modality_specs = {
    "low_dim": {
        "Lift": [
            {
                "obs": {
                    "low_dim": [
                        "robot0_eef_pos",
                        "robot0_eef_quat",
                        "robot0_gripper_qpos",
                        "object",
                    ],
                    "rgb": [],
                    "depth": [],
                    "scan": [],
                },
                "goal": {"low_dim": [], "rgb": [], "depth": [], "scan": []},
            }
        ],
        "PickPlaceCan": [
            {
                "obs": {
                    "low_dim": [
                        "robot0_eef_pos",
                        "robot0_eef_quat",
                        "robot0_gripper_qpos",
                        "object",
                    ],
                    "rgb": [],
                    "depth": [],
                    "scan": [],
                },
                "goal": {"low_dim": [], "rgb": [], "depth": [], "scan": []},
            }
        ],
        "NutAssemblySquare": [
            {
                "obs": {
                    "low_dim": [
                        "robot0_eef_pos",
                        "robot0_eef_quat",
                        "robot0_gripper_qpos",
                        "object",
                    ],
                    "rgb": [],
                    "depth": [],
                    "scan": [],
                },
                "goal": {"low_dim": [], "rgb": [], "depth": [], "scan": []},
            }
        ],
        "TwoArmTransport": [
            {
                "obs": {
                    "low_dim": [
                        "robot0_eef_pos",
                        "robot0_eef_quat",
                        "robot0_gripper_qpos",
                        "robot1_eef_pos",
                        "robot1_eef_quat",
                        "robot1_gripper_qpos",
                        "object",
                    ],
                    "rgb": [],
                    "depth": [],
                    "scan": [],
                },
                "goal": {"low_dim": [], "rgb": [], "depth": [], "scan": []},
            }
        ],
    },
    "image": {
        "Lift": [
            {
                "obs": {
                    "low_dim": [
                        "robot0_eef_pos",
                        "robot0_eef_quat",
                        "robot0_gripper_qpos",
                    ],
                    "rgb": ["agentview_image", "robot0_eye_in_hand_image"],
                    "depth": [],
                    "scan": [],
                },
                "goal": {"low_dim": [], "rgb": [], "depth": [], "scan": []},
            }
        ],
        "PickPlaceCan": [
            {
                "obs": {
                    "low_dim": [
                        "robot0_eef_pos",
                        "robot0_eef_quat",
                        "robot0_gripper_qpos",
                    ],
                    "rgb": ["agentview_image", "robot0_eye_in_hand_image"],
                    "depth": [],
                    "scan": [],
                },
                "goal": {"low_dim": [], "rgb": [], "depth": [], "scan": []},
            }
        ],
        "NutAssemblySquare": [
            {
                "obs": {
                    "low_dim": [
                        "robot0_eef_pos",
                        "robot0_eef_quat",
                        "robot0_gripper_qpos",
                    ],
                    "rgb": ["agentview_image", "robot0_eye_in_hand_image"],
                    "depth": [],
                    "scan": [],
                },
                "goal": {"low_dim": [], "rgb": [], "depth": [], "scan": []},
            }
        ],
        "TwoArmTransport": [
            {
                "obs": {
                    "low_dim": [
                        "robot0_eef_pos",
                        "robot0_eef_quat",
                        "robot0_gripper_qpos",
                        "robot1_eef_pos",
                        "robot1_eef_quat",
                        "robot1_gripper_qpos",
                    ],
                    "rgb": [
                        "shouldercamera0_image",
                        "robot0_eye_in_hand_image",
                        "shouldercamera1_image",
                        "robot1_eye_in_hand_image",
                    ],
                    "depth": [],
                    "scan": [],
                },
                "goal": {"low_dim": [], "rgb": [], "depth": [], "scan": []},
            }
        ],
    },
}


observation_spaces = {
    "low_dim": {
        "Lift": S.Dict(
            {
                "robot0_eef_pos": S.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "robot0_eef_quat": S.Box(low=-np.inf, high=np.inf, shape=(4,)),
                "robot0_gripper_qpos": S.Box(low=-np.inf, high=np.inf, shape=(2,)),
                "object": S.Box(low=-np.inf, high=np.inf, shape=(10,)),
            }
        ),
        "PickPlaceCan": S.Dict(
            {
                "robot0_eef_pos": S.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "robot0_eef_quat": S.Box(low=-np.inf, high=np.inf, shape=(4,)),
                "robot0_gripper_qpos": S.Box(low=-np.inf, high=np.inf, shape=(2,)),
                "object": S.Box(low=-np.inf, high=np.inf, shape=(14,)),
            }
        ),
        "NutAssemblySquare": S.Dict(
            {
                "robot0_eef_pos": S.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "robot0_eef_quat": S.Box(low=-np.inf, high=np.inf, shape=(4,)),
                "robot0_gripper_qpos": S.Box(low=-np.inf, high=np.inf, shape=(2,)),
                "object": S.Box(low=-np.inf, high=np.inf, shape=(14,)),
            }
        ),
        "TwoArmTransport": S.Dict(
            {
                "robot0_eef_pos": S.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "robot0_eef_quat": S.Box(low=-np.inf, high=np.inf, shape=(4,)),
                "robot0_gripper_qpos": S.Box(low=-np.inf, high=np.inf, shape=(2,)),
                "robot1_eef_pos": S.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "robot1_eef_quat": S.Box(low=-np.inf, high=np.inf, shape=(4,)),
                "robot1_gripper_qpos": S.Box(low=-np.inf, high=np.inf, shape=(2,)),
                "object": S.Box(low=-np.inf, high=np.inf, shape=(41,)),
            }
        ),
    },
    "image": {
        "Lift": S.Dict(
            {
                "robot0_eef_pos": S.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "robot0_eef_quat": S.Box(low=-np.inf, high=np.inf, shape=(4,)),
                "robot0_gripper_qpos": S.Box(low=-np.inf, high=np.inf, shape=(2,)),
                "agentview_image": S.Box(
                    low=0, high=1, shape=(3, 84, 84), dtype=np.float32
                ),
                "robot0_eye_in_hand_image": S.Box(
                    low=0, high=1, shape=(3, 84, 84), dtype=np.float32
                ),
            }
        ),
        "PickPlaceCan": S.Dict(
            {
                "robot0_eef_pos": S.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "robot0_eef_quat": S.Box(low=-np.inf, high=np.inf, shape=(4,)),
                "robot0_gripper_qpos": S.Box(low=-np.inf, high=np.inf, shape=(2,)),
                "agentview_image": S.Box(
                    low=0, high=1, shape=(3, 84, 84), dtype=np.float32
                ),
                "robot0_eye_in_hand_image": S.Box(
                    low=0, high=1, shape=(3, 84, 84), dtype=np.float32
                ),
            }
        ),
        "NutAssemblySquare": S.Dict(
            {
                "robot0_eef_pos": S.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "robot0_eef_quat": S.Box(low=-np.inf, high=np.inf, shape=(4,)),
                "robot0_gripper_qpos": S.Box(low=-np.inf, high=np.inf, shape=(2,)),
                "agentview_image": S.Box(
                    low=0, high=1, shape=(3, 84, 84), dtype=np.float32
                ),
                "robot0_eye_in_hand_image": S.Box(
                    low=0, high=1, shape=(3, 84, 84), dtype=np.float32
                ),
            }
        ),
        "TwoArmTransport": S.Dict(
            {
                "robot0_eef_pos": S.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "robot0_eef_quat": S.Box(low=-np.inf, high=np.inf, shape=(4,)),
                "robot0_gripper_qpos": S.Box(low=-np.inf, high=np.inf, shape=(2,)),
                "robot1_eef_pos": S.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "robot1_eef_quat": S.Box(low=-np.inf, high=np.inf, shape=(4,)),
                "robot1_gripper_qpos": S.Box(low=-np.inf, high=np.inf, shape=(2,)),
                "shouldercamera0_image": S.Box(
                    low=0, high=1, shape=(3, 84, 84), dtype=np.float32
                ),
                "robot0_eye_in_hand_image": S.Box(
                    low=0, high=1, shape=(3, 84, 84), dtype=np.float32
                ),
                "shouldercamera1_image": S.Box(
                    low=0, high=1, shape=(3, 84, 84), dtype=np.float32
                ),
                "robot1_eye_in_hand_image": S.Box(
                    low=0, high=1, shape=(3, 84, 84), dtype=np.float32
                ),
            }
        ),
    },
}
