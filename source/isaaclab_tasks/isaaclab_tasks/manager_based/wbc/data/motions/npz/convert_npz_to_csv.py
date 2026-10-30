import os
import numpy as np

anchor_name = "torso_link"
body_names = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
anchor_idx = body_names.index(anchor_name)


def convert(npz_path: str):
    data = np.load(npz_path)

    joint_pos = data["joint_pos"]  # (T, 29)
    joint_vel = data["joint_vel"]  # (T, 29)
    body_pos_w = data["body_pos_w"]  # (T, 30, 3)
    body_quat_w = data["body_quat_w"]  # (T, 30, 4)
    anchor_pos_w = body_pos_w[:, anchor_idx, :]  # (T, 3)
    anchor_quat_w = body_quat_w[:, anchor_idx, :]  # (T, 4)
    timesteps = joint_pos.shape[0]

    arrays = [
        anchor_pos_w.reshape(timesteps, -1), anchor_quat_w.reshape(timesteps, -1), 
        joint_vel.reshape(timesteps, -1), joint_vel.reshape(timesteps, -1), 
        ]
    # arrays = [anchor_pos_w.reshape(timesteps, -1), anchor_quat_w.reshape(timesteps, -1), joint_vel.reshape(timesteps, -1)]
    combined = np.concatenate(arrays, axis=1)  # (T, 268)
    csv_path = npz_path.replace(".npz", ".csv")
    np.savetxt(csv_path, combined, delimiter=",")
    print(f"Saved {combined.shape} -> {csv_path}")


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.abspath(__file__))
    for fname in os.listdir(dir_path):
        if fname.endswith(".npz"):
            convert(os.path.join(dir_path, fname))
