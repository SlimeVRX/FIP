r"""
    Config for paths, joint set, and normalizing scales.
"""
amass_data = ['ACCAD', 'BioMotionLab_NTroje', 'BMLhandball', 'BMLmovi', 'CMU', 'DanceDB', 'DFaust_67',
            'Eyes_Japan_Dataset', 'HumanEva', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU', 'SSM_synced']


class paths:
    raw_dipimu_dir = 'data/dataset_raw/DIP_IMU'   # raw DIP-IMU dataset path (raw_dipimu_dir/s_01/*.pkl)
    dipimu_dir = 'data/dataset_work/DIP_IMU'      # output path for the preprocessed DIP-IMU dataset

    # DIP recalculates the SMPL poses for TotalCapture dataset. You should acquire the pose data from the DIP authors.
    raw_totalcapture_dip_dir = 'data/dataset_raw/TotalCapture/DIP_recalculate'  # contain ground-truth SMPL pose (*.pkl)
    raw_totalcapture_official_dir = 'data/dataset_raw/TotalCapture/official'    # contain official gt (S1/acting1/*.txt)
    totalcapture_dir = 'data/dataset_work/TotalCapture'          # output path for the preprocessed TotalCapture dataset

    raw_amass_dir = 'data/dataset_raw/AMASS'   # raw DIP-IMU dataset path (raw_dipimu_dir/s_01/*.pkl)
    amass_dir = 'data/dataset_work/AMASS'      # output path for the preprocessed DIP-IMU dataset

    example_dir = 'data/example'                    # example IMU measurements
    male_smpl_file = 'models/SMPL_male.pkl'              # official SMPL model path
    female_smpl_file = 'models/SMPL_female.pkl'              # official SMPL model path
    weights_file = 'data/weights.pt'                # network weight file


class joint_set:
    leaf = [7, 8, 12, 20, 21]
    full = list(range(1, 24))
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]

    sensor = [18, 19, 4, 5, 15, 0, 1, 2, 9] # 传感器穿戴位置顺序
    dip_imu = [7, 8, 11, 12, 0, 2, 9, 10, 1]
    VERTEX_IDS = [1962, 5431, 1096, 4583, 412, 3021, 949, 4434, 3506]
    SMPL_SENSOR = ['L_Elbow', 'R_Elbow', 'L_Knee', 'R_Knee', 'Head', 'Pelvis']

    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)


acc_scale = 30
vel_scale = 3
