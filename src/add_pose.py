import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    # 1. Calculate relative odometry: turn 45 deg, move 2m, turn 45 deg
    dx = 2.0 * math.cos(math.radians(45))
    dy = 2.0 * math.sin(math.radians(45))
    dtheta = math.radians(90)
    
    odometry_measurement = gtsam.Pose2(dx, dy, dtheta)
    
    # add the odometry factor between X(3) and X(4) to the graph
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry_measurement, ODOMETRY_NOISE))

    # find the initial estimate for the pose of X(4). 
    ideal_pose_3 = gtsam.Pose2(4.0, 0.0, 0.0)
    pose_4 = ideal_pose_3.compose(odometry_measurement)
    
    # add it to the graph
    initial_estimate.insert(X(4), pose_4)
    
    return graph, initial_estimate