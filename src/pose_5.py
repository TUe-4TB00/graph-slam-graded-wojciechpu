import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))


def add_pose(graph, initial_estimate, pose_5):
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate


def add_landmark_measurement(graph, result, pose_5, landmark):
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph


def optimize(graph, initial_estimate):
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    return result



def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    lowest_sum = float("inf")

    for pose_key, pose_5 in pose_options.items():
        for landmark in [1, 2]:

            graph_copy = gtsam.NonlinearFactorGraph(graph)
            estimate_copy = gtsam.Values(initial_estimate)

            graph_copy, estimate_copy = add_pose(graph_copy, estimate_copy, pose_5)
            result = optimize(graph_copy, estimate_copy)

            graph_copy = add_landmark_measurement(graph_copy, result, pose_5, landmark)
            result = optimize(graph_copy, estimate_copy)

            marginals = gtsam.Marginals(graph_copy, result)

            total_cov = marginals.marginalCovariance(L(landmark)).sum()

            if total_cov < lowest_sum:
                lowest_sum = total_cov
                best_pose = pose_key
                best_landmark = landmark

    pose_5 = pose_options[best_pose]

    graph_copy = gtsam.NonlinearFactorGraph(graph)
    estimate_copy = gtsam.Values(initial_estimate)

    graph_copy, estimate_copy = add_pose(graph_copy, estimate_copy, pose_5)
    result = optimize(graph_copy, estimate_copy)

    graph_copy = add_landmark_measurement(graph_copy, result, pose_5, best_landmark)
    result = optimize(graph_copy, estimate_copy)

    marginals = gtsam.Marginals(graph_copy, result)

    final_sum = (
        marginals.marginalCovariance(L(1)).sum() +
        marginals.marginalCovariance(L(2)).sum()
    )

    return best_pose, best_landmark, final_sum





def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    sum_of_errors = float("inf")

    gt = {
        1: (0.0, 0.0, 0.0),
        2: (2.0, 0.0, 0.0),
        3: (4.0, 0.0, 0.0),
    }

    for pose_key, pose_5 in pose_options.items():
        for landmark in [1, 2]:

            graph_copy = gtsam.NonlinearFactorGraph(graph)
            estimate_copy = gtsam.Values(initial_estimate)

            graph_copy, estimate_copy = add_pose(graph_copy, estimate_copy, pose_5)
            result = optimize(graph_copy, estimate_copy)

            graph_copy = add_landmark_measurement(graph_copy, result, pose_5, landmark)
            result = optimize(graph_copy, estimate_copy)

            error_sum = 0
            for i in [1, 2, 3]:
                pose = result.atPose2(X(i))
                x_gt, y_gt, theta_gt = gt[i]

                error = (
                    abs(pose.x() - x_gt) +
                    abs(pose.y() - y_gt) +
                    abs(pose.theta() - theta_gt)
                )

                error_sum += error

            if error_sum < sum_of_errors:
                sum_of_errors = error_sum
                best_pose = pose_key
                best_landmark = landmark

    return best_pose, best_landmark, sum_of_errors