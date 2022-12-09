import numpy as np
import math
from scipy.ndimage.morphology import distance_transform_edt as edt
w=0.05
def distance_bin(distance_map, predicted_map, step=w):
    distance_max = np.max(distance_map)
    distance_min = np.min(distance_map)
    distance_normalize = (distance_map - distance_min) / (distance_max - distance_min)
    bin = int(1 / step)
    distance_dict = {}
    # distance_list=[]
    # distance_dict_min={}
    n = predicted_map.shape[3]
    probability_values = []
    # weight={}
    for j in range(bin):
        bin_min = j * step
        bin_max = (j + 1) * step
        id = np.argwhere(np.logical_and(distance_normalize > bin_min, distance_normalize <= bin_max))
        # print('id:',id.shape)
        for k in range(n):
            probability_map = predicted_map[:, :, :, k]
            probability_values.extend(([probability_map[e[0]][e[1]][e[2]] for e in id]))
        # print('uncertainty_values_shape',len(probability_values))
        if len(probability_values) > 0:
            second_probability = probability_count(np.array(probability_values))
            # second_probability=np.average(np.array(probability_values))
            distance_dict[bin_max] = second_probability
            # weight[bin_max]=
            # draw_uncertainty(second_probability,bin_max)

        # else:
        # print('uncertainty_values shape',probability_values.shape)
        #    distance_dict[bin_max]=0

    return distance_dict


def probability_count(probability_values, probability_step=0.1):
    # print('min_probability:',np.min(probability_values))
    n = probability_values.shape[0]
    bin = int(1 / probability_step)
    probability_dict = {}
    for i in range(bin):
        bin_min = i * probability_step
        bin_max = (i + 1) * probability_step
        id = np.argwhere(np.logical_and(probability_values >= bin_min, probability_values < bin_max))
        probability_dict[bin_max] = len(id) / n
    return probability_dict  # {0.5:0.8,0.52:0.9}


def from_type2_to_type1(distance_dict):
    type2_to_type1 = {}
    for k, v in distance_dict.items():
        result = 0
        if type(v) == type(distance_dict):
            for u, weight in v.items():
                result = result + u * weight
        type2_to_type1[k] = result
    return type2_to_type1


def type1(distance_dict):
    result1 = 0
    # num=0
    for k, v in distance_dict.items():
        result1 = result1 + (2 * v - 1) ** 2
    uncertianty = 1 - math.sqrt(result1 / (len(distance_dict) + 1e-7))
    return uncertianty


def fuzzy_measure(argu_output, sum_arge_image):
    #argu_output is the average of all predicted images.
    #sum_arge_image includes all predicted images.
    fg_mask = argu_output.squeeze().cpu().detach().numpy()
    fg_dist = edt(fg_mask)
    # sum_arge_image_array = np.array(sum_arge_image).squeeze().transpose((1, 2, 3,0))
    # print('sum_arge_image_array:',sum_arge_image_array.shape)
    # print('fg_dist:',fg_dist.shape)
    distance_dict = distance_bin(fg_dist, sum_arge_image)
    # draw_threeD(distance_dict)
    type1_dict = from_type2_to_type1(distance_dict)
    type1_uncertainty = type1(type1_dict)

    return type1_uncertainty
