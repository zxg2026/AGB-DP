import time
import numpy as np
from munkres import Munkres
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

def calculate_center_and_radius(gb):
    data_no_label = gb[:, :] 
    center = data_no_label.mean(axis=0) 
    radius = np.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    return center, radius

def division2(hb_list):
    gb_list_new = []
    for hb in hb_list:
        if len(hb) == 1:
            gb_list_new.append(hb)
            continue
        if len(hb) >= 8:
            ball_1, ball_2 = spilt_ball(hb)
            DM_parent = get_DM(hb)
            DM_child_1 = get_DM(ball_1)
            DM_child_2 = get_DM(ball_2)
            w = len(ball_1) + len(ball_2)
            w1 = len(ball_1) / w
            w2 = len(ball_2) / w
            w_child = (w1 * DM_child_1 + w2 * DM_child_2)
            t2 = (w_child < DM_parent)  
            if t2:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_new.append(hb)
        else:
            gb_list_new.append(hb)
    return gb_list_new

def get_DM(hb):
    num = len(hb)
    if num == 0 or num == 1:
        return 1
    center = np.mean(hb, axis=0)
    distances = np.linalg.norm(hb - center, axis=1)
    sum_radius = np.sum(distances)
    if num > 1:
        DM = sum_radius / num
    return DM

def spilt_ball(data):
    dist_matrix = squareform(pdist(data, metric='euclidean'))
    r, c = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    ball1 = []
    ball2 = []
    for i in range(len(data)):
        if dist_matrix[i, r] < dist_matrix[i, c]:
            ball1.append(data[i])
        else:
            ball2.append(data[i])
    return np.array(ball1), np.array(ball2)

def normalized_ball(hb_list, radius_detect):
    hb_list_temp = []
    for hb in hb_list:
        if len(hb) < 2:
            hb_list_temp.append(hb)
        else:
            ball_1, ball_2 = spilt_ball(hb)
            _, rl = calculate_center_and_radius(hb)
            if rl <= 2 * radius_detect:
                hb_list_temp.append(hb)
            else:
                hb_list_temp.extend([ball_1, ball_2])
    return hb_list_temp


def get_ball_quality(gb, center):
    N = gb.shape[0]
    ball_quality = get_DM(gb)
    mean_r = np.mean(((gb - center) ** 2) ** 0.5)
    return ball_quality, mean_r, N

def ball_density(radiusAD, ball_qualitysA, ball_mA):
    N = radiusAD.shape[0]
    ball_dens = np.zeros(shape=N)
    for i in range(N):
        if radiusAD[i] == 0:
            ball_dens[i] = 0
        else:
            ball_dens[i] = ball_mA[i] / (ball_qualitysA[i])
    return ball_dens

def ball_distance(centersAD):
    Y1 = pdist(centersAD)
    ball_distAD = squareform(Y1)
    return ball_distAD

def ball_min_dist(ball_distS, ball_densS):
    N3 = ball_distS.shape[0]
    ball_min_distAD = np.zeros(shape=N3)
    ball_nearestAD = np.zeros(shape=N3)
    index_ball_dens = np.argsort(-ball_densS)
    for i3, index in enumerate(index_ball_dens):
        if i3 == 0:
            continue
        index_ball_higher_dens = index_ball_dens[:i3]
        ball_min_distAD[index] = np.min([ball_distS[index, j] for j in index_ball_higher_dens])
        ball_index_near = np.argmin([ball_distS[index, j] for j in index_ball_higher_dens])
        ball_nearestAD[index] = int(index_ball_higher_dens[ball_index_near])
    ball_min_distAD[index_ball_dens[0]] = np.max(ball_min_distAD)
    if np.max(ball_min_distAD) < 1:
        ball_min_distAD = ball_min_distAD * 10
    return ball_min_distAD, ball_nearestAD

def ball_draw_decision(ball_densS, ball_min_distS, nc):
    centers = []
    rho = ball_densS * ball_min_distS
    indices = np.argsort(-rho)[:nc]
    for i in range(ball_densS.shape[0]):
        if i in indices:
            centers.append(i)
    return np.array(centers)

def ball_cluster(ball_densS, ball_centers, ball_nearest, ball_min_distS):
    K1 = len(ball_centers)
    if K1 == 0:
        print('no centers')
        return
    N5 = ball_densS.shape[0]
    ball_labs = -1 * np.ones(N5).astype(int)
    for i5, cen1 in enumerate(ball_centers):
        ball_labs[cen1] = int(i5 + 1)
    ball_index_density = np.argsort(-ball_densS)
    for i5, index2 in enumerate(ball_index_density):
        if ball_labs[index2] == -1:
            ball_labs[index2] = ball_labs[int(ball_nearest[index2])]
    return ball_labs

def update_point_labels(data, ball_labs, gb_list):
    labels = -np.ones(data.shape[0], dtype=int)
    gb_dict = {}
    for i6 in range(len(gb_list)):
        for j6, point in enumerate(gb_list[i6]):
            gb_dict[tuple(point)] = ball_labs[i6]
    for i, data1 in enumerate(data):
        if tuple(data1) in gb_dict and labels[i] == -1:
            labels[i] = gb_dict[tuple(data1)]
    return labels

def evaluation(y_true, y_pred):
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    if num_class1 != num_class2:
        print('error')
        return
    cost = np.zeros((num_class1, num_class2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = accuracy_score(y_true, new_predict)
    return acc, nmi


if __name__ == "__main__":
  
    np.set_printoptions(threshold=1e16)
    file_names = ['zoo','iris']

    for i, file_name in enumerate(file_names):
        df = np.loadtxt(f'Data\{file_name}.txt')
        data = df[:, :-1]
        data_label = df[:, -1]
        nc = len(np.unique(data_label))

        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        
        start = time.time()
        
        gb_list = [data]
        while 1:
            ball_number_1 = len(gb_list)
            gb_list = division2(gb_list)
            ball_number_2 = len(gb_list)
            if ball_number_1 == ball_number_2:
                break

        radius = []
        for hb in gb_list:
            if len(hb) >= 2:
                _, rl1 = calculate_center_and_radius(hb)
                radius.append(rl1)

        radius_median = np.median(radius)
        radius_mean = np.mean(radius)
        radius_detect = max(radius_median, radius_mean)

        while 1:
            ball_number_old = len(gb_list)
            gb_list = normalized_ball(gb_list, radius_detect)
            ball_number_new = len(gb_list)
            if ball_number_new == ball_number_old:
                break

        centers = []
        radiuss = []
        ball_num = []
        ball_qualitys = []
        mean_rs = []
        ball_m = []

        for gb in gb_list:
            center, radius = calculate_center_and_radius(gb)
            ball_quality, mean_r, m = get_ball_quality(gb, center)
            ball_qualitys.append(ball_quality)
            ball_m.append(m)

            mean_rs.append(mean_r)
            centers.append(center)
            radiuss.append(radius)

        centersA = np.array(centers)
        radiusA = np.array(radiuss)
        ball_qualitysA = np.array(ball_qualitys)
        ball_mA = np.array(ball_m)
        
        ball_densS = ball_density(radiusA, ball_qualitysA, ball_mA)
        ball_distS = ball_distance(centersA)
        ball_min_distS, ball_nearest = ball_min_dist(ball_distS, ball_densS)

        ball_centers = ball_draw_decision(ball_densS, ball_min_distS, nc)
        ball_labs = ball_cluster(ball_densS, ball_centers, ball_nearest, ball_min_distS)
        
        end = time.time()
        times = end - start

        label = update_point_labels(data, ball_labs, gb_list)
        ACC, NMI = evaluation(data_label, label)

        print(f"Dataname：{file_name}")
        print("ACC:", f"{ACC:.3f}", "\nNMI:", f"{NMI:.3f}")
        print('runtime: {:.3f}s'.format(times))
        print('\n')

