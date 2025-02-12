import numpy as np


def brute_force_ball_query(points, queries, radius, max_neighbors):
    M, N = queries.shape[0], points.shape[0]
    dim = points.shape[1]

    neighbors = [[] for _ in range(M)]
    
    for i in range(M):
        q_i = queries[i]
        for j in range(N):
            p_j = points[j]

            dist2 = 0.0

            for d in range(dim):
                diff = q_i[d] - p_j[d]
                dist2 +=  diff * diff
            if dist2 <= radius*radius:
                neighbors[i].append(j)
                if len(neighbors[i]) >= max_neighbors:
                    break

    return neighbors



def brute_force_knn(points, queries, K):
    M, N = queries.shape[0], points.shape[0]
    dim = points.shape[1]
    knn_indices = [ [] for _ in range(M) ]
    for i in range(M):
        qi = queries[i]
        # Compute distance to every point
        distances = []
        for j in range(N):
            pj = points[j]
            dist2 = 0.0
            for d in range(dim):
                diff = qi[d] - pj[d]
                dist2 += diff * diff
            distances.append((dist2, j))
        # Sort by distance and take K smallest
        distances.sort(key=lambda x: x[0])
        knn_indices[i] = [idx for (_, idx) in distances[:K]]
    return knn_indices




def main():
    num_points = 1000
    points = np.random.rand(num_points, 3)

                     
    idx = np.random.choice(points.shape[0], num_points, replace=False)
    query_points = points[idx]
    neighbors = brute_force_ball_query(points,query_points,radius=0.2,max_neighbors=3)
    print(neighbors)

    neighbors = brute_force_knn(points,query_points,3)

    print(neighbors)


if __name__ == "__main__":

    main()