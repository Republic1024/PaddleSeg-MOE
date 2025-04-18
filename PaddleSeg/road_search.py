import numpy as np
from queue import PriorityQueue


def a_star_pathfinding(start_idx, end_idx, cost_map):
    """
    使用 A* 算法在 cost_map 上进行路径规划。

    参数：
        start_idx (tuple): 起点坐标 (x, y)
        end_idx (tuple): 终点坐标 (x, y)
        cost_map (np.ndarray): 代价地图，数值越高表示越难通行
        mask_val (np.ndarray): 可选，背景图像用于绘制路径（必须为2D）
        thickness (int): 路径可视化的线条粗细，默认2个像素

    返回：
        path_array (np.ndarray): 路径点组成的数组，shape=(n, 2)
        seg_rgb (np.ndarray): 可视化路径后的 RGB 图像（如果提供了 mask_val）
    """

    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    h, w = cost_map.shape
    visited = np.zeros((h, w), dtype=bool)
    prev_node = np.empty((h, w), dtype=object)
    pq = PriorityQueue()
    start_tup = tuple(start_idx)
    end_tup = tuple(end_idx)
    pq.put((heuristic(start_tup, end_tup), 0.0, start_tup))
    found = False

    while not pq.empty():
        f, g, current = pq.get()
        if visited[current]:
            continue
        visited[current] = True
        if current == end_tup:
            found = True
            break
        x, y = current
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                    ng = g + cost_map[nx, ny]
                    pq.put((ng + heuristic((nx, ny), end_tup), ng, (nx, ny)))
                    if prev_node[nx, ny] is None:
                        prev_node[nx, ny] = (x, y)

    if not found:
        print("[❌] 未找到从起点到终点的路径！")
        path = []
    else:
        path = []
        cur = end_tup
        while cur != start_tup and prev_node[cur] is not None:
            path.append(cur)
            cur = prev_node[cur]
        path.append(start_tup)
        path = path[::-1]
        if path[-1] == end_tup:
            print("[✅] 成功找到从起点到终点的路径！")
        else:
            print("[❌] 回溯路径未正确连接起点与终点！")

    print("Path length (pixels):", len(path))

    path_array = np.array([[x, y] for (x, y) in path]) if path else np.array([])

    return path
