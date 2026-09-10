import sys
import heapq
from collections import deque

# AtCoderにおける再帰上限エラー対策
sys.setrecursionlimit(10**9)

class Graph:
    def __init__(self, n: int):
        """
        頂点数nで初期化。頂点は 0 から n-1 とします。
        後から add_vertex で任意の頂点IDを追加することも可能です。
        """
        self.graph = {i: {} for i in range(n)}
        self.INF = 10**36

    def add_vertex(self, x: int) -> None:
        """頂点xを追加する"""
        if x not in self.graph:
            self.graph[x] = {}

    def add_directed_edge(self, x: int, y: int, weight: int = 1) -> None:
        """x から y への有向辺を追加する"""
        self.add_vertex(x)
        self.add_vertex(y)
        # 多重辺がある場合は、最小の重みを採用する
        if y in self.graph[x]:
            self.graph[x][y] = min(self.graph[x][y], weight)
        else:
            self.graph[x][y] = weight

    def add_undirected_edge(self, x: int, y: int, weight: int = 1) -> None:
        """x と y の間に無向辺を追加する"""
        self.add_directed_edge(x, y, weight)
        self.add_directed_edge(y, x, weight)

    def cycle(self) -> list[int]:
        """
        閉路を持つならそのうちの一つを返す（有向閉路を探索）。
        持たない場合はValueErrorを投げる。
        """
        visited = {v: 0 for v in self.graph} # 0:未訪問, 1:訪問中, 2:訪問済
        parent = {}

        def dfs(curr: int) -> list[int]:
            visited[curr] = 1
            for nxt in self.graph[curr]:
                if visited[nxt] == 0:
                    parent[nxt] = curr
                    res = dfs(nxt)
                    if res: return res
                elif visited[nxt] == 1:
                    # 閉路を検出
                    cycle_path = [nxt]
                    p = curr
                    while p != nxt:
                        cycle_path.append(p)
                        p = parent[p]
                    cycle_path.append(nxt)
                    return cycle_path[::-1]
            visited[curr] = 2
            return []

        for v in self.graph:
            if visited[v] == 0:
                res = dfs(v)
                if res: return res
                
        raise ValueError("No cycle found")

    def has_cycle(self) -> bool:
        """閉路を持つかを判定する"""
        try:
            self.cycle()
            return True
        except ValueError:
            return False

    def MST(self) -> 'Graph':
        """
        Kruskal法により最小全域木(MST)を構築して返す。
        元のグラフを無向グラフとみなして計算します。
        """
        edges = []
        for u in self.graph:
            for v, w in self.graph[u].items():
                if u <= v:  # 無向辺の重複を防ぐ
                    edges.append((w, u, v))
        edges.sort()

        parent_uf = {v: v for v in self.graph}
        
        def find(i: int) -> int:
            if parent_uf[i] == i:
                return i
            parent_uf[i] = find(parent_uf[i])
            return parent_uf[i]
            
        def union(i: int, j: int) -> bool:
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent_uf[root_i] = root_j
                return True
            return False

        mst_graph = Graph(0)
        for v in self.graph:
            mst_graph.add_vertex(v)

        for w, u, v in edges:
            if union(u, v):
                mst_graph.add_undirected_edge(u, v, w)

        return mst_graph

    def is_DAG(self) -> bool:
        """DAG(有向非巡回グラフ)かどうか判定する"""
        return not self.has_cycle()

    def topological_sort(self) -> list[int]:
        """
        トポロジカルソートを行い、配列を返す。
        不可能（閉路が存在）な場合はValueErrorを投げる。
        """
        indegree = {v: 0 for v in self.graph}
        for u in self.graph:
            for v in self.graph[u]:
                indegree[v] += 1
                
        queue = deque([v for v in self.graph if indegree[v] == 0])
        res = []
        
        while queue:
            u = queue.popleft()
            res.append(u)
            for v in self.graph[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    queue.append(v)
                    
        if len(res) != len(self.graph):
            raise ValueError("Graph is not a DAG, cannot topologically sort.")
        return res

    def SCC(self) -> list[list[int]]:
        """
        強連結成分分解(SCC: Strongly Connected Components)。
        Kosarajuのアルゴリズムを使用。
        """
        order = []
        visited = set()

        # 1回目のDFS
        def dfs1(u: int):
            visited.add(u)
            for v in self.graph[u]:
                if v not in visited:
                    dfs1(v)
            order.append(u)

        for v in self.graph:
            if v not in visited:
                dfs1(v)

        # 辺の向きを逆にしたグラフを構築
        reversed_graph = {v: [] for v in self.graph}
        for u in self.graph:
            for v in self.graph[u]:
                reversed_graph[v].append(u)

        visited.clear()
        scc_list = []

        # 2回目のDFS
        def dfs2(u: int, comp: list[int]):
            visited.add(u)
            comp.append(u)
            for v in reversed_graph[u]:
                if v not in visited:
                    dfs2(v, comp)

        for v in reversed(order):
            if v not in visited:
                comp = []
                dfs2(v, comp)
                scc_list.append(comp)

        return scc_list

    def dijkstra(self, start: int, is_same_dist: bool = False) -> dict[int, int]:
        """
        startからの最短距離を求める。到達不可は self.INF。
        is_same_dist=True の場合は 01-BFS で高速に計算する。
        """
        dist = {v: self.INF for v in self.graph}
        dist[start] = 0

        if is_same_dist:
            # 01-BFS (重みが0と1のグラフ、または等コストグラフ用)
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v, w in self.graph[u].items():
                    if dist[v] > dist[u] + w:
                        dist[v] = dist[u] + w
                        if w == 0:
                            queue.appendleft(v)
                        else:
                            queue.append(v)
        else:
            # 通常のダイクストラ法 (重みが非負のグラフ用)
            pq = [(0, start)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                for v, w in self.graph[u].items():
                    if dist[v] > dist[u] + w:
                        dist[v] = dist[u] + w
                        heapq.heappush(pq, (dist[v], v))
                        
        return dist

    def Warshall_Floyd(self) -> dict[int, dict[int, int]]:
        """全点対最短経路をワーシャルフロイド法で求める"""
        dist = {u: {v: self.INF for v in self.graph} for u in self.graph}
        for v in self.graph:
            dist[v][v] = 0
        for u in self.graph:
            for v, w in self.graph[u].items():
                dist[u][v] = min(dist[u][v], w)

        nodes = list(self.graph.keys())
        for k in nodes:
            for i in nodes:
                for j in nodes:
                    if dist[i][k] != self.INF and dist[k][j] != self.INF:
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
                        
        return dist

    def has_path(self, from_node: int, to_node: int) -> bool:
        """from_node から to_node へのパスが存在するかを判定する"""
        if from_node not in self.graph or to_node not in self.graph:
            return False
            
        visited = {from_node}
        queue = deque([from_node])
        
        while queue:
            u = queue.popleft()
            if u == to_node:
                return True
            for v in self.graph[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
                    
        return False

    def path(self, from_node: int, to_node: int) -> list[int]:
        """
        from_node から to_node への最短パス(頂点のリスト)を返す。
        パスが存在しない場合は ValueError を投げる。
        """
        if from_node not in self.graph or to_node not in self.graph:
            raise ValueError("Node not found in graph.")
            
        dist = {v: self.INF for v in self.graph}
        prev = {v: None for v in self.graph}
        dist[from_node] = 0
        pq = [(0, from_node)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == to_node:
                break
            for v, w in self.graph[u].items():
                if dist[v] > dist[u] + w:
                    dist[v] = dist[u] + w
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))

        if dist[to_node] == self.INF:
            raise ValueError("No path exists.")

        # 経路の復元
        path_res = []
        curr = to_node
        while curr is not None:
            path_res.append(curr)
            curr = prev[curr]
            
        return path_res[::-1]