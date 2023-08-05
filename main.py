import heapq
import networkx as nx
import matplotlib.pyplot as plt

graph = {
  'A': [('B', 5), ('C', 2), ('D', 1)],
  'B': [('A', 5), ('E', 4), ('F', 7)],
  'C': [('A', 2), ('F', 7), ('G', 3)],
  'D': [('A', 1), ('G', 2), ('H', 2)],
  'E': [('B', 4)],
  'F': [('B', 7), ('C', 7)],
  'G': [('C', 3), ('D', 2), ('H', 3)],
  'H': [('D', 2), ('G', 3)],
}
start_node = 'A'
goal_node = 'H'


def uniform_cost_search(graph, start, goal):
  priority_queue = [(0, start, [start])]
  explored = set()

  while priority_queue:
    current_cost, current_node, path = heapq.heappop(priority_queue)

    if current_node in explored:
      continue

    explored.add(current_node)

    if current_node == goal:

      return path, current_cost

    for neighbor, cost in graph[current_node]:
      if neighbor not in explored:
        new_path = path + [neighbor]
        heapq.heappush(priority_queue,
                       (current_cost + cost, neighbor, new_path))

  return None, None


def display_graph(graph):
  G = nx.DiGraph()
  for node, neighbors in graph.items():
    for neighbor, weight in neighbors:
      G.add_edge(node, neighbor, weight=weight)

  pos = nx.spring_layout(G)
  nx.draw(G,
          pos,
          with_labels=True,
          node_size=1000,
          node_color='skyblue',
          font_size=10,
          font_weight='bold',
          arrows=True)
  labels = nx.get_edge_attributes(G, 'weight')
  nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

  plt.title("Graph")
  plt.show()


def main():
  display_graph(graph)

  path, cumulative_cost = uniform_cost_search(graph, start_node, goal_node)

  if path is not None:
    print(
      f"Minimum cost from '{start_node}' to '{goal_node}' is: {cumulative_cost}"
    )
    print(
      f"Traversal from '{start_node}' to '{goal_node}': {' -> '.join(path)}")
  else:
    print(f"No path found from '{start_node}' to '{goal_node}'.")


if __name__ == "__main__":
  main()
