

import json

def generate_collatz_graph(start_node, max_depth):
    """
    Generates a graph of reversed Collatz sequences using breadth-first search.

    Args:
        start_node (int): The number to start the reverse sequence from.
        max_depth (int): The maximum depth to traverse the tree.

    Returns:
        dict: An adjacency list representation of the Collatz graph.
    """
    graph = {}
    queue = [(start_node, 0)]
    # Keep track of nodes to add to the queue to avoid duplicates
    nodes_to_process = {start_node}

    while queue:
        current_node, depth = queue.pop(0)

        if depth >= max_depth:
            continue

        if current_node not in graph:
            graph[current_node] = []

        # Reverse operation 1: n/2 -> n, so the predecessor is 2*n
        pred_even = current_node * 2
        graph[current_node].append(pred_even)
        if pred_even not in nodes_to_process:
            queue.append((pred_even, depth + 1))
            nodes_to_process.add(pred_even)

        # Reverse operation 2: 3n+1 -> n, so the predecessor is (n-1)/3
        # This is only valid if the result is an odd integer > 1.
        if (current_node - 1) % 3 == 0:
            pred_odd = (current_node - 1) // 3
            if pred_odd > 1 and pred_odd % 2 != 0:
                graph[current_node].append(pred_odd)
                if pred_odd not in nodes_to_process:
                    queue.append((pred_odd, depth + 1))
                    nodes_to_process.add(pred_odd)
    return graph

def format_as_text_tree(graph, node, prefix="", is_last=True):
    """
    Formats the graph into a visually appealing, indented text tree.

    Args:
        graph (dict): The graph to format.
        node (int): The current node to process.
        prefix (str): The prefix for the current line's indentation.
        is_last (bool): True if the node is the last child of its parent.

    Returns:
        str: The formatted text tree string.
    """
    output = prefix
    if prefix: # Not for the root node
        output += "+-- " if is_last else "+-- "
    
    output += str(node) + "\n"

    children = sorted(graph.get(node, []))
    for i, child in enumerate(children):
        is_last_child = (i == len(children) - 1)
        new_prefix = prefix + ("    " if (is_last or not prefix) else "|   ")
        output += format_as_text_tree(graph, child, new_prefix, is_last_child)

    return output

def to_hierarchical_dict(graph, node):
    """
    Converts the graph from an adjacency list to a hierarchical dictionary
    suitable for JSON conversion (e.g., for D3.js).

    Args:
        graph (dict): The graph to convert.
        node (int): The current node to process.

    Returns:
        dict: A hierarchical dictionary.
    """
    children = sorted(graph.get(node, []))
    if not children:
        return {"name": str(node)}

    return {
        "name": str(node),
        "children": [to_hierarchical_dict(graph, child) for child in children]
    }

if __name__ == "__main__":
    START_NUMBER = 1
    MAX_DEPTH = 14

    print(f"Generating reverse Collatz tree starting from {START_NUMBER} to depth {MAX_DEPTH}\n")

    # 1. Generate the graph data
    collatz_graph = generate_collatz_graph(START_NUMBER, MAX_DEPTH)

    # 2. Print the visually appealing text tree to the console
    print("--- Text Mindmap ---")
    text_tree = format_as_text_tree(collatz_graph, START_NUMBER)
    print(text_tree)

    # 3. Convert to hierarchical dictionary and save as a JSON file
    print("\n--- JSON Output ---")
    hierarchical_data = to_hierarchical_dict(collatz_graph, START_NUMBER)
    json_filepath = "collatz_mindmap.json"
    
    with open(json_filepath, "w") as f:
        json.dump(hierarchical_data, f, indent=4)
        
    print(f"Successfully saved hierarchical data to {json_filepath}")
    print("You can use this JSON file with visualization tools like D3.js or other mind map applications.")


