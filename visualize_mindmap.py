import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

import json
import graphviz
import sys

def add_nodes_edges(graph, data):
    """
    Recursively adds nodes and edges to the graph from the hierarchical data.
    """
    node_name = data['name']
    graph.node(node_name, style='filled', fillcolor='lightblue', shape='box', fontname='Helvetica')

    if 'children' in data:
        for child in data['children']:
            child_name = child['name']
            graph.edge(node_name, child_name)
            add_nodes_edges(graph, child)

def main():
    """
    Main function to generate the mind map visualization.
    """
    json_filepath = 'collatz_mindmap.json'
    output_filename = 'graphs/collatz_mindmap_visualization' # Will be saved as .svg

    try:
        with open(json_filepath, 'r') as f:
            hierarchical_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_filepath}' was not found.")
        print("Please run the 'collatz_mindmap.py' script first to generate the data.")
        sys.exit(1)

    # Create a new directed graph
    dot = graphviz.Digraph('CollatzMindmap', comment='Reverse Collatz Conjecture Mind Map')
    # Increase size for larger graphs and set output format to svg
    dot.attr(rankdir='LR', size='100,100', splines='curved', bgcolor='transparent')
    dot.attr('node', shape='ellipse', style='filled', color='skyblue', fontname='Helvetica')
    dot.attr('edge', color='gray', arrowhead='vee')


    # Add nodes and edges from the JSON data
    add_nodes_edges(dot, hierarchical_data)

    try:
        # Render the graph to a file (e.g., SVG)
        # The format is determined by the extension of the output filename
        dot.render(output_filename, format='svg', view=False, cleanup=True)
        print(f"Successfully created the mind map visualization: {output_filename}.svg")
        print("You can find it in the 'graphs' directory.")

    except graphviz.backend.execute.ExecutableNotFound:
        print("\n---")
        print("ERROR: Graphviz executable not found.")
        print("To generate the mind map image, you need to install the Graphviz software.")
        print("\nInstructions:")
        print("1. Download and install Graphviz from: https://graphviz.org/download/")
        print("2. Make sure the Graphviz 'bin' directory is in your system's PATH environment variable.")
        print("   (The installer usually gives you an option to do this automatically).")
        print("3. After installation, try running this script again.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
