import os
import json
import sys
from neo4j import GraphDatabase

# Add parent directory to Python path to find minirag package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minirag.utils import xml_to_json

# Constants
WORKING_DIR = "./LiHua-World"
BATCH_SIZE_NODES = 500
BATCH_SIZE_EDGES = 100

# Neo4j connection credentials
NEO4J_URI = "neo4j+s://383cd4cc.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "20REQOBXBDD2d31iLaUFkuDXAZdOu1-HnFJybelPDcU"


def convert_xml_to_json(xml_path, output_path):
    """Convert XML to JSON and save to disk."""
    if not os.path.exists(xml_path):
        print(f"Error: File not found - {xml_path}")
        return None

    json_data = xml_to_json(xml_path)
    if json_data:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"JSON file created: {output_path}")
        return output_path
    else:
        print("Failed to create JSON data")
        return None


def read_json_in_batches(json_path, key, batch_size):
    """
    Stream JSON array (nodes or edges) in batches to avoid memory overload.
    Expects file structure: {"nodes": [...], "edges": [...]}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        items = data.get(key, [])
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]


def process_nodes_batch(tx, batch):
    query = """
    UNWIND $nodes AS node
    MERGE (e:Entity {id: node.id})
    SET e.entity_type = node.entity_type,
        e.description = node.description,
        e.source_id = node.source_id,
        e.displayName = node.id
    REMOVE e:Entity
    WITH e, node
    CALL apoc.create.addLabels(e, [node.id]) YIELD node AS labeledNode
    RETURN count(*)
    """
    tx.run(query, nodes=batch)


def process_edges_batch(tx, batch):
    query = """
    UNWIND $edges AS edge
    MATCH (source {id: edge.source})
    MATCH (target {id: edge.target})
    WITH source, target, edge,
         CASE
            WHEN edge.keywords CONTAINS 'lead' THEN 'lead'
            WHEN edge.keywords CONTAINS 'participate' THEN 'participate'
            WHEN edge.keywords CONTAINS 'uses' THEN 'uses'
            WHEN edge.keywords CONTAINS 'located' THEN 'located'
            WHEN edge.keywords CONTAINS 'occurs' THEN 'occurs'
            ELSE REPLACE(SPLIT(edge.keywords, ',')[0], '\"', '')
         END AS relType
    CALL apoc.create.relationship(source, relType, {
      weight: edge.weight,
      description: edge.description,
      keywords: edge.keywords,
      source_id: edge.source_id
    }, target) YIELD rel
    RETURN count(*)
    """
    tx.run(query, edges=batch)


def main():
    # File paths
    xml_file = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")
    json_file = os.path.join(WORKING_DIR, "graph_data.json")

    # Convert XML → JSON
    if not os.path.exists(json_file):
        if convert_xml_to_json(xml_file, json_file) is None:
            return

    # Create driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        with driver.session() as session:
            # Stream nodes in batches
            for node_batch in read_json_in_batches(json_file, "nodes", BATCH_SIZE_NODES):
                session.execute_write(process_nodes_batch, node_batch)
                print(f"Inserted batch of {len(node_batch)} nodes")

            # Stream edges in batches
            for edge_batch in read_json_in_batches(json_file, "edges", BATCH_SIZE_EDGES):
                session.execute_write(process_edges_batch, edge_batch)
                print(f"Inserted batch of {len(edge_batch)} edges")

            # Final label/displayName setup
            session.run("""
            MATCH (n)
            SET n.displayName = n.id
            WITH n
            CALL apoc.create.setLabels(n, [n.entity_type]) YIELD node
            RETURN count(*)
            """)

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        driver.close()
        print("Neo4j connection closed.")


if __name__ == "__main__":
    main()
