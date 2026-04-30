import os

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable


class Neo4jClient:
    """Manages connections and CRUD operations against the Neo4j knowledge graph."""

    def __init__(self) -> None:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def verify_connection(self) -> bool:
        """Return True if the database is reachable."""
        try:
            self._driver.verify_connectivity()
            return True
        except ServiceUnavailable:
            return False

    # ── Read ──────────────────────────────────────────────────────────────────

    def execute_cypher(self, query: str, params: dict | None = None) -> list[dict]:
        """Run a read-only Cypher query and return results as a list of dicts."""
        with self._driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    # ── Write ─────────────────────────────────────────────────────────────────

    def ingest_graph_data(self, data: dict) -> None:
        """Merge extracted entities and relationships into the graph."""
        with self._driver.session() as session:
            for entity in data.get("entities", []):
                self._merge_node(
                    session,
                    label=entity["type"],
                    name=entity["name"],
                    props=entity.get("properties") or {},
                )
            for rel in data.get("relationships", []):
                self._merge_relationship(
                    session,
                    source=rel["source"],
                    target=rel["target"],
                    rel_type=rel["type"],
                )

    def _merge_node(self, session, label: str, name: str, props: dict) -> None:
        query = (
            f"MERGE (n:{label} {{name: $name}}) "
            "SET n += $props "
            "RETURN n"
        )
        session.run(query, name=name, props=props)

    def _merge_relationship(
        self, session, source: str, target: str, rel_type: str
    ) -> None:
        query = (
            "MATCH (a {name: $source}), (b {name: $target}) "
            f"MERGE (a)-[:{rel_type}]->(b)"
        )
        session.run(query, source=source, target=target)

    # ── Schema ────────────────────────────────────────────────────────────────

    def create_constraints(self) -> None:
        """Create uniqueness constraints for all core node labels."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Regulation)   REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:CapitalRatio)  REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Jurisdiction)  REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Institution)   REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Metric)        REQUIRE n.name IS UNIQUE",
        ]
        with self._driver.session() as session:
            for stmt in constraints:
                session.run(stmt)

    def clear_graph(self) -> None:
        """Delete all nodes and relationships — use only in development/testing."""
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
