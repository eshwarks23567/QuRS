ENTITY_EXTRACTION_PROMPT = """You are a regulatory knowledge extraction assistant.

Extract all entities and relationships from the text below.

Return ONLY valid JSON in this exact format:
{{
  "entities": [
    {{"type": "Regulation", "name": "...", "properties": {{"description": "...", "issuing_body": "..."}}}},
    {{"type": "CapitalRatio", "name": "...", "properties": {{"minimum_threshold": null, "description": "..."}}}},
    {{"type": "Jurisdiction", "name": "...", "properties": {{"region": "..."}}}}
  ],
  "relationships": [
    {{"source": "...", "target": "...", "type": "AFFECTS"}},
    {{"source": "...", "target": "...", "type": "APPLIES_IN"}}
  ]
}}

Valid entity types: Regulation, CapitalRatio, Jurisdiction, Institution, Metric
Valid relationship types: AFFECTS, APPLIES_IN, REFERENCES, DEFINES, REQUIRES

Rules:
- Only extract entities explicitly mentioned in the text
- Relationship source and target must both be entity names you extracted
- Return empty lists if nothing relevant is found
- Do not wrap in markdown fences

Text:
{text}

JSON output:"""


CYPHER_GENERATION_PROMPT = """You are a Neo4j Cypher expert for a regulatory knowledge graph.

Graph Schema:
{schema_context}

Convert the following natural language query into a valid read-only Cypher query.
- Use only MATCH and RETURN — no CREATE, MERGE, SET, DELETE, or REMOVE
- Prefer case-insensitive matching with toLower() where names are compared
- Always include a LIMIT (default 50) unless the query asks for all results
- Return the Cypher inside a ```cypher block and nothing else

Query: {user_query}

```cypher"""


REASONING_PROMPT = """You are a regulatory analysis assistant with access to a knowledge graph.

User Query:
{user_query}

Knowledge Graph Results:
{graph_results}

Additional Context from Documents:
{additional_context}

Instructions:
- Answer directly and concisely based on the data above
- If graph results are empty, state that clearly and suggest what data might be missing
- Structure the answer with bullet points if multiple regulations or ratios are involved
- Do not hallucinate regulations or ratios not present in the data

End your response with exactly one line in this format:
Confidence: <score between 0.0 and 1.0>"""


QUERY_UNDERSTANDING_PROMPT = """Analyze the following regulatory query and extract structured metadata.

Query: {user_query}

Respond ONLY with valid JSON (no markdown fences):
{{
  "intent": "one-sentence description of what the user wants",
  "entity_types": ["list of relevant entity types from: Regulation, CapitalRatio, Jurisdiction, Institution, Metric"],
  "relationships": ["list of relevant relationship types from: AFFECTS, APPLIES_IN, REFERENCES, DEFINES, REQUIRES"],
  "jurisdictions": ["list of jurisdictions mentioned, empty list if none"]
}}"""
