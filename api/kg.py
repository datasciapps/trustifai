from yacman import FutureYAMLConfigManager as YAMLConfigManager
from neo4j import AsyncGraphDatabase
from dataclasses import dataclass, field
from typing import Dict, Any
from box import Box


config = Box(YAMLConfigManager.from_yaml_file("config.yaml").to_dict())
to_camel = lambda s: s.replace(' ', '_').lower()
from_camel = lambda s: s.replace('_', ' ').title()


@dataclass
class Query:
    body: str
    parameters: Dict[str, Any] = field(default_factory=dict)


queries = {
    'get subgraph': lambda label: Query("""
            CALL apoc.cypher.run('
                MATCH (connected)-[e]->(n {label: $label})
                RETURN connected.label AS connected_node, type(e) AS edge_name',
                {label: $label}
            ) YIELD value
            RETURN value.connected_node AS connected_node, value.edge_name AS edge_name
        """, 
        parameters={
            'label': label 
        }), 
    'get tasks': Query("""
            CALL apoc.cypher.run( 
            'MATCH (p:' + $label + ')
            CALL apoc.path.subgraphAll(p, {maxLevel: -1}) YIELD nodes, relationships 
            RETURN nodes, relationships', 
            {} ) YIELD value 
            RETURN value.nodes AS nodes, value.relationships AS relationships
        """, parameters={'label': 'data_science_task'}),   
        #TODO: recursively scan for all the children of the data_science_task node 
        #TODO: given two nodes return all the pathways between these two nodes

    'get requirements': Query("""
            CALL apoc.cypher.run( 
            'MATCH (p:' + $label + ')
            CALL apoc.path.subgraphAll(p, {maxLevel: -1}) YIELD nodes, relationships 
            RETURN nodes, relationships', 
            {} ) YIELD value 
            RETURN value.nodes AS nodes, value.relationships AS relationships
        """, parameters={'label': 'requirement'}),
    'get operators': lambda parent: Query("""
            CALL apoc.cypher.run( 'MATCH (n:' + $label + ')
            CALL apoc.path.expand( n, null, null, 0, $depth )
            YIELD path WITH DISTINCT last(nodes(path)) AS connected, relationships(path) AS rels
            UNWIND rels AS rel RETURN connected.label AS connected_node, type(rel) AS edge_name', {depth: $depth} )
            YIELD value RETURN value.connected_node AS connected_node, value.edge_name AS edge_name
        """, parameters={'label': parent, 'depth': 1}),
    'get threat': lambda operator: Query("""
            CALL apoc.cypher.run( 
                'MATCH (start {label: $start_node}), (end {label: $end_node}) 
                CALL apoc.path.expandConfig(start, { 
                    endNodes: [end], 
                    relationshipFilter: ">",  // Only outgoing relationships 
                    maxLevel: -1, 
                    uniqueness: "NODE_GLOBAL" 
                }) YIELD path 
                RETURN count(path) > 0 AS is_connected', 
                {start_node: $start_node, end_node: $end_node} 
            ) YIELD value 
            RETURN value.is_connected AS is_connected 
        """, parameters={'label': operator}),
    'get action': lambda threat: Query("""
        """, parameters={'label': threat}),
    'delete node': Query("""
        MATCH p=(:`test_source`)-[]->() DETACH DELETE p ;
    """,
    parameters={

    }),
    'create node': lambda src, rel, dst, sentence, doc, page: Query(
        """ 
            CALL apoc.merge.node([$source], {label: $source}) YIELD node AS source 
            CALL apoc.merge.node([$destination], {label: $destination}) YIELD node AS destination 
            CALL apoc.merge.relationship( 
            source,  
            $relationship,  
            {sentence: $sentence, page_number: $page_number, document: $document},  
            {},  
            destination,  
            {} 
            ) YIELD rel 
            RETURN source, destination, rel 
        """,
        parameters = {
            'source': src,
            'destination': dst,
            'relationship': rel,
            'sentence': sentence,
            'document': doc,
            'page_number': page,
        }
    ),
    'get pathways': lambda source, destination: Query("""
        CALL apoc.cypher.run(
        '
        MATCH p = (start {label: $source})-[*]->(end {label: $destination})
        RETURN p
        ',
        {source: $source, destination: $destination}
        ) YIELD value
        RETURN value.p AS path
        """, parameters={
            'source': source,
            'destination': destination
        }),

    'get attributes of a person': Query("""
    CALL apoc.cypher.run(
        'MATCH (n)-[r: `is_attribute_of`]->(p:`person`)
         RETURN n
        ', {}
    ) YIELD value
    RETURN value.n AS nodes
""", parameters={}),
        # CALL apoc.cypher.run(
        #     'MATCH (n:' + $label + ')
        #     CALL apoc.path.expand( n, null, null, 0, $depth )
        #     YIELD path WITH DISTINCT last(nodes(path)) AS connected, relationships(path) AS rels
        #     UNWIND rels AS rel
        #     RETURN connected.label AS connected_node, type(rel) AS edge_name', {depth: $depth} )
        #     YIELD value RETURN value.connected_node AS connected_node, value.edge_name AS edge_name
        # """,
        # parameters={
        #     'label': ,
        #     'depth': ,
        # })
}

actions = {
    'immediate alternatives': lambda label: Query("""
            CALL apoc.cypher.run(' 
                // Step 1: Find the target node of the is_subclass_of relationship 
                MATCH (start {label: $start_node})-[:is_subclass_of]->(target) 
                // Step 2: Find all nodes connected to the target by is_subclass_of relationships 
                MATCH (connected)-[:is_subclass_of]->(target) 
            // Step 3: Exclude the start node from connected nodes  
                WHERE connected.label <> $start_node 
                // Step 3: Return the target node and all connected nodes 
                RETURN target.label AS target_node, collect(connected.label) AS connected_nodes 
                ', 
                {start_node: $start_node} 
            ) YIELD value 
            RETURN value.target_node AS target_node, value.connected_nodes AS connected_nodes 
        """,
        parameters={
            'start_node': label
        })
}


async def execute_query(query: Query):
    async with AsyncGraphDatabase.driver(config.neo4j.uri, auth=(config.neo4j.username, config.neo4j.password)) as driver:
        async with driver.session(database=config.neo4j.dbname) as session:
            try:
                result = await session.run(query.body, query.parameters)
                records = []
                async for record in result:
                    records.append(record.data())
                return records
            except Exception as e:
                print(f"An error occurred: {e}")
                return []

# await execute_query(queries['create node']('immediate alternatives', 'might mitigate', 'bias', 'The Immediate Alternatives mitigation strategy might mitigate bias', '', 0))
# spo = await execute_query(Query("""MATCH (s)-[p]->(o) RETURN s, p, o;"""))
# print(__pprint(spo, 'data_science_task'))
# print(f'{len(set(el['label'] for rel in spo for el in [rel['s'], rel['o']]))} nodes, {len(spo)} rels')
    
# kg = (await execute_query(queries['get requirements']))[0].get('relationships', [])
# for el in kg:
#     match el:
#         case {'label': s}, p, {'label': o}:
#             print(s, p, o)
#         case {'s': {'label': s}, 'p': ({'label': _s}, p, {'label': _o}), 'o': {'label': o}}:
#             assert (s, o) == (_s, _o), (s, _s, p, o, _o)
#             print(from_camel(s), from_camel(p), from_camel(o))
#         case unknown:
#             print(unknown)

# def __pprint(spo, root):
#     _nodes, done = set(el['label'] for rel in spo for el in [rel['s'], rel['o']]), []
#     out = []
#     def _foo(nid, depth=0):
#         if (not nid) or (nid not in _nodes): return ''
#         out.append(f'{" " * depth}{nid}')
#         for el in spo:
#             match el:
#                 case {'s': {'label': s}, 'p': ({'label': _s}, 'is_subclass_of', {'label': _o}), 'o': {'label': o}}:
#                     if o != nid: continue
#                     if s in done: continue
#                     done.append(s)
#                     res = _foo(s, depth+1)
#                     if res:
#                         out.append(f'{"  " * (depth+1)}{res}')
#         return ''

#     _foo(root)         
#     return '\n'.join(out)