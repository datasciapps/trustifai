from typing import Dict, Any, Sequence, Literal, Optional, get_args
from neo4j import AsyncGraphDatabase
from dataclasses import dataclass, field
from collections import namedtuple
from inflection import underscore
from uuid import uuid4
import re
from api.knowledge import knowledge

from backend.config import config


async def populate(ontology):
    """
    Populate the Neo4j database from a NetworkX graph G.

    Assumes that the database is already empty. 
    If not, you can clear it by calling `delete_neo4j_database` first.
    """
    async with AsyncGraphDatabase.driver(config.local.uri, auth=(config.local.username, config.local.password)) as driver:
        async with driver.session(database=config.local.dbname) as session:
            try:
                tx = await session.begin_transaction()
                await tx.run("MATCH (n) DETACH DELETE n")
                # await tx.run("""
                #             CREATE CONSTRAINT node_id IF NOT EXISTS
                #             FOR (n:Node)
                #             REQUIRE n.id IS UNIQUE;
                #             """
                # )
                for n in ontology.nodes:
                    await tx.run("CREATE (n:Node {name: $name, label: $name})", name=n.type)
                    # await tx.run("""
                    #                 MATCH (n:Node)
                    #                 WHERE n.name IS NOT NULL
                    #                 CALL apoc.create.addLabels(n, [n.name]) YIELD node
                    #                 RETURN node
                    #             """)

                for e in ontology.edges:
                    if e.type == 'hasname': continue
                    await tx.run("""
                        MATCH (from {name: $s}), (to {name: $o})
                        CALL apoc.create.relationship(from, $p, {}, to)
                        YIELD rel
                        RETURN rel
                        """,
                        s=e.source,
                        p=underscore(e.type),
                        o=e.destination,
                    )
                await tx.commit()
                tx = await session.begin_transaction()
                for e in ontology.edges:
                    if e.type == 'hasname':
                        await tx.run("""
                            MATCH (n {name: $name})
                            SET
                              n.uuid = $name, 
                              n.name = $method
                            RETURN n;
                            """,
                            name=e.source,
                            method=e.destination,
                        )
                await tx.commit()
                tx = await session.begin_transaction()
                for e in ontology.edges:
                    if e.type == 'calls':
                        await tx.run("""
                            MATCH (n:Node {uuid: $name})
                            REMOVE n:Node
                            SET n:Method
                            RETURN n;
                            """,
                            name=e.destination,
                        )
                        await tx.run("""
                            MATCH (n:Node {name: $name})
                            REMOVE n:Node
                            SET n:Method, n.uuid = $uuid
                            RETURN n;
                            """,
                            name=e.destination,
                            uuid=uuid4().hex
                        )
                await tx.commit()                
            except Exception as e:
                print(f"An error occurred: {e}")
                return []



# TODO: Skip triggers fallback

Predicate = Literal['IsSubclassOf', 'Implements', 'IsEquivalentTo', 'HasParameter', 'AppliesTo', 'IsAn', 'IsA',
                    'IsOfType', 'Fallback', 'WithDescription', 'WithParameter', 'Ensures', 'ContributesTo', 'Calls',
                    'MightIntroduce', 'IsThreatTo', 'IsDimensionOf', 'AttributesTo', 'ShouldEnsure', 'MightMitigate', 'IsSynonymOf', 'IsProtectedAttribute', 'BelongsTo', 'InheritsFrom',
                    'ContainArgument', 'OfReturnType', 'ContainAttribute', 'ContainMethod']
Statement = namedtuple('Statement', 'subject predicate object')

separator = ';'
camel_case = re.compile(r'(?<!^)(?=[A-Z])')



class UnknownRelation(Exception):
    pass


def parse(statement: str) -> Sequence[Statement]:

    try:
        assert 'has name' not in get_args(Predicate)
        edges = list(map(underscore, get_args(Predicate)))
        part, *rest = statement.split(separator)
        for edge in edges:
            e = edge.replace('_', ' ')
            if e in part:
                s, o = part.split(e)
                if rest:
                    _id = uuid4().hex
                    return [Statement(subject=s.strip(), predicate=edge, object=_id),
                            Statement(subject=_id, predicate='has name', object=o.strip())] \
                            + parse(_id + separator.join(rest))
                else:
                    return [Statement(subject=s.strip(), predicate=edge, object=o.strip())]
                
        raise UnknownRelation(statement)
    except Exception as error: 
        #print(f"An error occurred: {error}")
        print(part)


@dataclass
class Node:
    type: str
    attr: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __hash__(self): return hash(self.type)


@dataclass
class Edge:
    source: str
    destination: str
    type: Predicate
    attr: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class Ontology:
    nodes: Sequence[Node]
    edges: Sequence[Edge]


def to_camel(expr: str) -> str:
    # return re.sub(r'\s(\w)', lambda x: x[1].upper(), expr.title())
    return expr.replace(' ', '')


def get_ontology() -> Ontology:
    stmts = [el for line in knowledge.strip().split('\n') if line and not line.startswith('#') for el in parse(line)]
    return Ontology(
        nodes=list(set([Node(type=el) for st in stmts for el in [st.subject, st.object]])),
        edges=list(map(lambda el: Edge(source=el.subject,
                                       destination=el.object,
                                       type=to_camel(el.predicate)
                                       ), stmts))
    )


if __name__ == "__main__":
    print(get_ontology())

    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(populate(get_ontology()))
