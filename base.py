from neo4j import AsyncGraphDatabase
from dataclasses import dataclass, field
from collections import namedtuple
from typing import Dict, Sequence, Any, Literal, Optional, get_args
from pathlib import Path
from inflection import underscore
import re


from backend.config import config


async def populate(ontology):
    """
    Populate the Neo4j database from a NetworkX graph G.

    Assumes that the database is already empty. 
    If not, you can clear it by calling `delete_neo4j_database` first.
    """
    async with AsyncGraphDatabase.driver(config.knowledgeBase.uri, auth=(config.knowledgeBase.username, config.knowledgeBase.password)) as driver:
        async with driver.session(database=config.knowledgeBase.dbname) as session:
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
                    await tx.run(
                        """
                        CREATE (n:Node {id: $node_id})
                        """,
                        node_id=n.type.title(),
                    )

                for e in ontology.edges:
                    await tx.run(
                        f"""
                        MATCH (n1:Node {{id: $u}}), (n2:Node {{id: $v}})
                        CREATE (n1)-[r:{underscore(e.type)}]->(n2)
                        """,
                        u=e.source.title(),
                        v=e.destination.title(),
                    )
                await tx.commit()
            except Exception as e:
                print(f"An error occurred: {e}")
                return []
        

EdgeType = Literal['IsA', 'IsSubclassOf', 'Implements', 'IsEquivalentTo', 'HasParameter']

def to_camel(expr: str) -> str:
    # return re.sub(r'\s(\w)', lambda x: x[1].upper(), expr.title())
    return expr.replace(' ', '')


Statement = namedtuple('Statement', 'subject predicate object')
camel_case = re.compile(r'(?<!^)(?=[A-Z])')
knowledge = """
# Data Science Task is a Core Concept
Data Preprocessing is subclass of Data Science Task
Supervised Learning is subclass of Data Science Task
Unsupervised Learning is subclass of Data Science Task
# TODO: binary vs multiclass classification
Classification is subclass of Supervised Learning
Regression is subclass of Supervised Learning
Clustering is subclass of Unsupervised Learning
Data Collection is subclass of Data Science Task
Data Cleaning is subclass of Data Preprocessing
Data Augmentation is subclass of Data Preprocessing
Data Integration is subclass of Data Preprocessing
Data Exploration is subclass of Data Science Task
Data Visualization is subclass of Data Science Task
Statistical Analysis is subclass of Data Science Task
Natural Language Processing is subclass of Data Science Task
Image Processing is subclass of Data Science Task
Video Processing is subclass of Data Science Task
Time Series Analysis is subclass of Data Science Task
Anomaly Detection is subclass of Data Science Task
Fraud Detection is subclass of Anomaly Detection
Network Intrusion Detection is subclass of Anomaly Detection
Model Selection is subclass of Data Science Task
Model Evaluation is subclass of Data Science Task
Model Deployment is subclass of Data Science Task

TrainTestSplit is subclass of Model Selection
KFold is subclass of Model Selection
StratifiedKFold is subclass of Model Selection
Data Normalization is subclass of Data Preprocessing
MinMaxScaler is subclass of Data Normalization
StandardScaler is subclass of Data Normalization
Data Encoding is subclass of Data Preprocessing
OneHotEncoder is subclass of Data Encoding
SVM is subclass of Classification

TrainTestSplit might introduce Bias

# TODO: libraries and versions
sklearn.model_selection.train_test_split is an Operator
sklearn.model_selection.KFold is an Operator
sklearn.model_selection.StratifiedKFold is an Operator
sklearn.preprocessing.MinMaxScaler is an Operator
sklearn.preprocessing.StandardScaler is an Operator
sklearn.preprocessing.OneHotEncoder is an Operator
sklearn.svm.SVC is an Operator

sklearn.model_selection.train_test_split implements TrainTestSplit
sklearn.model_selection.KFold implements KFold
sklearn.model_selection.StratifiedKFold implements StratifiedKFold
sklearn.preprocessing.MinMaxScaler implements MinMaxScaler
sklearn.preprocessing.StandardScaler implements StandardScaler
sklearn.preprocessing.OneHotEncoder implements OneHotEncoder
sklearn.svm.SVC implements SVM

# Requirement is a Core Concept
# https://digital-strategy.ec.europa.eu/en/library/ethics-guidelines-trustworthy-ai
# TODO: Add 'has context' with descriptions and definitions
Fairness is a Requirement
Lawfulness is a Requirement
Human Agency is a Requirement
Human Oversight is a Requirement
Safety is a Requirement
Technical Robustness is a Requirement
Privacy is a Requirement
Data Governance is a Requirement
Transparency is a Requirement
Diversity is a Requirement
Non-discrimination is a Requirement
Fairness is a Requirement
Societal Well-being is a Requirement
Environmental Well-being is a Requirement
Accountability is a Requirement
Acceptance is a Requirement

Equality is subclass of Fairness
Equity is subclass of Fairness
Bias is threat to Fairness
Equality is dimension of Fairness
Equity is dimension of Fairness
Bias is a Risk
Algorithmic Bias is subclass of Bias
Historical Bias is subclass of Bias

Person is an Entity
First Name attributes to Person
Middle Name attributes to Person
Last Name attributes to Person
Date of Birth attributes to Person
Age attributes to Person
Gender attributes to Person
Race attributes to Person
Ethnicity attributes to Person
Disability status attributes to Person
Religion attributes to Person
Sexual orientation attributes to Person
National origin attributes to Person
Marital status attributes to Person
Socioeconomic status attributes to Person

# Gender is protected against Bias
# Race is protected against Bias

Classification should ensure Fairness

Tabular Data is subclass of Data

# Metrics, e.g., https://www.kaggle.com/code/alexisbcook/ai-fairness
Demographic Parity is subclass of Equality
Demographic Parity contributes to Equality
Equality of Opportunity  is subclass of Equality
Equality of Opportunity contributes to Equality
Equal Accuracy is subclass of Equality
Equal Accuracy contributes to Equality
Group Unaware is subclass of Equality
Group Unaware contributes to Equality

# https://github.com/understandable-machine-intelligence-lab/Quantus
Explainability is a Requirement
Explanation Robustness is subclass of Explainability
Explanation Consistency is a Measure
Quantus_Consistency is a Metric
Explanation Consistency contributes to Explanation Robustness
Quantus_Consistency implements Explanation Consistency
Quantus_Consistency applies to Tabular Data

# X implements Demographic parity

Predictive Performance is a Measure
Accuracy is a Metric
Recall is a Metric
ROC AUC Score is a Metric
Accuracy contributes to Predictive Performance
Recall contributes to Predictive Performance
ROC AUC Score contributes to Predictive Performance
"""


class UnknownRelation(Exception):
    pass


def parse(statement: str) -> Statement:
    # edges = [camel_case.sub(r' ', edge).lower() for edge in get_args(EdgeType)]
    edges = list(map(underscore, get_args(EdgeType)))
    for edge in edges:
        e = edge.replace('_', ' ')
        if e in statement:
            s, o = statement.split(e)
            return Statement(subject=s.strip(), predicate=edge, object=o.strip())
    raise UnknownRelation(statement)


@dataclass
class Node:
    type: str
    attr: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __hash__(self): return hash(self.type)


@dataclass
class Edge:
    source: str
    destination: str
    type: EdgeType
    attr: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class Ontology:
    nodes: Sequence[Node]
    edges: Sequence[Edge]


def get_ontology() -> Ontology:
    stmts = [parse(line) for line in knowledge.strip().split('\n') if line and not line.startswith('#')]
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