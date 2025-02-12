from typing import Dict, Any, Sequence, Literal, Optional, get_args
from neo4j import AsyncGraphDatabase
from dataclasses import dataclass, field
from collections import namedtuple
from inflection import underscore
from uuid import uuid4
import re

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
                    await tx.run("CREATE (n:Node {name: $name})", name=n.type)

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

Predicate = Literal['IsSubclassOf', 'Implements', 'IsEquivalentTo', 'HasParameter', 'AppliesTo', 'Has', 'IsAn', 'IsA',
                    'IsOfType', 'Fallback', 'WithDescription', 'WithParameter', 'Ensures', 'ContributesTo', 'Calls',
                    'MightIntroduce', 'IsThreatTo', 'IsDimensionOf', 'AttributesTo', 'ShouldEnsure', 'MightMitigate', 'IsSynonymOf', 'IsProtectedAttribute']
Statement = namedtuple('Statement', 'subject predicate object')

separator = ';'
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
Data Loading is subclass of Data Science Task
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

Predictive Performance is a Measure
Accuracy is a Metric
Recall is a Metric
ROC AUC Score is a Metric
Accuracy contributes to Predictive Performance
Recall contributes to Predictive Performance
ROC AUC Score contributes to Predictive Performance

# Evaluation Procedure is a Core Concept

# TODO: 'conditions' vs 'preferences'
# E.g., for smaller datasets, use cross-validation or LOOCV to make the best use of limited data

# Interface is a Core Concept
Sklearn Estimator is an Interface
Sklearn Estimator calls __init__; calls fit; calls predict

Sklearn Transformer is an Interface
Sklearn Transformer calls __init__; calls fit; calls transform

Function is an Interface

TrainTestSplit might introduce Bias

# TODO: libraries and versions
sklearn.model_selection.train_test_split is an Operator
sklearn.model_selection.train_test_split implements TrainTestSplit
sklearn.model_selection.train_test_split is a Function

sklearn.model_selection.KFold is an Operator
sklearn.model_selection.KFold implements KFold
sklearn.model_selection.KFold is a Function

sklearn.model_selection.StratifiedKFold is an Operator
sklearn.model_selection.StratifiedKFold implements StratifiedKFold
sklearn.model_selection.StratifiedKFold is a Function

sklearn.preprocessing.MinMaxScaler is an Operator
sklearn.preprocessing.MinMaxScaler is a Sklearn Transformer
sklearn.preprocessing.MinMaxScaler implements MinMaxScaler

sklearn.preprocessing.StandardScaler is an Operator
sklearn.preprocessing.StandardScaler is a Sklearn Transformer
sklearn.preprocessing.StandardScaler implements StandardScaler

sklearn.preprocessing.OneHotEncoder is an Operator
sklearn.preprocessing.OneHotEncoder is a Sklearn Transformer
sklearn.preprocessing.OneHotEncoder implements OneHotEncoder

sklearn.svm.SVC is an Operator
sklearn.svm.SVC is a Sklearn Estimator
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

#Concepts of Fairness
Equality is subclass of Fairness
Equity is subclass of Fairness
Transparency is subclass of Fairness
Confidentiality is subclass of Fairness
Voice is subclass of Fairness
Timeliness is subclass of Fairness
Impartiality is subclass of Fairness
Rationality is subclass of Fairness
Accountability is subclass of Fairness
Flexibility is subclass of Fairness
Dignity is subclass of Fairness
Bias is threat to Fairness
Equality is dimension of Fairness
Equity is dimension of Fairness

#Measures of fairness
Fairness Measure is a Measure
Demographic Parity is a Fairness Measure
Statistical Parity is synonym of Demographic Parity
Equalized odds is a Fairness Measure
Equal Opportunity is a Fairness Measure
Predictive Parity is a Fairness Measure
Error-Rate Parity is a Fairness Measure
Accuracy Parity is a Fairness Measure
Individual Fairness is a Fairness Measure
Distance-based Fairness is synonym of Individual Fairness
Counterfactual Fairness is a Fairness Measure
Causal Fairness is a Fairness Measure
Calibration Fairness is a Fairness Measure
Ranking Fairness is a Fairness Measure
Equal mis-opportunity is a Fairness Measure
Predictive Equality is synonym of Equal mis-opportunity
Balanced Group is a Fairness Measure
Average odds is a Fairness Measure

#Metrics of Fairness


Bias is a Risk
Algorithmic Bias is subclass of Bias
Historical Bias is subclass of Bias
Data Bias is subclass of Bias
Representation Bias is subclass of Bias
Measurement Bias is subclass of Bias
Omitted variable Bias is subclass of Bias
Evaluation Bias is subclass of Bias
Aggregation Bias is subclass of Bias
User interaction bias is subclass of Bias
Population Bias is subclass of Bias
Deployment Bias is subclass of Bias
Feedback Loop contributes to Bias
Unconscious Bias is subclass of Bias
Cognitive Bias is subclass of Bias
Confirmation Bias is subclass of Bias
Selection Bias is subclass of Bias
Reporting Bias is subclass of Bias


Person is an Entity
First Name attributes to Person
Middle Name attributes to Person
Last Name attributes to Person
Date of Birth attributes to Person
Age attributes to Person
Gender attributes to Person
Sex attributes to Person
Race attributes to Person
Ethnicity attributes to Person
Disability status attributes to Person
Religion attributes to Person
Sexual orientation attributes to Person
National origin attributes to Person
Marital status attributes to Person
Socioeconomic status attributes to Person

Gender is a Protected Attribute
Race is a Protected Attribute
Sexual orientation is a Protected Attribute
Race is a Protected Attribute
Religion is a Protected Attribute
Ethnicity is a Protected Attribute
Sex is a Protected Attribute
Disability status is a Protected Attribute
National origin is a Protected Attribute

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

Immediate Alternatives is a Mitigation Action
Immediate Alternatives might mitigate Bias
"""


class UnknownRelation(Exception):
    pass


def parse(statement: str) -> Sequence[Statement]:
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
