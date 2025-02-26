from dataclasses import dataclass
from itertools import repeat
from pipe import select
from lark import Lark, Tree
from box import Box
import difflib
import asyncio
import sys
import inquirer
import pandas as pd
from api.kg import Query, execute_query, queries, actions, from_camel, to_camel
from api.events import Event, emit
from time import sleep
from termcolor import colored
from api.eda import basicStatistics
import json
import seaborn as sns
import matplotlib.pyplot as plt
import math
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset


@dataclass
class Pipeline:
    spec: str
    tree: Tree


lower = lambda ll: [l.lower() for l in ll]
approx_match = lambda token, options: difflib.get_close_matches(token, options, n=5, cutoff=0.2)
exact_match = lambda token, options: token if token.lower() in lower(options) else False
cleanup = lambda obj, key: [[el['label'] if isinstance(el, dict) else el for el in path[key]] for path in obj]


def pipeline(spec: str) -> Pipeline:
    grammar = '''
    DIGIT: "0".."9"
    INT: DIGIT+
    FLOAT: INT? "." DIGIT+
    NUMBER: FLOAT | INT
    WORD: /[a-zA-Z_]+/
    STRING: /"[^"]*"/
    start: expression (">" expression)*
    key: WORD
    value: WORD | NUMBER | STRING
    kwarg: key "=" value
    attribute: WORD
    callable: attribute "(" kwarg? ("," kwarg)* ")"
    expression: attribute | callable

    // imports from terminal library
    // %import common.WORD
    %import common.WS
    %ignore WS
    '''

    l = Lark(grammar)
    tree = l.parse(spec)
    return Pipeline(spec, tree)



async def debug(pipeline: Pipeline) -> None:
    emit(Event('DebuggingInitiated', {'pipeline': pipeline.spec}))

    state = Box({
        'pipeline': {
            'steps': list(pipeline.tree.find_data('attribute') | select(lambda attr: attr.children[0].value)),
        },
        'tasks': list((await execute_query(queries['get tasks']))[0].get('nodes', []) | select(lambda n: from_camel(n['label']))),
    })

    emit(Event('OperatorsExtracted', {'operators': list(state.pipeline.steps)}))

    to_check, tasks = [], []

    for step in state.pipeline.steps:
        approxmatches = None
        exactmatch = None
        answers = None
        operator = None

        #find the exact match first
        exactmatch = exact_match(step, state.tasks)
        if exactmatch != False and exactmatch is not None:
            operator = exactmatch

        else:
            approxmatches = approx_match(step, state.tasks)

            #list all the matches and ask the user to select the correct one
            if not approxmatches:
                emit(Event('No matches found', {'task': step}))
            else:
                #list the matches and ask the user to select the correct one
                questions = [inquirer.List('task', message=colored("Approximate matches found for step {}, please select the best option".format(step), "magenta"), choices=approxmatches + ["None of the above"])]
                answers = inquirer.prompt(questions)
        
            operator = answers['task']
            if operator == "None of the above":
                emit(Event('TaskNotFound', {'task': step}))

        paths = cleanup((await execute_query(queries['get directed pathways'](operator, 'Data Science Task'))), 'path')
        print(paths)
        paths = [l.split(':') for l in set(":".join(el) for el in paths)]
        if paths:
            emit(Event('TaskRecognized', {'task': step, 'paths': paths}))
            #sleep(0.5)
            tasks.append(operator)
            print(tasks)
            to_check.extend(paths)
    
    emit(Event('TasksIdentified', {'tasks': tasks}))

    actions_to_check = []
    for task in tasks:
        risks = cleanup((await execute_query(queries['get directed pathways'](task, 'Risk'))), 'path')
        if risks:
            emit(Event('RiskRelationshipsIdentified', {'task': task, 'risks': risks}))
            name_of_risks = [risk[2] for risk in risks]
            emit(Event('RisksIdentified', {'task': task, 'risks': name_of_risks}))
            for risk in risks:
                actions_to_check.extend(zip(repeat(task), risk[::2][1:-2]))

    riskAndMitigationActions = {}
    for task, risk in actions_to_check:
        #TODO: Finish this one today.
        #print(task, risk)
        candidates = (await execute_query(Query(f"""MATCH p=(s)-[:`might_mitigate`]->(r) where r.label = "{risk}" RETURN s;""")))
        #candidates = (await execute_query(actions["Immediate Alternatives"](task)))
        candidates = [s['s']['label'] for s in candidates]
        #candidates = candidates[0]['connected_nodes']

        if candidates:
            riskAndMitigationActions[risk] = candidates
            #emit(Event('MitigationActionsIdentified', {'task': task, 'risk': risk, 'actions': candidates}))
            #for candidate in candidates:
                #alternatives = cleanup(await execute_query(actions[candidate](task)), 'connected_nodes')
                #emit(Event('SuggestionRendered', {'task': task, 'risk': risk, 'action': candidate, 'actions': alternatives}))
    emit(Event('MitigationActionsIdentified', {'risk_and_actions': riskAndMitigationActions}))


async def main():
    return await debug(pipeline('DataLoading(path="data/test.csv") > TrainTestSplit(test_size=.2, random_state=42) > MinMaxScaler > OneHotEncoder > SVM'))

async def checkData():
    #load the dataset without the index column
    fileName = 'credit.csv'
    data = pd.read_csv(fileName)
    emit(Event('DataLoaded', {'data': data}))
    sleep(0.5)
    emit(Event('FeaturesNamesExtracted', {'features': list(data.columns)}))
    sleep(0.5)
    state = list((await execute_query(queries['get protected attributes'])))#[0].get('nodes', []) | select(lambda n: from_camel(n['label'])))
    attributes = [from_camel(n['nodes']['label']) for n in state]
    emit(Event('ProtectedAttributes', {'attributes': attributes}))
    prelimQs = [inquirer.List('q1', message=colored("Data {} has informaton about people.".format(fileName), "magenta"), choices=["Yes", "No"]), 
                 inquirer.List('q2', message=colored("Data {} has at least one of the protected attributes listed below, or has any proxy attribute that may identify any one of the protected attributes. \n {}".format(fileName, attributes), "magenta"), choices=["Yes", "No"])]
    prelimAs = inquirer.prompt(prelimQs)
    if prelimAs['q1'] == "No" and prelimAs['q2'] == "No":
        emit(Event('Alert', {'message': "Data does not contain any information about people. Check for fairness is not required."}))
        return

    sleep(0.5)
    features = list(data.columns)
    protectedFeatures = []
    emit(Event('FindingProtectedAttributes', {'attributes': attributes}))
    for feature in features:
        exactmatch = exact_match(feature, attributes)
        if exactmatch != False and exactmatch is not None:
            protectedFeature = exactmatch.lower()
            protectedFeatures.append(protectedFeature)
        else:
            approxmatches = approx_match(feature, attributes)
            if not approxmatches:
                emit(Event('No matches found', {'task': feature}))
                continue
            else:
                questions = [inquirer.List('task', message=colored("Approximate matches found for feature {}, please select the best option".format(feature), "magenta"), choices=approxmatches + ["Not a protected attribute"])]
                answers = inquirer.prompt(questions)
            protectedFeature = answers['task']
            if protectedFeature != "Not a protected attribute":
                protectedFeatures.append(feature)
            else:
                emit(Event('NotAProtectedFeature', {'attribute': feature}))
    emit(Event('ProtectedAttributesIdentified', {'attributes': protectedFeatures}))
    sleep(0.5)
    emit(Event('AuditingData', {'data': data, 'protected': protectedFeatures}))
    sleep(0.5)
    distinct_values = await basicStatistics(data=data)

    ##check for categorical data
    haveCategorical = input("Do you have any categorical data? (yes/no): ")
    if haveCategorical.lower() == "yes":
        #ask for file name where information about categorical data is stored
        filePath = input("Please enter the file path where information about categorical data is stored: ")
        emit(Event('CategoricalInformation', {'File': filePath}))
        sleep(0.5)
        #load the json file
        with open(filePath, 'r') as f:
            categoricalIndicator = json.load(f)
        emit(Event('CategoricalIndicatorsLoaded', {'data': categoricalIndicator}))
        sleep(0.5)
        #based on the categorical indicators assign Categorical or Continuous to the features
        categorical_columns = []
        numerical_columns = []
        for feature in features:
            if categoricalIndicator[feature] == True:
                categorical_columns.append(feature)
                data[feature] = data[feature].astype('category')
            else:
                numerical_columns.append(feature)
        emit(Event('CategoricalDataAssigned', {'data': data}))
        emit(Event('SummaryStatistics', {'data_types': data.dtypes}))

        protectedCategorical = [i for i in protectedFeatures if i in categorical_columns]
        protectedNumerical = [i for i in protectedFeatures if i in numerical_columns]

        #plots
        # Create a figure with subplots
        total_plots = len(protectedCategorical) + len(protectedNumerical)

        # Create a scrollable figure using Plotly subplots
        fig = sp.make_subplots(rows=total_plots, cols=1, subplot_titles=protectedCategorical + protectedNumerical)

        # Add categorical features as bar charts
        for i, col in enumerate(protectedCategorical):
            counts = data[col].value_counts()
            fig.add_trace(
                go.Bar(x=counts.index, y=counts.values, name=col),
                row=i+1, col=1
            )

        # Add numerical features as histograms
        # for i, col in enumerate(protectedNumerical):
        #     fig.add_trace(
        #         go.Histogram(x=data[col], nbinsx=20, name=col),
        #         row=len(categorical_columns) + i + 1, col=1
        #     )

        # Update layout for scrolling
        fig.update_layout(
            height=300 * total_plots,  # Adjust height to fit all subplots
            showlegend=False,
            title_text="Feature Distribution Visualization",
            xaxis=dict(title="Values"),
            yaxis=dict(title="Count"),
            hovermode="x",
        )

        # Show the figure by asking the user if they want to see the plots
        show_plots = input("Do you want to see the plots? (yes/no): ")
        if show_plots.lower() == "yes":
            fig.show()

        #ask for privileged and unprivileged groups for each protected attribute and provide the options to choose from the distinct values
        #TODO: maybe find automatically which groups are privileged and which are not. 

        privQ = [inquirer.List(feature, message=colored("Select the privileged group for feature '{}'".format(feature), "magenta"), choices=data[feature].unique().tolist()) for feature in protectedCategorical]
        privA = inquirer.prompt(privQ)

        unprivQ = [inquirer.List(feature, message=colored("Select the unprivileged group for feature '{}'".format(feature), "magenta"), choices=data[feature].unique().tolist()) for feature in protectedCategorical]
        uprivA = inquirer.prompt(unprivQ)

        label_column = input("Enter the name of the label column: ")
        protected_attributes = protectedFeatures
        
        aifDataset = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=data, label_names=[label_column], protected_attribute_names=protected_attributes)
        print(type(privA), type(uprivA))
        metric = BinaryLabelDatasetMetric(aifDataset, 
                                  privileged_groups=[privA], 
                                  unprivileged_groups=[uprivA])

        print(f"Disparate Impact: {metric.disparate_impact()}")
        print(f"Statistical Parity Difference: {metric.statistical_parity_difference()}")
        print(f"Mean Difference: {metric.mean_difference()}")

    





async def tryStuff():
    path =  await execute_query(queries['get protected attributes'])
    print(path)
    
if __name__ == "__main__":
    #asyncio.run(tryStuff())
    asyncio.run(checkData())
    #asyncio.run(main())