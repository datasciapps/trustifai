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
            #sleep(2)
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
    #TODO: the protected features need to be in the knowledge base. For now, we will use the attributes of a person as a placeholder
    data = pd.read_csv('credit.csv', index_col=0)
    emit(Event('DataLoaded', {'data': data}))
    sleep(2)
    emit(Event('FeaturesNamesExtracted', {'features': list(data.columns)}))
    sleep(2)
    state = list((await execute_query(queries['get protected attributes'])))#[0].get('nodes', []) | select(lambda n: from_camel(n['label'])))
    attributes = [from_camel(n['nodes']['label']) for n in state]
    emit(Event('FindingProtectedAttributes', {'attributes': attributes}))
    sleep(2)
    features = list(data.columns)
    protectedFeatures = []

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
                protectedFeatures.append(protectedFeature)
            else:
                emit(Event('NotAProtectedFeature', {'attribute': feature}))
    emit(Event('ProtectedAttributesIdentified', {'attributes': protectedFeatures}))


async def tryStuff():
    path =  await execute_query(queries['get protected attributes'])
    print(path)
    
if __name__ == "__main__":
    #asyncio.run(tryStuff())
    #asyncio.run(checkData())
    asyncio.run(main())