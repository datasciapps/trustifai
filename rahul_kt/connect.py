from neo4j import GraphDatabase
from yacman import FutureYAMLConfigManager as YAMLConfigManager
from box import Box
from dataclasses import dataclass, field

config = Box(YAMLConfigManager.from_yaml_file("config.yaml").to_dict())
URI = config.neo4j.uri
AUTH = (config.neo4j.username, config.neo4j.password)

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    print(driver.verify_connectivity())
