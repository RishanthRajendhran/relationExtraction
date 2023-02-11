import logging

#FunctionName: 
#   extractRelationInstances
#Input:
#   fileName    :   Path to txt file containing relation instances
#   debugMode   :   Boolean variable to enable debug mode
#                   Default: False
#Output:
#   entities    :   Dictionary with MIDs as keys; Every MID is 
#                   associated with a dictionary with relation 
#                   names as keys and list of related entity MIDs
#                   as values
#   relations   :   Dictionary with relation names as keys and 
#                   list of related entity tuples as values
#Description:
#   This function is used to extract relation instances
#Notes:
#   fileName is assumed to be a txt file path
def extractRelationInstances(fileName, debugMode=False):
    data = []
    with open(fileName, "r") as f:
        data = f.readlines()
    
    numRepeats = 0
    entities = {}
    relations = {}
    for instance in data:
        items = [i.replace("\n","").strip() for i in instance.split("\t")]
        if len(items) != 3:
            raise RuntimeError(f"Invalid relation instance: {instance}")
        entity_1 = items[0]
        reln = items[1]
        entity_2 = items[2]
        if entity_1 not in entities.keys():
            entities[entity_1] = {}
        if reln not in relations.keys():
            relations[reln] = []
        if entity_2 not in entities.keys():
            entities[entity_2] = {}
        
        #Add entity_2 to the list of entities related to entity_1 by the relation reln
        if reln not in entities[entity_1].keys():
            entities[entity_1][reln] = []
        entities[entity_1][reln].append(entity_2)
        
        #Add this reln relation instance to the relations dictionary
        #Raise a warning if both forward and reverse relation instances are present 
        if (entity_2, entity_1) in relations[reln]:
            numRepeats +=1 
            logging.warning(f"Both forward (({entity_2}, {reln}, {entity_1})) and reverse ({instance}) instances present!")
        relations[reln].append((entity_1, entity_2))
    avgNumRelnInsts = 0
    for rel in relations.keys():
        avgNumRelnInsts += len(relations[reln])
    avgNumRelnInsts /= len(relations)
    
    if debugMode:
        logging.info(f"No. of relation instances: {len(data)}")
        logging.info(f"No. of unique entities: {len(entities)}")
        logging.info(f"No. of unique relations: {len(relations)}")
        logging.info(f"Percentage of repetitions: {round((numRepeats/len(data))*100,2)}")
        logging.info(f"Average no. of instances per relation: {round(avgNumRelnInsts,1)}")
    
    return entities, relations
        
