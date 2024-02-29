import pandas as pd

def classify_relation(distance, position, interruption):
    
    # # load excel file in dataframe
    # df = pd.read_excel("results.xlsx")
    
    # # get the values from the dataframe
    # distance = df.at[1, "Distance"].astype(float)
    # position = df.at[1, "Position"]
    # interruption = df.at[1, "Interruption"]
    
    # If else statements for classification
    
    if distance == 0.0 and interruption == "Negative" and position == "None":
        relation = "Class 0"
        
    elif position == "Buccal" or position == "Apical" and interruption == "Negative":
        if distance >= 2.0:
            relation = "Class 1A"
        else:
            relation = "Class 1B"
    
    elif position == "Lingual" and interruption == "Negative":
        if distance >= 2.0:
            relation = "Class 2A"
        else:
            relation = "Class 2B"
    
    elif position == "Buccal" or position == "Apical":
       if distance == 0.0 and interruption == "Negative":
           relation = "Class 3A"
       elif distance == 0.0 and interruption == "Positive":
           relation = "Class 3B"
       else:
            relation = "Unclassified Relation"
           
    elif position == "Lingual":
        if distance == 0.0 and interruption == "Negative":
            relation = "Class 4A"
        elif distance == 0.0 and interruption == "Positive":
            relation = "Class 4B"
        else:
            relation = "Unclassified Relation"
    
    else:
        relation = "Unclassified Relation"
    
    print(f"Relation: {relation}")
    
    return relation

def classify_risk(relation):
    
    # load excel file in dataframe
    #TODO: Change to fetch to db
    # df = pd.read_excel("results.xlsx")
    
    # # get the values from the dataframe
    # relation = df.at[1, "Relation"]
    
    # If else statements for classification
    if relation == "Class 0":
        risk = "N.0 (Non-determinant)"
    
    elif relation == "Class 1A" or relation == "Class 1B" or relation == "Class 2A" or relation == "Class 2B" or relation == "Class 4A":
        risk = "N.1 (Low)"
    
    elif relation == "Class 3A" or relation == "Class 3B":
        risk = "N.2 (Medium)"
    
    elif relation == "Class 4B":
        risk = "N.3 (High)"
    
    else:
        risk = "Unclassified Risk"
    
    print(f"Risk: {risk}")
    
    return risk
