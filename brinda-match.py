#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from glob import glob
from datetime import date

#model = SentenceTransformer('all-MiniLM-L6')
model = SentenceTransformer("/Users/k33988/Downloads/all-MiniLM-L6-v2")

def load_csv_file(file_path):
    return pd.read_csv(file_path)

def load_excel_file(file):
    return pd.read_excel(file)

def load_previous_matches(csv_files_path):
    all_files = glob(os.path.join(csv_files_path, "*.csv"))
    filtered_files = [f for f in all_files if not os.path.basename(f).startswith("deduplicated")]

    data_frames = pd.concat((pd.read_csv(f) for f in filtered_files))
    
    return data_frames
    
def load_matches(csv_files_path):

    """
    Loads and processes match data from output CSV file currently from 2023
    
    Args: 
        csv_file_path(str): The path to the directory containing the CSV files. 
    
    Returns: 
        dict: A dictionary where the key are email addresses 
        and the values are sets of email addresses that the key has been matched with
    """
    matches_df = load_previous_matches(csv_files_path)
    
    
    #Initiates and Iterates through the rows of the combined DataFrame
    previous_matches = {}
    for index, row in matches_df.iterrows():
        person = row['Email Address']
        matched_with = row['Matched Email Address']

        if person not in previous_matches: 
            previous_matches[person] = set()
        if matched_with not in previous_matches:
            previous_matches[matched_with] = set()

        previous_matches[person].add(matched_with)
        previous_matches[matched_with].add(person)
        
    #converts emails to all lowercase to be detected as the same email (there were some that were capitalized)
    previous_matches = {key.lower():{email.lower() for email in value} for key, value in previous_matches.items()}
    return previous_matches



def generate_pool(data,freq_column,start_date):
    today = date.today()
    #number of months passed
    curr_round_num = (today.year - start_date.year)* 12 + (today.month - start_date.month)
    mapping = {'Monthly':1,'Bi-monthly':2,'Quarterly':3}

    data_pool = data.copy()
    data_pool[freq_column + '_std'] = data_pool[freq_column].map(mapping)
    #includes or excludes people
    data_pool['pool_fl'] = data_pool[freq_column + '_std'].apply(lambda x: 1 if curr_round_num % x ==0 else 0)
    data_pool = data_pool[data_pool['pool_fl'] == 1]

    return data_pool

def get_embedding(text):
    return model.encode(text)

def calculate_similarity(employee1, employee2, attribute):

    """
    Calculates the similarity score between two employees based on the three attributes given
        Args: 
            employee1 (pandas.Series): A row from a dataframe representing the first employee.
            employee2 (pandas.Series): A row from a dataframe representing the second employee.
            attribute (list): A list of column names from the input dataframe that represents the attributes to be used for calculating similarity.
            
        Returns: 
            dict: A dictionary containing the similarities and matched text for each attribute.
            """
    similarities ={}
    matched_texts = {}

    for x in attribute:

        #gets the text for each attribute for the two employees
        text1 = str(employee1[x])
        text2 = str(employee2[x])

        
        embedding1 = get_embedding(text1)
        embedding2 = get_embedding(text2)
    
        #avoidng index error by reshapping embeddings
        embedding1 = np.array(embedding1).reshape(1,-1)
        embedding2 = np.array(embedding2).reshape(1,-1)

        similarities[x] = cosine_similarity(embedding1, embedding2)[0][0]

        
        #Split the text into individual phrases and get the embeddings for them
        words1 = text1.split(",")
        words2 = text2.split(",")

        word_embedding1 = get_embedding(words1)
        word_embedding2 = get_embedding(words2)

        #This is for finding the matched texts 
        matched_words = []
        for i, word1 in enumerate(words1):
            for j, word2 in enumerate(words2):
                word_similarity = cosine_similarity(
                    word_embedding1[i].reshape(1,-1),
                    word_embedding2[j].reshape(1,-1)
                )[0][0]
                if word_similarity > 0.80:   #threshold is for finding similarity
                    matched_words.append(word1)
                    break
        matched_texts[x] = " ".join(matched_words)

    similarities['Aggregate'] = np.mean(list(similarities.values()))

    return similarities, matched_texts


#FIX: need to add managers
def find_best_match(employee, already_matched, all_people, attribute, previous_matches):

    """
    Finds the best match for a given employee among the unmatched people. 
    
    Args: 
        employee (pandas.Series): A row from the monthly input dataframe representing the employees to find a match for
        already_matched (set): A set of emails representing the people that have already been paired (helps with unique pairs)
        all_people (pandas.DataFrame): The dataframe containing information about all employees, AKA monthly input dataframe
        attribute (list): A list of column names from the input dataframe that represents the three columns of texts employees wil be matched on
        previous_matches (dict): A dictionary where keys are emails representing employees and values are everyone they have previously matched with

    Returns: 
        tuple: A tuple containing the BEST MATCH and a dictionary with the best similarities and matched texts.
        """
    
    person_email = employee['Email Address'].lower()
    #Initialize the variables to store the best match and similarity
    best_match = None
    best_similarity = -1
    best_similarities = None
    best_matched_texts = None

    #Adds newcomers to previous matches
    if person_email not in previous_matches:
        previous_matches[person_email] = set()

    previous_matched_people = previous_matches[employee['Email Address'].lower() ]
    
    #Using the already_matched set that removes people as they are getting matched, remove previous matches
    unmatched_people = already_matched - previous_matched_people
    unmatched_people = all_people[all_people['Email Address'].isin(unmatched_people)]

    #where I am trying to make sure they don't get matched with managers
    #itierate through all possible contenders
    for _, candidate in unmatched_people.iterrows():
        candidate_email = candidate['Email Address'].lower()

        #skip if it is themself
        if person_email == candidate_email:
            continue
        
    
        similarities, matched_texts = calculate_similarity(employee, candidate, attribute)

        #Update the best match and similarity if teh current candidate i better
        if similarities['Aggregate'] > best_similarity: #compares overall similarity score aganist best seen so far
            best_similarity = similarities['Aggregate'] # new best similarity score (keeps changing until finding the best)
            best_match = candidate 
            
            best_similarities = similarities  #similarities dictionary of current candidate
            best_matched_texts = matched_texts
    return best_match,{'similarities': best_similarities, 'matched_texts' : best_matched_texts }
    
def create_pairs(data_pool, attribute, previous_matches):
    """
    Creates a dataframe of matched pair
    
    Args: 
        data (pandas.DataFrame): The dataframe containing information about all employees, AKA monthly input dataframe
        attribute (list): A list of column names from the input dataframe that represents the three columns of texts employees wil be matched on
        previous_matches (dict): A dictionary where keys are emails representing employees and values are everyone they have previously matched with
        
    Returns:
        pandas.DataFrame: The Dataframe containing the matched paira and associating information.
        """
    #Made a copy to avoid modifying the original 
    all_people = data_pool.copy()
    all_people = all_people.fillna('')
    all_people = all_people[all_people['Active Flag'].str.lower() == 'active']

    pairs = []
    already_matched = set(all_people['Email Address']) #Initializes a list of people who haven't been matched yet

    #Coninues the loop until there is one or zero left
    while len(already_matched) >= 2:
        #Get the next unmatched employee
        employee = all_people[all_people['Email Address'].isin(already_matched)].iloc[0]
        employee_email = employee['Email Address']

        if employee_email not in previous_matches:
            previous_matches[employee_email] = set()

        #Find the best match
        best_match, similarities = find_best_match(employee, already_matched, all_people, attribute, previous_matches)

        if best_match is not None: #forces to find a match 
            best_match_email = best_match['Email Address']
        
            pairs.append({
                'Name': employee['Name'],
                'Email Address': employee_email,
                'Matched Name': best_match['Name'],
                'Matched Email Address': best_match_email,
                'Matched Text Aggregate': similarities['matched_texts'],
                'Matched Score Aggregate': similarities['similarities']['Aggregate'],
                
                #did ** instead of a for loop
                **{f'Matched Text {x}':similarities['matched_texts'][x] for x in attribute},
                **{f'Matched Score {col}': similarities['similarities'][col] for col in attribute} 
                })
            # Remove the employee and best match from the already_matched set
            already_matched.remove(employee_email)
            already_matched.remove(best_match_email)

        else: #odd number
            pairs.append({
                'Name': employee['Name'],
                'Email Address': employee_email,
                'Matched Name': "",
                'Matched Email Address': "",
                'Matched Text Aggregate': "",
                'Matched Score Aggregate': "",
                
                #did ** instead of a for loop
                **{f'Matched Text {x}':" " for x in attribute},
                **{f'Matched Score {col}': " " for col in attribute} 
                })
            # Remove the employee and best match from the already_matched set
            already_matched.remove(employee_email)
    
    return pd.DataFrame(pairs)

def main():
    input_file = './input/2024-07-01-profiles.xlsx'
    previous_pairs_file = './output'
    #managers_file = '/Users/k33988/Documents/summer-coffee-hour/exclude-managers/manager-email.xlsx'

    data = load_excel_file(input_file)
    #managers = load_excel_file(managers_file)
    previous_matches = load_matches(previous_pairs_file)

    

    attribute = ['Professional Interests', 'Hobbies', 'Topics you would like to learn more about']
    start_date = date(2024, 7, 1)

   
    data_pool = generate_pool(data, 'Match Frequency', start_date)
    results = create_pairs(data_pool, attribute, previous_matches)


    output_order = ['Name', 'Email Address', 'Matched Name', 'Matched Email Address',
                    'Matched Text Aggregate',
                    'Matched Score Aggregate',
                    'Matched Text Professional Interests',
                    'Matched Score Professional Interests',
                    'Matched Text Hobbies', 
                    'Matched Score Hobbies',
                    'Matched Text Topics you would like to learn more about',
                    'Matched Score Topics you would like to learn more about']
    results = results[output_order]
    results.to_csv('output_matches-august.csv', index = False)

    

if __name__ == "__main__":
    main()

# input_file = '/Users/k33988/Documents/summer-coffee-hour/input/2024-07-01-profiles.xlsx'
# data = load_excel_file(input_file)
# employee = data.iloc[0]
# print('DEBUG')
# print(employee['Email Address'])

# previous_pairs_file = '/Users/k33988/Documents/summer-coffee-hour/output'
# previous_matches = load_matches(previous_pairs_file)
# ava_people = set(previous_matches.keys())
# matched_people = previous_matches[employee['Email Address'].lower()]
# print(employee)
# print(matched_people)
# print(data['Name'])
# unmatched_people = data[~data['Email Address'].isin(matched_people)]
# print(unmatched_people)
# person_email = employee['Email Address'].lower()

# managers_file = '/Users/k33988/Documents/summer-coffee-hour/exclude-managers/manager-email.xlsx'
# managers = load_excel_file(managers_file)
# for _,candidate in unmatched_people.iterrows():
#         if candidate == employee: