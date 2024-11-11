import random
import csv
import numpy as np
import pycountry
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pandas as pd
import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from plotly.offline import plot, iplot, init_notebook_mode
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from classes import *



def feature_engineering(df):
    df_result_1 = df[df['result'].isin([-1, 1])]
    df_result_0 = df[df['result'] == 0]

    # Group by uid and calculate average and standard deviation of year_released
    grouped_year_released_1 = df_result_1.groupby('uid')['year_released'].agg(['mean', 'std']).reset_index()
    grouped_year_released_0 = df_result_0.groupby('uid')['year_released'].agg(['mean', 'std']).reset_index()

    grouped_min_req_age_1 = df_result_1.groupby('uid')['min_required_age'].agg(['mean', 'std']).reset_index()
    grouped_min_req_age_0 = df_result_0.groupby('uid')['min_required_age'].agg(['mean', 'std']).reset_index()

    grouped_price_1 = df_result_1.groupby('uid')['price'].agg(['mean', 'std']).reset_index()
    grouped_price_0 = df_result_0.groupby('uid')['price'].agg(['mean', 'std']).reset_index()

    grouped_rating_1 = df_result_1.groupby('uid')['rating'].agg(['mean', 'std']).reset_index()
    grouped_rating_0 = df_result_0.groupby('uid')['rating'].agg(['mean', 'std']).reset_index()


    # Rename columns
    grouped_year_released_1.columns = ['uid', 'year_released_1_avg', 'year_released_1_std']
    grouped_year_released_0.columns = ['uid', 'year_released_0_avg', 'year_released_0_std']

    grouped_min_req_age_1.columns = ['uid', 'min_req_age_1_avg', 'min_req_age_1_std']
    grouped_min_req_age_0.columns = ['uid', 'min_req_age_0_avg', 'min_req_age_0_std']

    grouped_price_1.columns = ['uid', 'price_1_avg', 'price_1_std']
    grouped_price_0.columns = ['uid', 'price_0_avg', 'price_0_std']

    grouped_rating_1.columns = ['uid', 'rating_1_avg', 'rating_1_std']
    grouped_rating_0.columns = ['uid', 'rating_0_avg', 'rating_0_std']

    result_df = pd.merge(grouped_year_released_1, grouped_year_released_0, on='uid', how='outer')
    result_df = pd.merge(result_df, grouped_min_req_age_1, on='uid', how='outer')
    result_df = pd.merge(result_df, grouped_min_req_age_0, on='uid', how='outer')
    result_df = pd.merge(result_df, grouped_price_1, on='uid', how='outer')
    result_df = pd.merge(result_df, grouped_price_0, on='uid', how='outer')
    result_df = pd.merge(result_df, grouped_rating_1, on='uid', how='outer')
    result_df = pd.merge(result_df, grouped_rating_0, on='uid', how='outer')
    
    unique_categories = df['category'].unique()

    category_dataframes = {}

    for category in unique_categories:

        category_df = df[df['category'] == category].copy()  # Make a copy of the slice to avoid the warning
        category_df.loc[category_df['result'] == -1, 'result'] = 1
        category_counts = category_df.groupby('uid')['result'].value_counts().unstack(fill_value=-1)
        category_counts.columns = [f'result_{category}_0_count', f'result_{category}_1_count']
        category_counts.reset_index(inplace=True)
        category_dataframes[category] = category_counts
        
    result_df = pd.merge(result_df, category_dataframes['Action'], on='uid', how='left')
    result_df = pd.merge(result_df, category_dataframes['Adventure'], on='uid', how='left')
    result_df = pd.merge(result_df, category_dataframes['Sport'], on='uid', how='left')
    result_df = pd.merge(result_df, category_dataframes['Strategy'], on='uid', how='left')
    result_df = pd.merge(result_df, category_dataframes['Simulation'], on='uid', how='left')
    
    unique_has_offer_values = df['has_offer'].unique()


    has_offer_dataframes = {}

    for has_offer_value in unique_has_offer_values:
        has_offer_df = df[df['has_offer'] == has_offer_value].copy()  # Make a copy of the slice to avoid the warning
        has_offer_df.loc[has_offer_df['result'] == -1, 'result'] = 1
        has_offer_counts = has_offer_df.groupby('uid')['result'].value_counts().unstack(fill_value=-1)
        has_offer_counts.columns = [f'result_has_offer_{has_offer_value}_0_count', f'result_has_offer_{has_offer_value}_1_count']
        has_offer_counts.reset_index(inplace=True)
        has_offer_dataframes[has_offer_value] = has_offer_counts
        
    result_df = pd.merge(result_df, has_offer_dataframes[True], on='uid', how='left')
    result_df = pd.merge(result_df, has_offer_dataframes[False], on='uid', how='left')

    unique_min_system_req_values = df['min_system_req'].unique()

    min_system_req_dataframes = {}

    for min_system_req_value in unique_min_system_req_values:
        min_system_req_df = df[df['min_system_req'] == min_system_req_value].copy()  # Make a copy of the slice to avoid the warning
        min_system_req_df.loc[min_system_req_df['result'] == -1, 'result'] = 1
        min_system_req_counts = min_system_req_df.groupby('uid')['result'].value_counts().unstack(fill_value=-1)
        min_system_req_counts.columns = [f'result_min_system_req_{min_system_req_value}_0_count', f'result_min_system_req_{min_system_req_value}_1_count']
        min_system_req_counts.reset_index(inplace=True)
        min_system_req_dataframes[min_system_req_value] = min_system_req_counts
        
    result_df = pd.merge(result_df, min_system_req_dataframes['p'], on='uid', how='left')
    result_df = pd.merge(result_df, min_system_req_dataframes['pp'], on='uid', how='left')
    result_df = pd.merge(result_df, min_system_req_dataframes['ppp'], on='uid', how='left')
    result_df = pd.merge(result_df, min_system_req_dataframes['pppp'], on='uid', how='left')
    
    result_df = pd.merge(result_df,df[['uid','gender','age','country','platform_user','cluster']].drop_duplicates(subset=['uid']), on='uid', how='left')
    result_df_enc = result_df.copy()

    from sklearn.preprocessing import LabelEncoder

    # Initialize LabelEncoder
    encoder = LabelEncoder()
    result_df_enc['uid'] = encoder.fit_transform(result_df_enc['uid'])
    result_df_enc['gender'] = encoder.fit_transform(result_df_enc['gender'])
    result_df_enc['country'] = encoder.fit_transform(result_df_enc['country'])
    result_df_enc['platform_user'] = encoder.fit_transform(result_df_enc['platform_user'])

    # Assuming df is your DataFrame
    result_df_enc.fillna(-1, inplace=True)
    result_df_enc_no_cluster = result_df_enc.drop(columns=['cluster'])
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    result_df_enc_norm = pd.DataFrame(scaler.fit_transform(result_df_enc_no_cluster), columns=result_df_enc_no_cluster.columns)
    return result_df_enc,result_df_enc_norm,result_df


import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def feature_engineering_result_1(df):
    df = df[df['result'] ==1]
    df_result_1 = df[df['result'].isin([-1, 1])]

    # Group by uid and calculate average and standard deviation of year_released
    grouped_year_released_1 = df_result_1.groupby('uid')['year_released'].agg(['mean', 'std']).reset_index()
    # grouped_year_released_0 = df_result_0.groupby('uid')['year_released'].agg(['mean', 'std']).reset_index()

    grouped_min_req_age_1 = df_result_1.groupby('uid')['min_required_age'].agg(['mean', 'std']).reset_index()
    # grouped_min_req_age_0 = df_result_0.groupby('uid')['min_required_age'].agg(['mean', 'std']).reset_index()

    grouped_price_1 = df_result_1.groupby('uid')['price'].agg(['mean', 'std']).reset_index()
    # grouped_price_0 = df_result_0.groupby('uid')['price'].agg(['mean', 'std']).reset_index()

    grouped_rating_1 = df_result_1.groupby('uid')['rating'].agg(['mean', 'std']).reset_index()
    # grouped_rating_0 = df_result_0.groupby('uid')['rating'].agg(['mean', 'std']).reset_index()

    # Rename columns
    grouped_year_released_1.columns = ['uid', 'year_released_1_avg', 'year_released_1_std']
    # grouped_year_released_0.columns = ['uid', 'year_released_0_avg', 'year_released_0_std']

    grouped_min_req_age_1.columns = ['uid', 'min_req_age_1_avg', 'min_req_age_1_std']
    # grouped_min_req_age_0.columns = ['uid', 'min_req_age_0_avg', 'min_req_age_0_std']

    grouped_price_1.columns = ['uid', 'price_1_avg', 'price_1_std']
    # grouped_price_0.columns = ['uid', 'price_0_avg', 'price_0_std']

    grouped_rating_1.columns = ['uid', 'rating_1_avg', 'rating_1_std']
    # grouped_rating_0.columns = ['uid', 'rating_0_avg', 'rating_0_std']

    # Merge the two dataframes
    result_df = pd.merge(grouped_year_released_1, grouped_min_req_age_1, on='uid', how='outer')
    # result_df = pd.merge(result_df, grouped_min_req_age_1, on='uid', how='outer')
    # result_df = pd.merge(result_df, grouped_min_req_age_0, on='uid', how='outer')
    result_df = pd.merge(result_df, grouped_price_1, on='uid', how='outer')
    # result_df = pd.merge(result_df, grouped_price_0, on='uid', how='outer')
    result_df = pd.merge(result_df, grouped_rating_1, on='uid', how='outer')
    # result_df = pd.merge(result_df, grouped_rating_0, on='uid', how='outer')
    
    unique_categories = df['category'].unique()
    category_dataframes = {}

    for category in unique_categories:
        category_df = df[df['category'] == category].copy()  # Make a copy of the slice to avoid the warning
        category_df.loc[category_df['result'] == -1, 'result'] = 1
        category_counts = category_df.groupby('uid')['result'].value_counts().unstack(fill_value=-1)
        category_counts.columns = [f'result_{category}_1_count']
        category_counts.reset_index(inplace=True)
        category_dataframes[category] = category_counts
        
    result_df = pd.merge(result_df, category_dataframes['Action'], on='uid', how='left')
    result_df = pd.merge(result_df, category_dataframes['Adventure'], on='uid', how='left')
    result_df = pd.merge(result_df, category_dataframes['Sport'], on='uid', how='left')
    result_df = pd.merge(result_df, category_dataframes['Strategy'], on='uid', how='left')
    result_df = pd.merge(result_df, category_dataframes['Simulation'], on='uid', how='left')
    
    unique_has_offer_values = df['has_offer'].unique()
    has_offer_dataframes = {}

    for has_offer_value in unique_has_offer_values:
        has_offer_df = df[df['has_offer'] == has_offer_value].copy()  # Make a copy of the slice to avoid the warning
        has_offer_df.loc[has_offer_df['result'] == -1, 'result'] = 1
        has_offer_counts = has_offer_df.groupby('uid')['result'].value_counts().unstack(fill_value=-1)
        has_offer_counts.columns = [f'result_has_offer_{has_offer_value}_1_count']
        has_offer_counts.reset_index(inplace=True)
        has_offer_dataframes[has_offer_value] = has_offer_counts
        
    result_df = pd.merge(result_df, has_offer_dataframes[True], on='uid', how='left')
    result_df = pd.merge(result_df, has_offer_dataframes[False], on='uid', how='left')

    unique_min_system_req_values = df['min_system_req'].unique()
    min_system_req_dataframes = {}

    for min_system_req_value in unique_min_system_req_values:
        min_system_req_df = df[df['min_system_req'] == min_system_req_value].copy()  # Make a copy of the slice to avoid the warning
        min_system_req_df.loc[min_system_req_df['result'] == -1, 'result'] = 1
        min_system_req_counts = min_system_req_df.groupby('uid')['result'].value_counts().unstack(fill_value=-1)
        min_system_req_counts.columns = [f'result_min_system_req_{min_system_req_value}_1_count']
        min_system_req_counts.reset_index(inplace=True)
        min_system_req_dataframes[min_system_req_value] = min_system_req_counts
        
    result_df = pd.merge(result_df, min_system_req_dataframes['p'], on='uid', how='left')
    result_df = pd.merge(result_df, min_system_req_dataframes['pp'], on='uid', how='left')
    result_df = pd.merge(result_df, min_system_req_dataframes['ppp'], on='uid', how='left')
    result_df = pd.merge(result_df, min_system_req_dataframes['pppp'], on='uid', how='left')
    
    result_df = pd.merge(result_df,df[['uid','gender','age','country','platform_user','cluster']].drop_duplicates(subset=['uid']), on='uid', how='left')
    result_df_enc = result_df.copy()

    encoder = LabelEncoder()
    result_df_enc['uid'] = encoder.fit_transform(result_df_enc['uid'])
    result_df_enc['gender'] = encoder.fit_transform(result_df_enc['gender'])
    result_df_enc['country'] = encoder.fit_transform(result_df_enc['country'])
    result_df_enc['platform_user'] = encoder.fit_transform(result_df_enc['platform_user'])

    result_df_enc.fillna(-1, inplace=True)
    result_df_enc_no_cluster = result_df_enc.drop(columns=['cluster'])

    scaler = MinMaxScaler()
    result_df_enc_norm = pd.DataFrame(scaler.fit_transform(result_df_enc_no_cluster), columns=result_df_enc_no_cluster.columns)

    return result_df_enc, result_df_enc_norm, result_df


def confusion_matrix(result_df,clusters):
    result_df['cluster_gen'] = clusters

    cluster_lists = {}
    for cluster_id in result_df['cluster'].unique():
        cluster_data = result_df[result_df['cluster'] == cluster_id]
        uids = cluster_data['uid'].tolist()
        cluster_lists[f'list{cluster_id}'] = uids
    list1 = cluster_lists['list1']
    list2 = cluster_lists['list2']
    list3 = cluster_lists['list3']
    list4 = cluster_lists['list4']
    cluster_lists = {}
    for cluster_id in result_df['cluster_gen'].unique():
        cluster_data = result_df[result_df['cluster_gen'] == cluster_id]
        uids = cluster_data['uid'].tolist()
        cluster_lists[f'list{cluster_id}'] = uids
    list1_gen = cluster_lists['list0']
    list2_gen = cluster_lists['list1']
    list3_gen = cluster_lists['list2']
    list4_gen = cluster_lists['list3']


    def count_cluster_labels_in_list(uid_list, cluster_label):
        cluster_count = 0
        for uid in uid_list:
            uid_cluster_label = uid[-2:]
            if uid_cluster_label == cluster_label:
                cluster_count += 1

        return cluster_count
    cluster_counts_per_list = {'list1': {}, 'list2': {}, 'list3': {}, 'list4': {}}

    for list_name, uid_list in zip(cluster_counts_per_list.keys(), [list1, list2, list3, list4]):
        for label in ['c1', 'c2', 'c3', 'c4']:
            count = count_cluster_labels_in_list(uid_list, label)
            cluster_counts_per_list[list_name][label] = count

    # for list_name, counts in cluster_counts_per_list.items():
    #     print(f"Counts for {list_name}:")
    #     for label, count in counts.items():
    #         print(f"{label}: {count}")

    
    cluster_counts_per_list = {'list1_gen': {}, 'list2_gen': {}, 'list3_gen': {}, 'list4_gen': {}}

    for list_name, uid_list in zip(cluster_counts_per_list.keys(), [list1_gen, list2_gen, list3_gen, list4_gen]):
        for label in ['c1', 'c2', 'c3', 'c4']:
            count = count_cluster_labels_in_list(uid_list, label)
            cluster_counts_per_list[list_name][label] = count
    for list_name, counts in cluster_counts_per_list.items():
        print(f"Counts for {list_name}:")
        for label, count in counts.items():
            print(f"{label}: {count}")
            
            
            
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = ['c1', 'c2', 'c3', 'c4']
    conf_matrix = np.array([[cluster_counts_per_list[list_name][label] for label in labels] for list_name in cluster_counts_per_list.keys()])
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=cluster_counts_per_list.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Cluster Label')
    plt.ylabel('List')
    plt.show()
    return 

#FUNCTIONS FOR THE GENERATION OF THE DATA

def truncated_gaussian(mu, sigma, min_val, max_val,seed=42):
    while True:
        value = random.gauss(mu, sigma)
        if min_val <= value <= max_val:
            return round(value, 1)
            
def countries_gen():
    countries = list(pycountry.countries)
    country_names = [country.name for country in countries]
    return country_names

def truncated_gaussian_age(mu, sigma, min_val, max_val,seed=42):
    while True:
        value = random.gauss(mu, sigma)
        if min_val <= value <= max_val:
            return int(value)

def score_beta_per_year(year_released, alpha, beta, current_year=2023,seed=42):
    """
    Calculate the score of a game based on its release year using a beta distribution.
    """
    np.random.seed(42)
    age = max(0, current_year - year_released)  # Ensure age is non-negative
    score = np.random.beta(alpha, beta) * (1 - age / (current_year - 2000))
    return max(0, min(1, score))

def generate_price_exp(min_price, max_price, scale,seed=42):
    """
    Generate a single random price from an exponential distribution within the specified range.
    
    Parameters:
        min_price (float): The minimum price.
        max_price (float): The maximum price.
        scale (float): The scale parameter of the exponential distribution.
    
    Returns:
        float: A randomly generated price within the specified range.
    """
    while True:
        new_price = np.random.exponential(scale)
        if min_price <= new_price <= max_price:
            return new_price

def price_distribution(size,seed=42):
    # Probability of getting 60
    prob_60 = 0.15
    # Probability of getting 0
    prob_0 = 0.15
    # Probability of getting any other number
    prob_other = 1 - prob_60 - prob_0
    
    # Generate random numbers based on probabilities
    price = np.random.choice([60, 0, truncated_gaussian(30, 30, 1, 59)], size=size, p=[prob_60, prob_0, prob_other])
    
    return price

def exponential_distribution(x,seed=42):
    return np.exp((2*x - 2023) / 10)

def generate_games(games_num: int = 100, seed: int = 42,
                   min_system_req_p: float=0.1,
                   min_system_req_pp: float=0.3,
                   min_system_req_ppp: float=0.4,
                   min_system_req_pppp: float=0.2,
                   rating_mean: float = 8,
                   rating_std: float = 3,
                   min_required_age_10_prob: float = 0.3,
                   min_required_age_15_prob: float = 0.3,
                   min_required_age_18_prob: float = 0.4,
                   macos_prob: float = 0.2,
                   linux_prob: float = 0.1
                  ):
    games = []

    for i in range(games_num):
        gameid = i
        
        category = random.sample(['Adventure', 'Action', 'Sport', 'Simulation', 'Strategy'], 1)[0]
        
        if category == "Adventure":
            min_system_req = random.choice(["ppp", "pppp"])
        else:
            com_power_sample = np.random.multinomial(1, [min_system_req_p, min_system_req_pp, min_system_req_ppp, min_system_req_pppp]).argmax()
            if com_power_sample == 0:
                min_system_req = "p"
            elif com_power_sample == 1:
                min_system_req = random.choice(["pp"])
            elif com_power_sample == 2:
                min_system_req = random.choice(["ppp"])
            else:
                min_system_req = random.choice(["pppp"])
        
        has_offer = random.choice([True, False])
        
        # year_released = random.randint(2000, 2023)
        year_released = [random.choices(range(2000, 2024), weights=[exponential_distribution(year) for year in range(2000, 2024)])[0] for _ in range(1)][0]
        
        rating = truncated_gaussian(rating_mean, rating_std, 0, 10)
        
        min_required_age = random.choices([10, 15, 18], weights=[min_required_age_10_prob, min_required_age_15_prob, min_required_age_18_prob])[0]
        
        # platform_num = 3
        # platform = random.sample(['Windows', 'MacOS', 'Linux'], platform_num)
        platform = ['Windows']
        remaining_platforms = ['MacOS', 'Linux']
        
        if random.random() < macos_prob:#if its going to be the game available for that os
            platform.append('MacOS')
        if random.random() < linux_prob:
            platform.append('Linux')

        
        price = price_distribution(2)[0]
        
        games.append(Games(gameid,year_released, rating, category, min_required_age, price, platform, has_offer, min_system_req))

    return games

def generate_users_segment1(user_num:int=10000, seed: int = 42,
                           age_mean: float = 16,
                           age_std: float=4,
                           user_prob_windows: float = 0.6,
                           user_prob_macos: float = 0.6,
                           user_prob_linux: float = 0.6,
                           ):
    """
    SEGMENT 1: Poor Gamer
    --------------------------------
    * Age: Gaussian distro with a mean of 16
    * price < 30
    * min_system_req <= (p,pp)
    * HAS_OFFER: 90% influence
    * 90% consistent
    """
    
    users=[]
    
    for i in range(user_num):
        
        uid = str(i) + "c1"
        
        gender=random.choice(["M", "F"]) # pick a random gender 
        
        country = random.choice(countries_gen())
    
        age = truncated_gaussian_age(16, 4, 10, 23)

        platform = random.choices(['Windows', 'MacOS', 'Linux'], weights=[user_prob_windows, user_prob_macos,user_prob_linux], k=1)[0]
        
        cluster = 0
        
        users.append(User(uid,gender,age,country,platform,cluster))
        
    return users



def generate_ratings_segment1(users:list, games:list,seed = 42,
                             max_price: float = 30,
                             has_offer_influ: float = 0.9,
                             consistancy: float = 0.9):
    """
    SEGMENT 1: Poor Gamer
    --------------------------------
    * Age: Gaussian distro with a mean of 16,ok
    * price < 30,ok
    * min_system_req <= (p,pp),ok
    * HAS_OFFER: 90% influence,ok
    * 90% consistent,ok
    """

    with open('segment1.csv', 'w', newline='', encoding='utf-8') as fw:
        writer = csv.writer(fw)
        
        writer.writerow(['uid','gender','age','country','platform_user','cluster','year_released','category',
                         'min_required_age','price','platform_game',
                          'has_offer','min_system_req','rating','result','gameid'])
        
        cnt = 0
        total = 0
        for usr in users: # for each user
            
            rating_num = random.randint(1, len(games)) # get the number of ratings
            total += rating_num
            my_games = random.sample(games, rating_num) # sample games to be rated
        
            for game in my_games: # for each game
                
                result = 0 # initialize to negative rating
                cluster = 1
                
                if usr.age >= game.min_required_age: 
                    if usr.platform_user in game.platform_game:
                        # if the game meets the criteria
                        if game.min_system_req in ['p','pp']:
                            
                            if game.price < max_price: 
                                # check if it's an offer
                                if random.random() or (game.has_offer and random.random() <= has_offer_influ):
                                    result = 1
                                    cnt += 1
                         
                if random.random() < (1 - consistancy): # consistency switch
                    result *= -1
                    
                new_row = [usr.uid,usr.gender, usr.age, usr.country, usr.platform_user,cluster, game.year_released, game.category,
                           game.min_required_age, game.price, game.platform_game,
                           game.has_offer, game.min_system_req,game.rating, result,game.gameid]

                
                writer.writerow(new_row)
                
        print('ones', cnt / total)





def generate_users_segment2(user_num:int=10000, seed: int = 42,
                           age_mean: float = 35,
                           age_std: float=5,
                           user_prob_windows: float = 1,
                           user_prob_macos: float = 0,
                           user_prob_linux: float = 0,
                           ):
    """
    SEGMENT 2: Epic Gamer - graphics lover
    --------------------------------
    * Age: Gaussian distro with a mean of 35
    * rating >9 if minimum system requirements (ppp)
    * rating >8 if minimum system requirements (pppp)
    * platform must be Windows
    * 95% yes if its Adventure 
    * prefers the latest releases (beta distribution)
    * 85% consistent
    """
    
    users=[]
    
    for i in range(user_num):
        
        uid = str(i) + "c2"
        
        gender=random.choice(["M", "F"]) # pick a random gender 
        
        country = random.choice(countries_gen())
    
        age = truncated_gaussian_age(age_mean, age_std, 20, 55)

        platform = random.choices(['Windows', 'MacOS', 'Linux'], weights=[user_prob_windows, user_prob_macos, user_prob_linux], k=1)[0]
        
        cluster = 0
        
        users.append(User(uid,gender,age,country,platform,cluster))
        
    return users

def generate_ratings_segment2(users:list, games:list,seed = 42,
                             adv_categ_cons: float = 0.60,
                             min_game_rate_ppp: float = 9,
                             min_game_rate_pppp: float = 8,
                             consistency: float = 0.95
                             ):
    """
    SEGMENT 2: Epic Gamer - graphics lover
    --------------------------------
    * Age: Gaussian distro with a mean of 35, ok
    * rating >9 if minimum system requirements (ppp),ok
    * rating >8 if minimum system requirements (pppp),ok
    * platform must be Windows,ok
    * 60% yes if its Adventure and minimum system requirements (pppp),ok
    * prefers the latest releases (beta distribution),ok
    * 85% consistent , ok
    """

    with open('segment2.csv', 'w', newline='', encoding='utf-8') as fw:
        writer = csv.writer(fw)
        
        writer.writerow(['uid','gender','age','country','platform_user','cluster','year_released','category',
                         'min_required_age','price','platform_game',
                          'has_offer','min_system_req','rating','result','gameid'])
        
        cnt = 0
        total = 0
        for usr in users: # for each user
            
            rating_num = random.randint(1, len(games)) # get the number of ratings
            total += rating_num
            my_games = random.sample(games, rating_num) # sample games to be rated
        
            for game in my_games: # for each game
                
                result = 0 # initialize to negative rating
                cluster = 2
                if usr.age >= game.min_required_age: 
                    if game.category == 'Adventure' and game.min_system_req in ('ppp','pppp') and random.random() <= adv_categ_cons:
                        result = 1
                        cnt+=1
                    
                        if result ==0 and game.category in ('Action','Strategy'):
                        
                            if game.platform == 'Windows':
                        
                                if (game.rating>=min_game_rate_ppp and game.min_system_req == 'ppp') or (game.rating>=min_game_rate_pppp and game.min_system_req == 'pppp'):
                    
                                    if random.random() <= game_score_beta(game.year_released, 100, 1):
                                        result = 1
                                        cnt+=1
                
                
                if random.random() < (1 - consistency): # consistency switch
                    result *= -1
                
                
                new_row = [usr.uid,usr.gender, usr.age, usr.country, usr.platform_user,cluster, game.year_released, game.category,
                           game.min_required_age, game.price, game.platform_game,
                           game.has_offer, game.min_system_req,game.rating, result,game.gameid]

                
                writer.writerow(new_row)
                
        print('ones', cnt / total)





def generate_users_segment3(user_num:int=10000, seed: int = 42,
                            male_poss_thresh: float = 0.85,
                           age_mean: float = 25,
                           age_std: float=5,
                           user_prob_windows: float = 0.8,
                           user_prob_macos: float = 0.1,
                           user_prob_linux: float = 0.1,
                           ):
    """
    SEGMENT 3: Action - Shooting Gamer 
    --------------------------------
    * category : Action
    * HAS_OFFER: 70% influence
    * price as cheap as possible
    * gender 95% Male
    """
    
    users=[]
    
    for i in range(user_num):
        
        uid = str(i) + "c3"
        
        #gender=random.choice(["M", "F"]) # pick a random gender 
        gender = "M" if random.random() < male_poss_thresh else "F"
        
        country = random.choice(countries_gen())
    
        age = truncated_gaussian_age(age_mean, age_std, 18, 60)

        platform = random.choices(['Windows', 'MacOS', 'Linux'], weights=[user_prob_windows, user_prob_macos, user_prob_linux], k=1)[0]
        
        cluster = 0
        
        users.append(User(uid,gender,age,country,platform,cluster))
        
    return users






def generate_ratings_segment3(users:list, games:list,seed:int = 42,
                             game_categ_must: str = 'Action',
                             has_offer_cons: float = 0.9):
    """
    SEGMENT 3: Action - Shooting Gamer 
    --------------------------------
    * category : Action,ok
    * HAS_OFFER: 70% influence,ok
    * price as cheap as possible,ok
    * gender 95% Male,ok
    """

    with open('segment3.csv', 'w', newline='', encoding='utf-8') as fw:
        writer = csv.writer(fw)
        
        writer.writerow(['uid','gender','age','country','platform_user','cluster','year_released','category',
                         'min_required_age','price','platform_game',
                          'has_offer','min_system_req','rating','result','gameid'])
        
        cnt = 0
        total = 0
        for usr in users: # for each user
            
            rating_num = random.randint(1, len(games)) # get the number of ratings
            total += rating_num
            my_games = random.sample(games, rating_num) # sample games to be rated
        
            for game in my_games: # for each game
                
                result = 0 # initialize to negative rating
                cluster = 3
                
                if usr.age >= game.min_required_age: 
                    
                    if usr.platform_user in game.platform_game:
                    
                        if game.category in (game_categ_must):
                        
                                # if random.random() or (game.has_offer and random.random() <= 0.95):
                                if (game.has_offer and random.random() <= has_offer_cons):
                                
                                     if game.price <= generate_price_exp(0, 60, 30):#distribution for as cheap as possible based on the preferences
                                
                                        result = 1
                                        cnt += 1
                                
                
                new_row = [usr.uid,usr.gender, usr.age, usr.country, usr.platform_user,cluster, game.year_released, game.category,
                           game.min_required_age, game.price, game.platform_game,
                           game.has_offer, game.min_system_req,game.rating, result,game.gameid]

                
                writer.writerow(new_row)
                
        print('ones', cnt / total)





def generate_users_segment4(user_num:int=10000, seed: int = 42,
                           age_mean: float = 50,
                           age_std: float = 5,
                           user_prob_windows: float = 0.5,
                           user_prob_macos: float = 0.35,
                           user_prob_linux: float = 0.15,
                           ):
    """
    SEGMENT 4: Casual - Free time pass Gamer 
    --------------------------------
    * Age: Gaussian distro with a mean of 50, ok
    * price = 0 
    * min_system_req (p,pp)
    * rating > 7
    * category (strategy,Simulation)
    """
    
    users=[]
    
    for i in range(user_num):
        
        uid = str(i) + "c4"
        
        gender=random.choice(["M", "F"]) # pick a random gender 
        
        country = random.choice(countries_gen())
    
        age = truncated_gaussian_age(age_mean, age_std, 10, 60)

        platform = random.choices(['Windows', 'MacOS', 'Linux'], weights=[user_prob_windows, user_prob_macos, user_prob_linux], k=1)[0]
        
        cluster = 0
        
        users.append(User(uid,gender,age,country,platform,cluster))
        
    return users

def generate_ratings_segment4(users:list, games:list,seed = 42,
                              min_game_rating: float = 5,
                              min_price_accepted: float = 0,
                              min_sys_req_acc: tuple = ('p','pp','ppp'),
                              consistency: float = 0.95
                             ):
    """
    SEGMENT 4: Casual - Free time pass Gamer 
    --------------------------------
    * Age: Gaussian distro with a mean of 50, ok
    * price = 0, ok
    * min_system_req (p,pp),ok
    * rating > 7,ok
    * category (strategy,Simulation),ok
    * 75% consistent,ok
    """

    with open('segment4.csv', 'w', newline='', encoding='utf-8') as fw:
        writer = csv.writer(fw)
        
        writer.writerow(['uid','gender','age','country','platform_user','cluster','year_released','category',
                         'min_required_age','price','platform_game',
                          'has_offer','min_system_req','rating','result','gameid'])
        
        cnt = 0
        total = 0
        for usr in users: # for each user
            
            rating_num = random.randint(1, len(games)) # get the number of ratings
            total += rating_num
            my_games = random.sample(games, rating_num) # sample games to be rated
        
            for game in my_games: # for each game
                
                result = 0 # initialize to negative rating
                cluster = 4
                #if game.category in ('Strategy','Simulation'):
                if usr.age >= game.min_required_age: 
                    if usr.platform_user in game.platform_game:    
                        if game.category in ('Strategy','Simulation'):
                            if game.rating > min_game_rating:
                    
                                if game.price <=min_price_accepted:
                            
                                    if game.min_system_req in min_sys_req_acc:
                            
                                        result = 1
                                        cnt += 1
                
                if random.random() < (1-consistency): # consistency switch
                    result *= -1
                
                
                new_row = [usr.uid,usr.gender, usr.age, usr.country, usr.platform_user,cluster, game.year_released, game.category,
                           game.min_required_age, game.price, game.platform_game,
                           game.has_offer, game.min_system_req,game.rating, result,game.gameid]

                
                writer.writerow(new_row)
                
        print('ones', cnt / total)
        
        
        
        
        
        
        
        
        
        
        
