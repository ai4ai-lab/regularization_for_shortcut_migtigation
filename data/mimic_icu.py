import os
from scipy.stats import pearsonr

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    """Load and preprocess the initial MIMIC-IV datasets"""
    df = pd.read_csv(f"{path}/admissions.csv.gz")
    df_pat = pd.read_csv(f"{path}/patients.csv.gz")
    df_diagcode = pd.read_csv(f"{path}/diagnoses_icd.csv.gz")
    df_icu = pd.read_csv(f"{path}/icustays.csv.gz")
    return df, df_pat, df_diagcode, df_icu


def process_los(df):
    """Process length of stay data"""
    df['ADMITTIME'] = pd.to_datetime(df['admittime'])
    df['DISCHTIME'] = pd.to_datetime(df['dischtime'])
    df['LOS'] = (df['DISCHTIME'] - df['ADMITTIME']).dt.total_seconds()/86400
    df = df[df['LOS'] > 0]
    df.drop(columns=['dischtime', 'DISCHTIME', 'edregtime', 'edouttime', 'hospital_expire_flag'], inplace=True)
    df['DECEASED'] = df['deathtime'].notnull().map({True:1, False:0})
    return df


def process_diagnosis_codes(df_diagcode):
    """Process diagnosis codes and create categories"""
    # ICD-9 processing
    df_diagcode['recode'] = df_diagcode['icd_code'][df_diagcode['icd_version'] == 9]
    df_diagcode['recode'] = df_diagcode['recode'][~df_diagcode['recode'].str.contains("[a-zA-Z]").fillna(False)]
    df_diagcode['recode'].fillna(value='999', inplace=True)
    df_diagcode['recode'] = df_diagcode['recode'].str.slice(start=0, stop=3, step=1)
    df_diagcode['recode'] = df_diagcode['recode'].astype(int)

    # Define ranges and categories
    icd9_ranges = [(1, 140), (140, 240), (240, 280), (280, 290), (290, 320), (320, 390), 
                   (390, 460), (460, 520), (520, 580), (580, 630), (630, 680), (680, 710),
                   (710, 740), (740, 760), (760, 780), (780, 800), (800, 1000), (1000, 2000)]
    
    diag_dict = {0: 'infectious', 1: 'neoplasms', 2: 'endocrine', 3: 'blood',
                 4: 'mental', 5: 'nervous', 6: 'circulatory', 7: 'respiratory',
                 8: 'digestive', 9: 'genitourinary', 10: 'pregnancy', 11: 'skin', 
                 12: 'muscular', 13: 'congenital', 14: 'prenatal', 15: 'misc',
                 16: 'injury', 17: 'misc'}

    for num, cat_range in enumerate(icd9_ranges):
        df_diagcode['recode'] = np.where(df_diagcode['recode'].between(cat_range[0],cat_range[1]), 
                num, df_diagcode['recode'])
    
    df_diagcode['cat'] = df_diagcode['recode'].replace(diag_dict)
    return df_diagcode


def process_icu_data(df_icu):
    """Process ICU data"""
    df_icu['first_careunit'].replace({
        'Coronary Care Unit (CCU)': 'ICU',
        'Neuro Stepdown': 'NICU',
        'Neuro Intermediate': 'NICU',
        'Cardiac Vascular Intensive Care Unit (CVICU)': "ICU",
        'Neuro Surgical Intensive Care Unit (Neuro SICU)': 'ICU',
        'Medical/Surgical Intensive Care Unit (MICU/SICU)': 'ICU',
        'Medical Intensive Care Unit (MICU)': 'ICU',
        'Surgical Intensive Care Unit (SICU)': 'ICU',
        'Trauma SICU (TSICU)': 'ICU'
    }, inplace=True)
    return df_icu


def create_feature_matrix(df, df_diagcode, df_icu):
    """Create feature matrix from processed data"""
    # Process diagnoses
    hadm_list = df_diagcode.groupby('hadm_id')['cat'].apply(list).reset_index()
    hadm_item = pd.get_dummies(hadm_list['cat'].apply(pd.Series).stack())
    hadm_item = hadm_item.groupby(level=0).sum()
    hadm_item = hadm_item.join(hadm_list['hadm_id'], how="outer")
    df = df.merge(hadm_item, how='inner', on='hadm_id')

    # Process ICU data
    df_icu['cat'] = df_icu['first_careunit']
    icu_list = df_icu.groupby('hadm_id')['cat'].apply(list).reset_index()
    icu_item = pd.get_dummies(icu_list['cat'].apply(pd.Series).stack())
    icu_item = icu_item.groupby(level=0).sum()
    icu_item[icu_item >= 1] = 1
    icu_item = icu_item.join(icu_list['hadm_id'], how="outer")
    df = df.merge(icu_item, how='outer', on='hadm_id')

    df['ICU'].fillna(value=0, inplace=True)
    df['NICU'].fillna(value=0, inplace=True)
    return df


def generate_shortcut_features(df_final, shortcut_dim=10):
    """Generate shortcut features"""
    torch.manual_seed(42)
    fc_nn = nn.Linear(1, shortcut_dim)
    fc_nn.weight.data.normal_(1, 0.5)
    fc_nn.bias.data.normal_(0, 0.5)

    shortcut_features = fc_nn(torch.tensor(df_final['LOS'].values, dtype=torch.float).reshape(-1, 1))
    shortcut_features = pd.DataFrame(shortcut_features.detach().numpy())
    shortcut_features.columns = ['shortcut_' + str(i) for i in range(shortcut_dim)]

    np.random.seed(0)
    for i in range(shortcut_dim):
        shortcut_features['shortcut_' + str(i)] += np.random.normal(0, 1, len(df_final))

    return shortcut_features


def save_processed_data(root_path, data_final, train_data_scaled, test_data_scaled):
    """Save processed data to CSV files if they don't exist"""
    if os.path.exists('./mimic_icu/train_data_scaled.csv') and os.path.exists('./mimic_icu/test_data_scaled.csv') and os.path.exists('./mimic_icu/los_prediction_all.csv'):
        print("Data files already exist")
        return False
    else:
        data_final.to_csv(f'{root_path}/los_prediction_all.csv', index=False)
        train_data_scaled.to_csv(f'{root_path}/train_data_scaled.csv', index=False)
        test_data_scaled.to_csv(f'{root_path}/test_data_scaled.csv', index=False)
        print("Data files saved successfully")
        return True
    

def load_preprocessed_data(root_path):
    """Load preprocessed data if it exists"""
    all_path = f'{root_path}/los_prediction_all.csv'
    train_path = f'{root_path}/train_data_scaled.csv'
    test_path = f'{root_path}/test_data_scaled.csv'
    
    if os.path.exists(all_path) and os.path.exists(train_path) and os.path.exists(test_path):
        return True
    else:
        return False



if __name__ == "__main__":

    root_path = "./data/mimic_icu"

    # Check for existing preprocessed data
    if load_preprocessed_data(root_path):
         print("Preprocessed data already exists...")

    else:
        print("Processing raw data...")

        # Load data
        path = f"./{root_path}/icu_stays"
        df, df_pat, df_diagcode, df_icu = load_data(path)

        # Process length of stay
        df = process_los(df)

        # Process diagnosis codes
        df_diagcode = process_diagnosis_codes(df_diagcode)

        # Process ICU data
        df_icu = process_icu_data(df_icu)

        # Create feature matrix
        df = create_feature_matrix(df, df_diagcode, df_icu)

        # Clean up unnecessary columns
        df.drop(columns=['admission_location', 'subject_id', 'hadm_id', 'ADMITTIME', 'admittime',
                        'discharge_location', 'language', 'DECEASED', 'deathtime'], inplace=True)

        # Filter LOS
        df = df[df['LOS'] < 40]

        # Select final features
        df_final = df[['LOS', 'blood', 'circulatory', 'congenital', 'digestive',
            'endocrine', 'genitourinary', 'infectious', 'injury', 'mental', 'misc',
            'muscular', 'neoplasms', 'nervous', 'pregnancy', 'prenatal',
            'respiratory', 'skin', 'ICU', 'NICU']]
        df_final = df_final.reset_index(drop=True)

        # Generate shortcut features
        shortcut_features = generate_shortcut_features(df_final)
        df_final = df_final.join(shortcut_features, how="inner")

        # Split and scale data
        train_data, test_data = train_test_split(df_final, test_size=0.5, random_state=42)
        scalar = StandardScaler()
        
        train_data_scaled = scalar.fit_transform(train_data.drop(['LOS'], axis=1))
        test_data_scaled = scalar.transform(test_data.drop(['LOS'], axis=1))

        train_data_scaled = pd.DataFrame(train_data_scaled, columns=train_data.drop(['LOS'], axis=1).columns)
        test_data_scaled = pd.DataFrame(test_data_scaled, columns=test_data.drop(['LOS'], axis=1).columns)

        # Modify test data shortcut variables
        np.random.seed(42)
        for i in range(10):
            test_data_scaled[f'shortcut_{i}'] = np.random.normal(0, 1, test_data_scaled.shape[0])

        # Add LOS back to scaled data
        train_data_scaled['LOS'] = train_data['LOS'].values
        test_data_scaled['LOS'] = test_data['LOS'].values

        # Save processed data
        save_processed_data(root_path=root_path, 
                            data_final=df_final, 
                            train_data_scaled=train_data_scaled, 
                            test_data_scaled=test_data_scaled)