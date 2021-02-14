import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random as rn
import tensorflow as tf

from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


####### STUDENTS FILL THIS OUT ######
#Question 3
# should be OK
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    
    #this function does the lookup in the table, x is the rdc_code
    def get_generic_name(x):
        condition = (ndc_df['NDC_Code'] == x)
        
        # lookup in table
        vet_rit = ndc_df[condition]['Non-proprietary Name'].unique()
        
        # check if list is empty
        if len(vet_rit):
            # remove the list
            val_rit = vet_rit[0]
        else:
            val_rit = 'NA'
            
        return val_rit
    
    # apply with map to the entire DF
    # it takes a cople of minutes (140K records)
    df['generic_drug_name'] = df['ndc_code'].map(get_generic_name)
    
    return df

#Question 4
# should be OK
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    HOW_MANY = 1
    first_encounter_df = df.sort_values(by=['patient_nbr', 'encounter_id'], ascending=True).groupby(by=['patient_nbr']).head(HOW_MANY)
    return first_encounter_df

# helper, for question 6
def compute_fraction(num, den):
    return round(num*100./den, 1) 

def print_dataset_summary(df, name, n_total):
    print('Number of records in', name,  len(df), ', fraction is:', compute_fraction(len(df), n_total), '%')
    
#Question 6
# I have decided to use SKlearn GroupKfold.
# It divides the given dataframe in N splits, ensuring that we have diffrents groups in each split
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    # first shuffle the data
    df = df.sample(frac = 1)
    
    n_total = df.shape[0]
    
    # we're going to divide the df to have 80% in train-validation and 20% in the test set
    # we will take 4 out of 5 fold for train-val, the rest for test
    # for this reason n_split for the first split is 5
    gkf = GroupKFold(n_splits=5)
    
    # here we define the differents groups id, on which we're splitting to have non overlapping
    # get indexes
    train_val_idx, test_idx = next(gkf.split(df, groups=df[patient_key].unique()))
    
    test = df.iloc[test_idx]
    
    # now test is 20% of the total 
    # need another split to divide in train and validation
    
    train_val = df.iloc[train_val_idx]
    
    # remaining 80% to split in 4
    gkf2 = GroupKFold(n_splits=4) 
        
    train_idx, val_idx = next(gkf2.split(train_val , groups=train_val[patient_key].unique()))
    
    train = train_val.iloc[train_idx]
    validation = train_val.iloc[val_idx]
    
    print_dataset_summary(train, 'Train', n_total)
    print_dataset_summary(validation, 'Validation', n_total)
    print_dataset_summary(test, 'Test', n_total)
    
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        # count # of items in file
        num_items = sum(1 for line in open(vocab_file_path))
        
        # print(c, num_items)
        
        tf_categorical_feature_column = tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_file(key=c, vocabulary_file=vocab_file_path, 
                                                                      vocabulary_size=num_items,
                                                                      num_oov_buckets=1))
        
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    def norm_func(col):
        col = tf.cast(col, tf.float32)
        
        return (col - mean)/std
    
    return norm_func

def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    tf_numeric_feature = tf.feature_column.numeric_column(col, dtype=tf.dtypes.float32, default_value=default_value, 
                                                          normalizer_fn=normalize_numeric_with_zscore(MEAN, STD))
    
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    
    return m, s

# Question 10
def get_student_binary_prediction(df, col, threshold=5.0):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    # From requirements: we want patients staying for at least 5 day in hospital
    THRESHOLD = threshold
    
    student_binary_prediction = (df[col] >= THRESHOLD).values.astype(int)
    
    print('get_student_binary_prediction---> Predicted positive:', sum(student_binary_prediction))
    
    return student_binary_prediction

# other utilities and helper functions

# to analyze distinct values of categorical columns
def count_unique_values(df, cat_col_list):
    cat_df = df[cat_col_list]
    val_df = pd.DataFrame({'column': cat_df.columns, 
                       'cardinality': cat_df.nunique() } )
    return val_df

# plt hist of numerical features
def show_dist(df, col, bins, kde=False):
    # plt.title(col)
    sns.histplot(df[col], bins=bins, kde=kde)
    plt.grid()

def plot_hist_numerical(df, num_feat_list, vet_bins):
    plt.figure(figsize=(16,10))
    
    for i, col in enumerate(num_feat_list):
        plt.subplot(3, 4, i + 1)
        # in student_utils
        show_dist(df, col, bins=vet_bins[i])
        plt.grid()
    
    plt.tight_layout()
    
# to count the missing values in the fields

def count_missing_values(df, chr_used, list_fields):
    n_count = []
    n_perc = []
    
    n_rows = df.shape[0]
    for i, col in enumerate(list_fields):
        condition = (df[col] == chr_used[i])
        count = len(df[condition])
        n_count.append(count)
        perc = round(count*100./float(n_rows), 1)
        
        n_perc.append(perc)
    
    # build the return dataframe
    count_df = pd.DataFrame({'columns': list_fields, 
                       '# of nulls': n_count, "perc": n_perc } )
    
    return count_df
# to count zeros in numerical fields

def count_zeros(df, list_fields):
    n_count = []
    n_perc = []
    
    n_rows = df.shape[0]
    for i, col in enumerate(list_fields):
        condition = (df[col] == 0)
        count = len(df[condition])
        n_count.append(count)
        perc = round(count*100./float(n_rows), 1)
        
        n_perc.append(perc)
    
    # build the return dataframe
    count_df = pd.DataFrame({'columns': list_fields, 
                       '# of zeros': n_count, "perc": n_perc } )
    
    return count_df

# compute the ROC AUC using the trapeze formula
def compute_area(fpr_reversed, tpr_reversed):
    n_point = len(fpr_reversed)
    
    area = 0.
    
    for i in range(0, n_point - 1):
        area_ti = (tpr_reversed[i] + tpr_reversed[i + 1]) * (fpr_reversed[i + 1] - fpr_reversed[i]) * 0.5
        
        area += area_ti
    
    return round(area, 3)

# compute all the metrics (acc, precision, recall, etc) for various thresholds
def compute_ml_metrics(prob_output_df, threshold_list):
    # it will populate these lists, for plotting
    # returns a dataframe with all metrics
    # do the computation for different thresholds

    # it will populate these lists, for plotting
    acc_list = []
    prec_list = []
    rec_list = []
    f1_list = []
    conf_mat_list = []
    tpr_list = []
    fpr_list = []
    
    for thr in threshold_list:
        print('Compute for THRESHOLD:', thr)
        student_binary_prediction = get_student_binary_prediction(prob_output_df, 'pred_mean', threshold=thr)
        # pred_test_df = add_pred_to_test(d_test, student_binary_prediction, ['race', 'gender'])
        
        # extracts label and score
        # labels
        y_true = (prob_output_df['actual_value'].values >= 5).astype(int)
        # score
        y_preds = student_binary_prediction
        
        # compute metrics
        conf_mat = confusion_matrix(y_true, y_preds)
        tn, fp, fn, tp = conf_mat.ravel()
        
        acc = accuracy_score(y_true, y_preds)
        precision = precision_score(y_true, y_preds, average='weighted')
        recall = recall_score(y_true, y_preds, average='weighted')
        f1 = f1_score(y_true, y_preds, average='weighted')
        tpr = round(tp/float(tp + fn), 2)
        fpr = round(fp/float(fp + tn), 2)
        
        acc_list.append(acc)
        prec_list.append(precision)
        rec_list.append(recall)
        f1_list.append(f1)
        conf_mat_list.append([tn, fp, fn, tp])
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
        # build the results dataframe
        stats_df = pd.DataFrame(list(zip(threshold_list, conf_mat_list, acc_list, prec_list, rec_list, f1_list, tpr_list, fpr_list)), 
                                columns =['thr', 'conf_mat [tn, fp, fn, tp]', 'acc', 'prec', 'rec', 'f1', 'tpr', 'fpr'])
    
    return stats_df

def enable_reproducibility(seed):
    SEED = seed
    os.environ['PYTHONHASHSEED'] = '0'
    # The below is needed for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(SEED)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(SEED)
    tf.random.set_seed(SEED)

    # to plot loss vs epochs during training
def plot_loss(hist, skip):
    plt.figure(figsize=(14,6))
    
    plt.plot(hist.history['loss'][skip:], label='Training loss')
    plt.plot(hist.history['val_loss'][skip:], label='Validation loss')
    plt.title('Loss')
    plt.legend(loc='upper right')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.show();

def plot_zeros(df):
    plt.title('Perc. of zeros')
    sns.barplot(y=df['columns'], x=df['perc'])

def plot_missing_values(df):
    plt.title('Perc. of missing values')
    sns.barplot(y=df['columns'], x=df['perc'])

def plot_f1_score(thr_list, df):
    plt.figure(figsize=(8, 6))
    plt.plot(thr_list, df['f1'].values, '*-')
    plt.title('F1 score vs threshold')
    plt.xlabel('threshold')
    plt.ylabel('F1-score')
    plt.grid(True)
    plt.show()

def plot_prec_rec(thr_list, df, xlim=(2.2, 6.0), ylim=(0.1, 0.8)):
    plt.figure(figsize=(8, 6))
    plt.plot(thr_list, df['prec'].values, '*-', label='prec.')
    plt.plot(thr_list, df['rec'].values, '+-', label='rec.')
    plt.title('prec/rec vs threshold')
    plt.legend(loc='lower right')
    plt.xlabel('threshold')
    plt.ylabel('prec/rec')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True)
    plt.show()

def plot_roc(df):
    plt.figure(figsize=(8, 6))
    plt.plot(df['fpr'].values, df['tpr'].values, '*-')
    plt.title('ROC curve')
    plt.xlim((0.1, 0.7))
    plt.ylim((0.1, 0.9))
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.grid(True)
    plt.show()
    