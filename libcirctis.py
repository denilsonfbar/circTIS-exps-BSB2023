import os
import random
import numpy as np

import shogun as sg
import pandas as pd
from Bio import SeqIO
from sklearn import metrics as mt
from matplotlib import pyplot
import seaborn as sns


# Create the main dataset with a FASTA file with circRNA seqs and a TSV file with one ou more TIS annotation for each circRNA
def create_main_dataset(all_circrnas_fasta_file, df_tis, output_fasta_file, output_tsv_file):

    df_tsv_file = pd.DataFrame(columns=['circrna_id', 'TIS_type', 'TIS_position'])

    circrna_seqs_file = SeqIO.index(all_circrnas_fasta_file, "fasta")
    
    circrna_counter = 0
    tis_counter = 0
    unique_seqs = set()
    unique_circrnas = []

    for i,row in df_tis.iterrows():

        circrna = circrna_seqs_file[row['transcirc_id']]
        circrna_na = str(circrna.seq)

        if circrna_na not in unique_seqs:  # removing circRNAs with diffents ids, but same na sequence

            circrna_counter += 1
            unique_seqs.add(circrna_na)
            unique_circrnas.append(circrna)

            for i,row in df_tis.loc[(df_tis['transcirc_id'] == circrna.id)].iterrows():

                tis_counter += 1
                tis_position_zi = row['tis_position_in_circrna'] - 1  # index correction
                tis_type = circrna_na[tis_position_zi : tis_position_zi + 3]

                df_tsv_file.loc[df_tsv_file.shape[0]] = [circrna.id, tis_type, tis_position_zi + 1]  # index correction
                
    circrna_seqs_file.close()

    SeqIO.write(unique_circrnas, output_fasta_file, 'fasta')
    df_tsv_file.to_csv(output_tsv_file, index=False, sep='\t')

    return circrna_counter, tis_counter


def count_sequences_in_fasta_file(fasta_file):

    count = 0
    with open(fasta_file, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            count += 1

    return count

def remove_repeated_sequences_in_fasta_file(fasta_file):
    
    unique_seqs = set()
    records_to_keep = []

    for record in SeqIO.parse(fasta_file, "fasta"):
        
        if str(record.seq) not in unique_seqs:
            unique_seqs.add(str(record.seq))
            records_to_keep.append(record)

    with open(fasta_file, "w") as output_handle:
        SeqIO.write(records_to_keep, output_handle, "fasta")


def split_fasta_randomly(fasta_file, output_path, num_files, seed):
    
    random.seed(seed)

    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    file_names = []

    # Shuffle the sequences randomly
    random.shuffle(sequences)

    # Split the sequences into groups
    groups = [sequences[i::num_files] for i in range(num_files)]

    # Write the sequences into separate output files
    for i, group in enumerate(groups):

        output_split_path = output_path + str(i+1) + '/'
        os.mkdir(output_split_path)

        output_file = output_split_path + 'seqs.fa'
        
        file_names.append(output_file)

        SeqIO.write(group, output_file, 'fasta')

    return file_names

def fasta_tsv_inner_join(fasta_file, tsv_file, output_tsv_file):

    fasta_records = list(SeqIO.parse(fasta_file, "fasta"))
    df_fasta = pd.DataFrame([{'circrna_id': record.id} for record in fasta_records])

    df_tsv = pd.read_csv(tsv_file, sep='\t', header=0)

    # Perform the inner join between the DataFrames
    df_merged = pd.merge(df_tsv, df_fasta, on='circrna_id', how='inner')

    # Write the result to the output TSV file
    df_merged.to_csv(output_tsv_file, sep='\t', index=False)


def merge_fasta_files(fasta_files, output_fasta_file):

    with open(output_fasta_file, 'w') as output:
            for fasta_file in fasta_files:
                with open(fasta_file, 'r') as fasta:
                    output.write(fasta.read())

def merge_tsv_files(tsv_files, output_tsv_file):

    dfs = []
    for tsv_file in tsv_files:
        df = pd.read_csv(tsv_file, sep='\t', header=0)
        dfs.append(df)
    
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_tsv_file, sep='\t', index=False)


# Extract a circRNA subsequence around a position, according length windows. 
# If necessary, loops through the circRNA to complete total sample length.
def extract_circrna_subseq_around_position(circrna_na, upstream_length, downstream_length, position):

    up_subseq = circrna_na[:position]
    down_subseq = circrna_na[position:]

    while len(up_subseq) < upstream_length:
        up_subseq = circrna_na + up_subseq
    up_subseq = up_subseq[-upstream_length:]

    while len(down_subseq) < downstream_length:
        down_subseq = down_subseq + circrna_na
    down_subseq = down_subseq[:downstream_length]

    full_subseq = up_subseq + down_subseq

    return full_subseq

def extract_circrna_samples(circrna_fasta_file, tis_tsv_file, output_samples_file):
    
    sample_upstream_length = 100
    sample_downstream_length = 303 
    tis_types = ['ATG', 'CTG', 'GTG', 'TTG']

    df_samples_columns = ['circrna_id', 'TIS_type', 'TIS_position', 
                          'sample_upstream_length', 'sample_downstream_length',
                          'sample_na', 'sample_label']
    df_samples = pd.DataFrame(columns=df_samples_columns)

    df_tis = pd.read_csv(tis_tsv_file, sep='\t', header=0)

    print('\nFASTA file:', circrna_fasta_file)
    n_circrnas = count_sequences_in_fasta_file(circrna_fasta_file)

    circrna_counter = 0
    for circrna in SeqIO.parse(circrna_fasta_file, "fasta"):

        circrna_counter += 1
        print('Extracting:', circrna_fasta_file, '\t', circrna_counter, 'of', n_circrnas)

        df_tis_circrna = df_tis.loc[(df_tis['circrna_id']==circrna.id)]
        ls_tis_circrna = df_tis_circrna['TIS_position'].to_list()

        circrna_na = str(circrna.seq)
        for test_position_zi in range(len(circrna_na)-3):

            sample_label = 0

            test_codon = circrna_na[test_position_zi:test_position_zi+3]

            if test_codon in tis_types:

                test_position = test_position_zi + 1  # index correction

                if test_position in ls_tis_circrna:  # real TIS
                    sample_label = 1
            
                else:  # false TIS
                    sample_label = -1
                    
            if sample_label != 0:

                sample_na = extract_circrna_subseq_around_position(circrna_na, sample_upstream_length, sample_downstream_length, test_position_zi)

                df_samples.loc[df_samples.shape[0]] = [circrna.id, circrna_na[test_position_zi:test_position_zi+3], test_position, 
                                                       sample_upstream_length, sample_downstream_length, 
                                                       sample_na, sample_label]

    df_samples.to_csv(output_samples_file, sep='\t', index=False)

def decrease_length_samples(samples, upstream_size, downstream_size):

    tis_start_idx = 100

    up_first_idx = tis_start_idx - upstream_size
    down_last_idx = tis_start_idx + downstream_size

    trans_samples = []
    for sample in samples:
        new_sample = sample[up_first_idx:tis_start_idx] + sample[tis_start_idx:down_last_idx]
        trans_samples.append(new_sample)

    return trans_samples


def prepare_exp_data(parameters, df_samples_train, df_samples_test):
   
    # Train data
    n_samples_train = len(df_samples_train)
    n_samples_train_pos = df_samples_train[df_samples_train['sample_label'] == 1].shape[0]
    n_samples_train_neg = df_samples_train[df_samples_train['sample_label'] == -1].shape[0]
    X_train = df_samples_train['sample_na'].tolist()
    y_train = np.array(df_samples_train['sample_label'])

    # Test data
    n_samples_test = len(df_samples_test)
    n_samples_test_pos = df_samples_test[df_samples_test['sample_label'] == 1].shape[0]
    n_samples_test_neg = df_samples_test[df_samples_test['sample_label'] == -1].shape[0]
    X_test = df_samples_test['sample_na'].tolist()
    y_test = np.array(df_samples_test['sample_label'])

    # Samples resize
    X_train = decrease_length_samples(X_train, parameters['up_sample_size'], parameters['down_sample_size'])
    X_test = decrease_length_samples(X_test, parameters['up_sample_size'], parameters['down_sample_size'])

    samples_info = {}
    samples_info['n_samples_train'] = n_samples_train
    samples_info['n_samples_train_pos'] = n_samples_train_pos
    samples_info['n_samples_train_neg'] = n_samples_train_neg
    samples_info['n_samples_test'] = n_samples_test
    samples_info['n_samples_test_pos'] = n_samples_test_pos
    samples_info['n_samples_test_neg'] = n_samples_test_neg
    samples_info['sample_size'] = parameters['up_sample_size'] + parameters['down_sample_size']

    return X_train, y_train, X_test, y_test, samples_info

def train_svm(parameters, X_train_raw, y_train):
    
    # Creating Shogun features objects
    X_train = sg.StringCharFeatures(sg.DNA)
    X_train.set_features(X_train_raw)
    y_train = sg.BinaryLabels(y_train)

    # Create kernel
    if parameters['kernel'] == 'WD':
        kernel = sg.WeightedDegreeStringKernel(X_train, X_train, int(parameters['degree']))
 
    # Create SVM and set regularization parameters
    C = parameters['C1']
    svm = sg.LibSVM(C, kernel, y_train)
    svm.set_C(parameters['C1'], parameters['C2'])

    # Train SVM
    svm.train()

    return svm

def svm_predict(svm, X_test_raw):

    # Creating Shogun features objects
    X_test = sg.StringCharFeatures(sg.DNA)
    X_test.set_features(X_test_raw)
    
    # Predicting
    prediction = svm.apply(X_test)
    y_pred_labels = prediction.get_labels()
    y_pred_scores = prediction.get_values()

    return y_pred_labels, y_pred_scores


def calc_metrics_labels(dict_metrics, y_real_labels, y_pred_labels):

    TN, FP, FN, TP = mt.confusion_matrix(y_real_labels, y_pred_labels).ravel()

    f1_score = mt.f1_score(y_real_labels, y_pred_labels)
    precision = mt.precision_score(y_real_labels, y_pred_labels)
    recall = mt.recall_score(y_real_labels, y_pred_labels)  # sensitivity

    specificity = mt.recall_score(y_real_labels, y_pred_labels, pos_label=-1)
    accuracy = mt.accuracy_score(y_real_labels, y_pred_labels)

    dict_metrics['TP'] = TP
    dict_metrics['FP'] = FP
    dict_metrics['FN'] = FN
    dict_metrics['TN'] = TN

    dict_metrics['precision'] = precision
    dict_metrics['recall'] = recall
    dict_metrics['f1_score'] = f1_score

    dict_metrics['specificity'] = specificity
    dict_metrics['accuracy'] = accuracy

    return dict_metrics

def calc_metrics_scores(dict_metrics, y_real_labels, y_pred_scores):

    AUROC = mt.roc_auc_score(y_real_labels, y_pred_scores)
    AUPR = mt.average_precision_score(y_real_labels,y_pred_scores)

    dict_metrics['AUROC'] = AUROC
    dict_metrics['AUPR'] = AUPR

    return dict_metrics


def create_df_evaluation():

    df_eval = pd.DataFrame(columns=[
        'fold',
        'up_size',
        'down_size',
        'degree',
        'C1',
        'C2',
        'kernel',
        'TP',
        'FP',
        'FN',
        'TN',
        'F1-score',
        'AUPR',
        'Precision',
        'Recall',
        'Specificity',
        'Accuracy',
        'AUROC',
        'n_train',
        'n_train_pos',
        'n_train_neg',
        'n_test', 
        'n_test_pos',
        'n_test_neg',
        'sample_size',
        'train_time',
        'pred_time'
    ])

    return df_eval

def add_result_in_df_evaluation(df_eval, parameters, samples_info, times, metrics):

    df_eval.loc[df_eval.shape[0]] = [
        parameters['fold'],
        parameters['up_sample_size'],
        parameters['down_sample_size'],
        parameters['degree'],
        parameters['C1'],
        parameters['C2'],
        parameters['kernel'],
        metrics['TP'],
        metrics['FP'],
        metrics['FN'],
        metrics['TN'],
        metrics['f1_score'],
        metrics['AUPR'],
        metrics['precision'],
        metrics['recall'],
        metrics['specificity'],
        metrics['accuracy'],
        metrics['AUROC'],
        samples_info['n_samples_train'],
        samples_info['n_samples_train_pos'],
        samples_info['n_samples_train_neg'],
        samples_info['n_samples_test'],
        samples_info['n_samples_test_pos'],
        samples_info['n_samples_test_neg'],
        samples_info['sample_size'],
        times['train_time'],
        times['pred_time']
    ]
            
    return df_eval


def plot_line_chart(df_eval, x_axis, y_axis):

    dims = (20, 12)
    fig, ax = pyplot.subplots(figsize=dims)
    pyplot.grid()

    for col in y_axis:
        sns.lineplot(data=df_eval, x=x_axis, y=col, legend='brief',label=col, marker="o")
