import pandas as pd
from time import time
import datetime as dt
import libcirctis

n_folds = 5
output_file = 'outputs/09.tsv'

parameters = {}
parameters['kernel'] = 'WD'

parameters['up_sample_size'] =
parameters['down_sample_size'] =
parameters['degree'] = parameters['up_sample_size'] + parameters['down_sample_size']

C_values = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]


df_eval = libcirctis.create_df_evaluation()

for C in C_values:

    for fold in range(1, n_folds+1):

        start_t = time()

        parameters['fold'] = fold
        parameters['C1'] = parameters['C2'] = C

        print('\n' + dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + str(parameters['up_sample_size']) + '\t' 
        + str(parameters['down_sample_size']) + '\t' + str(parameters['degree']) + '\t' + str(parameters['C1'])
        + '\t' + str(parameters['fold']))

        train_samples_file = 'datasets/cross_validation/fold_' + str(fold) + '/train/samples.tsv'
        test_samples_file =  'datasets/cross_validation/fold_' + str(fold) + '/validation/samples.tsv'

        # Balancing training data
        df_samples_train_all = pd.read_csv(train_samples_file, sep='\t', header=0)
        df_samples_pos_train = df_samples_train_all.loc[(df_samples_train_all['sample_label'] == 1)]
        df_samples_neg_train = df_samples_train_all.loc[(df_samples_train_all['sample_label'] == -1)]
        df_samples_neg_train = df_samples_neg_train.sample(frac=1, random_state=721379)
        df_samples_neg_train = df_samples_neg_train.head(df_samples_pos_train.shape[0])
        df_samples_train = pd.concat([df_samples_pos_train, df_samples_neg_train])

        df_samples_test = pd.read_csv(test_samples_file, sep='\t', header=0)

        X_train, y_train, X_test, y_test, samples_info = libcirctis.prepare_exp_data(parameters, df_samples_train, df_samples_test)

        svm = libcirctis.train_svm (parameters, X_train, y_train)

        train_t = time()
        train_time = train_t - start_t
        print(f'Train time: {train_time:.3f} secs')

        y_pred_labels, y_pred_scores = libcirctis.svm_predict(svm, X_test)

        pred_time = time() - train_t
        print(f'Prediction time: {pred_time:.3f} secs')

        times = {}
        times['train_time'] = train_time
        times['pred_time'] = pred_time

        metrics = {}
        metrics = libcirctis.calc_metrics_labels(metrics, y_test, y_pred_labels)
        metrics = libcirctis.calc_metrics_scores(metrics, y_test, y_pred_scores)

        libcirctis.add_result_in_df_evaluation(df_eval, parameters, samples_info, times, metrics)

df_eval.to_csv(output_file, sep='\t', index=False)
