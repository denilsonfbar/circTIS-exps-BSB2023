from time import time
import datetime as dt
import libcirctis

n_folds = 5
output_file = 'outputs/06.tsv'

upstream_length = 100
downstream_length = 103

degree_values = range(1, 204)


df_eval = libcirctis.create_df_evaluation()

for degree in degree_values:

    for fold in range(1, n_folds+1):

        start_t = time()

        parameters = libcirctis.default_exp_parameters()
        parameters['fold'] = fold
        parameters['sample_size'] = upstream_length + downstream_length
        parameters['up_sample_size'] = upstream_length
        parameters['down_sample_size'] = downstream_length
        parameters['degree'] = degree
        print('\n' + dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + str(parameters['up_sample_size']) + '\t' 
              + str(parameters['down_sample_size']) + '\t' + str(parameters['degree']) + '\t' + str(parameters['fold']))

        X_train, y_train, X_test, y_test, samples_info = libcirctis.prepare_fold_data(parameters)

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
