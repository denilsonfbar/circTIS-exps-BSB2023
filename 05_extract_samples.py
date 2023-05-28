import libcirctis

circrna_fasta_file = 'datasets/splits/test/seqs.fa'
tis_tsv_file = 'datasets/splits/test/tis.tsv'
output_samples_file = 'datasets/splits/test/samples.tsv'
libcirctis.extract_circrna_samples(circrna_fasta_file, tis_tsv_file, output_samples_file)

n_folds = 5
cv_path = 'datasets/cross_validation/'
for i in range(1, n_folds+1):

    circrna_fasta_file = cv_path + 'fold_' + str(i) + '/train/seqs.fa'
    tis_tsv_file = cv_path + 'fold_' + str(i) + '/train/tis.tsv'
    output_samples_file = cv_path + 'fold_' + str(i) + '/train/samples.tsv'
    libcirctis.extract_circrna_samples(circrna_fasta_file, tis_tsv_file, output_samples_file)

    circrna_fasta_file = cv_path + 'fold_' + str(i) + '/validation/seqs.fa'
    tis_tsv_file = cv_path + 'fold_' + str(i) + '/validation/tis.tsv'
    output_samples_file = cv_path + 'fold_' + str(i) + '/validation/samples.tsv'
    libcirctis.extract_circrna_samples(circrna_fasta_file, tis_tsv_file, output_samples_file)
