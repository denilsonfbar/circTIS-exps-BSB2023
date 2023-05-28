import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from liftover import get_lifter
from lxml import etree


N_TOTAL_CIRCRNAS =     -1  # -1 for all circRNAs preprocessing
LENGTH_SUBSEQ_GENOME = 30  # to locate the position of the TISs in the circRNAs


# Loading Transcirc metadata
df_transcirc = pd.read_csv('data_raw/transcirc_metadata.tsv.gz', sep='\t', compression='gzip')
print('Total circRNAs in Transcirc:', len(df_transcirc))

# Selecting circRNAs with TIS annotation
df_transcirc.drop(df_transcirc[df_transcirc['TIS'] != 1.0].index, axis=0, inplace=True)
print('Selected circRNAs with TIS annotation:', len(df_transcirc))

# Limiting the number of circRNAs preprocessed, for testing only
if N_TOTAL_CIRCRNAS != -1:
    df_transcirc = df_transcirc[:N_TOTAL_CIRCRNAS]

# Loading TISdb data
df_tisdb = pd.read_csv('data_raw/human_tisdb_data_1.0.csv.gz', compression='gzip')
print('Total annoted TIS in TISdb:', len(df_tisdb))

# Mapping the positions of TIS annotated in TISdb onto circRNA sequences
converter = get_lifter('hg19', 'hg38')
df_circrna_tis = pd.DataFrame(columns=['transcirc_id', 'circ_strand', 'circ_chrom', 'tis_coordinate_hg38', 'tis_position_in_circrna'])
circrna_seqs_file = SeqIO.index('data_raw/transcirc_sequence.fa.bgz', 'fasta')

for i,row in df_transcirc.iterrows():
    
    transcirc_id = row['TransCirc_ID']
    circ_gene = row['gene']
    circ_strand = row['strand']
    circ_start_coord = row['start']
    circ_end_coord = row['end']
    circ_seq = circrna_seqs_file[transcirc_id].seq

    last_tis_located_coord = -1

    # Locating the TIS annotated in TISdb (0..n) in the same gene as the circRNA
    for i,row in df_tisdb.loc[(df_tisdb['Gene']==circ_gene)].sort_values(by=['Coordinate start codon']).iterrows():
        
        chrom_tisdb = row['Chr']
        coord_tisdb = row['Coordinate start codon']

        # In TISdb, the same start codon appears annotated in several transcripts, probably due to alternative splicing
        if coord_tisdb != last_tis_located_coord:

            ## Checking if TIS is in circRNA and saving position
            
            # Converting TIS start position in TISdb from hg19 to hg38
            result = converter[chrom_tisdb][coord_tisdb]
            tis_coordinate_hg38 = result[0][1]

            # Obtaining a genome subsequence starting at TIS
            if circ_strand == '+':
                subseq_start = tis_coordinate_hg38
                subseq_end = tis_coordinate_hg38 + LENGTH_SUBSEQ_GENOME - 1
            else:  # antisense
                subseq_start = tis_coordinate_hg38 - LENGTH_SUBSEQ_GENOME + 1
                subseq_end = tis_coordinate_hg38 

            query_das = 'http://genome.ucsc.edu/cgi-bin/das/hg38/dna?segment=' + str(chrom_tisdb) + ':' + str(subseq_start) + ',' + str(subseq_end)
            doc = etree.parse(query_das, parser=etree.XMLParser())
            subseq = doc.xpath('SEQUENCE/DNA/text()')[0]
            subseq = subseq.replace('\n','')
            subseq = subseq.upper()

            if circ_strand == '-':  # getting the reverse complement of the subsequence
                subseq_rc = Seq(subseq)
                subseq = subseq_rc.reverse_complement()
                subseq = str(subseq)
            
            # Aligning the genome subsequence on the circRNA and getting the position of the TIS on the circRNA
            dup_circ_seq = circ_seq + circ_seq  # for the case of TIS near the downstream end
            tis_position_in_circ = dup_circ_seq.find(subseq)
            if (tis_position_in_circ != -1): 
                tis_position_in_circ += 1  # index correction
                df_circrna_tis.loc[df_circrna_tis.shape[0]] = [transcirc_id, circ_strand, chrom_tisdb, tis_coordinate_hg38, tis_position_in_circ]

            last_tis_located_coord = coord_tisdb
            print(transcirc_id, '\t', circ_strand, '\t', subseq, '\t', circ_seq)

circrna_seqs_file.close()

df_circrna_tis.to_csv('data_raw/circRNA_TIS_1.tsv', index=False, sep='\t')
print('Number of mapped TIS in circRNAs:', len(df_circrna_tis))
print('Number of circRNAs with at least one mapped TIS:', len(df_circrna_tis.groupby(['transcirc_id']).count()))


'''
Output:
Total circRNAs in Transcirc: 328080
Selected circRNAs with TIS annotation: 9394
Total annoted TIS in TISdb: 11923
Number of mapped TIS in circRNAs: 9691
Number of circRNAs with at least one mapped TIS: 8500
'''
