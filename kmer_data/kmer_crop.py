def split_rna_to_kmers(input_file, output_file, k):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            rna_seq = line.strip()
            kmers = [rna_seq[i:i+k] for i in range(0, len(rna_seq), k)]
            for kmer in kmers:
                outfile.write(kmer + ' ')
            outfile.write('\n')
if __name__ == "__main__":
    input_file = 'RNAlist.txt'
    k=3
    output_file = str(k) + 'mer_output.txt'
    split_rna_to_kmers(input_file, output_file, k)
    print("Done!")
