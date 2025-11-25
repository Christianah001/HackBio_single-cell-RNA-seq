# Task 1: Write a python function for translating DNA to protein
# Sample DNA to Protein Translator

def translate_dna_to_protein(dna_seq):
    """Translate a DNA sequence into a protein sequence."""
    codon_table = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
    }

    protein_seq = ""
    for i in range(0, len(dna_seq) - 2, 3):
        codon = dna_seq[i:i+3]
        protein_seq += codon_table.get(codon, 'X')  # X = unknown/invalid codon
    return protein_seq

# --- Example DNA sequence (TRPV6 gene fragment) ---
dna_seq = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAGTGAAGCCACCATGGAGCCCATGCTGCTGCTGGAGCCTG"

# Translate
protein_seq = translate_dna_to_protein(dna_seq)

print("DNA sequence length:", len(dna_seq), "bp")
print("Protein sequence length:", len(protein_seq), "aa")
print("Protein sequence:\n", protein_seq)

# Task 2: Write a python function for calculating the hamming distance between your slack username and twitter/X  handle.

def hamming_distance(str1, str2):
    """
    Calculate the Hamming distance between two strings.
    If they have different lengths, pad the shorter one with underscores ('_').
    """
    max_len = max(len(str1), len(str2))
    str1 = str1.ljust(max_len, '_')
    str2 = str2.ljust(max_len, '_')

    distance = sum(c1 != c2 for c1, c2 in zip(str1, str2))
    return distance

# Given handles
slack_handle = "Funmilayo Ligali"
twitter_handle = "FunmilayoC"

# Compute
distance = hamming_distance(slack_handle, twitter_handle)

print(f"Slack Handle: {slack_handle}")
print(f"Twitter Handle: {twitter_handle}")
print(f"Hamming Distance: {distance}")
