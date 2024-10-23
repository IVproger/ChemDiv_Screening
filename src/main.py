import polars as pl
from transformers import BertTokenizerFast, BertModel,BertTokenizer
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import sys

def create_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

import re
# Function to generate embeddings for SMILES
def generate_embedding_smiles(smile):
    tokens = tokenizer(smile, return_tensors='pt')
    embedding = smiles_tokenizer(**tokens)
    return embedding[0].detach().numpy().flatten().tolist() 

# Function to generate embeddings for protein sequences
def generate_embedding_protein(sequence):
    sequence_Example = re.sub(r"[UZOB]", "X", sequence)
    encoded_input = tokenizer_rostlab(sequence_Example, return_tensors='pt')
    embedding = model_rostlab(**encoded_input)
    return embedding[0].detach().numpy().flatten().tolist() 


schema = pa.schema([
    ('ID', pa.string()),
    ('encoding', pa.list_(pa.float64()))
])

# Universal function to write embeddings to Parquet
def write_embeddings_to_parquet(data, batch_size, parquet_file, generate_embedding_func):
    num_newlines = len(data)
    total_batches = (num_newlines + batch_size - 1) // batch_size

    writer = None

    for batch in tqdm(create_batches(data, batch_size), total=total_batches, desc="Processing batches", file=sys.stdout, ascii=True):
        embeddings = [generate_embedding_func(item) for item in batch]
        df = pd.DataFrame({'ID': batch, 'encoding': embeddings})
        
        table = pa.Table.from_pandas(df, schema=schema)
        
        if writer is None:
            writer = pq.ParquetWriter(parquet_file, schema)
        
        writer.write_table(table)

    if writer:
        writer.close()

if __name__ == "__main__":
    df = pl.read_parquet('data/BindingDB_predprocessed/BindingDB_v0.parquet')

    dataic50 = df[["Ligand SMILES","IC50 (nM)","BindingDB Target Chain Sequence"]].drop_nulls() 

    ligand_smiles = dataic50['Ligand SMILES'].to_list()
    target_chain_sequence = dataic50['BindingDB Target Chain Sequence'].to_list()
    
    checkpoint = 'unikei/bert-base-smiles'
    tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
    smiles_tokenizer = BertModel.from_pretrained(checkpoint)
    
    tokenizer_rostlab = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
    model_rostlab = BertModel.from_pretrained("Rostlab/prot_bert")
    
    batch_size = 512
    parquet_file_smiles = 'data/embeddings/ligand_embeddings.parquet'
    parquet_file_protein = 'data/embeddings/protein_embeddings.parquet'
    
    print("Generating embeddings for SMILES and protein sequences")
    # Write SMILES embeddings to Parquet
    write_embeddings_to_parquet(ligand_smiles, batch_size, parquet_file_smiles, generate_embedding_smiles)

    print("Generating embeddings for protein sequences")
    # Write protein sequence embeddings to Parquet
    write_embeddings_to_parquet(target_chain_sequence, batch_size, parquet_file_protein, generate_embedding_protein)


