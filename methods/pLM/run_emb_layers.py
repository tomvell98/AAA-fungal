import os
import torch
import argparse # Not used, but kept
import numpy as np
from Bio import SeqIO
from esm.models.esmc import ESMC # Using local ESMC
# from esm.sdk.api import ESMProtein, LogitsConfig # Not needed for local model
from pathlib import Path
from tqdm import tqdm
# from getpass import getpass # Not needed for local model

def load_sequences_from_fasta(fasta_file_path):
    """Load protein sequences from a FASTA file."""
    sequences = {}
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

def get_mean_embedding_from_local_model_layer(model, sequence_str, layer_index, device, batch_converter):
    """
    Get mean embedding for a protein sequence from a specific hidden layer of a local ESMC model.
    layer_index should be 0-indexed for accessing the hidden_states tuple.
    If layer 0 is input embeddings, then layer N is output of Nth transformer block.
    To get the "12th layer" (output of 12th block), use layer_index=12.
    """
    data = [("protein_id_dummy", sequence_str)] # Model expects a list of tuples
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        # Request all hidden states
        results = model(batch_tokens, repr_layers=[], return_contacts=False, output_hidden_states=True)
    
    # hidden_states is a tuple of tensors: (num_layers + 1) x (batch_size, seq_len, embed_dim)
    # The first element (index 0) is often the input embeddings.
    # Subsequent elements are outputs of each transformer layer.
    all_hidden_states = results['hidden_states'] 
    
    if not (0 <= layer_index < len(all_hidden_states)):
        raise ValueError(f"Layer index {layer_index} is out of bounds. Model has {len(all_hidden_states)-1} transformer layers.")

    # Select the specified layer's embeddings
    # Shape: (batch_size, seq_len + 2, embed_dim) -- +2 for <cls> and <eos> tokens
    layer_embeddings_with_tokens = all_hidden_states[layer_index]
    
    # Remove <cls> and <eos> tokens (first and last token representations)
    # Assuming batch_size is 1 as we process one sequence at a time here
    # Squeeze out batch dimension
    layer_embeddings_for_sequence = layer_embeddings_with_tokens.squeeze(0)[1:-1, :] 
    # Now shape is (seq_len, embed_dim)

    mean_embedding = layer_embeddings_for_sequence.mean(dim=0)  # Average across sequence length
    
    return mean_embedding

def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from a local ESMC model for a FASTA file.')
    parser.add_argument('--model_name', required=True, help='Local model name (e.g., esmc_600m)')
    parser.add_argument('--fasta_path', required=True, help='Path to input FASTA file')
    parser.add_argument('--layer', type=int, required=True, help='0-indexed layer to extract')
    parser.add_argument('--output_dir', required=True, help='Directory to save output embeddings and IDs')
    args = parser.parse_args()

    model_name = args.model_name
    fasta_path = args.fasta_path
    layer_to_extract_0_indexed = args.layer
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(fasta_path).stem
    embeddings_output_path = output_dir / f"{base_name}_layer{layer_to_extract_0_indexed}_embeddings.npy"
    ids_output_path = output_dir / f"{base_name}_layer{layer_to_extract_0_indexed}_ids.txt"

    # Check if CUDA is available and set to GPU 1
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str == "cpu":
        print("CUDA is not available, using CPU instead")
        device = torch.device("cpu")
    else:
        # It's generally better to let the user specify the GPU via CUDA_VISIBLE_DEVICES
        # or handle it more flexibly. Forcing device 1.
        if torch.cuda.device_count() > 1:
            torch.cuda.set_device(1) 
            device = torch.device("cuda:1")
            print(f"Using CUDA device 1: {torch.cuda.get_device_name(1)}")
            print(f"GPU Memory Available: {torch.cuda.get_device_properties(1).total_memory / 1024**2:.0f}MB")
        else:
            device = torch.device("cuda:0") # Default to GPU 0 if only one is available
            print(f"Using CUDA device 0: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")


    # Load the ESM model and alphabet/batch_converter
    print(f"Loading local model {model_name}...")
    # For ESMC, from_pretrained might return model and alphabet, or just model
    # Checking ESMC source, it should just be the model.
    # ESMC.from_pretrained() is the correct way for these new models.
    # The older esm1b etc. used esm.pretrained.load_model_and_alphabet()
    try:
        model = ESMC.from_pretrained(model_name)
    except Exception as e:
        print(f"Failed to load model with ESMC.from_pretrained: {e}")
        print("Attempting with older esm.load_model_and_alphabet_hub if it's an older model type...")
        import esm # For older model loading
        model, alphabet = esm.pretrained.load_model_and_alphabet_hub(model_name)

    model = model.to(device)
    model.eval() # Set model to evaluation mode

    # Replace direct call to get_batch_converter with fallback
    if hasattr(model, 'get_batch_converter'):
        batch_converter = model.get_batch_converter()
    elif hasattr(model, 'alphabet') and hasattr(model.alphabet, 'get_batch_converter'):
        batch_converter = model.alphabet.get_batch_converter()
    else:
        from esm.tokenization import get_esmc_model_tokenizers
        batch_converter = get_esmc_model_tokenizers()
    
    print(f"Model {model_name} loaded successfully.")
    num_layers_in_model = model.num_layers # Check how many layers the model reports
    print(f"The loaded model reports {num_layers_in_model} layers.")
    print(f"Attempting to extract from 0-indexed layer {layer_to_extract_0_indexed}.")
    print(f"This typically corresponds to the output of the {layer_to_extract_0_indexed}th transformer block if layer 0 is input embeddings.")


    # Load sequences from FASTA file
    print(f"Loading sequences from {fasta_path}...")
    sequences = load_sequences_from_fasta(fasta_path)
    print(f"Loaded {len(sequences)} sequences")
    
    # Process each sequence
    embeddings_list = []
    ids_list = []
    print(f"Extracting embeddings from layer {layer_to_extract_0_indexed} (0-indexed)...")
    for i, (protein_id, sequence_str) in tqdm(enumerate(sequences.items()), total=len(sequences)):
        try:
            mean_embedding = get_mean_embedding_from_local_model_layer(
                model, sequence_str, layer_to_extract_0_indexed, device, batch_converter
            )
            embeddings_list.append(mean_embedding.cpu().detach().numpy())
            ids_list.append(protein_id)
        except Exception as e:
            print(f"Error processing {protein_id}: {e}")
            import traceback
            traceback.print_exc()

    # Convert list to numpy array and save embeddings
    if embeddings_list:
        embeddings_array = np.array(embeddings_list)
        print(f"Saving embeddings array with shape {embeddings_array.shape} to {embeddings_output_path}...")
        np.save(embeddings_output_path, embeddings_array)
    else:
        print("No embeddings were generated.")

    # Save IDs to a text file
    if ids_list:
        print(f"Saving {len(ids_list)} protein IDs to {ids_output_path}...")
        with open(ids_output_path, 'w') as f:
            for protein_id in ids_list:
                f.write(f"{protein_id}\n")
    else:
        print("No IDs to save.")

    print("Done!")

if __name__ == "__main__":
    main()