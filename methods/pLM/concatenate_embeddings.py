os.makedirs(output_dir, exist_ok=True)
enzyme_order_file = os.path.join(output_dir, "enzymes_order.txt")
import numpy as np
import os
import re
import argparse

def main():
    parser = argparse.ArgumentParser(description='Concatenate and average enzyme embeddings across layers.')
    parser.add_argument('--base_dir', required=True, help='Base directory containing enzyme subfolders')
    parser.add_argument('--output_dir', required=True, help='Directory for output files')
    parser.add_argument('--enzymes', nargs='+', required=True, help='List of enzyme names')
    args = parser.parse_args()

    base_dir = args.base_dir
    output_dir = args.output_dir
    enzymes = args.enzymes
    os.makedirs(output_dir, exist_ok=True)
    enzyme_order_file = os.path.join(output_dir, "enzymes_order.txt")
    if not os.path.exists(enzyme_order_file):
        with open(enzyme_order_file, "w") as f:
            for e in enzymes:
                f.write(f"{e}\n")

    enzyme_dir = os.path.join(base_dir, enzymes[0])
    layer_files = [f for f in os.listdir(enzyme_dir) if re.match(rf"{enzymes[0]}_layer_\d+_embeddings.npy", f)]
    layers = sorted({
        int(m.group(1))
        for f in layer_files
        if (m := re.search(r"layer_(\d+)_embeddings", f))
    })

    print(f"Found layers: {layers}")

    for layer in layers:
        embedding_files = [f"{base_dir}/{e}/{e}_layer_{layer}_embeddings.npy" for e in enzymes]
        id_files = [f"{base_dir}/{e}/{e}_layer_{layer}_ids.txt" for e in enzymes]

        all_ids = []
        all_embeddings = []
        for emb_file, id_file in zip(embedding_files, id_files):
            with open(id_file) as f:
                ids = [line.strip() for line in f]
            all_ids.append(ids)
            all_embeddings.append(np.load(emb_file))

        ref_ids = all_ids[0]
        for i, ids in enumerate(all_ids[1:], 1):
            if ids != ref_ids:
                raise ValueError(f"ID order mismatch in {id_files[i]}")

        emb_stack = np.stack(all_embeddings)
        np.save(f"{output_dir}/concat_layer_{layer}_embeddings.npy", emb_stack)
        concat_2d = np.concatenate(all_embeddings, axis=1)
        np.save(f"{output_dir}/concat2d_layer_{layer}_embeddings.npy", concat_2d)
        mean_embeddings = np.mean(emb_stack, axis=0)
        np.save(f"{output_dir}/mean_layer_{layer}_embeddings.npy", mean_embeddings)
        with open(f"{output_dir}/mean_layer_{layer}_ids.txt", "w") as f:
            for id in ref_ids:
                f.write(f"{id}\n")
        print(f"Done! Saved mean embeddings for layer {layer}.")

    print("All layers processed.")

if __name__ == "__main__":
    main()