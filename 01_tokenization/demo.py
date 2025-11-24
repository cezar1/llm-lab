
"""
Tokenization & Embeddings Demo
Build byte-pair encoder + train subword vocab + visualize tokens and embeddings
"""

from bpe import BPE
from embeddings import TokenEmbedder
import matplotlib.pyplot as plt

def demo_bpe_training():
    """Demonstrate BPE training and vocabulary building"""
    print("=" * 60)
    print("1. BPE TRAINING & VOCABULARY BUILDING")
    print("=" * 60)

    # Sample corpus
    corpus = [
        "hello", "world", "help", "held", "world",
        "data", "database", "day", "way", "play",
        "there", "where", "here", "their"
    ]

    # Train BPE with 20 merges
    bpe = BPE(merges=20)
    rules = bpe.train(corpus)

    print(f"Corpus: {corpus}")
    print(f"Vocabulary size: {len(bpe.vocab)}")
    print(f"Top 10 tokens: {sorted(bpe.vocab.items(), key=lambda x: x[1], reverse=True)[:10]}")
    print(f"Merge rules applied: {len(rules)}")

    # Plot vocabulary statistics
    print("\nPlotting vocabulary frequency distribution...")
    bpe.plot_vocab_stats()

    # Plot merge history
    print("Plotting merge history...")
    bpe.plot_merge_history()

    return bpe


def demo_encoding_decoding(bpe):
    """Demonstrate encoding and decoding"""
    print("\n" + "=" * 60)
    print("2. ENCODING & DECODING")
    print("=" * 60)

    texts = [
        "hello world",
        "where is there",
        "playing in data"
    ]

    for text in texts:
        print(f"\nOriginal text: '{text}'")
        token_ids = bpe.encode(text)
        print(f"Token IDs: {token_ids}")

        # Decode back
        decoded = bpe.decode(token_ids)
        print(f"Decoded: '{decoded}'")
        print(f"Match: {decoded == text}")


def demo_token_visualization(bpe):
    """Demonstrate token visualization"""
    print("\n" + "=" * 60)
    print("3. TOKEN VISUALIZATION")
    print("=" * 60)

    text = "the quick brown fox jumps over the lazy dog"
    print(f"\nVisualizing tokens for: '{text}'")
    bpe.token_ids_to_viz(text)


def demo_embedding_analysis(bpe):
    """Demonstrate embedding space analysis"""
    print("\n" + "=" * 60)
    print("4. EMBEDDING SPACE ANALYSIS")
    print("=" * 60)

    # Encode some text
    text = "hello world help held where there"
    token_ids = bpe.encode(text)

    if len(token_ids) == 0:
        print("Could not encode text. Adjusting...")
        # Use simpler text that fits vocab
        token_ids = list(range(min(10, len(bpe.vocab))))

    print(f"Analyzing embeddings for tokens: {token_ids}")

    # Create embedder
    embedder = TokenEmbedder(len(bpe.vocab), embed_dim=64)

    # Compare embeddings
    print("\nComparing one-hot vs learned embeddings...")
    embedder.compare_embeddings(token_ids)

    # Plot learned embeddings in 2D
    print("Projecting embeddings to 2D...")
    embedder.visualize_embeddings_2d(token_ids)

    # Plot cosine similarity for learned embeddings
    print("Plotting cosine similarity heatmap...")
    embedder.plot_embedding_comparison(token_ids, 'learned')


def demo_compression_ablation(bpe):
    """Ablation: How many merges are needed for good compression?"""
    print("\n" + "=" * 60)
    print("5. ABLATION: MERGES vs COMPRESSION")
    print("=" * 60)

    test_text = "the quick brown fox jumps over the lazy dog and the brown fox runs"

    compression_ratios = []
    merge_counts = range(0, len(bpe.rules) + 1, 5)

    for num_merges in merge_counts:
        bpe_temp = BPE(merges=num_merges)
        bpe_temp.train(["hello", "world", "test"])  # minimal corpus
        bpe_temp.rules = bpe.rules[:num_merges]  # use learned rules
        bpe_temp.token_to_id = bpe.token_to_id
        bpe_temp.id_to_token = bpe.id_to_token

        try:
            token_ids = bpe_temp.encode(test_text)
            ratio = 100 * len(token_ids) / len(test_text)
            compression_ratios.append(ratio)
            print(f"Merges: {num_merges:3d} | Tokens: {len(token_ids):4d} | Compression: {ratio:.1f}%")
        except:
            pass

    # Plot compression curve
    if compression_ratios:
        plt.figure(figsize=(10, 5))
        plt.plot(list(merge_counts)[:len(compression_ratios)], compression_ratios, 'o-')
        plt.xlabel('Number of Merges')
        plt.ylabel('Compression Ratio (%)')
        plt.title('Compression Ratio vs Training Merges')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Run all demos
    bpe = demo_bpe_training()
    demo_encoding_decoding(bpe)
    demo_token_visualization(bpe)
    demo_embedding_analysis(bpe)
    demo_compression_ablation(bpe)

    print("\n" + "=" * 60)
    print("TOKENIZATION DEMO COMPLETE")
    print("Key insights:")
    print("- BPE learns frequent subword patterns")
    print("- Merges reduce character-level redundancy")
    print("- Learned embeddings capture similarity better than one-hot")
    print("- Token compression improves with more merges (diminishing returns)")
    print("=" * 60)
