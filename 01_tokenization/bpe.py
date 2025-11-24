
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

class BPE:
    def __init__(self, merges=50):
        self.merges = merges
        self.rules = []
        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.merge_history = []

    def stats(self, words):
        pairs = Counter()
        for w in words:
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += 1
        return pairs

    def merge_words(self, words, pair):
        a, b = pair
        token = a + b
        new = []
        for w in words:
            i = 0
            out = []
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == pair:
                    out.append(token)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            new.append(out)
        return new

    def train(self, corpus):
        """Train BPE on corpus and build vocabulary"""
        words = [list(w) for w in corpus]
        self.vocab = {}

        for _ in range(self.merges):
            p = self.stats(words)
            if not p:
                break
            pair = max(p, key=p.get)
            self.rules.append(pair)
            self.merge_history.append((pair, p[pair]))
            words = self.merge_words(words, pair)

        # Build vocabulary from final words
        for word in words:
            for token in word:
                self.vocab[token] = self.vocab.get(token, 0) + 1

        # Create token<->id mappings
        self.token_to_id = {token: i for i, token in enumerate(sorted(self.vocab.keys()))}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

        return self.rules

    def encode(self, text):
        """Encode text to token IDs"""
        # First split into characters
        tokens = list(text)

        # Apply merge rules
        for a, b in self.rules:
            merged = a + b
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == a and tokens[i + 1] == b:
                    tokens = tokens[:i] + [merged] + tokens[i + 2:]
                else:
                    i += 1

        # Convert to IDs
        return [self.token_to_id.get(t, -1) for t in tokens if t in self.token_to_id]

    def decode(self, token_ids):
        """Decode token IDs back to text"""
        tokens = [self.id_to_token.get(tid, '') for tid in token_ids]
        return ''.join(tokens)

    def plot_vocab_stats(self):
        """Plot vocabulary frequency distribution"""
        if not self.vocab:
            print("No vocabulary built. Run train() first.")
            return

        tokens = sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)[:20]
        names, freqs = zip(*tokens)

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(names)), freqs)
        plt.xticks(range(len(names)), [str(n)[:20] for n in names], rotation=45, ha='right')
        plt.ylabel('Frequency')
        plt.title('Top 20 Tokens by Frequency')
        plt.tight_layout()
        plt.show()

    def plot_merge_history(self):
        """Plot merge frequency over training steps"""
        if not self.merge_history:
            print("No merge history. Run train() first.")
            return

        frequencies = [freq for _, freq in self.merge_history]

        plt.figure(figsize=(10, 5))
        plt.plot(frequencies, alpha=0.7)
        plt.xlabel('Merge Step')
        plt.ylabel('Pair Frequency')
        plt.title('BPE Merge Frequency Over Training')
        plt.grid(True, alpha=0.3)
        plt.show()

    def token_ids_to_viz(self, text, max_len=50):
        """Create visualization of token IDs for a given text"""
        token_ids = self.encode(text)[:max_len]
        tokens = [self.id_to_token.get(tid, '?') for tid in token_ids]

        print(f"Text: {text[:100]}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Compression ratio: {len(text)} chars -> {len(token_ids)} tokens ({100*len(token_ids)/len(text):.1f}%)")

        # Visualize
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

        # Token ID sequence
        ax1.bar(range(len(token_ids)), token_ids, color='steelblue')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Token ID')
        ax1.set_title(f'Token ID Sequence (length={len(token_ids)})')
        ax1.grid(True, alpha=0.3, axis='y')

        # Token names
        colors = plt.cm.tab20(np.linspace(0, 1, len(set(token_ids))))
        color_map = {tid: colors[i % len(colors)] for i, tid in enumerate(set(token_ids))}
        ax2.bar(range(len(tokens)), [1]*len(tokens),
                color=[color_map[tid] for tid in token_ids])
        ax2.set_xticks(range(len(tokens)))
        ax2.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Token')
        ax2.set_title('Tokens by Position')
        ax2.set_ylim(0, 1.5)

        plt.tight_layout()
        plt.show()
