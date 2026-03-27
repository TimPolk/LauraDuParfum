import matplotlib.pyplot as plt


def plot_top_binary_columns(df, prefix, top_n=15, title="Top Items"):
    # safety buffer if prefix is valid but not lowercase
    prefix = prefix.lower()
    if prefix != "note_" and prefix != "accord_":
        print("ERROR: Choose either 'note_' or 'accord_'")
        return
    # Target only the specific multi-hot encoded columns
    cols = [c for c in df.columns if c.startswith(prefix)]
    item_counts = df[cols].sum().sort_values(ascending=False).head(top_n)

    # Clean up the labels for graph
    clean_labels = [label.replace(prefix, "").capitalize() for label in item_counts.index]

    plt.figure(figsize=(10, 6))
    y_positions = range(len(item_counts) - 1, -1, -1)

    plt.barh(y_positions, item_counts.values, color='steelblue')
    plt.yticks(y_positions, clean_labels)

    plt.title(f"{title} (Top {top_n})")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.show()

