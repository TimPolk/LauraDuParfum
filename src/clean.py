import pandas as pd
import ast
import threading


path = "./Data/fra_perfumes.csv"


def encode_gender(df, results):
    # Converts categorical to nominal
    # 0 = unisex, 1 = for women, 2 = for men

    gender_map = {
        "for women and men": 0,
        "for women": 1,
        "for men": 2,
    }
    col = df["Gender"].str.strip().map(gender_map)
    results["gender"] = col.astype("int64")


def encode_rating_count(df, results):
    # Removes commas if any value > 999
    col = (
        df["Rating Count"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype("int64")
    )
    results["rating_count"] = col


def multi_hot_main_accords(df, results):
    # Parses the string representation of each accord list into an actual list
    parsed = df["Main Accords"].apply(
        lambda x: ast.literal_eval(x)
         if isinstance(x, str) else []
    )

    # Builds a complete vocab set of every unique accord across the dataset
    # so every possible accord is guaranteed a column when encoding
    vocab_set = set()
    for accords in parsed:
        for accord in accords:
            vocab_set.add(accord.strip())
    vocab = sorted(vocab_set)

    # Binary encodes each row against the fixed vocab
    # 1 if the perfume has that accord, 0 if not
    def row_to_binary(accords):
        accord_set = {a.strip() for a in accords}
        binary = []
        for accord in vocab:
            if accord in accord_set:
                binary.append(1)
            else:
                binary.append(0)
        return binary

    encoded = pd.DataFrame(
        parsed.apply(row_to_binary).tolist(),

        columns=[f"accord_{v}" for v in vocab],

        index=df.index,
    )

    results["main_accords"] = encoded


def multi_hot_notes(df, results):
    def extract_from_description(text):
        if not isinstance(text, str):
            return []
        
        notes_found = []
        # Standardize text for easier searching
        text_lower = text.lower()
        
        # Look for the section after 'notes are' and before the next category or period
        parts = text_lower.split("notes are ")
        
        for part in parts[1:]:  # Skip the text before the first 'notes are'
            # Stop at a period or a semicolon
            clean_part = part.split('.')[0].split(';')[0]
            # Split by commas and 'and'
            items = clean_part.replace(' and ', ',').split(',')
            notes_found.extend([i.strip() for i in items if i.strip()])
            
        return notes_found

    # Create a Series where each row contains a list of all notes for that perfume
    all_notes_series = df["Description"].apply(extract_from_description)

    # Establish a 1% frequency threshold to manage high-dimensional data
    threshold = len(df) * 0.01

    # Flatten the list of lists to count total occurrences of every unique note
    all_items = [item for sublist in all_notes_series for item in sublist]
    item_counts = pd.Series(all_items).value_counts()

    # Create the final vocabulary: only keep notes that meet the threshold
    valid_vocab = sorted(item_counts[item_counts >= threshold].index.tolist())

    # Inner helper function to convert a list of notes into a binary vector (1s and 0s)
    def row_to_binary(notes):
        note_set = set(notes)
        return [1 if v in note_set else 0 for v in valid_vocab]

    encoded = pd.DataFrame(
        all_notes_series.apply(row_to_binary).tolist(),
        columns=[f"note_{v}" for v in valid_vocab], 
        index=df.index,
    )

    results["notes"] = encoded


def main():
    df = pd.read_csv(path)
    df = df.dropna()
    df = df.reset_index(drop=True)

    # Remove gender from the name of the fragrance
    labels = ["for women and men", "for women", "for men"]
    for label in labels:
        df["Name"] = df["Name"].str.replace(label, "", regex=False).str.strip()

    # Shared dict, no race conditions
    results = {}

    threads = [
        threading.Thread(target= encode_gender, args=(df, results)),
        threading.Thread(target= encode_rating_count, args=(df, results)),
        threading.Thread(target= multi_hot_main_accords, args=(df, results)),
        threading.Thread(target=multi_hot_notes, args=(df, results)),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Reassemble
    clean_df = pd.concat(
        [
            df[["Name", "Rating Value"]],
            results["gender"].rename("Gender"),
            results["rating_count"].rename("Rating Count"),
            results["notes"],
            # No more 'Main Accords' column cause of multi-hot encoding
            results["main_accords"],
        ],
        axis=1,
    ).reset_index(drop=True)

    # Drop rows where gender mapping returned NaN 
    clean_df = clean_df.dropna(subset=["Gender"])

    print(clean_df.dtypes)
    print(clean_df.head())

    clean_df.to_csv("./Data/fragrance_cleaned.csv", index=False)

if __name__ == "__main__":
    main()
