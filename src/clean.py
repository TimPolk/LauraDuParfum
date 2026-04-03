import pandas as pd
import ast
import threading
from sklearn.preprocessing import MinMaxScaler

path = "./Data/fra_perfumes.csv"


def encode_gender(df, results):
    # One-hot encodes gender to match multi-hot style of accords/notes
    gender_col = df["Gender"].str.strip()
    encoded = pd.DataFrame(
        {
            "gender_unisex": (gender_col == "for women and men").astype(int),
            "gender_women": (gender_col == "for women").astype(int),
            "gender_men": (gender_col == "for men").astype(int),
        },
        index=df.index,
    )
    results["gender"] = encoded


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
    # (TP) Extracts notes by pyramid level (top, middle, base) instead of flat pooling
    # This allows the clustering model to weight note positions differently
    def extract_pyramid(text):
        """Parses description text and returns a dict with top, middle, and base note lists."""
        top, middle, base = [], [], []

        if not isinstance(text, str):
            return {"top": top, "middle": middle, "base": base}

        text_lower = text.lower()

        # (TP) Match both plural "notes are" and singular "note is" patterns
        # Each pattern maps to its target list
        patterns = [
            ("top notes are ",    top),
            ("top note is ",      top),
            ("middle notes are ", middle),
            ("middle note is ",   middle),
            ("base notes are ",   base),
            ("base note is ",     base),
        ]

        for trigger, target in patterns:
            idx = text_lower.find(trigger)
            if idx == -1:
                continue
            # Grab everything after the trigger up to the next period or semicolon
            after = text_lower[idx + len(trigger):]
            clean_part = after.split('.')[0].split(';')[0]
            items = clean_part.replace(' and ', ',').split(',')
            target.extend([i.strip() for i in items if i.strip()])

        return {"top": top, "middle": middle, "base": base}

    # Extract pyramid for every row
    pyramids = df["Description"].apply(extract_pyramid)

    # Establish a 1% frequency threshold to manage high-dimensional data
    threshold = len(df) * 0.01

    # (TP) Build separate vocabularies for each pyramid level
    # so top_note_lemon and base_note_lemon are distinct columns
    encoded_frames = {}
    for level in ["top", "middle", "base"]:
        all_items = [item for row in pyramids for item in row[level]]
        item_counts = pd.Series(all_items).value_counts()
        valid_vocab = sorted(item_counts[item_counts >= threshold].index.tolist())

        def row_to_binary(pyramid, level=level, vocab=valid_vocab):
            note_set = set(pyramid[level])
            return [1 if v in note_set else 0 for v in vocab]

        encoded = pd.DataFrame(
            pyramids.apply(row_to_binary).tolist(),
            columns=[f"{level}_note_{v}" for v in valid_vocab],
            index=df.index,
        )
        encoded_frames[level] = encoded

    results["top_notes"] = encoded_frames["top"]
    results["middle_notes"] = encoded_frames["middle"]
    results["base_notes"] = encoded_frames["base"]


def main():
    df = pd.read_csv(path)
    df = df.dropna()
    df = df.reset_index(drop=True)

    # (TP) Remove rows with no usable description — can't extract note pyramid without it
    # Catches empty strings and descriptions with no note keywords
    has_notes = df["Description"].str.lower().str.contains("notes are|note is", na=False)
    before = len(df)
    df = df[has_notes].reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with no extractable notes ({len(df)} remaining)")

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

    # (TP) Reassemble with separate top/middle/base note columns instead of flat notes
    clean_df = pd.concat(
        [
            df[["Name", "Rating Value"]],
            results["gender"],
            results["rating_count"].rename("Rating Count"),
            results["top_notes"],
            results["middle_notes"],
            results["base_notes"],
            results["main_accords"],
        ],
        axis=1,
    ).reset_index(drop=True)

    # Drop rows where gender mapping returned NaN 
    clean_df = clean_df[
        clean_df[["gender_unisex", "gender_women", "gender_men"]].any(axis=1)]
    
    scaler = MinMaxScaler()
    clean_df[["Rating Value", "Rating Count"]] = scaler.fit_transform(
        clean_df[["Rating Value", "Rating Count"]]
    )

    print(clean_df.dtypes)
    print(clean_df.head())

    clean_df.to_csv("./Data/fragrance_cleaned.csv", index=False)


if __name__ == "__main__":
    main()