import pandas as pd
import src.constants as const


def preprocessing(input_df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows that are inconsistent (double label) and duplicate texts."""

    df = input_df.copy()
    # Compute the number of words in the text column
    df["nb_words"] = df[const.TEXT_COLUMN].apply(lambda x: len(x.split()))
    # Compute number of labels for the same text
    df["nb_labels"] = df.groupby(const.ID)[const.TARGET].transform("nunique")
    # Keep rows that have only one label
    df = df[df["nb_labels"] == 1].reset_index()
    # Remove duplicates for the title+sentence
    df = df.drop_duplicates(subset=[const.ID]).reset_index()

    return df
