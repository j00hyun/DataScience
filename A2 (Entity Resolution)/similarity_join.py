import re
import pandas as pd

class SimilarityJoin:
    def __init__(self, data_file1, data_file2):
        self.df1 = pd.read_csv(data_file1)
        self.df2 = pd.read_csv(data_file2)
          
    def preprocess_df(self, df, cols):
        """
        Input:
            df: a pandas DataFrame
            cols: a list of column names in df to be concatenated and tokenized

        Output:
            new_df: a new DataFrame that is the same as df, but with an extra column:
                    - joinKey: a list of tokens for each record
        """
        # Make a copy so we do not change the original dataframe
        new_df = df.copy()

        def build_joinkey(row):
            # Join the selected columns into one string
            concat_str = " ".join(
                str(row[c]) if pd.notna(row[c]) else ""
                for c in cols
            )

            # Split the string into tokens and convert to lower-case
            tokens = [t.lower() for t in re.split(r'\W+', concat_str) if t != ""]

            return tokens

        # Apply the function to every row to create joinKey
        new_df["joinKey"] = new_df.apply(build_joinkey, axis=1)

        return new_df


    def filtering(self, df1, df2):
        """
        Input: df1, df2 have a 'joinKey' column (list of tokens) and an 'id' column
        Output: cand_df with columns: id1, joinKey1, id2, joinKey2
                Keep only pairs that share at least one token
        """

        # Build an inverted index for df1
        # token -> list of row indices in df1
        inv_index = {}

        for i, tokens in enumerate(df1["joinKey"]):
            # use set() to avoid duplicate tokens in one record
            for t in set(tokens):
                if t not in inv_index:
                    inv_index[t] = []
                inv_index[t].append(i)

        # Find candidate pairs using df2
        # use set to remove duplicate (i, j) pairs
        cand_pairs = set()

        for j, tokens2 in enumerate(df2["joinKey"]):
            for t in set(tokens2):
                if t in inv_index:
                    for i in inv_index[t]:
                        cand_pairs.add((i, j))

        # Build candidate DataFrame
        rows = []

        for i, j in cand_pairs:
            rows.append({
                "id1": df1.iloc[i]["id"],
                "joinKey1": df1.iloc[i]["joinKey"],
                "id2": df2.iloc[j]["id"],
                "joinKey2": df2.iloc[j]["joinKey"]
            })

        cand_df = pd.DataFrame(rows, columns=["id1", "joinKey1", "id2", "joinKey2"])

        return cand_df


    def verification(self, cand_df, threshold):
        """
        Input:
            cand_df: the output DataFrame from filtering()
                     It has columns: id1, joinKey1, id2, joinKey2
            threshold: float in (0, 1]

        Output:
            result_df: a new DataFrame with columns:
                       id1, joinKey1, id2, joinKey2, jaccard
                       Keep only rows with jaccard >= threshold
        """

        rows = []

        for _, row in cand_df.iterrows():
            tokens1 = row["joinKey1"]
            tokens2 = row["joinKey2"]

            # Jaccard is defined on sets (remove duplicates)
            set1 = set(tokens1)
            set2 = set(tokens2)

            inter = set1.intersection(set2)
            union = set1.union(set2)

            # avoid division by zero (just in case)
            jacc = 0.0 if len(union) == 0 else len(inter) / len(union)

            # keep only pairs whose jaccard is no smaller than threshold
            if jacc >= threshold:
                rows.append({
                    "id1": row["id1"],
                    "joinKey1": row["joinKey1"],
                    "id2": row["id2"],
                    "joinKey2": row["joinKey2"],
                    "jaccard": jacc
                })

        result_df = pd.DataFrame(rows, columns=["id1", "joinKey1", "id2", "joinKey2", "jaccard"])
        return result_df


    def evaluate(self, result, ground_truth):
        """
        Input:
            result: a list of matching pairs found by your ER algorithm
                    e.g., [["a1","g1"], ["a2","g2"], ...]
            ground_truth: a list of true matching pairs (human-labeled)

        Output:
            (precision, recall, fmeasure)
        """

        # Make them sets of tuples so we can compare easily
        # (Also removes duplicates automatically)
        result_set = set(tuple(p) for p in result)
        gt_set = set(tuple(p) for p in ground_truth)

        # True positives = pairs that appear in BOTH result and ground truth
        true_positive = result_set.intersection(gt_set)

        # Compute precision and recall (handle divide-by-zero)
        precision = len(true_positive) / len(result_set) if len(result_set) > 0 else 0.0
        recall = len(true_positive) / len(gt_set) if len(gt_set) > 0 else 0.0

        # Compute F-measure (harmonic mean)
        fmeasure = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return (precision, recall, fmeasure)


    def jaccard_join(self, cols1, cols2, threshold):
        new_df1 = self.preprocess_df(self.df1, cols1)
        new_df2 = self.preprocess_df(self.df2, cols2)
        print ("Before filtering: %d pairs in total" %(self.df1.shape[0] *self.df2.shape[0])) 
        
        cand_df = self.filtering(new_df1, new_df2)
        print ("After Filtering: %d pairs left" %(cand_df.shape[0]))
        
        result_df = self.verification(cand_df, threshold)
        print ("After Verification: %d similar pairs" %(result_df.shape[0]))
        
        return result_df
       
        

if __name__ == "__main__":
    er = SimilarityJoin("Amazon_sample.csv", "Google_sample.csv")
    amazon_cols = ["title", "manufacturer"]
    google_cols = ["name", "manufacturer"]
    result_df = er.jaccard_join(amazon_cols, google_cols, 0.5)

    result = result_df[['id1', 'id2']].values.tolist()
    ground_truth = pd.read_csv("Amazon_Google_perfectMapping_sample.csv").values.tolist()
    print ("(precision, recall, fmeasure) = ", er.evaluate(result, ground_truth))