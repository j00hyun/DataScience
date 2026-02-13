import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import ast


class AnomalyDetection():
    
    def scaleNum(self, df, indices):
        """
        Standardize numerical features at given indices
        """
        # Make a copy so we don't change the original DataFrame
        df = df.copy()
    
        # Get the "features" column as a list of feature vectors
        feature_list = df["features"].tolist()

        # Loop over each feature index that we want to scale
        for idx in indices:
            # Collect all values at this index from every feature vector
            values = [f[idx] for f in feature_list]

            # Compute the mean (average)
            mean = sum(values) / len(values)

            # Compute the standard deviation
            std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
    
            # Avoid division by zero if all values are the same
            if std == 0:
                std = 1
    
            # Apply standardization to each feature vector
            for f in feature_list:
                f[idx] = (f[idx] - mean) / std

        # Put the scaled features back into the DataFrame
        df["features"] = feature_list

        # Return the updated DataFrame
        return df

    
    def cat2Num(self, df, indices):
        """
        Convert categorical features at given indices into one-hot encoding
        """

        # Make a copy of the DataFrame so the original is not changed
        df = df.copy()

        # Get the "features" column as a list of feature vectors
        feature_list = df["features"].tolist()

        # This dictionary will store category → index mappings
        category_maps = {}

        # Build mapping for each categorical index
        for idx in indices:
            categories = []
            seen = set()

            # Collect unique category values in the order they appear
            for f in feature_list:
                v = f[idx]
                if v not in seen:
                    categories.append(v)
                    seen.add(v)

            # Create mapping: category → position in one-hot vector
            category_maps[idx] = {cat: i for i, cat in enumerate(categories)}

        new_feature_list = []

        # Convert each feature vector
        for f in feature_list:
            new_f = []

            # Go through each value in the feature vector
            for i, val in enumerate(f):

                # If this position is categorical → apply one-hot encoding
                if i in indices:
                    mapping = category_maps[i]

                    # Create one-hot vector
                    one_hot = [0] * len(mapping)
                    one_hot[mapping[val]] = 1

                    # Add one-hot values instead of original value
                    new_f.extend(one_hot)
                else:
                    # Keep numeric values unchanged
                    new_f.append(val)

            new_feature_list.append(new_f)

        # Replace the original features with encoded features
        df["features"] = new_feature_list

        # Return updated DataFrame
        return df


    def detect(self, df, k, t):
        """
        Detect anomalies using K-means clustering (small cluster = anomaly).
        """

        # Make a copy so the original DataFrame is not changed
        df = df.copy()

        # Convert feature lists into a numeric matrix
        X = np.array(df["features"].tolist(), dtype=float)

        # Run K-means clustering
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(X)
        
        # Count how many points are in each cluster
        counts = np.bincount(labels)          

        # Get cluster size for each row
        sizes = counts[labels]                
    
        # Convert cluster size into anomaly score
        # Smaller cluster → larger score
        Nmax, Nmin = sizes.max(), sizes.min()
        
        if Nmax == Nmin:
            score = np.zeros_like(sizes, dtype=float)
        else:
            score = (Nmax - sizes) / (Nmax - Nmin)

        df["score"] = score

        # keep only rows whose score >= t
        out = df[df["score"] >= t]

        # Reset the index to bring "id" back as a column,
        # and return only the id, features, and anomaly score columns
        return out.reset_index()[["id", "features", "score"]]
    
 
if __name__ == "__main__":
    df = pd.read_csv('A5-data/logs-features-sample.csv').set_index('id')

    # Convert string representations of lists in the "features" column into actual Python lists
    df["features"] = df["features"].apply(ast.literal_eval)
    ad = AnomalyDetection()

    df1 = ad.cat2Num(df, [0,1])
    print(df1)

    df2 = ad.scaleNum(df1, [6])
    print(df2)

    df3 = ad.detect(df2, 8, 0.97)
    print(df3)
