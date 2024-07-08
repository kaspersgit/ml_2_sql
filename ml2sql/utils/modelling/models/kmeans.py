import random
from sklearn.cluster import KMeans


def save_kmeans_model_to_sql(kmeans_model, file_path):
    """
    Save a trained K-Means model as an SQL file.

    Args:
        kmeans_model (KMeans): A trained K-Means model from scikit-learn.
        file_path (str): The path to save the SQL file.
    """
    n_features = kmeans_model.cluster_centers_.shape[1]
    # n_clusters = kmeans_model.n_clusters
    centroids = kmeans_model.cluster_centers_
    feature_names = getattr(kmeans_model, "feature_names", None)

    # Save the SQL code to a file
    with open(file_path, "w") as file:
        # Create table for cluster centroids
        print("CREATE TABLE cluster_centroids (", file=file)
        print("    cluster_id INT PRIMARY KEY,", file=file)
        for i in range(n_features):
            feature_name = feature_names[i] if feature_names else f"centroid_x{i+1}"
            print(f"    {feature_name} FLOAT,", file=file)
        print(");", file=file)
        print("", file=file)

        # Insert cluster centroids
        print("INSERT INTO cluster_centroids (cluster_id, ", file=file, end="")
        for i in range(n_features):
            feature_name = feature_names[i] if feature_names else f"centroid_x{i+1}"
            print(f"{feature_name}, ", file=file, end="")
        print(") VALUES", file=file)
        for i, centroid in enumerate(centroids):
            print(f"    ({i}, {', '.join(str(val) for val in centroid)}),", file=file)
        print(";", file=file)
        print("", file=file)

        # Create temporary table for squared distances
        print("CREATE TEMPORARY TABLE squared_distances (", file=file)
        print("    data_id INT,", file=file)
        print("    cluster_id INT,", file=file)
        print("    squared_distance FLOAT", file=file)
        print(");", file=file)
        print("", file=file)

        # Precompute squared distances
        print(
            "INSERT INTO squared_distances (data_id, cluster_id, squared_distance)",
            file=file,
        )
        print("SELECT", file=file)
        print("    st.id AS data_id,", file=file)
        print("    cc.cluster_id,", file=file)
        squared_distance = ""
        for i in range(n_features):
            feature_name = feature_names[i] if feature_names else f"x{i+1}"
            centroid_name = feature_names[i] if feature_names else f"centroid_x{i+1}"
            squared_distance += (
                f"    POW(sd.{feature_name} - cc.{centroid_name}, 2) +\n"
            )
        squared_distance = (
            squared_distance[:-3] + " AS squared_distance\n"
        )  # Remove the trailing " +\n"
        print(squared_distance, file=file, end="")
        print("FROM", file=file)
        print("    <source_table> st", file=file)
        print("    CROSS JOIN cluster_centroids cc;", file=file)
        print("", file=file)

        # Create temporary table for cluster assignments
        print("CREATE TEMPORARY TABLE cluster_assignments (", file=file)
        print("    data_id INT,", file=file)
        print("    cluster_id INT", file=file)
        print(");", file=file)
        print("", file=file)

        # Assign data points to nearest centroids based on squared distances
        print("INSERT INTO cluster_assignments (data_id, cluster_id)", file=file)
        print("SELECT", file=file)
        print("    sd.data_id,", file=file)
        print("    (SELECT cluster_id", file=file)
        print("     FROM squared_distances sd2", file=file)
        print("     WHERE sd2.data_id = sd.data_id", file=file)
        print("     ORDER BY squared_distance", file=file)
        print("     LIMIT 1)", file=file)
        print("FROM", file=file)
        print("    squared_distances sd;", file=file)
        print("", file=file)

        # Query to get cluster assignments
        print("-- Query to get the cluster assignments", file=file)
        print("SELECT", file=file)
        for i in range(n_features):
            feature_name = feature_names[i] if feature_names else f"x{i+1}"
            print(f"    st.{feature_name},", file=file)
        print("    ca.cluster_id", file=file)
        print("FROM", file=file)
        print("    <source_table> st", file=file)
        print("    JOIN cluster_assignments ca ON st.id = ca.data_id", file=file)
        print("ORDER BY", file=file)
        print("    ca.cluster_id;", file=file)

    print(f"SQL file saved to {file_path}")


# Generate demo data with 1000 records and 5 variables
X = [[random.uniform(-10, 10) for _ in range(5)] for _ in range(1000)]
feature_names = ["var1", "var2", "var3", "var4", "var5"]
kmeans = KMeans(n_clusters=5, random_state=42).fit(X)
kmeans.feature_names = feature_names
save_kmeans_model_to_sql(kmeans, "kmeans_model.sql")
