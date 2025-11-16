# Text Clustering using K-Means and TF-IDF

### A project to segment the Netflix content library using unsupervised NLP

This project demonstrates a complete unsupervised machine learning workflow for text clustering. It uses **TF-IDF** to vectorize text data and **K-Means Clustering** to segment a large, unstructured content library (the Netflix Movies & TV Shows dataset) into distinct, thematic groups.

## Project Summary

The primary goal of this project is to apply K-Means clustering to a high-dimensional text dataset. The business problem this technique solves is creating a content-based grouping of movies and TV shows from the Netflix catalog based on features like genre, description, director, and cast.

The model successfully segments the entire catalog into 10 distinct clusters, which can be used to power a "similar content" recommendation feature or provide a high-level strategic analysis of the content library's composition.

## Technical Workflow

The project follows a structured NLP and clustering pipeline:

1.  **Feature Engineering:** A `content_soup` feature was created by combining all relevant text columns (director, cast, listed_in, description) into a single document per title.
2.  **Text Vectorization:** The `content_soup` text was converted into a numerical matrix using **TF-IDF (Term Frequency-Inverse Document Frequency)**, limited to the top 25,000 features.
3.  **Dimensionality Reduction:** **TruncatedSVD** was applied to reduce the 25,000-dimension TF-IDF matrix down to 200 components. This captures the most variance while making K-Means computationally efficient and effective.
4.  **Model Training:**
    * The **Elbow Method** was used to determine that **10 clusters (k=10)** was the optimal number.
    * A **K-Means Clustering** model was trained on the reduced 200-component data.
5.  **Model Evaluation:** As an unsupervised task, evaluation was primarily qualitative. The resulting clusters were analyzed and found to be highly coherent and thematically distinct.

## Model Performance & Cluster Analysis

The model's **Silhouette Score** was **0.0xxx** *(fill in from your notebook)*, which is common and expected for high-dimensional, sparse text data.

The true success of the model is in the **qualitative analysis** of the clusters. The K-Means algorithm successfully identified clear, interpretable themes:

* **Cluster 0:** US TV Shows (Dramas & Comedies)
* **Cluster 1:** Stand-Up Comedy
* **Cluster 2:** International Movies (Dramas)
* **Cluster 3:** International Movies (Dramas & Comedies)
* **Cluster 4:** Documentaries
* **Cluster 5:** Indian Movies (Dramas & International)
* **Cluster 6:** Thrillers & Horror Movies
* **Cluster 7:** Kids' TV & Children's Content
* **Cluster 8:** International TV Shows (Dramas & Crime)
* **Cluster 9:** Action & Adventure Movies

## Tech Stack

* **Data Analysis:** Pandas, NumPy, SciPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning / NLP:** Scikit-learn (TfidfVectorizer, TruncatedSVD, KMeans)
* **Model Saving:** Joblib
