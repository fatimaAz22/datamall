import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

df = pd.read_csv('Mall_Customers.csv')

print("--- 1. Structure des données et types de variables ---")
print(df.info())

print("\nFeatures:", df.columns.tolist())

print("\n--- 2. Statistiques descriptives ---")
print(df.describe())

print("\n--- 3. Matrice de Corrélation ---")
numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['CustomerID'])
corr_matrix = numeric_df.corr()
print(corr_matrix)

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matrice de Corrélation (Heatmap)')
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['Age'], kde=True, ax=axes[0], color='skyblue').set_title('Distribution de l\'Âge')
sns.histplot(df['Annual Income (k$)'], kde=True, ax=axes[1], color='salmon').set_title('Distribution du Revenu Annuel')
sns.histplot(df['Spending Score (1-100)'], kde=True, ax=axes[2], color='lightgreen').set_title('Distribution du Score de Dépenses')
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(y=df['Age'], ax=axes[0], color='skyblue').set_title('Boxplot de l\'Âge')
sns.boxplot(y=df['Annual Income (k$)'], ax=axes[1], color='salmon').set_title('Boxplot du Revenu Annuel')
sns.boxplot(y=df['Spending Score (1-100)'], ax=axes[2], color='lightgreen').set_title('Boxplot du Score de Dépenses')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', data=df, palette='Pastel1')
plt.title('Répartition par Genre (Variable Catégorielle)')
plt.show()

valeurs_manquantes = df.isnull().sum()
print(valeurs_manquantes)

if valeurs_manquantes.sum() > 0:
    df.dropna(inplace=True)
    print("Valeurs manquantes supprimées.")
else:
    print("Aucune valeur manquante .")

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

colonnes_selectionnees = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
donnees_entree = df[colonnes_selectionnees]

mise_a_echelle = StandardScaler()
donnees_transformees = mise_a_echelle.fit_transform(donnees_entree)

df_final = pd.DataFrame(donnees_transformees, columns=colonnes_selectionnees)

print("Données après Encodage et Standardisation :")
print(df_final.head())

Q1 = df['Annual Income (k$)'].quantile(0.25)
Q3 = df['Annual Income (k$)'].quantile(0.75)
IQR = Q3 - Q1

limite_inferieure = Q1 - 1.5 * IQR
limite_superieure = Q3 + 1.5 * IQR

outliers = df[(df['Annual Income (k$)'] < limite_inferieure) | (df['Annual Income (k$)'] > limite_superieure)]
print("Nombre d'outliers détectés :", len(outliers))

df = df[(df['Annual Income (k$)'] >= limite_inferieure) & (df['Annual Income (k$)'] <= limite_superieure)]

print("Taille du dataset après traitement :", df.shape)

features = ['Annual Income (k$)', 'Spending Score (1-100)']
X_raw = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', color='purple')
plt.title('Méthode du Coude (Elbow Method)')
plt.xlabel('Nombre de clusters (K)')
plt.ylabel('WCSS')
plt.show()

k_optimal = 5
kmeans = KMeans(n_clusters=k_optimal, init='k-means++', n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

plt.figure(figsize=(10, 7))
sns.scatterplot(x=df[features[0]], y=df[features[1]], hue=df['Cluster'], 
                palette='viridis', s=100, data=df)

plt.title('Segmentation des Clients')
plt.xlabel('Revenu Annuel (k$)')
plt.ylabel('Score de Dépenses (1-100)')
plt.legend(title='Cluster')
plt.show()

print(df.shape)
print(df.head())

dbscan = DBSCAN(eps=0.4, min_samples=6)
clusters_dbscan = dbscan.fit_predict(X_scaled)
df['Cluster_DBSCAN'] = clusters_dbscan

cluster_names = {
    -1: "Bruit (Outliers)",
    0: "Revenu Faible / Score Moyen",
    1: "Revenu Moyen / Score Moyen",
    2: "Haut Revenu / Haut Score",
    3: "Revenu Faible / Haut Score",
    4: "Haut Revenu / Faible Score"
}

df['Noms_Clusters'] = df['Cluster_DBSCAN'].map(cluster_names)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=df[features[0]], y=df[features[1]], hue=df['Noms_Clusters'], 
                palette='bright', s=100, data=df)

plt.title('Segmentation des Clients via DBSCAN')
plt.xlabel('Revenu Annuel (k$)')
plt.ylabel('Score de Dépenses (1-100)')

plt.legend(title='Catégories de Clients', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print(df['Cluster_DBSCAN'].value_counts())

plt.figure(figsize=(15, 8))
dendrogram = sch.dendrogram(
    sch.linkage(X_scaled, method='ward'),
    no_labels=True, 
    leaf_rotation=90,
    distance_sort='descending',
    show_leaf_counts=True
)

plt.title('Dendrogramme pour le Clustering Hiérarchique')
plt.xlabel('Clients (Indices regroupés)')
plt.ylabel('Distances Euclidiennes')
plt.axhline(y=7, color='r', linestyle='--')
plt.show()

hc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
clusters_hc = hc.fit_predict(X_scaled)
df['Cluster_HC'] = clusters_hc

hc_names = {
    0: "Score Moyen / Revenu Moyen",
    1: "Score Élevé / Revenu Faible",
    2: "Score Faible / Revenu Élevé",
    3: "Score Élevé / Revenu Élevé",
    4: "Score Faible / Revenu Faible"
}
df['Noms_HC'] = df['Cluster_HC'].map(hc_names)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=df[features[0]], y=df[features[1]], hue=df['Noms_HC'], 
                palette='viridis', s=100, data=df)

plt.title('Résultat du Clustering Hiérarchique (K=5)')
plt.xlabel('Revenu Annuel (k$)')
plt.ylabel('Score de Dépenses (1-100)')
plt.legend(title='Segments de Clients', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

print(df['Cluster_HC'].value_counts())

km = KMeans(n_clusters=5, init='k-means++', random_state=42)
labels_km = km.fit_predict(X_scaled)

agg = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
labels_agg = agg.fit_predict(X_scaled)

db = DBSCAN(eps=0.5, min_samples=5)
labels_db = db.fit_predict(X_scaled)

def get_metrics(labels, name):
    actual_labels = labels[labels != -1]
    actual_X = X_scaled[labels != -1]
    
    if len(set(actual_labels)) > 1:
        return {
            'Algorithme': name,
            'Silhouette (↑)': round(silhouette_score(actual_X, actual_labels), 4),
            'Davies-Bouldin (↓)': round(davies_bouldin_score(actual_X, actual_labels), 4),
            'Calinski-Harabasz (↑)': round(calinski_harabasz_score(actual_X, actual_labels), 4)
        }
    else:
        return {'Algorithme': name, 'Silhouette (↑)': "N/A", 'Davies-Bouldin (↓)': "N/A", 'Calinski-Harabasz (↑)': "N/A"}

metrics_list = [
    get_metrics(labels_km, "K-Means"),
    get_metrics(labels_agg, "Agglomerative (Hiérarchique)"),
    get_metrics(labels_db, "DBSCAN")
]

df_comparaison = pd.DataFrame(metrics_list)

print("\n" + "="*70)
print("   COMPARAISON DES MODÈLES : ÉVALUATION INTERNE")
print("="*70)
print(df_comparaison.to_string(index=False))
print("="*70)

algorithms = ['K-Means', 'Agglomerative', 'DBSCAN']
silhouette_values = [
    get_metrics(labels_km, "")['Silhouette (↑)'],
    get_metrics(labels_agg, "")['Silhouette (↑)'],
    get_metrics(labels_db, "")['Silhouette (↑)']
]

silhouette_values = [0 if v == "N/A" else v for v in silhouette_values]

plt.figure(figsize=(10, 6))
colors = ['#3498db', '#9b59b6', '#2ecc71']
bars = plt.bar(algorithms, silhouette_values, color=colors)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, yval, ha='center', va='bottom', fontweight='bold')

plt.title('Comparaison de la Qualité du Clustering (Silhouette Score)', fontsize=14)
plt.xlabel('Algorithmes', fontsize=12)
plt.ylabel('Score de Silhouette', fontsize=12)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

hc_names = {
    0: "Score Moyen / Revenu Moyen",
    1: "Score Élevé / Revenu Faible",
    2: "Score Faible / Revenu Élevé",
    3: "Score Élevé / Revenu Élevé",
    4: "Score Faible / Revenu Faible"
}

df_pca_final = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca_final['Segment'] = pd.Series(labels_agg).map(hc_names)

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='PC1', 
    y='PC2', 
    hue='Segment', 
    data=df_pca_final, 
    palette='viridis', 
    s=120,
    alpha=0.8,
    edgecolor='w'
)

plt.title('Évaluation Visuelle Final : Projection PCA des Clusters (Agglomerative)', fontsize=16, pad=20)
plt.xlabel('Composante Principale 1 (PC1)', fontsize=12)
plt.ylabel('Composante Principale 2 (PC2)', fontsize=12)

plt.legend(title='Profils des Clients', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("ANALYSE DE LA VISUALISATION PCA :")
print("="*80)
print("- Cohésion : Les points d'un même segment sont bien regroupés.")
print("- Séparation : Les 5 groupes sont distincts, confirmant la validité du clustering.")
print("- Interprétation : La PCA confirme visuellement la logique des métriques calculées.")
print("="*80)
