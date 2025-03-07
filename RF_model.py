import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and prepare the training shapefile
gdf = gpd.read_file("Training_data/Training_data.shp")

# Feature engineering
gdf['area'] = gdf.geometry.area
gdf['perimeter'] = gdf.geometry.length
gdf['centroid_x'] = gdf.geometry.centroid.x
gdf['centroid_y'] = gdf.geometry.centroid.y

# Split data into labeled data
labeled_gdf = gdf[gdf['classify'].notnull()]

# Features and target
X_train = labeled_gdf[['area', 'perimeter', 'centroid_x', 'centroid_y']]
y_train = labeled_gdf['classify']

# Train the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save the model to a .pkl file
joblib.dump(rf, "RF_model.pkl")

print("✅ Model trained and saved as 'RF_model.pkl'.")
