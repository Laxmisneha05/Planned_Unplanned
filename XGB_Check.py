import geopandas as gpd
import joblib

# Load model and label encoder
xgb_model = joblib.load("xgboost_classification_model.pkl")
label_encoder = joblib.load("xgb_label_encoder.pkl")

# Load new shapefile
new_gdf = gpd.read_file("Test Data/Test_data.shp")

# Create features
new_gdf['area'] = new_gdf.geometry.area
new_gdf['perimeter'] = new_gdf.geometry.length
new_gdf['centroid_x'] = new_gdf.geometry.centroid.x
new_gdf['centroid_y'] = new_gdf.geometry.centroid.y

# Predict
X_new = new_gdf[['area', 'perimeter', 'centroid_x', 'centroid_y']]
predictions = xgb_model.predict(X_new)
new_gdf['classify'] = label_encoder.inverse_transform(predictions)

# Save the output
new_gdf.to_file("XGBoost_Model_Output/XGB_Area_classified.shp")

print("✅ New shapefile classified with XGBoost and saved!")
