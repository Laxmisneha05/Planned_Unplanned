import geopandas as gpd
import joblib

# Load the trained model
rf = joblib.load("RF_model.pkl")

# Load a new shapefile
new_gdf = gpd.read_file("Test Data/Test_data.shp")

# Create necessary features
new_gdf['area'] = new_gdf.geometry.area
new_gdf['perimeter'] = new_gdf.geometry.length
new_gdf['centroid_x'] = new_gdf.geometry.centroid.x
new_gdf['centroid_y'] = new_gdf.geometry.centroid.y

# Prepare features
X_new = new_gdf[['area', 'perimeter', 'centroid_x', 'centroid_y']]

# Predict classifications
new_gdf['classify'] = rf.predict(X_new)

# Save the classified shapefile
new_gdf.to_file("RF_Model_output/RF_Area_classified.shp")

print("✅ New shapefile classified and saved as 'RF_Area_classified.shp'.")
