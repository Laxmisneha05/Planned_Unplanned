import geopandas as gpd
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder

# Load shapefile
gdf = gpd.read_file("Test Data/mumbai_classified.shp")

# Add features
gdf['area'] = gdf.geometry.area
gdf['perimeter'] = gdf.geometry.length
gdf['centroid_x'] = gdf.geometry.centroid.x
gdf['centroid_y'] = gdf.geometry.centroid.y

# Split into labeled and unlabeled data
labeled_gdf = gdf[gdf['classify'].notnull()]
unlabeled_gdf = gdf[gdf['classify'].isnull()]

# Encode target labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(labeled_gdf['classify'])

# Features
X_train = labeled_gdf[['area', 'perimeter', 'centroid_x', 'centroid_y']]

# Create XGBoost model
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=10,
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False
)

# Train the model
xgb_model.fit(X_train, y_train)

# Save the model and label encoder
joblib.dump(xgb_model, "xgboost_classification_model.pkl")
joblib.dump(label_encoder, "xgb_label_encoder.pkl")

print("✅ XGBoost model and label encoder trained and saved!")
