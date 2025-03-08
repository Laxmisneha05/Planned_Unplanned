# Overview

Planned_Unplanned is a spatial analysis tool that evaluates geographic areas provided as shapefiles. By calculating the area and perimeter of each region and assessing the organization of its road network, the tool classifies areas into two categories: **planned** (well-organized roads and infrastructure) and **unplanned** (lacking organized structures). The classification is achieved using two powerful machine learning algorithms: **Random Forest (RF)** and **XGBoost (XGB)**.

---

## Features

- **Shapefile Input:** Process and analyze geographic data in shapefile format.
- **Geometric Calculations:** Compute area and perimeter for each region.
- **Infrastructure Analysis:** Evaluate the organization of road networks.
- **Machine Learning Classification:** Classify regions as planned or unplanned using RF and XGB.
- **Flexible Execution:** Run specific scripts to test either the RF or the XGB model.

---

## Installation

Follow these steps to set up the project locally:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Laxmisneha05/Planned_Unplanned.git
   
2. Navigate into the Project Directory:
   ```bash
   cd Planned_Unplanned

3. Install Dependencies: Install the required libraries using pip:
   ```bash
   pip install -r requirements.txt

--- 

## Machine Learning Models
  1. Random Forest (RF): Utilizes an ensemble of decision trees to capture complex patterns and relationships within the data.
  
  2. XGBoost (XGB):  Leverages boosted decision trees for enhanced prediction accuracy and efficiency.

Both models provide robust insights into the urban planning and organization of geographic areas.

---

### Each script will:

   - Load the shapefile.
   - Calculate the area and perimeter.
   - Analyze the road network.
   - Classify the area as planned or unplanned using the selected machine learning model.


