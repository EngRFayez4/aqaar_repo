# app.py
from flask import Flask, render_template, request
import pandas as pd
import os
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

app = Flask(__name__)

# Replace settings.STATIC_ROOT with the directory for your static files
STATIC_ROOT = 'path_to_your_static_folder'

# Google Maps API Function to get district name
def get_district_name(lat, lng, api_key):
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={api_key}&language=ar"
    response = requests.get(geocode_url)
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
        if results:
            for component in results[0]['address_components']:
                if 'sublocality_level_1' in component['types'] or 'administrative_area_level_2' in component['types']:
                    return component['long_name']
    return "Unknown"

# Main function to process the recommendation
def pre(lat1, lng1, lat2, lng2, budget):
    # Define locations
    work_location = [lat1, lng1]
    wife_location = [lat2, lng2]
    user_price_per_meter = budget

    # Read CSV files from static directory
    merged_df_path = os.path.join(STATIC_ROOT, 'data', 'merged_df.csv')
    schools_df_path = os.path.join(STATIC_ROOT, 'data', 'final_schools_data.csv')
    mosques_df_path = os.path.join(STATIC_ROOT, 'data', 'mousq_final_data.csv')
    districts_df_path = os.path.join(STATIC_ROOT, 'data', 'final_district_data.csv')

    df = pd.read_csv(merged_df_path)
    schools_df = pd.read_csv(schools_df_path)
    mosques_df = pd.read_csv(mosques_df_path)
    districts_df = pd.read_csv(districts_df_path).drop(columns=['Unnamed: 0'])

    # Haversine function to calculate distance
    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        r = 6371  # Radius of Earth in kilometers
        return c * r

    # Fit and scale the neighborhood data
    X_mean = df['Mean'].values.reshape(-1, 1)
    scaler = StandardScaler()
    X_mean_scaled = scaler.fit_transform(X_mean)

    knn_mean = NearestNeighbors(n_neighbors=5).fit(X_mean_scaled)
    user_input_scaled = scaler.transform([[user_price_per_meter]])
    distances_mean, indices_mean = knn_mean.kneighbors(user_input_scaled)

    candidate_neighborhoods_mean = df.iloc[indices_mean[0]]

    # Calculate distances and nearby amenities
    candidate_neighborhoods_mean['Distance_to_Work'] = candidate_neighborhoods_mean.apply(
        lambda row: haversine(row['longitude'], row['latitude'], work_location[1], work_location[0]), axis=1)
    
    candidate_neighborhoods_mean['Distance_to_Wife'] = candidate_neighborhoods_mean.apply(
        lambda row: haversine(row['longitude'], row['latitude'], wife_location[1], wife_location[0]), axis=1)

    def count_nearby_schools(lon, lat):
        return sum(haversine(lon, lat, slon, slat) <= 3 for slat, slon in zip(schools_df['latitude'], schools_df['longitude']))

    def count_nearby_mosques(lon, lat):
        return sum(haversine(lon, lat, mlon, mlat) <= 3 for mlat, mlon in zip(mosques_df['latitude'], mosques_df['longitude']))

    candidate_neighborhoods_mean['Nearby_Schools'] = candidate_neighborhoods_mean.apply(
        lambda row: count_nearby_schools(row['longitude'], row['latitude']), axis=1)

    candidate_neighborhoods_mean['Nearby_Mosques'] = candidate_neighborhoods_mean.apply(
        lambda row: count_nearby_mosques(row['longitude'], row['latitude']), axis=1)

    # Sort by combined score
    candidate_neighborhoods_mean['Combined_Score'] = (candidate_neighborhoods_mean['Distance_to_Work'] +
                                                      candidate_neighborhoods_mean['Distance_to_Wife']) / 2
    candidate_neighborhoods_mean = candidate_neighborhoods_mean.sort_values(by='Combined_Score')

    # Select columns for the result
    recommended_neighborhoods = candidate_neighborhoods_mean[['District', 'Distance_to_Work', 'Distance_to_Wife', 'Nearby_Schools', 'Nearby_Mosques']].head(2)

    return recommended_neighborhoods

# Route for the homepage
@app.route('/')
def index():
    return render_template('test.html')

# Route to process the form submission
@app.route('/process_form', methods=['POST'])
def process_form():
    if request.method == 'POST':
        budgets = request.form.get('budget')
        your_location = request.form.get('your_location')
        relative_location = request.form.get('relative_location')

        your_lat, your_lng = map(float, your_location.split(','))
        rel_lat, rel_lng = map(float, relative_location.split(','))

        # Get the recommended neighborhoods
        recommended_neighborhoods_df = pre(lat1=your_lat, lng1=your_lng, lat2=rel_lat, lng2=rel_lng, budget=budgets)

        # Convert DataFrame to HTML table
        results_html = recommended_neighborhoods_df.to_html(classes='table table-striped', index=False, justify='center')

        return render_template('results.html', results_html=results_html)

if __name__ == '__main__':
    app.run(debug=True)
