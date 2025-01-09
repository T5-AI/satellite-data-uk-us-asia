import ee
import numpy as np
import matplotlib.pyplot as plt
import requests
import rasterio

# Authenticate and initialize Earth Engine
# Initialize Earth Engine
ee.Initialize()

def get_bounding_box(lat, lon, zoom, size=512):
    EARTH_RADIUS = 6378137.0
    INITIAL_RESOLUTION = 2 * np.pi * EARTH_RADIUS / 256.0
    origin_shift = 2 * np.pi * EARTH_RADIUS / 2.0

    resolution = INITIAL_RESOLUTION / (2 ** zoom)

    mx = lon * origin_shift / 180.0
    my = np.log(np.tan((90 + lat) * np.pi / 360.0)) / (np.pi / 180.0)
    my = my * origin_shift / 180.0

    half_size = (size / 2.0) * resolution
    minx = mx - half_size
    maxx = mx + half_size
    miny = my - half_size
    maxy = my + half_size

    def meters_to_latlon(mx, my):
        lon = (mx / origin_shift) * 180.0
        lat = (my / origin_shift) * 180.0
        lat = 180 / np.pi * (2 * np.arctan(np.exp(lat * np.pi / 180.0)) - np.pi / 2.0)
        return lat, lon

    min_lat, min_lon = meters_to_latlon(minx, miny)
    max_lat, max_lon = meters_to_latlon(maxx, maxy)

    return min_lon, min_lat, max_lon, max_lat

# Define parameters
lat, lon = 22.32606067545809, 114.16704408356823  # Center coordinates
zoom = 17
size = 512  # Image size in pixels
bbox = get_bounding_box(lat, lon, zoom, size)

# Define the bounding box as an Earth Engine geometry
region = ee.Geometry.Rectangle([bbox[0], bbox[1], bbox[2], bbox[3]])

# Load the GHSL dataset
ghsl_dataset = ee.ImageCollection("JRC/GHSL/P2023A/GHS_POP").filterBounds(region)

# Select the latest available population data (e.g., 2020)
ghsl_image = ghsl_dataset.filterDate('2020-01-01', '2020-12-31').mean()

# Clip the image to the bounding box
ghsl_cropped = ghsl_image.clip(region)

# Export the cropped GHSL image as a GeoTIFF
url = ghsl_cropped.getDownloadURL({
    'scale': 1,  # Match pixel size (e.g., 1 meter per pixel)
    'region': region.getInfo()['coordinates'],
    'format': 'GeoTIFF'
})
print(f"Download URL: {url}")

response = requests.get(url, stream=True)
output_file = "/Users/wangzhuoyulucas/Library/Mobile Documents/com~apple~CloudDocs/SMART/dataset/GGSL_POP/hk_cropped_ghsl.tiff"
with open(output_file, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print(f"GeoTIFF downloaded as '{output_file}'")

# Open the GeoTIFF file
# Path to your GeoTIFF file
tif_file = "/Users/wangzhuoyulucas/Library/Mobile Documents/com~apple~CloudDocs/SMART/dataset/GGSL_POP/hk_cropped_ghsl.tiff"

# Open the GeoTIFF file
with rasterio.open(tif_file) as src:
    # Read the first band
    band1 = src.read(1)  # Assuming the data is in the first band
    
    # Get metadata
    metadata = src.meta
    print("Metadata:", metadata)
    
    # Get the extent (bounding box) for plotting
    extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)

# Plot the image
plt.figure(figsize=(10, 10))
plt.imshow(band1, cmap='gray', extent=extent)
plt.colorbar(label="Pixel Intensity")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("GeoTIFF Image")
plt.grid(visible=True, linestyle="--", alpha=0.5)
plt.show()
