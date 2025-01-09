import ee
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from io import BytesIO
from matplotlib import cm

class GeoDataProcessor:
    def __init__(self, api_key):
        self.api_key = api_key
        ee.Authenticate()
        ee.Initialize()

    def get_bounding_box(self, lat, lon, zoom, size=512):
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

    # Function to download Google Static Map
    def download_google_static_map(api_key, lat, lon, zoom, size, output_image_file):
        """
        Download Google Static Map for a given location and save it as an image.
        """
        base_url = "https://maps.googleapis.com/maps/api/staticmap"
        params = {
            "center": f"{lat},{lon}",
            "zoom": zoom,
            "size": f"{size}x{size}",
            "maptype": "satellite",
            "key": api_key,
        }

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            with open(output_image_file, "wb") as f:
                f.write(response.content)
                print(f"Google Static Map saved to {output_image_file}")
        else:
            print("Failed to download Google Static Map:", response.content)
    

    def resample_tiff_to_size(self, src_path, dst_size):
        with rasterio.open(src_path) as src:
            data = src.read(
                1,  # First band
                out_shape=(dst_size[1], dst_size[0])  # New height, width
            )
            transform = src.transform * src.transform.scale(
                (src.width / dst_size[0]),
                (src.height / dst_size[1])
            )
        return data, transform

    def apply_colormap(self, band, alpha=0.6):
        normalized_band = (band - np.min(band)) / (np.max(band) - np.min(band))
        colormap = cm.viridis(normalized_band)
        colormap[:, :, -1] = alpha  # Set transparency
        return (colormap * 255).astype(np.uint8)

    def process_data(self, lat, lon, zoom, size, tiff_output, google_map_output, final_output):
        # Calculate bounding box
        bbox = self.get_bounding_box(lat, lon, zoom, size)
        region = ee.Geometry.Rectangle([bbox[0], bbox[1], bbox[2], bbox[3]])

        # Fetch GeoTIFF from Earth Engine
        ghsl_dataset = ee.ImageCollection("JRC/GHSL/P2023A/GHS_POP").filterBounds(region)
        ghsl_image = ghsl_dataset.filterDate('2020-01-01', '2020-12-31').mean()
        ghsl_cropped = ghsl_image.clip(region)
        url = ghsl_cropped.getDownloadURL({
            'scale': 30,
            'region': region.getInfo()['coordinates'],
            'format': 'GeoTIFF'
        })

        # Download GeoTIFF
        response = requests.get(url, stream=True)
        with open(tiff_output, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Resample GeoTIFF
        resampled_band, _ = self.resample_tiff_to_size(tiff_output, (size, size))

        # Fetch Google Map image
        self.download_google_static_map(lat, lon, zoom, size, google_map_output)

        google_map_image = Image.open(google_map_output)
        
        # Apply colormap and transparency
        colored_overlay = Image.fromarray(self.apply_colormap(resampled_band))
        overlay_image = Image.alpha_composite(
            google_map_image.convert("RGBA"), colored_overlay.convert("RGBA")
        )

        # Save the final image
        overlay_image.save(final_output)
        overlay_image.show()

# Example usage
if __name__ == "__main__":
    api_key = 
    lat, lon = 51.48210658285678, -0.19386476311346104
    zoom = 17
    size = 512

    processor = GeoDataProcessor(api_key)
    processor.process_data(
        lat,
        lon,
        zoom,
        size,
        "/Users/wangzhuoyulucas/SMART /images/population_density.tiff",
        "/Users/wangzhuoyulucas/SMART /images/google_static_map.png",
        "/Users/wangzhuoyulucas/SMART /images/output_overlay_plot.png"
    )
