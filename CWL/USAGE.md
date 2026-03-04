Usage:
```bash
python stac_swath_processor.py \
--stac_url <stac_url> \
--collection <collection_id> \
--bbox="<bbox>" \
--temporal_extent "<temporal_extent>" \
--properties "<list_of_variables>" \
--resolution <resolution> \
--resample_method <resample_method> \
--min_quality_level <min_quality_level> \
```

Where:
- `<stac_url>`: The URL of the STAC API to query.
- `<collection_id>`: The ID of the collection to query.
- `<bbox>`: The bounding box for the area of interest, formatted as "min_lon,min_lat,max_lon,max_lat".
- `<temporal_extent>`: The temporal extent for the query, formatted as "start_date/end_date" (e.g., "2020-01-01/2020-12-31").
- `<list_of_variables>`: A comma-separated list of variables to retrieve (e.g., "temperature,precipitation").
- `<resolution>`: The desired grid resolution in degree for the output data (e.g., 0.01).
- `<resample_method>`: The method to use for resampling the data (e.g., "nearest", "gauss", "custom" (median)).
- `<min_quality_level>`: The minimum quality_level (0-5) to use from source data (from the quality mask). 0 disables filtering.

Example usage:
```bash
python stac_swath_processor.py \
  --stac_url https://stac-pg-api.ifremer.fr \
  --collection AVHRR_SST_METOP_B_OSISAF_L2P_v1_0 \
  --bbox="-70,47,-55,63" \
  --temporal_extent "2024-06-01T00:00:00Z/2024-06-02T00:00:00Z" \
  --properties "sea_surface_temperature" \
  --resolution 0.01 \
  --resample_method nearest \
  --min_quality_level 3
```

```bash
python stac_swath_processor.py \
  --stac_url https://stac-pg-api.ifremer.fr \
  --collection AVHRR_SST_METOP_B_OSISAF_L2P_v1_0 \
  --bbox="-70,47,-55,63" \
  --temporal_extent "2024-06-01T00:00:00Z/2024-06-02T00:00:00Z" \
  --properties "sea_surface_temperature" \
  --resolution 0.01 \
  --resample_method nearest \
  --min_quality_level 3 \
  --keep_time
```