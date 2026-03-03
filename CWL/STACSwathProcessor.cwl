cwlVersion: v1.0
class: CommandLineTool

label: "STAC Swath Processor"
doc: |
  Downloads L2P swath tiles from a STAC catalog (with IFREMER alternate link
  support), then merges them onto a regular lat/lon grid with quality-aware
  mosaicking and antimeridian handling.

requirements:
  DockerRequirement:
    dockerPull: "images.geomatys.com/stac-swath-processor:latest"

baseCommand: ["python", "/stac_swath_processor.py"]

inputs:
  stac_url:
    type: string
    doc: "Root STAC API URL (e.g. https://stac.ifremer.fr)"
    inputBinding:
      prefix: --stac_url

  collection:
    type: string
    doc: "STAC Collection ID (e.g. AVHRR_SST_METOP_B_OSISAF_L2P_v1_0)"
    inputBinding:
      prefix: --collection

  bbox:
    type: string
    doc: "Bounding box as min_lon,min_lat,max_lon,max_lat (e.g. -70,47,-55,63)"
    inputBinding:
      prefix: --bbox

  temporal_extent:
    type: string
    doc: "Temporal range in ISO 8601 (e.g. 2024-01-01T00:00:00Z/2024-01-02T00:00:00Z)"
    inputBinding:
      prefix: --temporal_extent

  properties:
    type: string?
    doc: "Comma-separated variable names to regrid (e.g. sea_surface_temperature,quality_level). If omitted, all numeric variables are included."
    inputBinding:
      prefix: --properties

  resolution:
    type: float?
    default: 0.01
    doc: "Target grid resolution in degrees (default: 0.01)"
    inputBinding:
      prefix: --resolution

  resample_method:
    type:
      - "null"
      - type: enum
        symbols: [nearest, gauss, custom]
    default: nearest
    doc: "Resampling method: nearest, gauss, or custom/median (default: nearest)"
    inputBinding:
      prefix: --resample_method

  min_quality_level:
    type: int?
    default: 0
    doc: "Minimum quality_level 0-5, 0 disables filtering (default: 0)"
    inputBinding:
      prefix: --min_quality_level

outputs:
  merged_netcdf:
    type: File
    doc: "Merged NetCDF file on a regular lat/lon grid"
    outputBinding:
      glob: "merged_sst.nc"
