# New Features Documentation

## 1. Automated Zarr Generation Scheduler

### Overview
A scheduled task system that automatically generates Zarr files at 2:40 AM daily and logs execution time.

### Files Created
- `run_zarr_generation.sh` - Main script that runs the zarr generation process
- `setup_cron.sh` - Helper script to configure the cron job
- `zarr_generation_log.txt` - Log file (created automatically when the script runs)

### Setup Instructions

1. **Make scripts executable** (if not already done):
   ```bash
   chmod +x run_zarr_generation.sh setup_cron.sh
   ```

2. **Run the setup script to configure the cron job**:
   ```bash
   ./setup_cron.sh
   ```

3. **Verify the cron job**:
   ```bash
   crontab -l
   ```

   You should see:
   ```
   40 2 * * * /home/user/HDWI-Visualization-Forcast-Wind-View/run_zarr_generation.sh
   ```

### Manual Execution
You can also run the script manually for testing:
```bash
./run_zarr_generation.sh
```

### Log File Format
The log file (`zarr_generation_log.txt`) contains:
- Start timestamp
- All output from the zarr generation process
- End timestamp
- Execution time (hours, minutes, seconds)
- Exit status (SUCCESS/FAILED)

Example log entry:
```
========================================
Zarr Generation Started: 2026-01-04 02:40:00
========================================
[... script output ...]
Zarr Generation Completed: 2026-01-04 02:45:23
Exit Code: 0
Execution Time: 0h 5m 23s (Total: 323 seconds)
Status: SUCCESS
========================================
```

### Viewing Logs
To view the latest logs:
```bash
tail -f zarr_generation_log.txt
```

To view the last execution:
```bash
tail -n 50 zarr_generation_log.txt
```

---

## 2. Enhanced Area Analysis Features

### Overview
The `/area_analysis/` endpoint has been significantly enhanced with new features for advanced statistical analysis and visualization.

### New Request Parameters

The `AreaRequest` model now supports:

```python
{
    "lat_min": float,
    "lat_max": float,
    "lon_min": float,
    "lon_max": float,
    "variables": List[str],
    "timestamp": str,

    # NEW PARAMETERS
    "begin_time": str,              # Optional: Start time for analysis (ISO format)
    "end_time": str,                # Optional: End time for analysis (ISO format)
    "selected_statistics": List[str]  # Statistics to calculate ["mean", "min", "max", "std", "median"]
}
```

### New Features

#### 1. Custom Time Range Selection
You can now specify custom time ranges for analysis:

**Example Request**:
```json
{
    "lat_min": 35.0,
    "lat_max": 45.0,
    "lon_min": -10.0,
    "lon_max": 5.0,
    "variables": ["ws"],
    "timestamp": "2026-01-04T12:00:00",
    "begin_time": "2026-01-01T00:00:00",
    "end_time": "2026-01-04T23:59:59"
}
```

**Behavior**:
- If both `begin_time` and `end_time` are provided, analyzes that specific range
- If only `begin_time` is provided, analyzes from that time to the latest available
- If only `end_time` is provided, analyzes all data up to that time
- If neither is provided, uses the default 3.5 days from current time

#### 2. Selectable Statistical Features
You can choose which statistics to calculate and visualize:

**Example Request**:
```json
{
    "lat_min": 35.0,
    "lat_max": 45.0,
    "lon_min": -10.0,
    "lon_max": 5.0,
    "variables": ["ws"],
    "timestamp": "2026-01-04T12:00:00",
    "selected_statistics": ["mean", "std", "median"]
}
```

**Available Statistics**:
- `mean` - Average value
- `min` - Minimum value
- `max` - Maximum value
- `std` - Standard deviation
- `median` - Median value

#### 3. Individual Statistical Plots
Each selected statistic now gets its own dedicated plot, perfect for displaying in separate tabs.

**Response Structure**:
```json
{
    "variable_key": {
        "variable_name": "Wind Speed",
        "units": "m/s",
        "stats_current_frame": {
            "mean": 8.5,
            "min": 2.1,
            "max": 15.3,
            "std_dev": 3.2,
            "median": 8.1
        },
        "timeseries_plot": "base64_encoded_image",      # Combined plot
        "boxplot": "base64_encoded_image",              # Distribution plot
        "static_plot": "base64_encoded_image",          # NEW: Current frame heatmap
        "individual_stat_plots": {                       # NEW: Individual plots
            "mean": "base64_encoded_image",
            "std": "base64_encoded_image",
            "median": "base64_encoded_image"
        }
    }
}
```

#### 4. Static Spatial Heatmap
A new visualization showing the spatial distribution of the selected variable at the current timestamp.

Features:
- Color-coded heatmap
- Geographic coordinates (lat/lon)
- Colorbar with units
- Current timestamp in title

### Frontend Integration Guide

#### Example: Creating Tabs for Different Views

```javascript
// Example structure for frontend tabs
const tabs = [
    { name: "Overview", content: data.timeseries_plot },
    { name: "Distribution", content: data.boxplot },
    { name: "Current Frame", content: data.static_plot },
    ...Object.entries(data.individual_stat_plots).map(([stat, img]) => ({
        name: stat.charAt(0).toUpperCase() + stat.slice(1),
        content: img
    }))
];
```

#### Example: Time Range Selector

```javascript
function analyzeArea(bounds, variables, beginTime, endTime, statistics) {
    const request = {
        lat_min: bounds.south,
        lat_max: bounds.north,
        lon_min: bounds.west,
        lon_max: bounds.east,
        variables: variables,
        timestamp: currentTimestamp,
        begin_time: beginTime,
        end_time: endTime,
        selected_statistics: statistics
    };

    fetch('/area_analysis/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
    })
    .then(response => response.json())
    .then(data => {
        // Display results in tabs
        displayInTabs(data);
    });
}
```

#### Example: Statistics Selector

```html
<div class="statistics-selector">
    <label><input type="checkbox" value="mean" checked> Mean</label>
    <label><input type="checkbox" value="min" checked> Minimum</label>
    <label><input type="checkbox" value="max" checked> Maximum</label>
    <label><input type="checkbox" value="std"> Std Dev</label>
    <label><input type="checkbox" value="median"> Median</label>
</div>
```

### Visual Examples

#### Combined Time Series Plot
Shows all selected statistics on one plot with:
- Different colors for each statistic
- Different line styles (solid, dashed, dotted)
- Filled area between min-max if both selected
- Legend and grid

#### Individual Statistical Plots
Each statistic gets a dedicated plot with:
- Larger markers for better visibility
- Focused title showing the specific statistic
- Consistent color scheme
- Time on x-axis, statistic value on y-axis

#### Static Spatial Heatmap
Shows the current frame as a heatmap with:
- Viridis colormap (or any matplotlib colormap)
- Geographic extent matching selected area
- Colorbar showing value range
- Grid overlay

### Best Practices

1. **Time Range Selection**:
   - Use ISO 8601 format for timestamps: `"2026-01-04T12:00:00"`
   - Ensure begin_time < end_time
   - Don't select ranges too large (may cause performance issues)

2. **Statistics Selection**:
   - Default is `["mean", "min", "max"]`
   - Select only the statistics you need to reduce computation
   - For trend analysis, use `mean` and `std`
   - For extreme events, use `min` and `max`

3. **Frontend Display**:
   - Use tabs to organize different plots
   - Display the combined time series plot first
   - Group individual statistical plots together
   - Show the static heatmap for spatial context

### API Examples

#### Basic Analysis (Default Behavior)
```bash
curl -X POST "http://localhost:8000/area_analysis/" \
  -H "Content-Type: application/json" \
  -d '{
    "lat_min": 35.0,
    "lat_max": 45.0,
    "lon_min": -10.0,
    "lon_max": 5.0,
    "variables": ["ws"],
    "timestamp": "2026-01-04T12:00:00"
  }'
```

#### With Custom Time Range
```bash
curl -X POST "http://localhost:8000/area_analysis/" \
  -H "Content-Type: application/json" \
  -d '{
    "lat_min": 35.0,
    "lat_max": 45.0,
    "lon_min": -10.0,
    "lon_max": 5.0,
    "variables": ["ws"],
    "timestamp": "2026-01-04T12:00:00",
    "begin_time": "2026-01-01T00:00:00",
    "end_time": "2026-01-04T23:59:59"
  }'
```

#### With Selected Statistics
```bash
curl -X POST "http://localhost:8000/area_analysis/" \
  -H "Content-Type: application/json" \
  -d '{
    "lat_min": 35.0,
    "lat_max": 45.0,
    "lon_min": -10.0,
    "lon_max": 5.0,
    "variables": ["ws", "dir"],
    "timestamp": "2026-01-04T12:00:00",
    "selected_statistics": ["mean", "std", "median"]
  }'
```

---

## Migration Guide

### For Existing Frontends

The changes are **backward compatible**. Existing API calls will continue to work with default behavior:
- Default time range: 3.5 days from current time
- Default statistics: `["mean", "min", "max"]`

To use new features, simply add the new optional parameters to your requests.

### Response Changes

The response now includes additional fields:
- `stats_current_frame.median` - New median statistic
- `static_plot` - New spatial heatmap
- `individual_stat_plots` - New dictionary of individual plots

Existing fields remain unchanged for backward compatibility.

---

## Technical Details

### Dependencies
No new dependencies required. Uses existing:
- `xarray` - Data manipulation
- `pandas` - Time handling
- `numpy` - Statistics
- `matplotlib` - Plotting

### Performance Considerations
- Individual plots are generated in parallel during the same loop
- Statistics are calculated once and reused
- Caching is maintained at the dataset level
- Time range filtering happens early to reduce data processing

### Error Handling
- Invalid time ranges return appropriate error messages
- Unknown statistics are silently ignored
- Missing data is handled gracefully with NaN filtering
- Each variable is processed independently (one failure doesn't affect others)

---

## Support

For issues or questions:
1. Check the logs in `zarr_generation_log.txt`
2. Verify cron job setup with `crontab -l`
3. Test API endpoints manually with curl
4. Check FastAPI documentation at `http://localhost:8000/docs`
