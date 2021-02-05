import click
import dask
import datetime as dt
import numpy as np
import os
import xarray as xr

from dask import delayed
from dask.diagnostics import ProgressBar

from config import PATH_SEVERE_REPORT, PATH_INVENTORY, PATH_GLM_GRIDS
from util.coordinates import cookiecutter, convert_poly_to_fixedgrid
from grid_severe import add_xy_coords, filter_space
from util.io import make_goes_bucket_path, load_inventory, load_prelim_data, load_poly_data
from util.time import rows_in_date_range

XY_BUFFER = 0.001
MINUTES_BEFORE = 60
START = "2019-05-01"
END = "2019-05-02"
NULL_SUBSAMPLE_N = 200
GLM_VARIABLES = [
    'flash_extent_density',
    'flash_centroid_density',
    'average_flash_area',
    'total_energy',
    'group_extent_density',
    'group_centroid_density',
    'average_group_area',
    'minimum_flash_area'
]

def event_xy_range(event_x, event_y, buffer):
	"""
	Find x,y coordinates of corners around a severe/null event of a given size.
	
	Parameters
	----------
	event_x, event_y: x, y coordinates of event
	buffer: half the length of square box.
	
	Returns
	-------
	(tuple): (xmin, xmax), (ymin, ymax) square coordinates
	"""
    return (event_x - buffer, event_x + buffer), (event_y - buffer, event_y + buffer)


def event_time_before(event_end_time, minutes_before):
	"""
	Get the timestamps of event and X minutes before the event
	
	Parameters
	----------
	event_end_time: time of event
	minutes_before: desired minutes before event
	
	Returns
	-------
	start_time, end_time: Times X minutes before event and event time
	"""
    min_before = dt.timedelta(minutes=minutes_before)
    end_time = event_end_time
    start_time = end_time - min_before
    return start_time, end_time


def get_filenames(df_inv, start, end, glm_abi="glm", path_bucket=PATH_BUCKET):
    # Load GLM files leading up to event
    dir_dict = {"glm": "grid", "abi": "abi"}
    cond1 = df_inv["start_time_glm"] >= start
    cond2 = df_inv["end_time_glm"] <= end
    filenames = df_inv[cond1 & cond2][f"filename_{glm_abi}"]
    start_dates = df_inv[cond1 & cond2][f"start_time_glm"].to_list()
    end_dates = df_inv[cond1 & cond2][f"end_time_glm"].to_list()
    base_dir = os.path.join(path_bucket, dir_dict[glm_abi])
    files = [make_goes_bucket_path(base_dir, x) for x in filenames]
    return files, start_dates, end_dates


def filter_events(df_event, x_range, y_range):
	"""
	Filter severe report data to only those events within a box defined by
	x_range and y_range.
	
	Parameters
	----------
	df_event (DataFrame): Severe report data
	x_range, y_range (float): x and y (fixed grid) lengths of region of interest.
	
	Returns
	-------
	df_event: Severe report containing only those events in region of interest.
	"""
    # Filter severe reports to spatial and time range
    df_event = add_xy_coords(df_event)
    df_event = filter_space(df_event, x_range, y_range)
    return df_event


def preprocess_null(df_poly, start, end, x_range, y_range):
	"""
	Filter polygon data to those events within region of interest
	
	Parameters
	----------
	df_poly (DataFrame): Polygon event data
	start, end (datetime): start and end times to filter by
	x_range, y_range (float): x and y (fixed grid) lengths of region of interest.
	"""
    # reduce to polys within time of interest
    df_poly = rows_in_date_range(df_poly, "ISSUED_dt", start, end)
    # convert lat lon poly centroids to fixed grid
    df_poly = convert_poly_to_fixedgrid(df_poly)
    # filter based on subset region
    df_poly = filter_space(df_poly, xrange=x_range, yrange=y_range)
    print("Total number of severe/tornado warnings:", len(df_poly))
    return df_poly


def make_single_timeseries_df_from_files(files, start_dates, end_dates, x_range, y_range, length):
	"""
	Create timeseries of sum of GLM quantities within event box
	
	files (list): List of GLM grid files
	start_dates, end_dates (datetime): time interval of series
	x_range, y_range: x and y (fixed grid) lengths of event region of interest.
	lenth
	"""
    # Process
    xarrays = []
    for f in files:
    	# Grab GLM pixels around event
        cc = cookiecutter(xr.open_dataset(f), x_range, y_range)
        cc = cc.sum() # Sum over event box
        xarrays.append(cc) # Add to timeseries
    xr_event = xr.concat(xarrays, dim="dim_0") # Stack each time interval value
    df = xr_event[GLM_VARIABLES].to_dataframe().reset_index(drop=True) # convert to DF
    df = df[GLM_VARIABLES] # Keep only GLM variables of interest
    df["start_time"] = start_dates # set times
    df["end_time"] = end_dates
    df.set_index(["start_time", "end_time"], inplace=True)
    return df


def save_timeseries_df(df, episode_id, event_id, event_begin_time, path_output):
    """
    Save timeseries dataframe to file:
    ..../[sevtype]/[glm]/[datetime]_[episode_id]_[event_id].csv
    """
    os.makedirs(path_output, exist_ok=True)
    path_output = os.path.join(
        path_output,
        f"{event_begin_time.strftime('%Y%m%d%H%M%S')}_{episode_id}_{event_id}.csv"
    )
    df.to_csv(path_output)


def make_timeseries(df_event, df_inventory, path_output, xy_buffer, minutes_before):
	"""
	Create timeseries for all severe or null events
	"""
    print(f"{len(df_event)} events")

    # Track IDs for bookkeeping and filename conventions
    episode_ids = df_event["episode_id"].tolist()
    event_ids = df_event["event_id"].tolist()
    event_begin_times = df_event["UTC_event_begin_time"].tolist()
    # X and Y coordinate ranges for each event
    xy_ranges = [
        event_xy_range(lon, lat, buffer=xy_buffer)
        for lon, lat in zip(df_event["fixedlon"], df_event["fixedlat"])
    ]
    # Start and End times for each event
    times = [event_time_before(utc, minutes_before=minutes_before) for utc in event_begin_times]
    # Filenames and file start/end times for each event
    file_info = [get_filenames(df_inventory, t[0], t[1]) for t in times]
    # Ignore events without the correct number of GLM/ABI files to match minute-length
    # *Ought to save file of events that don't meet this criterion
    file_info = [{
        "files": f[0],
        "start": f[1],
        "end": f[2],
        "xrange": x[0],
        "yrange": x[1],
        "episode_id": ep,
        "event_id": ev,
        "time": t
    } for f, x, ep, ev, t in zip(file_info, xy_ranges, episode_ids, event_ids, event_begin_times)]
    file_info = [x for x in file_info if len(x["files"]) == minutes_before]
    print(f"Ignoring events ({len(df_event) - len(file_info)}) "
          "without files for all {minutes_before} minutes before event.")

    # load all files, reduce to area around event, and get timeseries before
    ts_dfs = [
        delayed(make_single_timeseries_df_from_files)(
            fi["files"], fi["start"], fi["end"], fi["xrange"], fi["yrange"], minutes_before
        )
        for fi in file_info
    ]

    # save each timeseries
    ts_dfs = [delayed(save_timeseries_df)(
        x,
        fi["episode_id"],
        fi["event_id"],
        fi["time"],
        path_output
    ) for x, fi in zip(ts_dfs, file_info)]

    with ProgressBar():
        dask.compute(*ts_dfs, scheduler="processes")


@click.command()
@click.option("--path-bucket", default=PATH_BUCKET)
@click.option("--path-inventory", default=PATH_INVENTORY)
@click.option("--path-event", default=PATH_SEVERE_REPORT)
@click.option("--path-output", required=True)
@click.option("--sevtype", default="tornado")
@click.option("--glm-abi", default="glm")
@click.option("--start", type=click.DateTime(formats=["%Y-%m-%d", "%Y%m%d", "%Y%m%d%H"]), default=START)
@click.option("--end", type=click.DateTime(formats=["%Y-%m-%d", "%Y%m%d"]), default=END)
@click.option("--x-range", default=X_RANGE)
@click.option("--y-range", default=Y_RANGE)
@click.option("--xy-buffer", default=XY_BUFFER)
@click.option("--minutes-before", default=MINUTES_BEFORE)
@click.option("--subsample_n", default=NULL_SUBSAMPLE_N)
def main(
    path_bucket,
    path_inventory,
    path_event,
    sevtype,
    glm_abi,
    start,
    end,
    path_output,
    x_range,
    y_range,
    xy_buffer,
    minutes_before,
    subsample_n
):
    df_inv = load_inventory(path_inventory, dropna=["filename_glm", "filename_abi"])
    if sevtype == "null":
        prelim_dropna = False
        prelim_sevtype = "all"
    else:
        prelim_dropna = True
        prelim_sevtype = sevtype

    df_event = load_prelim_data(path_event, sevtype=prelim_sevtype, dropna=prelim_dropna)
    df_event = rows_in_date_range(df_event, "UTC_event_begin_time", start, end)
    df_event = filter_events(df_event, x_range, y_range)

    if sevtype == "null":
        # poly data path
        polypath = os.path.join(
            path_bucket,
            f"grid/severe_data/wwa_{start.year}01010000_{start.year}12312359.shp")
        df_poly = load_poly_data(polypath, sevtype="both")
        df_poly = preprocess_null(df_poly, start, end, x_range, y_range)

        # report dates
        sevdata_dates = np.unique(df_event['UTC_event_begin_time'].dt.date)
        print("total number of report days:", sevdata_dates.shape)
        # polygon dates
        polygon_dates = np.unique(sorted(df_poly['ISSUED_dt'].dt.date))
        print("total number of polygon days:", polygon_dates.shape)

        null_dates = polygon_dates[~np.isin(polygon_dates, sevdata_dates)]
        df_poly = df_poly.loc[df_poly['ISSUED_dt'].dt.date.isin(null_dates)]
        df_poly.rename(columns={"ISSUED_dt": "UTC_event_begin_time"}, inplace=True)
        null_time = df_poly['UTC_event_begin_time']

        print("Total number of null warnings detected based on the predefined thresholds herein:", len(df_poly))

        print(f"Taking a subsample of {subsample_n} rows")
        rand_indx = np.random.permutation(len(null_time))[:subsample_n]
        df_poly = df_poly.iloc[rand_indx].sort_values("UTC_event_begin_time")
        df_event = df_poly

    make_timeseries(
        df_event,
        df_inv,
        path_output,
        xy_buffer=xy_buffer,
        minutes_before=minutes_before
    )


if __name__ == '__main__':
    main()
