import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.ndimage import gaussian_filter


# Global Day Dictionary/Colormap
day_dict = {0: ('Sun.', 'r'), 1: ('Mon.', 'g'), 2: ('Tues.', 'b'), 3: ('Wed.', 'c'), 4: ('Thurs.', 'm'),
            5: ('Fri.', 'y'), 6: ('Sat.', 'k')}


def setup_data():
    """
    Imports data from csv, applies filters to remove outliers and manipulated data to facilitate further processing.
    :return: DataFrame holding the filtered/enhanced dataset.
    """
    data = pd.read_csv('partA/bikes_October18.csv', ',')

    # Outlier Removal
    data = remove_outliers(data)

    # Add Column to indicate the day of bike hire
    date = [int(i.split(' ')[0].split('-')[2]) for i in data['started_at'].tolist()]
    day = [i % 7 for i in date]
    data['day'] = day  # Add new col to dataFrame

    # Compute journey distance and add as a column
    distance = [haversine_distance((row['start_station_latitude'], row['start_station_longitude']),
                                   (row['end_station_latitude'], row['end_station_longitude']))
                for index, row in data.iterrows()]
    data['distance'] = distance

    # Calculate and store duration in mins
    duration_mins = [row['duration'] / 60 for index, row in data.iterrows()]
    data['duration_mins'] = duration_mins

    return data


def haversine_distance(p1, p2):
    """
    Computes the distance (in Km) between two points given their coordinates in Lat/Lng format. Uses Haversine formula.
    https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude/43211266
    :param p1: Tuple in (lat, lng) format for the 1st point
    :param p2: Tuple in (lat, lng) format for the 2nd point
    :return: Distance between the two points.
    """
    # approximate radius of earth in km
    r = 6373.0

    # Convert to radians
    lat1 = math.radians(p1[0])
    lon1 = math.radians(p1[1])
    lat2 = math.radians(p2[0])
    lon2 = math.radians(p2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return r * c


def remove_outliers(dirty_data):
    """
    Applies a set of filters to the data to remove outliers and extreme values.
    :param dirty_data: DataFrame holding the "dirty" data.
    :return: DataFrame holding the filtered data.
    """
    # Time filter - each daily bike journey cannot exceed 24*60*60 seconds
    data = dirty_data.drop(dirty_data[dirty_data.duration > 24 * 60 * 60].index)
    return data


def plot_daily_duration_distribution(data):
    """
    Make a BoxPlot of the duration of the hire to illustrate the distribution of journey duration each day.
    :param data: DataFrame holding all the data
    """

    # Plot daily duration distribution
    fig1, ax1 = plt.subplots()

    ax1.set_title('Journey Duration Distribution per day')  # Add plot title
    # Plot duration in mins
    ax1.boxplot([data.loc[data['day'] == i]['duration_mins'].tolist() for i in range(0, 7)], showfliers=False)
    ax1.set_xticklabels([val[0] for val in day_dict.values()], rotation='vertical')  # Add X-axis tick labels
    plt.ylabel('Duration (mins)')
    plt.show()


def scatter_duration_distance(data):
    """
    Creates scatter plot for duration (x) and distance (y) for all journeys in the dataset. Each color indicates a
    different day of the week.
    :param data: DataFrame holding all the data
    """
    distances = data['distance'].tolist()
    durations = data['duration'].tolist()
    colors = [day_dict[i][1] for i in data['day'].tolist()]

    plt.scatter(distances, durations, c=colors, s=2)
    plt.show()


def scatter_avg_duration_distance(data):
    """
    Creates scatter plot for average duration (x) and distance (y) for all journeys in the dataset.
    Each color indicates a different day of the week.
    :param data: DataFrame holding all the data
    """
    for i in range(0, 7):
        avg_duration = data.loc[data['day'] == i]['duration_mins'].mean()
        avg_distance = data.loc[data['day'] == i]['distance'].mean()

        plt.scatter(avg_distance, avg_duration, c=day_dict[i][1], label=day_dict[i][0])

    plt.title("Average Distance/Duration for each day")
    plt.xlabel("Average Distance (km)")
    plt.ylabel("Average Duration (min)")
    plt.legend()
    plt.show()


def get_stations(data):
    """
    Creates a dictionary (station_id as key) holding information (name, location, traffic) for all the stations in the
    dataset. Regarding the traffic, the number of bikes departing from a station in hourly base is saved as well as
    number of bikes departing and arriving in hourly base.
    :param data: DataFrame holding all the data
    :return: Dictionary of type {station_id: (station_name, (station_lat, station_lng), traffic arrive & depart [1 X 24]
    , traffic depart [1 X 24], traffic depart weekend [1 X 24])}
    """
    # Get all the station ids that are present in the dataset
    station_ids = list(set(data['start_station_id'].tolist() + data['end_station_id'].tolist()))

    station_dict = {}
    # Add Name - Location for each station
    for station_id in station_ids:
        if station_id in data['start_station_id']:
            lat_lng_name = data.loc[data['start_station_id'] == station_id][['start_station_latitude',
                                                                             'start_station_longitude',
                                                                             'start_station_name']].iloc[0].tolist()

        else:
            lat_lng_name = data.loc[data['end_station_id'] == station_id][['end_station_latitude',
                                                                           'end_station_longitude',
                                                                           'end_station_name']].iloc[0].tolist()

        station_dict[station_id] = (lat_lng_name[2], (lat_lng_name[0], lat_lng_name[1]), np.zeros(24), np.zeros(24),
                                    np.zeros(24))

    # Add hourly traffic for each station
    for index, row in data.iterrows():

        start_station_id = row['start_station_id']  # get station id
        start_time = int(row['started_at'].split(' ')[1].split(':')[0])  # get start time of journey

        # Add journey to traffic table
        start_station_val = station_dict[start_station_id]
        start_station_val[2][start_time] += 1

        if (row['day'] == 0) or (row['day'] == 6):
            start_station_val[4][start_time] += 1
        else:
            start_station_val[3][start_time] += 1

        end_station_id = row['end_station_id']  # get station id
        end_time = int(row['ended_at'].split(' ')[1].split(':')[0])  # get start time of journey

        # Add journey to traffic table
        end_station_val = station_dict[end_station_id]
        end_station_val[2][end_time] += 1

    return station_dict


def edinburgh_heat_map(stations):
    """
    Approximately converts station locations from lat/lng to XY and plots them. Then creates a heat map to illustrate
    the bike traffic density for different hours of the day.
    https://stackoverflow.com/questions/16266809/convert-from-latitude-longitude-to-x-y
    :param stations: Dictionary of type {station_id: (station_name, (station_lat, station_lng), traffic arrive & depart
    [1 X 24], traffic depart [1 X 24])}
    :return A tuple holding the traffic spatial time 3d matrix and a list of tuples of the form (name, x, y) where name
    is the name of the station and x, y is the location of the station on the XY grid.
    traffic_spatial_time: 3d matrix, where x, y is spatial grid and z is time (specifically 24, one for each
    time of the day.
    """
    # approximate radius of earth in km
    r = 6373.0

    # reference angle for Edinburgh
    phi = np.cos(55.943894)

    # number of stations
    n = len(stations.keys())

    # Convert Lat/Lng coordinates to XY coordinates
    station_dict_xy = {station_id: (stations[station_id][0], (r * stations[station_id][1][1] * phi,
                                                              r * stations[station_id][1][0]))
                       for station_id in stations.keys()}

    # Shift points chose to coordinate origin
    x, y = zip(*[el[1] for el in station_dict_xy.values()])
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    station_dict_xy = {station_id: (station_dict_xy[station_id][0], (station_dict_xy[station_id][1][0] - mean_x,
                                                                     station_dict_xy[station_id][1][1] - mean_y))
                       for station_id in station_dict_xy.keys()}

    # Create traffic spatial map 3d - Each element indicates arrivals or departure in location (x, y) at time z
    x = x - mean_x
    y = y - mean_y
    granularity = 5
    im_x = math.ceil((max(x) - min(x)) / granularity)
    im_y = math.ceil((max(y) - min(y)) / granularity)

    traffic_spatial_time = np.zeros((im_x, im_y, 24))
    min_x = min(x)
    min_y = min(y)

    label_x_y = []
    for station_id in station_dict_xy.keys():
        # Find x, y for current station in traffic map
        cur_x = math.floor((station_dict_xy[station_id][1][0] - min_x) / granularity)
        cur_y = math.floor((station_dict_xy[station_id][1][1] - min_y) / granularity)

        label_x_y.append((stations[station_id][0], cur_x, cur_y))
        traffic_spatial_time[cur_x, cur_y, :] = stations[station_id][2]

    # Apply Gaussian filter (blur) for each time slice of the traffic_spatia_time matrix
    for h in range(24):
        traffic_spatial_time[:, :, h] = gaussian_filter(traffic_spatial_time[:, :, h], sigma=5)

    return traffic_spatial_time, label_x_y


def plot_edinburgh_heat_map(traffic_spatial_time, label_x_y, hour):
    """
    Plots the heat map for the given hour of the day.
    :param traffic_spatial_time: 3d matrix, where x, y is spatial grid and z is time (specifically 24, one for each
    time of the day.
    :param label_x_y: list of tuples of the form (name, x, y) where name is the name of the station and x, y is the
    location of the station on the XY grid
    :param hour: Integer specifying the hour of the day the plot will be about [0, 23]
    """
    # number of stations
    n = len(label_x_y)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(traffic_spatial_time[:, :, hour].transpose(), origin='lower')

    plt.title('Bike traffic at around %d:00' % hour)

    font = {'size': 8}
    plt.rc('font', **font)

    for ii in range(n):
        ax.annotate(label_x_y[ii][0], (label_x_y[ii][1], label_x_y[ii][2]))

    plt.show()


def station_graph_plot(stations, data):
    """
    Creates and plots the adjacency matrix, each station is a separate node and each edge is a journey.
    :param stations: Dictionary of type {station_id: (station_name, (station_lat, station_lng), traffic arrive & depart
    [1 X 24], traffic depart [1 X 24])}
    :param data: DataFrame holding all the data
    """
    station_names = [stations[station_id][0] for station_id in stations.keys()]

    # Get journeys made in Monday to Friday
    data_week = data.loc[(data['day'] == 0) & (data['day'] != 6)]
    # Get journeys made during the weekend
    data_weekend = data.loc[(data['day'] == 0) | (data['day'] == 6)]

    # Compute adjacency matrices for Week days and Weekend
    A_week = get_adjacency_matrix(data_week, station_names)
    A_weekend = get_adjacency_matrix(data_weekend, station_names)

    plot_adjacency_matrix(A_week, station_names, 'Adjacency matrix from Monday to Friday')
    plot_adjacency_matrix(A_weekend, station_names, 'Adjacency matrix for the Weekends')


def get_adjacency_matrix(data, nodes):
    """
    Creates the adjacency matrix
    :param data: DataFrame holding all the data
    :param nodes: List of strings as node labels (n)
    :return: Adjacency matrix
    """
    n = len(nodes)
    A = np.zeros((n, n))  # Init adjacency matrix

    for index, row in data.iterrows():

        start_station_name = row['start_station_name']  # get start station name
        end_station_name = row['end_station_name']  # get end station name
        start_index = nodes.index(start_station_name)
        end_index = nodes.index(end_station_name)
        A[start_index, end_index] += 1

    return A


def plot_adjacency_matrix(A, nodes, title):
    """
    Plots the adjacency matrix.
    :param A: Adjacency matrix (n -by- n)
    :param nodes: List of strings as node labels (n)
    :param title: Title for the plot
    """
    n = len(nodes)
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(A)

    # Show all ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    # Node labels
    ax.set_xticklabels(nodes)
    ax.set_yticklabels(nodes)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(A[i, j]), ha="center", va="center", color="w")

    ax.set_title(title)
    plt.show()


def plot_journey_start_time(stations):
    """
    Plot the absolute number of journeys started each hour of the day.
    :param stations: Dictionary of type {station_id: (station_name, (station_lat, station_lng), traffic arrive & depart
    [1 X 24], traffic depart [1 X 24])}
    """

    journey_per_hour = np.array([stations[station_id][3] for station_id in stations.keys()]).sum(axis=0)
    journey_per_hour = journey_per_hour / sum(journey_per_hour)

    journey_per_hour_weekend = np.array([stations[station_id][4] for station_id in stations.keys()]).sum(axis=0)
    journey_per_hour_weekend = journey_per_hour_weekend / sum(journey_per_hour_weekend)

    # Plot daily duration distribution
    fig1, ax1 = plt.subplots()
    ax1.set_xticks(np.arange(24))

    ax1.set_title('Journeys start per hour')  # Add plot title

    plt.plot(journey_per_hour, label='Monday to Friday')
    plt.plot(journey_per_hour_weekend, label='Weekend')


    plt.legend()
    plt.show()


# Main procedure
bike_data = setup_data()

# BoxPlots to illustrate the distribution of daily journey duration
plot_daily_duration_distribution(bike_data)

# Avg Scatter plot
scatter_avg_duration_distance(bike_data)

# Get station information
station_dict = get_stations(bike_data)

# Plot Bike stations on map
traffic_spatial_time, label_x_y = edinburgh_heat_map(station_dict)
plot_edinburgh_heat_map(traffic_spatial_time, label_x_y, 8)
plot_edinburgh_heat_map(traffic_spatial_time, label_x_y, 12)
plot_edinburgh_heat_map(traffic_spatial_time, label_x_y, 17)
plot_edinburgh_heat_map(traffic_spatial_time, label_x_y, 20)

# Plot station graph
station_graph_plot(station_dict, bike_data)

# What was the most common time(s) of day for journeys?
plot_journey_start_time(station_dict)
