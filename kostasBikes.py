import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt


def import_data():
    """
    Imports data from csv
    """
    data = pd.read_csv('partA/bikes_October18.csv', ',')
    return data


def plot_dur_dist(data):
    """
    Plots average duration and distance of bike trips on Tuesdays
    """
    sumDist = 0  # counter for distance
    sumDur = 0  # counter for duration
    n = 0   # counter for Tuesday bike journeys
    tuesdays = [i for i in range(2, 31, 7)]
    for index, row in data.iterrows():
        if int(row['started_at'].split(' ')[0].split('-')[2]) in tuesdays:
            sumDist += calc_distance(row['start_station_latitude'], row['start_station_longitude'],
                                          row['end_station_latitude'], row['end_station_longitude'])
            sumDur += row['duration']
            n += 1

    print('The average distance on Tuesdays is ' + str(sumDist / n) + 'km')
    print('The average duration on Tuesdays is ' + str(sumDur / n) + 'sec')


def calc_distance(start_lat, start_lng, end_lat, end_lng):
    """
    Distance in kilometers between the two points in Lat/Lng coordinate system.
    https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude/43211266
    """
    # approximate radius of earth in km
    r = 6373.0

    # Convert to radians
    lat1 = math.radians(start_lat)
    lon1 = math.radians(start_lng)
    lat2 = math.radians(end_lat)
    lon2 = math.radians(end_lng)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return r * c


def station_analysis(data):
    """
    Gets all stations and find most popular ones.
    """
    unique_stations = list(set(data['start_station_name'].tolist() + data['end_station_name'].tolist()))

    station_counter = {station : 0 for station in unique_stations}
    for index, row in data.iterrows():
        station_counter[row['start_station_name']] += 1

    print('List of all stations:')
    print(unique_stations)

    keys = list(station_counter.keys())
    vals = list(station_counter.values())
    indexArr = np.argsort(list(station_counter.values()))
    popularStations = []
    for i in reversed(indexArr):
        popularStations.append((keys[i], vals[i]))

    stations1, journeys = zip(*popularStations[0:10])
    plt.bar(stations1, journeys, 0.1)

    plt.xticks(stations1, rotation='vertical')
    plt.title('Popular stations')
    plt.xlabel('Station names')
    plt.ylabel('Journeys')

    plt.show()
    return station_counter


def common_times_journeys(data):
    """
    Plots the most common times to start a journey
    """
    hours = [0 for i in range(24)]
    for index, row in data.iterrows():
        curHour = int(row['started_at'].split(' ')[1].split(':')[0])
        hours[curHour] += 1

    plt.plot(range(24), hours)
    plt.xticks(range(24))
    plt.title('Peak hours')
    plt.xlabel('Hours')
    plt.ylabel('Journeys')
    plt.show()


def station_map(data, station_counter):
    """
    Plots station on the map.
    https://stackoverflow.com/questions/16266809/convert-from-latitude-longitude-to-x-y
    """
    r = 6373.0

    phi = np.cos(55.943894)

    stations = list(set(data['start_station_name'].tolist() + data['end_station_name'].tolist()))
    dict = {name: () for name in stations}
    for index, row in data.iterrows():
        dict[row['start_station_name']] = (row['start_station_latitude'], row['start_station_longitude'])
        dict[row['end_station_name']] = (row['end_station_latitude'], row['end_station_longitude'])
    # number of stations
    n = len(dict.keys())

    # Convert to XY coordinates
    dict_xy = {ids: (station_counter[ids], (r * dict[ids][1] * phi, r * dict[ids][0])) for ids in dict.keys()}

    journeys, loc = zip(*list(dict_xy.values()))
    x, y = zip(*list(loc))
    journeys = [i for i in journeys]

    fig, ax = plt.subplots(figsize=(13, 8))
    plt.scatter(x, y, s=journeys)

    for ii in dict_xy.keys():
        ax.annotate(ii, dict_xy[ii][1], dict_xy[ii][1])

    plt.show()


data = import_data()

# Q1
plot_dur_dist(data)
# Q2
station_counter = station_analysis(data)
# Q3
common_times_journeys(data)
# Q4
station_map(data, station_counter)

