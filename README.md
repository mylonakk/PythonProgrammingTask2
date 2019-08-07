# PythonProgrammingTask2

## Task 1 - Edinburgh cycle hire data
The first task is to investigate the dataset in the file `bikes_October18.csv` which contains anonymised data for all trips taken in October 2018 with the Edinburgh Cycle Hire bikes (found [here](https://edinburghcyclehire.com/open-data/historical)). This dataset is kindly supplied for use by Edinburgh Cycle Hire under the [Open Government License (OGL) v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

- What is the average journey time and distance of bike trips which started on a Tuesday?
- How many stations appear in the dataset? Which were the most common starting stations? Think about how to display/plot the results of your analysis.
- What was the most common time(s) of day for journeys to be undertaken?
- Can you process the geographic information (latitudes and longitudes of start and end stations), and display this in an interesting way?
- Can you use the data to answer any other interesting questions about the bike trips?

Please include all the code used to generate your answer and plots, and remember to investigate beyond the initial scope of the questions.

## Visual Results

![alt text](https://github.com/mylonakk/PythonProgrammingTask2/blob/master/imgs/download.png)

The above plot gives us an estimate about how much time do the people in Edinburgh spend on cycling depending on the day. The median duration during weekends seems to be higher than the other days at about 25 mins, which is reasonable. Moreover, during Saturday and Sunday 25% of the journeys take over 50 minutes and we see that maximum reaches 100 mins for Sunday and about 80 min for Saturday.

![alt text](https://github.com/mylonakk/PythonProgrammingTask2/blob/master/imgs/t1i2.png)

This scatter plot purpose is to illustrate the "speed" of each day. Each colored dot represents a different day (see the day-color map in legend). The x coordinate of each dot is equal to the average distance in km cyclists covered on the particular day of the month, while the y coordinate is equal to the average duration of the journeys. Therefore, essential we can derive the speed of each day. The first thing to notice is the gap in duration between the weekend and the other days. Secondly, we have low speed on Sunday (red), Saturday (black) is a bit faster but still slow. It is clear that Monday (green) is the fastest day, that's why we hate it.

![alt text](https://github.com/mylonakk/PythonProgrammingTask2/blob/master/imgs/t1i3.png)

Briefly the steps I followed to produce the above heat map are the following:
- find all the stations in the dataset
- convert (approximately) station location from latitude/longitude to Cartesian
- consider as traffic either arrival or departure
- quantize time keeping only hours (truncate mins)
- keep rectangle defined by the most north, west, south and east station
- make a grid of that rectangle
- create a 3D matrix, where each xy slice is a snapshot in time and xy is the rectangle (therefore we have 24 slices)
- Run over the whole dataset and add 1 to the cell corresponding to the arrival and departure station (x,y) of each row at corresponding time (z)
- now each slice of the matrix is sparse with non-zero values only at some stations, but since I have quantize the time to hours only, so I have to diffuse some of the traffic around the stations, therefore I apply gaussian blur to each time snapshot.

![alt text](https://github.com/mylonakk/PythonProgrammingTask2/blob/master/imgs/t1i8.png)

It is possible to represent the data as a directed graph, where each node is a different bike station, an edge from node A to node B represents the journey from station A to station B and the weight of that edge is the times this journey appears in the day. To extract some valuable information, I have plotted the adjacency matrix for this graph for weekends and for the rest of the days separately.

![alt text](https://github.com/mylonakk/PythonProgrammingTask2/blob/master/imgs/t1i9.png)

People generally start cycling from 7am to 9pm during weekend, with most of the journeys starting during the time span 11am to 3pm. During the other days of the week people start from one hour earlier than at 6am and keep starting journeys till around 10 or 11 pm. The two distribution seem of similar shape with two key differences, first during weekend peak start hour is 3pm while the other's distribution peak is at 5pm. Second, at the working days distribution there is a sudden increase at 6am obviously because of people cycling to work.
