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

## Task 2 - Text data analysis

The second task concerns the text file `midsummer.txt` which contains the text of William Shakespeare's play *A Midsummer Night's Dream*. The play contains 5 acts, each with 2 scenes, and the task is to investigate them, and in particular to decide how positive or negative each scene is.

To that end, there are two more data files: `positive-words.txt` and `negative-words.txt`, which contain lists of positive and negative words respectively. These words come from the paper *Minqing Hu and Bing Liu. "Mining and summarizing customer reviews." Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, Seattle, Washington, USA, Aug 22-25, 2004*. So whilst the words are not ideal for our purpose (since they are mainly to do with product review sentiments), they still represent an interesting way to examine our scenes.

- You should read in all three of the data files. You will need to split the play in 'midsummer.txt' up so that each scene can be considered individually.
- You need to invent a metric for how positive or negative a given scene is, based on how many of the words in it are in the positive/negative word lists. For instance, is a scene containing one positive and one negative word: overall positive, negative or neutral? - try and develop a single measure based on the word occurrences that will describe the positivity/negativity of the scene.
- Make a plot of the measure you have invented as a y-axis, with scene number as an x-axis.
- When a character starts speaking, their name appears in capitals, on its own line. Which character(s) speak most often?
- Can you use this data to answer any other questions about the data? For instance, could you compare different measures of positivity/negativity, or compare the pattern of positivity/negativity with that found in other plays? (You can find other texts on [the Project Gutenberg website](http://www.gutenberg.org/wiki/Main_Page).)

Note - this is a very simplistic way of doing this kind of text analysis, there are far more complex things that can be done, but I think even the basic approach is cool and can give quite interesting results.

Please include here all the code used to answer this question and generate any plots.

![alt text](https://github.com/mylonakk/PythonProgrammingTask2/blob/master/imgs/t2i1.png)

At the above plot I measure the sign and magnitude of the sentiment of the text of each scene. This measure consists of two steps:
- represent each text with a vector.
- using this vector assign a sentimental value.

I have implemented 3 representation schemes, all of them create vectors that live in the space that the given positive and negative words span $\mathbb{R}^d$, where d is the number of all words in the given dictionaries. Therefore, each text is represented by a vector $v \in \mathbb{R}^d$ and the their values depend on the selected scheme:

- binary: where $v_j = 1$ if the j-th word of the dictionary appears in the text, else $v_j = 0$.
- term frequency (tf): $v_j = \frac{n}{N}$, where n is the number the j-th word appears in the text and N is the number of words in the text.
- term frequency inverse document frequency (tf-idf): $v_j = tf(j)idf(j)$, where $idf(j) = \log\frac{D}{d}$, where D is the number of separate text in the corpus and d is the number of texts the j-th word appears.

Now, since we have a representation vector for the text, we need some decision strategies to assign a sentiment measure to this text according to its representation. To make our lives easier we multiply elementwise each text vector with a vector that has 1 where the corresponding word belongs to the positive dictionary and -1 when it belongs to the negative one. 

So the following decision strategies have been implemented:
- $L_1$-norm, which is essential the sum of the vector elements.
- $L_{\infty}$-norm, which is the max element of the vector.

So, the measures compared in this plot are all the possible combination of the aforementioned representation schemes and decision strategies, except the case of binary with $L_{\infty}$-norm which obviously has no meaning.

The first thing we notice is that binary measure magnitudes are larger than the other measures. Also, the measures binary, tf L1 and tf-idf L1 seem to follow the same trend, with the two latter to be highly correlated.

![alt text](https://github.com/mylonakk/PythonProgrammingTask2/blob/master/imgs/t2i3.png)
![alt text](https://github.com/mylonakk/PythonProgrammingTask2/blob/master/imgs/t2i4.png)

At the top plot, it is illustrated the frequency at which each character speaks in the play. The bottom plot is a correlation matrix for the characters of the play. Each element $A_{ij}$ of the matrix corresponds to the correlation of character i with character j, in terms that one appears to the same scenes that the other one also appears.

![alt text](https://github.com/mylonakk/PythonProgrammingTask2/blob/master/imgs/t2i5.png)

Using exactly the same code, we illustrate the sentimental measure for two other William Shakespeare plays, Hamlet and Romeo and Juliet. It seems that a common pattern is for the schene sentiment to change sign at almost every step.

[Hamlet reference link](http://www.gutenberg.org/ebooks/2265)

[Romeo and Juliet reference link](http://www.gutenberg.org/ebooks/1777)
The motivation for this plot is to identify closely related characters and measure the sentiment of their dialogues.
