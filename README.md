# ReboundingPrediction
A model that can predict who gets a rebound based on the current positions on the floor

This is a temporary ReadMe that will be replaced by end of project. Currently, just project proposal.

## Project Description
I love basketball, and based on a story I once heard about a player, I have had this idea up in my head. I think it's possible to create a workable model for who gets a rebound after a shot in a basketball game, because I think the placement of the shot is one of the most important factors in where the ball goes. Analytics have been heavily applied to shooting, but there are many avenues of the game which have plenty of room to improve in. I think that this model will help change how basketball is played in the future.

## Problem Solved

The goal of the project is to show how predictable rebounds are. I want to use very basic details about the game situation while making these predictions, specifically without taking into account aspects of the players. By approaching the problem from this way, only using positional data and ignoring metadata, we can take a look at how important position is to this aspect of the game.

The model generated by this project will be very useful for the basketball world, specifically for coaches and scouts. If I can create a successful model using this data, coaches can use it to improve their team's rebounding strategy both for individual player rebound coaching, and for how teams should move following shots.

## Presentation

I'm planning on doing 3 things: a ReadMe, a slide show, and a webapp. The ReadMe will be the full story. If possible, there will be an executable jupyter notebook. The slide show will be the cliffs notes on my process, and the results. The webapp will be a joint effort between me and one of the WDI students, where we will create a website that allows you to simulate a shot and rebound situation.

## Data Sources

The two data sources necessary for the model are the SportVU Tracking data from STATS, which I already have, and the game play by plays which I must scrape.

## Data Science Techniques

* *Modeling and Algorithms*: K-means, Logistic Regression, Neural Network, Random Forest, SVM, Matrix Factorization (possibly)

* *Data Skills*: Cleaning, Feature Engineering, Pipeline

* *Hacking Skills*: Webscraping, Regular Expressions, Web Hosting, Cloud Computing (possibly), Map Reduce (possibly), Blue Mix (possibly)

## Next Steps
At this point, I have my process outlined in this order:

#### 1. **Fix data pipelines**
 This involves creating an automatic process to unpack the tracking data, scrape a play by play, sort the plays properly, pull out the right rows from both charts, ending up with a dataframe with all of the original shot and rebound data from both data sets. The files are quite big, so I have to make it reasonably memory efficient. Each game is on average 100 rebounds, so I will have about 1000 datapoints per game.

  I am currently writing a pipeline to scrape a play by play using the file names of the games automatically. To design my single game cleaning pipeline, I'm using one of the games I've pinned down.

  I'm planning on using classes somehow to do this. Similar to our data cleaning package. I'll also have to add a test/train split, which will also go in the cleaning package.

  I need to finish this before the end of the first week.

#### 2. **Feature Engineering** Part 1
  After coordinating the data sets for each game, I need to add in some additional features to show additional aspects that are not directly shown in the positional data, including:

  * How many people they are boxing out
  * What angle and depth the shot is coming from
  * How they moved after the shot
  * How that changed their position and box out

  This will be pretty simple, and I'm going to use K-means clustering to do the boxing out just for fun. I'll be writing a custom class for this.

  I need to finish this by the beginning of the second week.

#### 3. **Building the Model**
  Next, I'll actually have to build the model. I'm going to start, on Ryan's suggestion, with the simplest model: Logistic Regression. I'm going to eliminate the colinear features, keeping the ones I think are most descriptive, and see how it performs. Also, as Ryan proposed, I will try a Neural Network with sigmoids as the smoothers. I also am interested in trying Random Forests and SVMs if there is time.

  I need to be finished with this by the late part of the second week, but I can tinker around later.

#### 4. Feature Engineering Part 2

In order to do the presentation as planned, I'll have to create a second model to simulate the features engineered in Step 2 with the data gathered in Step 1. Some of this will be trivial, just repeating functionality from earlier, but since I only have the initial position, and not the position at time of rebound, I'll have to find a way to model that. Worst case scenario, I can train a slightly dumber version of the main model, which will be trivial.

I need to finish this by the beginning of the third week.

#### 5. Presentation
After that, I create my slides, work on my ReadMe, and practice my talk. At this point, I'll be tinkering with the models to see if I can get the SVM and Random Forests working. Other things to do if I have time is apply the same analysis to NCAA games, which may also be available. See how it compares.
