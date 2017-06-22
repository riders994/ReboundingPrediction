# NBA Rebounding Model

### Rohan Vahalia, Galvanize G38DS Austin

## Overview

This project was inspired by a story about Dennis Rodman, who was one of the greatest rebounders in NBA history.

QUOTE HERE

Since players like Rodman and, more recently, Tristan Thompson are able to study rebounding and derive useful information, I wanted to see what would happen when machine learning was applied. I modeled rebounds based solely on where people are on the court, and how they move, and was able to successfully identify a rebounder 85% of the time.

This model can be used as a scouting and coaching tool, for evaluating player's ability to anticipate movement, and outperform their positioning.

## Data Sources

* NBA.com Play by Play API for play by play
* STATS SportVU tracking data for positional data ([github](https://github.com/riders994/BasketballData/tree/master/2016.NBA.Raw.SportVU.Game.Logs))

## Technology

* Python
  * Pandas and Numpy for data processing
  * SciKit Learn Logistic Regression and Random Forest Classifier
  * Keras Neural Networks for classification and regression
  * Flask for WebApp
* Javascript (@HT44)
  * D3 for visuals
  * AJAX for requests
* AWS for web hosting

## Data Cleaning

### SportVU

I started out with json files after extraction. Each row has a list of player + ball positions, as one of the columns, as well as time. I then extracted the ball's position and made that part of the row. Next, I used the balls height to determine when in time it started rising to pick out when shooting motions started. Then, I identified when the ball would approach the rim to use as a reference point between the two datasets. I kept the player position data enclosed in the dataframe as it would unnecessarily increase the size of the file.

### Play by Play

This was much easier to process, as the misses are properly labeled. I found the rows that indicated individual, rather than team, rebounds, and created a dataframe that was just the play by play readout for the missed shot and the rebound.


### Pairing

Since the shot is reported on the play by play as soon as it is counted as a make or miss, I used that time signature as the basis for searching. I would search the SportVU data for the point where the ball was near the rim, and closest to the time where the miss was registered. Then, I searched upward of that to find where the shooting motion started. After isolating both the pre and post positions, I created a dataframe that, for each shot, contained the time signatures of the shot and rebound as well each player's position at both moments, where each player was a single row. I also added ID numbers for the shots themselves so I could pick out a single shot easily.

I wrote a group of scripts using classes that randomly picked some number (400) of games to set up so it would be easy to randomly generate data for future validation.

## Modeling

### Initial Modeling

I started with the most basic model for this kind of task: Logistic Regression. Out of the box, it did not do much better than guessing 0 for every player, since for any given shot there is only 1 player who gets it, and 9 who don't. In order to overcome this, I tried artificially balancing the classes, as well as playing around with different weights, and it definitely improved.

I then played around with Random Forest classifiers, and a Neural Network classifier, and eventually decided on the random forest as the model to use. I wrote a quick custom function to score the shots by predicting each player's probability, and then seeing if the player with the highest probability was the actual rebounder. Running all my models (4 each of LR and RF with different weights, and 2 neural networks), the Random Forest with artificial weights performed the best. Logistic Regression topped out between 78 - 82% correct predictions, while Random Forest gave 85 - 90% on test sets, more accurate than on the training set.

### Secondary Modeling

My model takes in how players are oriented on the floor while the shot is in the air, and when it hits the rim. But, on the user interface for the web application, they can only put in the initial position of players. So, in order to get the app working, I had to model how the players moved on the court, and where they would move to. I ended up using a Dense Neural Network to do this, in Keras. It was largely successful, but this process needs to be refined for sure.

## Results

While the model was only able to predict an individual rebounder correctly a few times (10 out of 5000 in training set), when feeding in all 10 players' data for a shot, the player who has the highest predicted probability was correct 85% of the time. This shows that positioning is an incredibly important aspect of rebounding.

It was also interesting to look at the players who consistently out-performed the model, grabbing boards in situations they were predicted not to. The vast majority of them are known as "not great" players who are very good at a couple aspects of the game.

As for players who the model favored, and were unable to grab rebounds they were predicted for, they tended to be older or unskilled. To me, this says that the older players know instinctively how to move on the floor to grab the rebound, but are not physically able to.


Thank you for reading! I will be updating this repo with executable jupyter notebooks, as well as some scripts that you can run to take a look on your own.

Thanks again to Hayden (@HT44) for his help on the front-end of the webapp!
