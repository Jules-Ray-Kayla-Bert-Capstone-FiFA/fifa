#### Group Project By: Jules Morris, Eriberto Contreras, Raymond Cerna, and Kayla Brock | Codeup | Jemison Cohort | July 27, 2022

# _FIFA_

#### A project using linear regression to predict FIFA player contract amount

<hr style="border-top: 2px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

# Project Overview & EXECUTIVE SUMMARY 

<hr style="border-top: 1px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

#### _Project Goal_

The goal of this report is to create a linear regression model that can accurately estimate a player's contract amount within a 10% margin of error. This information can be used by investors to estimate the cost of starting and/or maintaining a team.

#### _Description_

_"Soccer, known as football outside of North America, is the 'beautiful game' to its multitude of fans. Soccer is played in every corner of the world and inspires its own frenzy every four years with the convening of the international championship known as the World Cup. There is no competition on earth that mobilizes such passion or that is as inclusive as the qualification for this event. The World Cup is entirely controlled through the international governing body for the sport of soccer, FIFA, the French language acronym for the Federation Internationale de Football Association. A testament to the status of soccer as the world's most popular sport. FIFA is unquestionably the most powerful and all-encompassing world sports body."_

       - https://www.encyclopedia.com/sports/sports-fitness-recreation-and-leisure-magazines/fifa-world-cup-soccer

This project aims to uncover the main attributes that influence FIFA player's contract amount. 

#### _Project Motivation:_

_In this section each member of the team discusses their reason why they chose this project._

#### Ray

_I was interested in this project because I think it sets a strong foundation for a financial type forecasting analysis. I know Football/Soccer is a subject a lot of individuals are very familiar with. In fact, it is the most popular sport in the entire world which makes it easily relatable. Also, since this was affiliated to financial structure and attributes of an individual, place, and team, I believed I may find a connection to utilize what we discover in future projects._

#### Eriberto

_This project peaked my interest because it gave me the opportunity to challenge my data science skills in a subject matter that I had no experience with prior. I hope to gain insight on to how teams budget their players and on how the wages range based on skillsets. I am excited to work on a data set that comes from such globally known sport and organization as big as FIFA. I hope this project will capture the attention of Data Scientist and FIFA fans and give them a fun learning experience._



#### Kayla 

_As a former teacher in central Phoenix, I can promise you 'professional soccer player' was the dream of at least 70% of my male students. While not everyone grows up to be the rich and famous Lionel Messi, I truly believed many of my students had what it took to make money playing the sport they loved. I am most excited about the opportunity to share this report with my former colleagues. I know the kids will love this!_ 

#### Jules

_I chose this project because I am passionate about one of the most intense, inclusive, fast-paced, and exhilarating games on Earth. I have been a fan of the game since I was young, and was excited about the possibility of combining some of the worlds most-famous players with my love of machine learning and predictive algorithms. This is a project that will capture the imagination of everyone who has a love of the game, the. players, or data science._

#### _Initial Questions_

- Does the league a player is with impact salary?
- Does the seniority a player has on a team impact salary?
- Does age impact salary?

#### _Executive Summary_

Overall:

RMSE Baseline: €20154.67

The features that were included in the overall model were:

International Reputation
Overall
Reactions
Potential
Test RMSE: €11095.90, which marked a 44.95% improvement from the original RMSE of this dataset.


Forwards {insert hyperlink to in-depth analysis}:

RMSE Baseline: €21889.30

The features included in this model:

Overall
Shooting
Ball control
Test RMSE: €12579.66, which marked a 42.53% improvement from the original RMSE of this dataset.

In this subset, predictive capacity did not show the same improvement as the inclusive overall model, it was beneficial for forwards to remain in the original dataset.

Midfielders:

RMSE Baseline: €21721.47

The features included in this model:

International reputation
Overall
Passing
Test RMSE: €11015.47, which marked a 49.29% improvement from the original RMSE of this dataset.


Defenders:

RMSE Baseline: €19629.66

The features included in this model:

Overall
Defending
Test RMSE: €10524.00, which marked a 46.39% improvement from the original RMSE of this dataset.


Goalkeepers:

RMSE Baseline: €15905.59

The features included in this model:

Overall
Goal keeping reflexes
Test RMSE: €9126.68, which marked a improvement 42.62% from the original RMSE of this dataset.
    
The exploration process revealed several nonlinear relationships, including a players overall score, their age, and their potential. Due to the lack of linearity, the linear regression model was less precise in predictions than was hoped for. For certain field positions such as midfielders and defenders, the test RMSE percentage improvement was greater than the inclusive dataset. Additionally, player salaries was not normally distributed, making predictions more challenging for regression models. Removing outliers is a strategy to reduce RMSE, though there is a trade-off, due to reducing the robustness of the model to new data.


#### _Deliverables_

Final Notebook, Readme, and Presentation

# II. Project Data 

<hr style="border-top: 2px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

#### _Data Dictionary_

The final DataFrame used to explore the data for this project contains the following variables (columns). The variables are defined below:

# III. Project PLAN

<hr style="border-top: 2px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

The following outlines the process taken through the data science pipeline to complete this project.

_Plan_

In the planning stage we: read project expectations, created a project outline, wrote a project goal to include how we would measure success or failure, reviewed the overview of the dataset, documented all initial thoughts, questions, and hypotheses, created a plan for completing the project, created a data dictionary to define features, created a local folder and github repository. 

_Acquire_

    In the acquire stage we: 
        - acquired data from kaggle: https://www.kaggle.com/bryanb/fifa-player-stats-database
        - used pd.read_excel to import the excel file
        - used pd.ExcelFile to read the excel file
        - separated pages of excel file by year
        - added a 'year' column to each page with associated year date
        - used pd.concat to concat the pages back together into one dataframe
        - created a CSV to store the data in our shared repository. 
    This dataframe can be called with acquire.py using the function get_fifa_data().

_Prepare_

- In the prepare stage we: 
       - dropped league_level '2, 3, and 4'
       - dropped nation_team_id and nation_position for having 90% null values 
       - dropped mentality_composure for having over 30 thousand values missing 
       - dropped columns that didn't add value: player_url, long_name, dob, league_level, club_jersey_number, club_loaned_from, nation_jersey_number, body_type, real_face, release_clause_eur, player_tags, player_traits, 'ls', 'st', 'rs', 'lw', 'lf', 'cf','rf', 'rw', 'lam', 'cam', 'ram','lm', 'lcm', 'cm', 'rcm', 'rm','lwb', 'ldm', 'cdm', 'rdm', 'rwb','lb','lcb', 'cb', 'rcb', 'rb','gk', 'player_face_url', 'club_logo_url','club_flag_url','nation_logo_url', 'nation_flag_url' 
       - imputed '0' for null values in goal keeping stats: goalkeeping_speed, defending, physic, dribbling, passing, shooting, and pace. 

_Explore_

During the exploratory data analysis process we used statistical testing, clustering, recursive feature elimination, and feature engineering to determine which features are most important in a salary prediction model.

Results were improved uniformly when outliers were removed in trial modeling, though the results were negligible. Additionally, this does make salary prediction of star players challenging and potentially reduces the ability of the model to be robust to new data.

_Model and Evaluate_

We used trained three regression models, Linear, Lasso + Lars, and a Generalized Linear Model using a Tweedie Regressor to see which model had the best predictive capacity to measure on the out of sample data. The regression models performed similarly with one another across the overall datasets and maintained the best performance even once the data was subset by field positions, resulting in the lowest overall Root Mean Squared Error and highest  score. The best performing model was the Generalized Linear Model using a Tweedie Regressor, and the baseline was calculated by taking the average of the Root Mean Squared Error from the train and validation sets.

_Deliver_

Final Notebook and Presentation

# IV. Supplementary Files 

<hr style="border-top: 2px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

- acquire.py - provides code to import the data 
- prepare.py - provides code to prepare, split, and encode the data 
- visuals.py - provides code for all visuals found in the notebook

# V. Steps to Reproduce

<hr style="border-top: 2px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

- Clone this repo (including acquire.py, prepare.py, visuals.py, CSV)
- Run Final Report Jupyter notebook to view the final product 
