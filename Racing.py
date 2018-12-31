
# coding: utf-8

# 
# ### Author Details:
# 
# <table>
#   <tr>
#     <th>Name</th>
#     <th>Address</th>
#     <th>Email</th>
#       <th>Organization</th>
#   </tr>
#   <tr>
#     <td>Subash Prakash</td>
#     <td>Ernst Lehmann Strasse-2, Magdeburg - 39106,Germany</td>
#     <td>subashprakash8@gmail.com</td>
#       <td>Master Student</td>
#   </tr>
# </table>

# <img src="https://static.digit.in/default/1ce17400832bc84923a5fc8598fa628e8a10535e.jpeg" />
# 
# 
# ## Racing
# 
# Racing on tracks could be easier somedays when the weather is good while sometimes it could be impossible to drive. 
# One such analysis of the dataset is racing.
# 
# Certain analysis questions to ask:
#     1. Does weather depend on winning/retiring races ?
#     2. How does fuel affect to winning?
#     3. How much money is spent on racing?
#     5. What are some yearly distributions?
#     4. Can we predict the status of a race given some conditions?
#     
# All of the above question makes it interesting to explore and understand the racing dataset.
# 
# Data Description:
# #### ID: 
# The ID of the entry
# 
# #### race_created: 
# When was the race scheduled?
# 
# #### race_driven: 
# When was the race held?
# 
# #### track_id: 
# ID of the race track on which the race was held.
# 
# #### challenger: 
# userID of the challenger
# 
# #### opponent: 
# userID of the opponent (challenged)
# 
# #### money: 
# use of the race (in EUR)
# 
# #### fuel_consumption: 
# Fuel consumption of participants during the race (in l)
# 
# #### winner: 
# winner of the race (userID)
# 
# #### status: 
# The status of the race. Possible are: waiting, finished, retired, declined
# 
# #### Forecast: 
# weather forecast for the race. The field (the CSV file is an export from a (My) SQL database) is a serialized array. 
# This means that an array data type is stored as a string in the database. The background is that you do not want to have your own field in the table of the database for each value in the array. 
# But now to the data itself. I want to explain data to you using the string from the "forecast" field of the first line: 
# a: 4: {s: 5: "sunny"; i: 10; s: 5: "rainy"; i: 70; s: 8: "thundery"; i: 0; s: 5: "snowy"; i: 20;}
# The a: 4 means that it is a serialized array of 4 pairs of data. The content of the array is then in the curly brackets. Each data pair represents a weather type and the probability of their occurrence. The weather types (entries in the array) are separated by semicolons. There are four types of weather (sunny, rainy, thundery, snowy). In front of it is "s:" and a number. The number only describes the length of the following string (sunny 5 characters, thundery 8, etc.). Behind the weather type is still an "i:" and a number. The number after the i (for integer) describes the percentage probability of occurrence of the respective weather. 
# 
# The above entry means broken down: 
# 10% sunny weather (probability)
# 70% rainy weather (probability) 
# 0% thunderstorm 
# 20% snowfall
# 
# #### weather: 
# Actual weather

# In[66]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[67]:


sns.set_palette("Dark2")
sns.set_style("whitegrid")


# #### Import the data from the location

# In[68]:


race = pd.read_csv("races.csv",sep=";")


# In[69]:


race.head(6)


# ## Information:
# As seen from head, now we are able to explore the dataset to see some values. If can see that some values for weather are Nan when status of the race is declined. Also, when declined race_driven date is dummy value.

# In[70]:


# To understand about each attributes we use info()
race.info()


# ### Distribution of Status accross the dataset:
# We can see from the plot that status of the race can be (finished, declined, retired, waiting). We can also observe that most of the dataset have either Finished races or retired races

# In[71]:


sns.countplot(x="status",data=race)


# ### An interesting observation can be made by plotting track_id frequencies. 
# We can observe that "track - 12" is widely used for racing and "track - 3" occationally. 
# Looks like track-6, track-13, track -14 are least used

# In[72]:


plt.figure(figsize=(15,8))
sns.countplot(x="track_id",data=race,hue="status")


# In[73]:


#### Create a creation year of races for more exploration on data


# In[74]:


race["race_created_year"] = race.race_created.apply(lambda x: x.split(".")[2])


# In[75]:


race.head(3)


# #### A Distribution plot over the years on races

# In[76]:


sns.distplot(race.race_created.apply(lambda x:int(x.split(".")[2])),bins=20)


# In[77]:


counts = race.groupby("race_created_year").count()
d= {"Year":counts.index,"Freq":counts.id}


# In[78]:


sum_money = race.groupby("race_created_year").mean()
d1= {"Year":sum_money.index,"Average_money":sum_money.money}


# In[79]:


race_years_counts = pd.DataFrame(d)
race_years_counts.head()


# In[80]:


race_years_avgMoney = pd.DataFrame(d1)
race_years_avgMoney.head()


# # Frequencies of races over the years
# It can be noted from the lineplot that an exponential decrease of races over the year. This could mean that either the application usage has become low or data contains more values of previous years as compared with the current year

# In[81]:


race_years_counts.plot(title="Frequency of races over the years",figsize=(10,6))


# # Average money utilized for races
# We can see a steady rise of races has occured in the year 2016. This could likely mean that users who raced in 2016 spent money over races and overall dataset contains less data for 2017 hence cannot be interpretted

# In[82]:


race_years_avgMoney.plot(x="Year",y="Average_money",title="Average Money spent over races in euro",figsize=(10,6))


# # Number of Races Created per year
# As seen from the plot it is very clear that year 2017 had very little races. An interesting observation is exponential decrease of races. Could also mean that dataset has large information of past years as compared to the present

# In[83]:


plt.figure(figsize=(10,5))
plt.title("Number of Races Created per year")
sns.barplot(x=race_years_counts.Year,y=race_years_counts.Freq)


# # Races Distribution by year

# In[84]:


plt.figure(figsize=(14,10))
plt.title("Race Distribution by year")
sns.countplot(x=race.race_created_year,data=race,hue="status")


# In[85]:


race.head(2)


# As observed from the data column for fuel_consumption in litres.
# A good notification was made on the data that it had to be cleaned to remove some bad data.
# Example:
#     1. A data value "Sep 75" is contained. For which, I have implemented a logic to keep the value 75 and remove Sep
#     2. A data value "03.Marz" is contained. For this, I have made it default 0 so that I can compute the mean, max and min
#     
# An apply function is used for the column and below function is defined to filter the values.

# In[86]:


def filter_fuel(x):
    numb = ""
    x_trim = ''.join(x.split())
    if x_trim.isalnum():
        for char in x_trim:
            if char.isdigit():
                numb+=str(char)
        return float(numb)
       
    elif "." in x_trim:
        if (x_trim.split(".")[0].isalpha()) or (x_trim.split(".")[1].isalpha()):
            return float(0)
        else:
            return float(x)
    return float(x)


# In[87]:


race["fuel_consumption"] = race.fuel_consumption.apply(filter_fuel)


# In[88]:


race.fuel_consumption.mean()


# In[89]:


race.head(5)


# In[90]:


race_fuel = race[race.fuel_consumption != 0.0]


# In[91]:


avg_fuel_year = race_fuel.groupby("race_created_year").mean()
d3= {"Year":sum_money.index,"Avg_fuel":avg_fuel_year.fuel_consumption}

race_years_avgFuel = pd.DataFrame(d3)
race_years_avgFuel.head()


# ## Average fuel spent over races per year
# This lineplot gives an indication that in the year 2013 there was high amount of usage of fuel. This could because of the races or because of the weather condition during that time

# In[92]:


race_years_avgFuel.plot(x="Year",y="Avg_fuel",title="Average fuel spent over races per year",figsize=(10,6))


# In[93]:


avg_fuel_status = race_fuel.groupby("status").mean()
d4 = {"status":avg_fuel_status.index,"Avg_fuel_consumption":avg_fuel_status.fuel_consumption}
avg_fuel_status_df = pd.DataFrame(d4)
avg_fuel_status_df.head()


# ## Average fuel consumed group by status (Finished, Declined, Retired, Waiting)
# An interesting observation to look is declined and waiting races have spent good amount of fuel

# In[94]:


plt.figure(figsize=(10,6))
plt.title("Average fuel consumption grouped by status of the race")
sns.barplot(x="Avg_fuel_consumption",y="status",data=avg_fuel_status_df)


# In[95]:


races_won = race[race.status == "finished"]


# In[96]:


races_declined = race[race.status == "declined"]


# In[97]:


races_declined.head(5)


# ## Parsing forecast:
# A logic for parsing forecast is implement. The serial array follows a pattern through which using string manipulations the value of the most probable forecast of the weather is extracted.
# This logic is very useful to extract the weather and separate columns for each of the attributes. Also, when the races are declined the status of the weather is unknown hence this logic puts a foundation to employ such a logic
# ### Logic:
# 1. In order to obtain the percentage of the weather probablities. The pattern of extraction involves to first locate the array position of respective weather using <code> find() </code>. Use this value to further take the <code> len(weather)</code> and add 4 to extract the value.
# Example: if String = a:4:{s:5:"sunny";i:10;s:5:"rainy";i:70;s:8:"thundery";i:0;s:5:"snowy";i:20;} and we have to extract say Sunny value which is 10. Then:
# 1. location of Sunny is at position 10
# 2. Add 10 + len(Sunny) = 5 + 4 (Because as seen 4 characters ahead we reach at the value). Store this number into a variable (Here: loc)
# 3. Use <code> x[loc:loc+3].replace(";","").replace("s","") </code> to extract the value where loc+3 indicate we can have percentages from 0 to 100 so 3 digit values and we will have to replace some unwanted character such as <code>";s}"</code> from the obtianed value
# 
# Below is the function behind the same logic

# In[98]:


forecast = race.forecast[0]
#a:4:{s:5:"sunny";i:10;s:5:"rainy";i:70;s:8:"thundery";i:0;s:5:"snowy";i:20;}'

def filter_forecast(x,weather):
    if "sunny" == weather:
        loc = x.find(weather) + len(weather) + 4
        return x[loc:loc+3].replace(";","").replace("s","")
    elif "rainy" == weather:
        loc = x.find(weather) + len(weather) + 4
        return x[loc:loc+3].replace(";","").replace("s","")
    elif "thundery" == weather:
        loc = x.find(weather) + len(weather) + 4
        return x[loc:loc+3].replace(";","").replace("s","")
    elif "snowy" == weather:
        loc = x.find(weather) + len(weather) + 4
        return x[loc:loc+3].replace(";","").replace("}","")


# In[99]:


# Just a test for the function before applying to the dataframe
filter_forecast(forecast,"snowy")


# ##### Extract the required values and add them into separate columns

# In[100]:


race["sunny"] = race.forecast.apply(filter_forecast,args=("sunny",))
race["rainy"] = race.forecast.apply(filter_forecast,args=("rainy",))
race["thundery"] = race.forecast.apply(filter_forecast,args=("thundery",))
race["snowy"] = race.forecast.apply(filter_forecast,args=("snowy",))


# In[101]:


race.head(2)


# #### To obtain the correct weather forecast for all the values lets take the max of the attribute columns ("sunny","rainy","thundery","snowy")

# In[102]:


race["weather"] = race[["sunny","rainy","thundery","snowy"]].idxmax(axis=1)


# In[103]:


race.head(3)


# ### Based on grouping weather we can extract the percentage of races that were finished, retired, declined, waiting

# In[104]:


race.groupby("weather")["status"].value_counts(normalize=True).rename("percentage").mul(100)


# ## How are the races distributed accross different weather conditions

# In[105]:


plt.figure(figsize=(12,8))
plt.title("Distribution of race status across weather")
sns.countplot(x="weather",data=race,hue="status")
plt.xlabel("Race Status")
plt.ylabel("Frequencies")


# In[106]:


race__group_df = (race.groupby("race_created_year")["status"].
                value_counts(normalize=True).
                mul(100).
                rename("percentage").
                reset_index())


# In[107]:


race__group_df


# In[108]:


finished_df = race__group_df[race__group_df.status == "finished"]
declined_df = race__group_df[race__group_df.status == "declined"]
waiting_df = race__group_df[race__group_df.status == "waiting"]
retired_df = race__group_df[race__group_df.status == "retired"]


# ## How does the race finished change over the years

# In[109]:


finished_df.plot(x="race_created_year",y="percentage",figsize=(12,6),title="% Races finished over the years")


# ## How does the race Decline change over the years in percentage

# In[110]:


declined_df.plot(x="race_created_year",y="percentage",figsize=(12,6),title="% Races declined over the years")


# ## How does the race waiting change over the years in percentage

# In[111]:


waiting_df.plot(x="race_created_year",y="percentage",figsize=(12,6),title="% Races waiting over the years")


# ## How does the races which are retired change over the years in percentage

# In[112]:


retired_df.plot(x="race_created_year",y="percentage",figsize=(12,6),title="% Race status retired over the years")


# In[113]:


race[race.race_created_year == "2015"]


# In[114]:


winner =race[((race.status != "declined") & (race.status != "retired") & (race.status != "waiting"))].winner.value_counts().head(10)


# In[115]:


d5 = {"UserIds":winner.index,"win_count":winner.values}
winner_df = pd.DataFrame(d5)
winner_df


# # Top 10 winners of the race

# In[116]:


plt.figure(figsize=(10,6))
plt.title("Top 10 race Winners")
sns.barplot(x="UserIds",y="win_count",data=winner_df,order=winner_df["UserIds"])
plt.xlabel("User Id of Winners")
plt.ylabel("# Wins")


# ### Number of races driven by year

# In[126]:


races_driven = race[race.status == "finished"]
races_driven.head(2)


# In[128]:


races_driven["race_driven_year"] = races_driven.race_driven.apply(lambda x:x.split(" ")[0].split(".")[2])


# In[130]:


races_driven.head(2)


# In[134]:


no_races_year =races_driven.groupby("race_created_year")["track_id"].sum()


# In[140]:


ndict = {"Year":no_races_year.index,"Counts":no_races_year.values}
df9 = pd.DataFrame(ndict)


# In[149]:


plt.figure(figsize=(10,6))
sns.barplot(x="Year",y="Counts",data=df9)
plt.title("Number of Races Created Per Year")


# In[164]:


def filter_driven_races(x):
    if "-" in x:
        x=x.replace("-",".")
    
    return x.split(" ")[0].split(".")[2]


# In[165]:





# In[166]:


race["race_driven_year"] = race.race_driven.apply(filter_driven_races)


# # Using scikit-learn package, we will try to predict the status based some features created
# 
# <img src="https://cdn-images-1.medium.com/max/1000/1*lkqc68a6b7_TLALs5fmI6A.png" align=left width=300 height=600/>
# 
# #### Scikit learn package is used to perform classification of data using machine learning algorithms. Scikit-learn is a python package and open source for usage

# In[167]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import itertools


# In[168]:


def cm_analysis(cm,labels, figsize=(10,8)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax,cmap="YlGn")


# In[169]:


dTree = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=600)


# In[170]:


race.columns


# In[172]:


X = race[['money', 'fuel_consumption', 'winner', 'race_driven_year','race_created_year','sunny', 'rainy', 'thundery', 'snowy']]
y = race["status"]


# In[173]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[174]:


dTree.fit(X_train,y_train)


# In[175]:


preds = dTree.predict(X_test)


# In[176]:


print(confusion_matrix(preds,y_test))


# In[177]:


print(classification_report(preds,y_test))


# In[178]:


rf.fit(X_train,y_train)


# In[179]:


predrf = rf.predict(X_test)


# In[180]:


clf_mat = classification_report(predrf,y_test)
print(clf_mat)
accuracy_score(y_test,predrf)


# In[181]:


# Compute confusion matrix for Random Forest
cnf_matrix = confusion_matrix(y_test, predrf)

class_names = race.status.unique()

# Plot confusion matrix
cm_analysis(cnf_matrix,sorted(class_names))


# In[183]:


# Compute confusion matrix for Decision Tree
cnf_matrix = confusion_matrix(y_test, preds)

class_names = race.status.unique()

# Plot confusion matrix
cm_analysis(cnf_matrix,sorted(class_names))


# In[184]:


from sklearn.model_selection import GridSearchCV
#class_weight=None, criterion='gini', max_depth=None,
 #          min_impurity_decrease=0.0, min_impurity_split=None,
  #          min_samples_leaf=1, min_samples_split=2,
   #         min_weight_fraction_leaf=0.0, presort=False, random_state=None,
    #        splitter='best'


# In[185]:


param_grid = {'criterion': ["gini","entropy"], 'max_depth': [3,5,7,11,15],'min_samples_leaf':[1,3,5,7]}


# In[186]:


gridcv = GridSearchCV(DecisionTreeClassifier(),param_grid,refit=True,verbose=2,cv=3)
gridcv.fit(X_train,y_train)


# In[187]:


gridcv.best_params_


# In[188]:


predictGridDT = gridcv.predict(X_test)


# In[189]:


print(classification_report(predictGridDT,y_test))
accuracy_score(predictGridDT,y_test)


# In[190]:


# Compute confusion matrix for Grid Based DT
cnf_matrix = confusion_matrix(y_test, predictGridDT)

class_names = race.status.unique()

# Plot confusion matrix
cm_analysis(cnf_matrix,sorted(class_names))


# In[191]:


## Resampling to the rescue
from imblearn.over_sampling import SMOTE


# In[192]:


sm = SMOTE(sampling_strategy='minority',random_state=101)


# In[193]:


x_train_res, y_train_res = sm.fit_sample(X_train, y_train)


# In[194]:


decisionTreeFit = gridcv.fit(x_train_res,y_train_res)


# In[195]:


gridcv.cv_results_


# In[210]:


predsSMOTE = gridcv.predict(X_test)


# In[212]:


print(classification_report(y_test,predsSMOTE))
accuracy_score(y_test,predsSMOTE)


# In[198]:


# Compute confusion matrix for DT SMOTE
cnf_matrix = confusion_matrix(y_test, predsSMOTE)

class_names = race.status.unique()

# Plot confusion matrix
cm_analysis(cnf_matrix,sorted(class_names))


# In[199]:


from sklearn.tree import export_graphviz
from sklearn.ensemble import GradientBoostingClassifier


# In[200]:


len(X_train)


# In[215]:


gbm = GradientBoostingClassifier(learning_rate=0.001, n_estimators=1500,max_depth=9, min_samples_split=1200, min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7,
warm_start=True)


# In[216]:


gbm.fit(x_train_res,y_train_res)


# In[217]:


predGBMSMOTE = gbm.predict(X_test)


# In[218]:


print(classification_report(predGBMSMOTE,y_test))
accuracy_score(predGBMSMOTE,y_test)


# In[219]:


# Compute confusion matrix for Random Forest
cnf_matrix = confusion_matrix(y_test, predGBMSMOTE)

class_names = race.status.unique()

# Plot confusion matrix
cm_analysis(cnf_matrix,sorted(class_names))


# <h3>Results:</h3>
# <p>To summarize the results obtained from the Classifications, it can be noted that Declined being treated as retired in most case having a low precision and high recall. From these observations, one option that could be done in future to the dataset is to either get more clear picture of each status through a new column which helps to discrimate it or convert the problem from a multi class prediction into a binary problem so that the predictions are either Finished or not Finished to improve the accuracy.</p>
# 
# <p>
# On the overall, still the accuracy stands good accross tree based classifiers. Also, a note should be taken that classes are imbalanced hence, a SMOTE technique is applied to the minority classes to balance the data for classification.
# </p>
# <p>
# Hyperparameter tunning is applied to each of the model built and below are the results in terms of Accuracy,precision,recall and F1-Score
# </p>
# <br>
# <table border=1>
#   <tr>
#     <th>Classifier</th>
#     <th>Accuracy</th>
#     <th>Precision</th>
#       <th>Recall</th>
#       <th>F1-Score</th>
#   </tr>
#       <tr>
#     <td>Gradient Boosting Machine (Learning Rate - 0.005)</td>
#     <td>90.4%</td>
#     <td>96%</td>
#     <td>90%</td>
#     <td>92%</td>
#   </tr>
#           <tr>
#     <td>Gradient Boosting Machine (Learning Rate - 0.001)</td>
#     <td>85.6%</td>
#     <td>90%</td>
#     <td>86%</td>
#     <td>86%</td>
#   </tr>
#   <tr>
#     <td>Decision Trees (Grid Search)</td>
#     <td>87.8%</td>
#     <td>88%</td>
#     <td>88%</td>
#     <td>87%</td>
#   </tr>
#     <tr>
#     <td>Random Forest</td>
#     <td>90.4%</td>
#     <td>94%</td>
#     <td>90%</td>
#     <td>92%</td>
#   </tr>
#   </table>
#   
# To conclude, this exploration helps to understand over the years about races. The machine learning model helps to identify quickly what are finished, declined, etc.. without manual intervention. In future, it would be great to provide more understanding through data how declined/retired is different so that classifier could be improved and used globally
