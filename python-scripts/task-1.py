# Data processing using Pandas...


import numpy as np
import pandas as pd

# Loading data from CSV file (with replacement of "/" values to NaN values)...
# Adjust file path!
data = pd.read_csv("../report.csv", na_values = "/")                                                                          					

# Reading first 5 rows...
print(data.head(10))                                                                                                        						 

# Reading last 10 rows...
print(data.tail(10))        
                               
# Reading specific column of DataFrame object                                                          						 
print(data["Ispit1"])                                                                                                        						 
print(data["Ispit2"])

# Reading rows with specific characteristic...
print(data.loc[data["Prisustvo"] == 0])     

# Replacing values (this is only for informative purpose, these values are already replaced)...
data.replace("/", np.nan, inplace = True)                                                                                    					 

# Reading specific columns of rows with specific characteristic...
print(data.loc[data["Ocjena"] > 7].loc[:, ["Indeks", "UKUPNO", "Ocjena"]])                                                 		      	 	 
	
# Deleting specific rows of DataFrame object...
data.dropna(subset = ["Ocjena"], inplace = True)                                                                             					 

# Adding new column (with specific values)...
temporaryStorage = data.replace(np.nan, -1)
data["Ispit1_final"] = np.maximum(temporaryStorage["Ispit1"], temporaryStorage["Ispit1_popravni"]).replace(-1, np.nan)                               
data["Ispit2_final"] = np.maximum(temporaryStorage["Ispit2"], temporaryStorage["Ispit2_popravni"]).replace(-1, np.nan)

# Deleting specific columns of DataFrame object...
data.drop(columns = ["Ispit1", "Ispit2", "Ispit1_popravni", "Ispit2_popravni"], inplace = True)                              			 

# Saving DataFrame object as CSV file...
data.to_csv("report-update.csv", sep = ";")

# Saving DataFrame object as Pickle file...                                                                                                                                                                     
data.to_pickle("report-update.p")                                                                                                                                                                                                                                      