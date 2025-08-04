POST CONSTRUCTION MODEL LOG


Tags:[LSTM,TCN,SCRAPER,AUTOMATION,LOADER,SEPERATOR,BILSTM]
___________________________

Default Model [TCN]

So the initial issue was validation loss was too high ,training loss(0.8) validation loss(3.1) ,and final MSE value was around (8) , this meant that the model was not generalising properly or that validation and test datasets had wildy varying values compared to training dataset , when manually reviewed this seemed to be the case , as the validation and test values had around 2 point difference to training values , training value(-1.2) , validation value (1.8) test value (2.4), this was deduced to be due to the fact that nifty 50 is a rapidly growing indice and training values were acquired from a slow bullish-bearish market and covid collapse , while validation was taken from a strong bullish market and test was taken from very strong bullish market

Training Loss(0.8)
Validation Loss(3.1)
MSE(8.3)

___________________________

Iteration 1 [Split the datasets in preprocessing] [TCN]

Initially the dataset was scaled in group and then split in the model program , this would mean that even though it increases accuracy a bit  (as validation and test values are reduced to match training to a degree) , it would not hold in real life as they all should be treated as independent entities to simulate real life market movements , 
the splitting caused even a sharper drop with validation loss and mse , with training loss being (0.8) validation loss being (6.2) and mse being (25) 

Training Loss(0.9)
Validation Loss(6.3)
MSE(25)

___________________________

Iteration 2 [Replaced Standard Scaler with Robust Scaler] [TCN]

Standard scaler would take in outliers at face value and not smooth them , which made the already large variances present between the training and validation and test datasets even larger , this issue was addressed by replacing standardscaler with robustscaler which evened out the outliers thus reducing variances in certain values but not solving overall issues 

No large differences in any values

__________________________

Iteration 3 [Log Transformation] [TCN]

Since it was groups of large values that were causing issues , log transformation was applied which reduced validation and test dataset values a by a good margin , most of train dataset were left intact due to most of it being negative thus skipping log transformation or being very small ,thus reducing loss and bringing MSE values closer to necessary requirement , but a concurrent issue facing all iterations till now has been the steady loss values , instead of dropping as epochs increases which indicates model learning , it is staying constant which means that model is not learning and is just blindly predicting 

Training Loss (0.5)
Validation Loss(1.2)
MSE(6.1)

_____________________________

Switched Model [LSTM]

Since the TCN model was constantly returning large loss and MSE values no matter how standardized and preprocessed the dataset was ,so after going through various research papers , i decided to test out a basic lstm system , since it was much more potent at understanding long sequences of data, after moving to just a basic lstm model, it could predict basic movement of the market but not the magnitude , as it really only understands temporal features it could capture movements but not the magnitude of these moves ,after it predicted the movements to a certain degree i moved onto the scraper 

Training Loss (0.5)
Validation Loss (0.8)
MSE(0.11)

This model worked , and returned a good enough output temporarily leaving it ,to build the rest of the system , will come back add technical indicators and polish it once the entire system is complete 

____________________________

Iteration 1 [BILSTM]

Decided to upgrade the lstm to a bilstm , since the bilstm would be able to capture much more temporal dependencies being of capable understanding both previous and future patterns from a certain point , also since training and validation,test having such a different values was a consistent issue , decided to add a custom loss function to the training dataset while keeeping original loss function for validationa and test

Training Loss (0.2)
Validation Loss(0.5)
MSE(0.05)

____________________________

Iteration 2 [BILSTM]

So there was an apparent issue with close being fed into validation and test dataset as well , which lead to a small data leakage , which skewed the results of the final model , after correcting this the results of final model were and added other metrics along mse for various different cases ,

Training Loss (0.2)
Validation Loss(0.5)
MSE: 0.0727, RMSE: 0.2696, MAE: 0.2419, R²: -0.5629, MAPE: 9.99%

__________________________

Iteration 3 [BILSTM]

Since the model now had somewhat good results and the overall architecture was good , i decided to begin fine tuning the hyperparameters , for this i implemented bayesian optimization ,initially wanted to do gaussian functions by myself but since that requires some time and acquisition function could be difficult ,i decided to go for optuna , but there was an issue where optuna constantly became stuck ,so added in a bunch of try catch blocks everywhere and added return(inf) to the optuna block which solved it

__________________________

Iteration 4 [BILSTM]

So i hit a black box issue in a way , basically optuna was getting stuck on random processes and sometimes it ran no trials , sometimes 2 and sometimes 3 , it was incredibly inconsistent, i added cpu logging to see if ram usage was an issue , logging also didnt work much because there was no explcit issue in code but more so inner workings failure , decided to set up a specific seed (42) to rule out any randomness and thus hopefully decrease random interal bugs but that didnt work well , also removed the manual runs before the optuna runs to save time

__________________________

Iteration 5 [BILSTM]

So the issue was that embarassingly i didnt save the models after each trials , and also ran the manual train on every iteration which was unnecessary , so after fixing those , the optuna began working well and gave solid results
it initially ran at t-1 which gave unsurprisingly near pinpoint results so increased gap t-20 and then finally t-45
which is where the model stands at right now , the model was run on google colab due to its gpu accomodation

Parameters
input_size=4, hidden_size=128, dropout=0.3, num_layers=2, batch_size=16 lr=5.401798979253006e-05 
epochs=75

Epoch [75/75] Train Loss: 4.9970 | Val Loss: 0.4735
Test Loss: 0.6706
MSE: 0.0550, RMSE: 0.2345, MAE: 0.2030, R²: 0.0449, MAPE: 8.31%

    Model is saved in Checkpoints folder 
