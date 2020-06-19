import pandas as pd
import numpy as np
import os
from sklearn import preprocessing

def read_data(file_name):
    file_path = file_name
    data = pd.read_csv(file_path,sep=',',names = ['time_counter','accel_base_X','accel_base_Y','accel_base_Z','gyro_base_X',
                                                  'gyro_base_Y','gyro_base_Z','accel_right_X','accel_right_Y','accel_right_Z',
                                                  'gyro_right_X','gyro_right_Y','gyro_right_Z','accel_left_X','accel_left_Y',
                                                  'accel_left_Z','gyro_left_X','gyro_left_Y','gyro_left_Z','fall_detection'])
    return data

def read_combined_csvdata(file_name, airbag_data):
    file_path = file_name
    data = pd.read_csv(file_path)
    data['Time Point Name'] = pd.to_numeric(data['Time Point Name'], errors='coerce')
    data = data[(data['Time Point Name']>100) & (data['Time Point Name'] < 200) &
                (data['Impact milisec'].notnull()) & ((data['Start milisec']) < (airbag_data.iloc[-1][0]))]
    data = data[['Time Point Name','Start milisec','Impact milisec', 'End Milisec','Duration']]
    return data

def add_new_fall_value(airbag_data, manual_data):
    # Create Column that is the magnitude of the acceleration vector from the data
    # Iterate through the combined data to find points at which time counter of the airbag_data is within the start and end milliseconds of the row
    # find the max of the 'accel_base_mag' within that time frame, set 'cal_impact_milisec' = 80
    airbag_data['accel_base_mag'] =  np.sqrt(airbag_data['accel_base_X']**2 + airbag_data['accel_base_Y']**2 +
                                             airbag_data['accel_base_Z']**2)
    for index, row in manual_data.iterrows():
        airbag_data.loc[(airbag_data['accel_base_mag'] == \
                                             (airbag_data.loc[(airbag_data['time_counter'] >= row['Start milisec']) & \
                                                                   (airbag_data['time_counter'] < row['End Milisec']),
                                                                   'accel_base_mag'].max())) & \
                                                 (airbag_data['time_counter'] >= row['Start milisec']) & \
                                                 (airbag_data['time_counter'] < row['End Milisec']),
                                                 'cal_impact_milisec'] = 80
    # print (sum(airbag_data['cal_impact_milisec'] == 80)) 

def extract_rnn_feature(data, window_time=1000, move_window = 0):
    """ 
    Arguments:
    Data -> From "Reading in Subject Data"
    window_time -> how large the window for extraction is
    move_window -> how big the window of the fall is said to be
    """
    max_ms = data['time_counter'].max() # Returns Maximum Time
    # Define Features
    feature = pd.DataFrame(columns={'time_counter','accel_base_X','accel_base_Y','accel_base_Z','accel_base_mag','fall_value'})

    # Get a list of times where falls occured
    impact_times = data.loc[(data['cal_impact_milisec'] == 80),'time_counter'].values 

    # Add Data to the feature set
    feature[['time_counter','accel_base_mag','accel_base_X','accel_base_Y','accel_base_Z']] = data[['time_counter','accel_base_mag','accel_base_X','accel_base_Y','accel_base_Z']]  
    
    # Set all fall values = 0
    feature['fall_value'] = np.zeros((feature.shape[0], 1))
    
    # Loop through each time in the impact_times
    for impact_time in impact_times:
      # print(data.loc[data['time_counter'] <= impact_time - move_window].tail(1).values)

      # Finds the last time <= to the impact_time shifted back by move_window
      fall_time_shifted = data.loc[data['time_counter'] <= impact_time - move_window].tail(1).values[0][0] 
      # Sets the time at the shifted time = 1 in order to predict move_window (ex: 10) ms into the future
      feature.loc[feature['time_counter'] == fall_time_shifted, 'fall_value'] = 1
    return feature

def EF_MAGNITUDE(subject, ts_len): 
  # Extract Magnitudes
  # Doesn't use a fall radius
  # Implement a 50% Overlap Algorithm
  # Start with a window size of ts_len
  # Experiment with window size and lead time
  # For Subject 1 with 1601594 rows of data
  # There should be 12812 features w/ 250 rows per feature
  impact_times = subject[subject['fall_value'] == 1].index.values.tolist() # All the indexes where falls occured
  subject = subject[['accel_base_mag', 'fall_value']] # Look at only the Accel_mag and fall_value (univariate)
  data_df = pd.DataFrame(columns=['X', 'Y']) # Create X and Y
  std_scale = preprocessing.StandardScaler().fit(subject[['accel_base_mag']])
  x_subject = std_scale.transform(subject[['accel_base_mag']])
  start_index = 0
  # (subject.shape[0] - ts_len)
  X = []
  Y = []

  while start_index < (subject.shape[0] - ts_len):
    x_val_seq = x_subject[start_index:start_index + ts_len]
    x_val_seq = x_val_seq.tolist()
    
    y_val = [1] if (sum(subject['fall_value'].iloc[start_index:(start_index + ts_len)]) > 0) else [0]

    X.append(x_val_seq)
    Y.append(y_val)
    start_index += (ts_len // 2)

  X = np.array(X)
  Y = np.array(Y)
  return X,Y

def EF_XYZ(subject, ts_len): 
  # Extract X,Y,Z coordinates
  # Doesn't use a fall radius
  # Implement a 50% Overlap Algorithm
  # Start with a window size of ts_len
  # Experiment with window size and lead time
  # For Subject 1 with 1601594 rows of data
  # There should be 12812 features w/ 250 rows per feature
  impact_times = subject[subject['fall_value'] == 1].index.values.tolist() # All the indexes where falls occured
  subject = subject[['accel_base_X', 'accel_base_Y', 'accel_base_Z', 'fall_value']] # Look at only the Accel_mag and fall_value (univariate)
  std_scale = preprocessing.StandardScaler().fit(subject[['accel_base_X', 'accel_base_Y', 'accel_base_Z']])
  x_subject = std_scale.transform(subject[['accel_base_X', 'accel_base_Y', 'accel_base_Z']])
  print(subject)
  start_index = 0
    # (subject.shape[0] - ts_len)
  X = []
  Y = []

  while start_index < (subject.shape[0] - ts_len):
    # x_val_seq = subject.iloc[start_index:(start_index + ts_len), 0:3]
    # x_val_seq = x_val_seq.values.tolist()

    x_val_seq = x_subject[start_index:start_index + ts_len]

    y_val = [1] if (sum(subject['fall_value'].iloc[start_index:(start_index + ts_len)]) > 0) else [0]

    X.append(x_val_seq)
    Y.append(y_val)
    start_index += (ts_len // 2)

  X = np.array(X)
  Y = np.array(Y)

  return X, Y

def create_df_XYZ_scaled(dataset, look_back, rem_range = 500, skip = 50):

  fall_sec = dataset[dataset['fall_value']==1].index.values.tolist()
  dataset = dataset[['accel_base_X','accel_base_Y','accel_base_Z','fall_value']]
  data_df = pd.DataFrame(columns=['X','Y'])
  std_scale = preprocessing.StandardScaler().fit(dataset[['accel_base_X', 'accel_base_Y', 'accel_base_Z']])
  x_subject = std_scale.transform(dataset[['accel_base_X', 'accel_base_Y', 'accel_base_Z']])
  print('Fall_sec: ',fall_sec)
  
  end_list = []  #edge condition for non-fall values
  start_list = [0]
  #prepare window frame 
  for i in range(len(fall_sec)):
      end_list.append(fall_sec[i] - rem_range)
      start_list.append(fall_sec[i] + rem_range)
  end_list.append(dataset.shape[0]-look_back-1)  
  
  #add fall values
  for i in range(len(fall_sec)):
    # print('row claculating fall: ',fall_sec[i])
    a = x_subject[fall_sec[i]-look_back:fall_sec[i]]
    a = numpy.reshape(a,(look_back,1,3)).tolist()
    b = dataset.iloc[fall_sec[i], 3]
    
    data_df = data_df.append({'X':a,'Y':b},ignore_index=True)
  print("Added total falls of ",str(len(fall_sec)))
  #add non-fall values
  for start,end in zip(start_list,end_list):
      for i in range(start,end,skip):
        # a = dataset.iloc[i:i+look_back, :3].values
        a = x_subject[i:i+look_back]
        a = numpy.reshape(a,(look_back, 1, 3)).tolist()
        b = dataset.iloc[i + look_back, 3]
        
        if(i%120023 == 0):
          print('row claculating',i) 
        data_df = data_df.append({'X':a,'Y':b},ignore_index=True)
  val = data_df.values
  X = np.reshape(val[:,0].tolist(),(val.shape[0],look_back,3))
  Y = np.reshape(val[:,1].tolist(),(val.shape[0],1))
  return X,Y
  # return data_df

def create_df_XYZ(dataset,look_back,rem_range = 500,skip = 50):

  fall_sec = dataset[dataset['fall_value']==1].index.values.tolist()
  dataset = dataset[['accel_base_X','accel_base_Y','accel_base_Z','fall_value']]
  data_df = pd.DataFrame(columns=['X','Y'])
  print('Fall_sec: ',fall_sec)
  
  end_list = []  #edge condition for non-fall values
  start_list = [0]
  #prepare window frame 
  for i in range(len(fall_sec)):
      end_list.append(fall_sec[i] - rem_range)
      start_list.append(fall_sec[i] + rem_range)
  end_list.append(dataset.shape[0]-look_back-1)  
  
  #add fall values
  for i in range(len(fall_sec)):
    # print('row claculating fall: ',fall_sec[i])
    a = dataset.iloc[fall_sec[i]-look_back:fall_sec[i], :3].values
    a = np.reshape(a,(look_back,1,3)).tolist()
    b = dataset.iloc[fall_sec[i], 3]
    
    data_df = data_df.append({'X':a,'Y':b},ignore_index=True)
  print("Added total falls of ",str(len(fall_sec)))
  #add non-fall values
  for start,end in zip(start_list,end_list):
      for i in range(start,end,skip):
        a = dataset.iloc[i:i+look_back, :3].values
        a = np.reshape(a,(look_back,1,3)).tolist()
        b = dataset.iloc[i + look_back, 3]
        
        if(i%120023 == 0):
          print('row claculating',i) 
        data_df = data_df.append({'X':a,'Y':b},ignore_index=True)
  val = data_df.values
  X = np.reshape(val[:,0].tolist(),(val.shape[0],look_back,3))
  Y = np.reshape(val[:,1].tolist(),(val.shape[0],1))
  return X,Y
  # return data_df

# READ IN DATA
data_path = '../Data/'

#S1
s1_data=read_data(data_path + 'S1-Airbag.csv')
s1_combined = read_combined_csvdata(data_path + 'S1-Combined.csv', s1_data)
add_new_fall_value(s1_data, s1_combined)

print(s1_data.loc[s1_data["cal_impact_milisec"] == 80])

#S2
s2_data=read_data(data_path + 'S2-Airbag.csv')
s2_combined = read_combined_csvdata(data_path + 'S2-Combined.csv', s2_data)
add_new_fall_value(s2_data, s2_combined)

# #S3
# s3_data=read_data(data_path + 'S3-Airbag.csv')
# s3_combined = read_combined_csvdata(data_path + 'S3-Combined.csv', s3_data)
# add_new_fall_value(s3_data, s3_combined)

# #C1
# c1_data=read_data(data_path + 'C1-Airbag.csv')
# c1_combined = read_combined_csvdata(data_path + 'C1-Combined.csv', c1_data)
# c1_data2 = pd.DataFrame.copy(c1_data, deep=True)
# add_new_fall_value(c1_data2, c1_combined)

# #C2
# c2_data=read_data(data_path + 'C2-Airbag.csv')
# c2_combined = read_combined_csvdata(data_path + 'C2-Combined.csv', c2_data)
# add_new_fall_value(c2_data, c2_combined)

# #C3
# c3_data=read_data(data_path + 'C3-Airbag.csv')
# c3_combined = read_combined_csvdata(data_path + 'C3-Combined.csv', c3_data)
# add_new_fall_value(c3_data, c3_combined)   


S1_rf = extract_rnn_feature(s1_data, 10)
S2_rf = extract_rnn_feature(s2_data, 10)

print(S1_rf)

s1x_ts, s1y_ts = create_df_XYZ(S1_rf, 50, 500, 25)
# print(s1x_ts)
# s2x_ts, s2y_ts = create_df_XYZ(S2_rf, 50, 500, 25)

# meta = "_seq_10ms_50ts" 
data_path = "./Havish/XYZ/std_scale/"
meta = "xyz_scaled_50_10"
# "50" is how long is the length of each feature
# 10 ms is the lead time previously extracted

# print(s2x_ts[:1])
# Convert the extracted arrays into numpy files

# with open(data_path + "S1" + meta + "_feat.npy", "wb") as f:
#     np.save(f, s1x_ts)

# with open(data_path + "S1" + meta + "_lab.npy", "wb") as f:
#     np.save(f, s1y_ts)

# with open(data_path + "S2" + meta + "_feat.npy", "wb") as f:
#     np.save(f, s2x_ts)

# with open(data_path + "S2" + meta + "_lab.npy", "wb") as f:
#     np.save(f, s2y_ts)
