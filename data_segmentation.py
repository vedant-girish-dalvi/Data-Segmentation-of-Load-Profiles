# Vedant Girish Dalvi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''CR_STATUS = True means CR segments present in system'''
CR_STATUS = False
CR_UPPER_LIMIT = 1000.0
CR_LOWER_LIMIT = 30.0

threshold_dict = {'bigfan': 0.1, 'drill': 0.5, 'dryer': 1.0, 'fanheater': 0.5, 'fanheater-1': 0.5, 'halogenlamp': 1.0, 'ledlamp': 0.00, 'ptcheater': 1.0, 'ricecooker': 1.0}

def insert_zero_current_rows(start_time:float, end_time:float, filepath:str, output_file:str):
    zero_row_added = False
    zero_row_count = 0
    last_row_added = False

    start = start_time
    end = end_time  
    df = pd.read_excel(f"{filepath}")

    # t = df.iterrows()
    # for (i, row1), (j, row2) in zip(t, t):
    #     print(row1, row2) 
    # Iterate through the DataFrame
    last = 0
    for index, row in df.iterrows():

        # first_row_voltage = df.loc[0, 'Voltage Peak (V)']
        current_number = row['Sr. No.']
        current_start_time = row['Start time (s)']
        current_end_time = row['End time(s)']
        previous_end_time = df.loc[df.index[index-1]]['End time(s)']
        # next_start_time = df.loc[df.index[index+1]]['Start time (s)']
        first_row_start_time = df.loc[0, 'Start time (s)']
        last_row_start_time = df.loc[df.index[-1]]['Start time (s)']
        last_row_end_time = df.loc[df.index[-1]]['End time(s)']
        # print(current_end_time, next_start_time)

        if first_row_start_time != start:
            # Create a new row with Mode as "Zero" and other values same as the previous row
            new_row = df.loc[0].copy()
            new_row['Sr. No.'] = current_number - 1
            new_row['Mode'] = 'Zero Current'
            new_row['Current Peak (A)'] = 0
            # new_row['Start Resistance (ohm)'] = 0
            # new_row['End Resistance (ohm)'] = 0
            new_row['Start time (s)'] = start
            new_row['End time(s)'] = current_start_time
            new_row['Time (s)'] = current_start_time
            # Insert the new row at the beginning
            df = pd.concat([pd.DataFrame([new_row]), df]).reset_index(drop=True)

        elif last_row_start_time != end and last_row_added == False:
            last_row = row.copy()
            last_row['Sr. No.'] = current_number + 1
            last_row['Mode'] = 'Zero Current'
            last_row['Current Peak (A)'] = 0
            last_row['Start time (s)'] = last_row_end_time
            last_row['End time(s)'] = end
            last_row['Time (s)'] = end - last_row_end_time
            # Insert the last row at the end
            df.loc[len(df)] = last_row

            # Reset the index
            df = df.reset_index(drop=True)
            last_row_added = True
                            
        # elif current_end_time != next_start_time and zero_row_added == False:
        #     # Create a new row with Mode as "Zero" and other values same as the previous row
        #     new_row = row.copy()
        #     new_row['Sr. No.'] = current_number + 1
        #     new_row['Mode'] = 'Zero Current'
        #     new_row['Current Peak (A)'] = 0
        #     new_row['Start time (s)'] = previous_end_time
        #     new_row['End time(s)'] = current_start_time
        #     df = pd.concat([df.loc[:index+1], pd.DataFrame([new_row]), df.loc[index+1:]]).reset_index(drop=False)
        #     print("Row Added")
        #     zero_row_count += 1
        #     zero_row_added = False
        #     continue

            # if current_mode == "Zero Current":  # Example condition, replace with your own
            # #     zero_row_added = True

        df = df.loc[:, ~df.columns.str.contains('level_0', case=False)]
        df = df.loc[:, ~df.columns.str.contains('index', case=False)]
        # Write the modified DataFrame back to the Excel file
        df.to_excel(f"{output_file}", index=False)
    print("DONE!")

def find_segments(time, current, voltage, threshold):
    segments = []
    start_time = None
    end_time = None
    segment_type = None
    zero_current_count = 0
    seq_count = 0
    voltage_values = []
    current_values = []
    
    for i in range(len(current)):
        if abs(current[i]) < threshold:
            zero_current_count += 1
            if zero_current_count >= 59:                    #Important Parameter : Change as per pattern
                if segment_type != "Zero Current":
                    if start_time is not None:
                        end_time = time[i-1]
                        seq_count += 1
                        # segment_type = "Zero Current"
                        segments.append((seq_count, segment_type, max_voltage, max_current, start_time, end_time, end_time - start_time))
                        # print(f"Mode={segment_type}",f"Time={time[i]}",f"Current={max_current}", f"Voltage={max_voltage}")
            
                        start_time = None
                        end_time = None
                        segment_type = None

            current_values.append(abs(current[i]))
            voltage_values.append(abs(voltage[i]))
            max_current = max(current_values)
            max_voltage = max(voltage_values)
            # print(f"Mode={segment_type}",f"Time={time[i]}",f"Current={max_current}", f"Voltage={max_voltage}")

        else:
            zero_current_count = 0
            if start_time is None:
                start_time = time[i]
            if current[i] > threshold:
                if segment_type is None:
                    segment_type = "CC"

            elif current[i] < -threshold:
                if segment_type is None:
                    segment_type = "CCR"

            current_values.append(abs(current[i]))
            voltage_values.append(abs(voltage[i]))
            max_current = max(current_values)
            max_voltage = max(voltage_values)
            # print(f"Mode={segment_type}",f"Time={time[i]}",f"Current={max_current}", f"Voltage={max_voltage}")
            

    if start_time is not None:
        end_time = time[-1]
        # time = end_time - start_time
        seq_count += 1
        segments.append((seq_count, segment_type, max_voltage, max_current, start_time, end_time, end_time - start_time))
        current_values.clear()
        voltage_values.clear()
    return segments


def find_CR_segments(time, current, voltage, resistance, threshold):
    segments = []
    start_time = None
    end_time = None
    segment_type = None
    zero_current_count = 0
    seq_count = 0
    maximum_current = max(current)
    voltage_values = []
    current_values = []
    time_values = []
    resistance_values = []
    last_time_value = time[-1]
    last_current_value = abs(current[-1])
    last_voltage_value = abs(voltage[-1])
    last_resistance_value = (float(last_voltage_value // last_current_value))
    cc_cr_resistance_values = []   
    cc_cr_time_values = []
    cc_cr_current_values = [] 
    cc_cr_voltage_values = []   
    current_indices = []
    voltage_indices = []

    for i in range(len(current)-1):

        if abs(current[i]) < threshold:
            zero_current_count += 1
            if zero_current_count >= 59:                            #Important Parameter : Change as per pattern
                if segment_type != "Zero Current":
                    if start_time is not None:
                        segment_type = "Zero Current"
                        # seq_count += 1
        

        elif (abs(current[i]) > threshold and abs(current[i]) <= maximum_current) and (abs(current[i+1]) > threshold and abs(current[i+1]) <= maximum_current):
            zero_current_count = 0
            if start_time is None:
                start_time = time[i]
                start_resistance = resistance[i]
                 
            time_values.append(abs(time[i]))
            current_values.append(abs(current[i]))
            current_indices = current_values.index(abs(current[i]))
            voltage_values.append(abs(voltage[i]))
            resistance_values.append(resistance[current_indices])
            max_cr_current = max(current_values)
            max_cr_voltage = max(voltage_values)

    current_values.append(last_current_value)
    if start_time is not None:
        end_time = time[i]
        end_resistance = resistance[i]
        
        # print(resistance_values)
    
        for current_value, next_value in zip(resistance_values, resistance_values[1:]):
            if CR_LOWER_LIMIT <= current_value <= CR_UPPER_LIMIT and CR_LOWER_LIMIT <= next_value <= CR_UPPER_LIMIT:
          
                segment_type = "CR"
                cc_cr_resistance_values.append(current_value)
                index = resistance_values.index(current_value)

                cc_cr_time_values.append(time_values[index])
                cc_cr_current_values.append(current_values[index])
                cc_cr_voltage_values.append(voltage_values[index])

                # resistance_values.pop(index)
                # time_values.pop(index)
                # current_values.pop(index)
                # voltage_values.pop(index)

            elif CR_LOWER_LIMIT <= current_value <= CR_UPPER_LIMIT or next_value > CR_UPPER_LIMIT or next_value < CR_LOWER_LIMIT:
                continue
            

        if CR_LOWER_LIMIT <= last_resistance_value <= CR_UPPER_LIMIT:
                cc_cr_resistance_values.append(last_resistance_value)
                cc_cr_time_values.append(last_time_value)
                cc_cr_current_values.append(last_current_value)
                cc_cr_voltage_values.append(last_voltage_value)
        else:
                resistance_values.append(last_resistance_value)
                time_values.append(last_time_value)
                current_values.append(last_current_value)
                voltage_values.append(last_voltage_value)
        
        max_voltage = max(cc_cr_voltage_values)
        max_current = max(cc_cr_current_values)

        # print(time_values[0], time_values[-1])
        # print(cc_cr_time_values[0], cc_cr_time_values[-1])
        
        # print(time_values)
        # print(cc_cr_time_values)

        # print(len(resistance_values))
        # print(len(cc_cr_resistance_values))

        # print(resistance_values[0], resistance_values[-1])
        # print(cc_cr_resistance_values[0], cc_cr_resistance_values[-1])

        # max_current = max()
        # print(len(time_values))
        # print(len(cc_cr_time_values))
                
    
        seq_count += 1

        segments.append((seq_count, segment_type, max_cr_voltage, max_cr_current, start_resistance, end_resistance, start_time, cc_cr_time_values[0], cc_cr_time_values[0] - start_time))

        seq_count += 1
        segments.append((seq_count, segment_type, max_voltage, max_current, end_resistance, last_resistance_value, cc_cr_time_values[0], end_time, end_time - cc_cr_time_values[0]))

        current_values.clear()
        voltage_values.clear()
        resistance_values.clear()
    return segments

def process_datafile(excel_file, threshold=1.0):                
    '''THRESHOLD : Change as per waveform'''

    df = pd.read_excel(excel_file)


    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['Current'] = pd.to_numeric(df['Current'], errors='coerce')
    df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')

    df.dropna(subset=['Time', 'Current', 'Voltage'], inplace=True)

    column_name = 'Current'
  
    waveform_values = df[column_name].values
 
    std_deviation = np.std(waveform_values)

    print(std_deviation)

    # threshold = 2
    # global CR_STATUS
    # if std_deviation > threshold:
    #     CR_STATUS = True
    # else:
    #     CR_STATUS = False

    time = df['Time'].to_list()
    start = time[0]
    end = time[-1]
    current = df['Current'].to_list()  
    voltage = df['Voltage'].to_list()
    resistance = []

    for i in range(len(current)):
        res = (voltage[i] // current[i])
        res = float(abs(res))
        resistance.append(res)

    # print(f"Resistance Values= {resistance}")
    if CR_STATUS == True:
        segments = find_CR_segments(time, current, voltage, resistance, threshold)

    else:
        segments = find_segments(time, current, voltage, threshold)
 
    return segments, start, end


if __name__ == "__main__":

    '''Read Data File'''
    excel_folder_path = r"C:/Users/DAV1SI/Downloads/V2L/"
    excel_file_path = "drill-2023-10-13-11-05-33-Transient-Scope.xlsx"
    segments, start, end = process_datafile(excel_folder_path + excel_file_path)
    print(CR_STATUS)
    '''Add Detected Segments to Dataframe & Save Excel file'''
    if CR_STATUS == True:
        report_df = pd.DataFrame(segments, columns=["Sr. No.","Mode", "Voltage Peak (V)", "Current Peak (A)", "Start Resistance (ohm)", "End Resistance (ohm)", "Start time (s)", "End time(s)", "Time (s)"])

    else:
        report_df = pd.DataFrame(segments, columns=["Sr. No.","Mode", "Voltage Peak (V)", "Current Peak (A)", "Start time (s)", "End time(s)", "Time (s)"])

    report_df.to_excel(f"{excel_file_path.split('-')[0]}_report.xlsx", index=False)

    '''Insert Zero Current Segments to Dataframe and Save FINAL RESULT'''
    result_df = insert_zero_current_rows(start, end, f"{excel_file_path.split('-')[0]}_report.xlsx",fr"C:/Users/DAV1SI/Downloads/{excel_file_path.split('-')[0]}_results.xlsx")

    print(f"Report saved to {excel_file_path.split('-')[0]}_results.xlsx")