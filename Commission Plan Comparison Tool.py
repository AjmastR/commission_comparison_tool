import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from tabulate import tabulate


def get_integer_input(prompt):
    while True:
        try:
            value = int(input(prompt))
            return value
        except ValueError:
            print("Please enter a valid integer.")

def get_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Please enter a valid float.")

def create_model(model_name):
    model = []  # Initialize the model as an empty list
    upper_band_temp = 0
    lower_band_temp = 0
    commission_temp = 0

    number_of_bands = get_integer_input(f"How many bands does {model_name} have? ")
    lower_band_temp = get_integer_input(f"What is the lower band for band 1 in {model_name}? ")
    
    for a in range(number_of_bands):
        b = a + 1
        band_data = []  # Initialize the sublist for each band

        if a == 0:
            band_data.append(lower_band_temp)
        else:
            band_data.append(upper_band_temp + 0.01)

        upper_band_temp = get_integer_input(f"What is the upper band for band {b} in {model_name}? ")

        while upper_band_temp <= band_data[0]:
            print("Sorry, the upper band must be larger than the lower band. Please try again.")
            upper_band_temp = get_integer_input(f"What is the upper band for band {b} in {model_name}? ")

        band_data.append(upper_band_temp)

        commission_temp = get_integer_input(f"What is the rate of commission for band {b} in {model_name}? ")

        while commission_temp < 0 or commission_temp > 100:
            print("Commission has to be less than 100% and larger than 0. Please try again.")
            commission_temp = get_integer_input(f"What is the rate of commission for band {b} in {model_name}? ")

        band_data.append(commission_temp)

        model.append(band_data)

    columns = ["Lower Bound", "Upper Bound", "Commission %"]
    df = pd.DataFrame(model, columns=columns)
    df.index = [f"Band {i+1}" for i in range(len(model))]
    df = df.rename_axis(model_name)

    print(f"\nDataFrame for {model_name}:")
    print(df)
    return model

def generate_data(max_value, start_point, num_points=11):
    """
    Generate a list of values within a specified range.
    
    Args:
        max_value (int): The maximum value.
        start_point (int): The starting point.
        num_points (int): The number of data points.
        
    Returns:
        list: A list of values within the specified range.
    """
    # Generate the list of values
    step = (max_value - start_point) / (num_points - 1)
    data = [start_point + i * step for i in range(num_points)]
    
    return data


def generate_model_data(model, data_list):
    model_data = []
    for income in data_list:
        commission = 0
        for lower_bound, upper_bound, commission_rate in model:
            if lower_bound <= income <= upper_bound:
                commission += (income - lower_bound) * commission_rate / 100
                break
            elif income > upper_bound:
                commission += (upper_bound - lower_bound) * commission_rate / 100
        model_data.append(commission)
    return model_data



def generate_table(models, data_list):
    headers = ["Income"] + [f"Model {i}" for i in range(1, len(models) + 1)]
    table_data = []
    for income in data_list:
        row = [f"£{income:.2f}"]
        for model in models:
            model_data = generate_model_data(model, [income])[0]
            row.append(f"£{model_data:.2f}")
        table_data.append(row)

    print("\nTable:")
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))
    

def plot_models(models, data_list, num_points=500):
    plt.figure(figsize=(10, 6))
    plt.xlabel('Income')
    plt.ylabel('Commission as Percentage of Total Income')
    plt.title('Commission vs. Income')

    for i, model in enumerate(models, start=1):
        model_name = f"Model_{i}"
        model_data = generate_model_data(model, data_list)
        commission_percentages = [commission / income * 100 if income != 0 else 0 for commission, income in zip(model_data, data_list)]

        # Perform spline interpolation
        spline = CubicSpline(data_list, commission_percentages)
        
        # Generate x values for the smooth curve
        x_smooth = np.linspace(min(data_list), max(data_list), num_points)

        # Calculate y values using the spline interpolation
        y_smooth = spline(x_smooth)

        # Plot the smooth curve
        plt.plot(x_smooth, y_smooth, label=f'Spline Interpolation_{i}')

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    while True:
        # Creating models
        num_models = get_integer_input("How many models do you want to create? (Enter -1 to exit) ")
        if num_models == -1:
            return
        models = []
        for i in range(num_models):
            model_name = f"Model {i+1}"
            models.append(create_model(model_name))

        # Generating data
        max_value = get_integer_input("What is the maximum total net income you would like to model till? (Enter -1 to exit) ")
        if max_value == -1:
            return
        start_point = get_integer_input("What income would you like to start from? (Enter -1 to exit) ")
        if start_point == -1:
            return
        data_list = generate_data(max_value, start_point)

        # Plotting models
        plot_models(models, data_list)

        # Generating table
        generate_table(models, data_list)

        choice = input("Do you want to restart or exit? (Type 'restart' or 'exit') ")
        if choice.lower() == 'restart':
            continue
        elif choice.lower() == 'exit':
            return
        else:
            print("Invalid choice. Exiting...")
            return


if __name__ == "__main__":
    main()
