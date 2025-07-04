import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta

def simulate_sensor_data():
    base_time = datetime(2025, 6, 1, 8)
    fill_levels = [20, 35, 50, 70, 85, 95]
    data = []
    for i, fill in enumerate(fill_levels):
        timestamp = base_time + timedelta(hours=i * 2)
        temp = 28 + i * 1.2
        data.append([timestamp, 'BIN01', fill, round(temp, 1)])
    df = pd.DataFrame(data, columns=['timestamp', 'bin_id', 'fill_level_percent', 'temperature'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def plot_fill_levels(df):
    plt.figure(figsize=(10,5))
    plt.plot(df['timestamp'], df['fill_level_percent'], marker='o', color='green')
    plt.title('Smart Dustbin Fill Level Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Fill Level (%)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("fill_level_plot.png")  # Save figure as PNG file
    plt.show()

def predict_fill_levels(df):
    df['elapsed_hours'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600
    X = df[['elapsed_hours']]
    y = df['fill_level_percent']
    model = LinearRegression()
    model.fit(X, y)
    
    future_hours = np.array([[h] for h in range(int(df['elapsed_hours'].max()) + 2, int(df['elapsed_hours'].max()) + 13, 2)])
    predicted_fill = model.predict(future_hours)

    plt.figure(figsize=(10,5))
    plt.plot(df['elapsed_hours'], y, 'go-', label='Actual')
    plt.plot(future_hours, predicted_fill, 'ro--', label='Predicted')
    plt.title('Actual vs Predicted Bin Fill Level')
    plt.xlabel('Elapsed Hours')
    plt.ylabel('Fill Level (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("predicted_fill_plot.png")  # Save predicted plot
    plt.show()

    return predicted_fill

def main():
    print("Simulating Smart Dustbin sensor data...")
    df = simulate_sensor_data()
    print(df)

    print("\nPlotting fill levels...")
    plot_fill_levels(df)

    print("\nPredicting future fill levels...")
    predicted_fill = predict_fill_levels(df)
    print("\nPredicted future fill levels (%):")
    print(predicted_fill.round(2))

if __name__ == "__main__":
    main()
