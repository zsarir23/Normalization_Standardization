# Example usage
if __name__ == "__main__":
    import pandas as pd
    from ft_StandarScaler import ft_StandardScaler
    from ft_MinMaxScaleer import ft_MinMaxscalar
    from sklearn.model_selection import train_test_split

    # Sample data
    data = {
        'km': [15000, 30000, 45000, 60000, 75000, 90000, 105000, 120000, 135000, 150000],
        'price': [25000, 22000, 20000, 18000, 16000, 15000, 14000, 13000, 12000, 11000]
    }
    df = pd.DataFrame(data)
    X = df[['km']].values
    y = df['price'].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and use the custom scaler
    # custom_scaler = ft_StandardScaler()
    custom_scaler = ft_MinMaxscalar()
    X_train_scaled = custom_scaler.fit_transform(X_train)
    X_test_scaled = custom_scaler.transform(X_test)

    print("X_train_scaled:")
    print(X_train_scaled)

    print("X_test_scaled:")
    print(X_test_scaled)
