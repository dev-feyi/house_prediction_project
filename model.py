import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os


class HousePriceModel:
    """
    House Price Prediction Model
    Uses Random Forest Regression for better accuracy with multiple features
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'square_feet', 'bedrooms', 'bathrooms',
            'age_years', 'garage_spaces', 'location_score'
        ]

    def load_data(self, file_path='data/housing_data.csv'):
        """
        Load housing data from CSV file
        Returns: X (features), y (target prices)
        """
        try:
            df = pd.read_csv(file_path)
            print(f"✓ Loaded {len(df)} housing records")

            # Separate features (X) and target (y)
            X = df[self.feature_names]
            y = df['price']

            return X, y
        except FileNotFoundError:
            print(f"✗ Error: {file_path} not found")
            return None, None

    def train(self, X, y):
        """
        Train the house price prediction model
        Uses 80% data for training, 20% for testing
        """
        print("\n--- Training Model ---")

        # Split data: 80% training, 20% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features (normalize values for better performance)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest model
        # n_estimators=100 means 100 decision trees
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=2
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model performance
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)

        train_mae = mean_absolute_error(y_train, train_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)

        print(f"\nTraining Results:")
        print(f"  Training MAE: ${train_mae:,.2f}")
        print(f"  Testing MAE: ${test_mae:,.2f}")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Testing R²: {test_r2:.4f}")

        # Show feature importance
        self._show_feature_importance()

        return test_mae, test_r2

    def _show_feature_importance(self):
        """Display which features matter most for predictions"""
        if self.model:
            importance = self.model.feature_importances_
            print("\nFeature Importance:")
            for name, imp in zip(self.feature_names, importance):
                print(f"  {name}: {imp:.4f}")

    def predict(self, square_feet, bedrooms, bathrooms, age_years,
                garage_spaces, location_score):
        """
        Predict house price based on input features
        Returns: Predicted price (float)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Create feature array
        features = np.array([[
            square_feet, bedrooms, bathrooms,
            age_years, garage_spaces, location_score
        ]])

        # Scale features using same scaler from training
        features_scaled = self.scaler.transform(features)

        # Make prediction
        prediction = self.model.predict(features_scaled)[0]

        return prediction

    def save_model(self, model_dir='model_files'):
        """Save trained model and scaler to disk"""
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, 'price_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        print(f"\n✓ Model saved to {model_path}")
        print(f"✓ Scaler saved to {scaler_path}")

    def load_model(self, model_dir='model_files'):
        """Load pre-trained model and scaler from disk"""
        model_path = os.path.join(model_dir, 'price_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"✓ Model loaded from {model_path}")
            return True
        except FileNotFoundError:
            print(f"✗ Model files not found in {model_dir}")
            return False


def train_and_save_model():
    """
    Main training function - run this to create the model
    """
    print("=" * 50)
    print("HOUSE PRICE PREDICTION MODEL TRAINING")
    print("=" * 50)

    # Initialize model
    hp_model = HousePriceModel()

    # Load data
    X, y = hp_model.load_data()
    if X is None:
        return

    # Train model
    hp_model.train(X, y)

    # Save trained model
    hp_model.save_model()

    # Test prediction
    print("\n--- Testing Sample Prediction ---")
    test_price = hp_model.predict(
        square_feet=2000,
        bedrooms=3,
        bathrooms=2.5,
        age_years=5,
        garage_spaces=2,
        location_score=8
    )
    print(f"Predicted price for sample house: ${test_price:,.2f}")

    print("\n" + "=" * 50)
    print("✓ Training Complete!")
    print("=" * 50)


if __name__ == "__main__":
    train_and_save_model()