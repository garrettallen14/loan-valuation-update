import logging
from typing import Union
import numpy as np
import pandas as pd
from pandas import Timestamp
from typing import List, Union, Optional
import logging
from tqdm import tqdm

class VectorHandler:
    """Handles processing and manipulation of rate vectors"""
    
    @staticmethod
    def process_vector(vector_input: Union[str, np.ndarray, float], 
                      loan_age: int = 0, 
                      extend_length: int = 0) -> np.ndarray:
        """Process rate vectors with comprehensive error handling"""
        try:
            # Convert input to numpy array
            if isinstance(vector_input, np.ndarray):
                vector = vector_input.copy()
            elif isinstance(vector_input, str):
                if ' ' in vector_input:
                    vector = np.array([float(x) for x in vector_input.split()])
                else:
                    vector = np.array([float(vector_input)])
            else:
                vector = np.array([float(vector_input)])

            # Convert percentages to decimals
            vector = np.where(vector > 1, vector / 100, vector)
            
            # Handle loan age trimming
            if loan_age > 0:
                if loan_age >= len(vector):
                    vector = np.array([vector[-1]])
                else:
                    vector = vector[loan_age:]
            
            # Ensure minimum length of 2
            while len(vector) < 2:
                vector = np.append(vector, vector[-1])
            
            # Extend to required length
            if extend_length > 0:
                current_length = len(vector)
                if extend_length > current_length:
                    vector = np.append(vector, [vector[-1]] * (extend_length - current_length))
            
            return vector
            
        except Exception as e:
            logging.error(f"Error processing vector: {str(e)}")
            # Return safe default vector
            return np.array([0.0] * max(2, extend_length))

    @staticmethod
    def calculate_monthly_rate(annual_rate: np.ndarray) -> np.ndarray:
        """Convert annual rate to monthly rate"""
        try:
            return 1 - (1 - annual_rate) ** (1/12)
        except Exception as e:
            logging.error(f"Error calculating monthly rate: {str(e)}")
            return np.zeros_like(annual_rate)

def normalize_next_payment_date(date: Timestamp, target_year: int = 2024, target_month: int = 11) -> Timestamp:
    """Normalize payment date to target month/year while preserving day (capped at 30)"""
    day = min(date.day, 30)  # Cap at 30 days
    return Timestamp(f'{target_year}-{target_month:02d}-{day:02d}')