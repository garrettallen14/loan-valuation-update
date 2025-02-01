"""Main workflow for loan valuation and cash flow projections.

This module orchestrates the loan valuation process by:
1. Loading and preparing loan data
2. Calculating yields and prices
3. Generating cash flow projections by monthboard
4. Exporting results to Excel
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
from loan_calculator import (
    CashFlowCalculator,
    LoanTerms,
    LoanDates,
    PrepaymentVectors
)

@dataclass
class Config:
    """Configuration for prepayment vectors and calculation target date.
    
    Attributes:
        cpr_vector: Space-separated string of CPR values
        cdr_vector: Space-separated string of CDR values
        severity_vector: Space-separated string of severity values
        coupon_mult_vector: Space-separated string of coupon multipliers
        target_year: Year to normalize payment dates to
        target_month: Month to normalize payment dates to
    """
    cpr_vector: str
    cdr_vector: str
    severity_vector: str
    coupon_mult_vector: str
    target_year: int
    target_month: int

class LoanDataProcessor:
    """Handles loan data loading, preparation, and calculations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.calculator = CashFlowCalculator()

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load loan data from Excel file and filter for active loans (ASTAT = 0)."""
        df = pd.read_excel(file_path)
        return df  # Filter active loans
        # return df[df['ASTAT'] == 0]  # Filter active loans

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare loan data for processing.
        
        1. Convert date columns to datetime
        2. Add prepayment vectors
        3. Normalize next payment dates to target month/year
        """
        # Convert date columns to datetime
        date_columns = ['ABKDTDT', 'ANXDTDT', 'ACODTDT', 'AUD3DT', 'AMTDTDT']
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])

        # Add prepayment vectors from config
        df['CPR'] = self.config.cpr_vector
        df['CDR'] = self.config.cdr_vector
        df['Severity'] = self.config.severity_vector
        df['COUPON_MULT'] = self.config.coupon_mult_vector

        # Normalize next payment dates to target month/year
        df['ANXDTDT'] = df['ANXDTDT'].apply(
            lambda x: self._normalize_next_payment_date(x)
        )

        return df

    def _normalize_next_payment_date(self, date: datetime) -> datetime:
        """Normalize payment date to November 2024 while preserving day of month.
        
        If November 2024 has fewer days than the original date's day,
        use the last day of November 2024 instead.
        """
        # Calculate last day of November 2024
        last_day = 30
        
        # Use original day if valid, otherwise use last day of month
        target_day = min(date.day, last_day)
        return datetime(2024, 11, target_day)

    def calculate_yields(self, df: pd.DataFrame) -> List[Optional[float]]:
        """Calculate yields for all loans with prices.
        
        For each loan:
        1. Create LoanTerms from loan data
        2. Create LoanDates from loan dates
        3. Create PrepaymentVectors from vectors
        4. Calculate yield using price
        """
        yields = []
        
        for _, row in tqdm(df.iterrows(), desc="Calculating Yields"):
            if pd.notna(row['PRICE']):
                # Create loan terms
                terms = LoanTerms(
                    monthly_payment=row['APMT1'],
                    remaining_payments=row['AOTRM'],
                    original_balance=row['AOFIN'],
                    current_balance=row['AOFIN'],
                    interest_rate=row['ARATE'],
                    original_term=row['AOTRM']
                )
                
                # Create loan dates
                dates = LoanDates(
                    calculation_date=row['ACODTDT'],
                    next_payment_date=row['AUD3DT'],
                    contract_date=row['ACODTDT'],
                    maturity_date=row['AMTDTDT']
                )

                # Create prepayment vectors
                vectors = PrepaymentVectors.from_strings(
                    cpr=row['CPR'],
                    cdr=row['CDR'],
                    severity=row['Severity'],
                    coupon_mult=row['COUPON_MULT']
                )
                
                # Calculate yield
                loan_yield = self.calculator.calculate_yield(terms, dates, vectors, row['PRICE'])
            else:
                loan_yield = None
                
            yields.append(loan_yield)
            
        return yields

    def calculate_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate new prices based on original yields.
        
        For each loan:
        1. Create LoanTerms from current loan state
        2. Create LoanDates from current dates
        3. Create PrepaymentVectors
        4. Calculate price using original yield
        5. Store processed vectors
        """
        results = {
            'prices': [],
            'cpr': [],
            'cdr': [],
            'severity': [],
            'coupon_mult': []
        }
        
        for _, row in tqdm(df.iterrows(), desc="Calculating Prices"):
            if pd.notna(row['PRICE']) and pd.notna(row['Original Yields']):
                # Create loan terms
                terms = LoanTerms(
                    monthly_payment=row['APMT1'],
                    remaining_payments=row['REMPMTS'],
                    original_balance=row['AOFIN'],
                    current_balance=row['ANETBAL'],
                    interest_rate=row['ARATE'],
                    original_term=row['AOTRM']
                )
                
                # Create loan dates
                dates = LoanDates(
                    calculation_date=row['ABKDTDT'],
                    next_payment_date=row['ANXDTDT'],
                    contract_date=row['ACODTDT'],
                    maturity_date=row['AMTDTDT'],
                    reference_date=row['ABKDTDT']
                )

                # Create prepayment vectors
                vectors = PrepaymentVectors.from_strings(
                    cpr=row['CPR'],
                    cdr=row['CDR'],
                    severity=row['Severity'],
                    coupon_mult=row['COUPON_MULT']
                )
                
                # Calculate price and get processed vectors
                npv, processed_vectors = self.calculator.calculate_price(
                    terms, dates, vectors, row['Original Yields']
                )
                
                # Convert NPV to price as percentage of current balance
                if row['ANETBAL'] == 0:
                    price = 0
                else:
                    price = npv * 100 / row['ANETBAL']
                
                # Store results
                results['prices'].append(price)
                results['cpr'].append(processed_vectors.cpr)
                results['cdr'].append(processed_vectors.cdr)
                results['severity'].append(processed_vectors.severity)
                results['coupon_mult'].append(processed_vectors.coupon_mult)
            else:
                for key in results:
                    results[key].append(None)
                    
        return results

class MonthboardProcessor:
    """Processes loans by monthboard to generate cash flow projections."""
    
    def __init__(self, config: Config):
        self.config = config
        self.calculator = CashFlowCalculator()

    def process_monthboard(self, df: pd.DataFrame, monthboard: int) -> Dict[str, pd.DataFrame]:
        """Process loans in a monthboard to generate cash flows.
        
        For each loan:
        1. Generate current cash flows from current state
        2. Generate original cash flows from origination
        3. Combine cash flows across loans
        
        Returns both current and original cash flows.
        """
        current_dfs = []
        original_dfs = []
        
        for _, row in tqdm(df[df['MONTHBOARD'] == monthboard].iterrows(),
                          desc=f"Processing Monthboard {monthboard}"):
            if pd.notna(row['PRICE']):

                vectors = PrepaymentVectors.from_strings(
                    cpr=row['CPR'],
                    cdr=row['CDR'],
                    severity=row['Severity'],
                    coupon_mult=row['COUPON_MULT']
                )
                
                # If the loan is a performing loan, we will add it to the current monthboard underwriting
                if row['ASTAT'] == 0:
                    # Generate current cash flows
                    current_terms = LoanTerms(
                        monthly_payment=row['APMT1'],
                        remaining_payments=row['REMPMTS'],
                        original_balance=row['AOFIN'],
                        current_balance=row['ANETBAL'],
                        interest_rate=row['ARATE'],
                        original_term=row['AOTRM']
                    )
                    
                    # For current cash flows, use bank date as reference
                    current_dates = LoanDates(
                        calculation_date=row['ABKDTDT'],
                        next_payment_date=row['ANXDTDT'],
                        contract_date=row['ACODTDT'],
                        maturity_date=row['AMTDTDT'],
                        reference_date=row['ABKDTDT']
                    )

                    
                    current_df, _, _, _ = self.calculator.project_cashflows(
                        current_terms, current_dates, vectors
                    )
                    current_dfs.append(current_df)
                
                # Generate original cash flows
                start_date = row['ACODTDT']  # 07/31/2023

                # Calculate how many months we need to push dates forward
                monthboard_date = datetime(monthboard // 100, monthboard % 100, 1)  # 10/1/2023
                aud3dt_period = pd.Period(row['AUD3DT'], freq='M')
                monthboard_period = pd.Period(monthboard_date, freq='M')
                months_to_push = max(0, (monthboard_period - aud3dt_period).n)
                
                # Adjust next payment date
                next_payment = row['AUD3DT'] if months_to_push == 0 else monthboard_date
                
                # Adjust maturity date by the same number of months
                adjusted_maturity = row['AMTDTDT'] + pd.DateOffset(months=months_to_push)
                
                if months_to_push > 0:
                    print(f"Pushing dates forward by {months_to_push} months for loan. "
                          f"Next payment moved from {row['AUD3DT'].strftime('%Y-%m-%d')} to {next_payment.strftime('%Y-%m-%d')}, "
                          f"Maturity moved from {row['AMTDTDT'].strftime('%Y-%m-%d')} to {adjusted_maturity.strftime('%Y-%m-%d')}")
                
                original_dates = LoanDates(
                    calculation_date=start_date,
                    next_payment_date=next_payment,
                    contract_date=row['ACODTDT'],
                    maturity_date=adjusted_maturity,
                    reference_date=start_date
                )
                
                original_terms = LoanTerms(
                    monthly_payment=row['APMT1'],
                    remaining_payments=row['AOTRM'],
                    original_balance=row['AOFIN'],
                    current_balance=row['AOFIN'],
                    interest_rate=row['ARATE'],
                    original_term=row['AOTRM']
                )
                
                original_df, _, _, _ = self.calculator.project_cashflows(
                    original_terms, original_dates, vectors
                )
                original_dfs.append(original_df)
        
        return {
            'current': self._combine_monthboard_dfs(current_dfs, monthboard_date),
            'original': self._combine_monthboard_dfs(original_dfs, monthboard_date)
        }

    def _combine_monthboard_dfs(self, dfs: List[pd.DataFrame], monthboard_date: datetime) -> pd.DataFrame:
        """Combine multiple cash flow DataFrames into monthly totals.
        
        1. Convert dates to datetime
        2. Set date as index
        3. Impute missing months with previous UPB values
        4. Combine all DataFrames
        5. Resample to month end:
           - Sum for cash flow columns
           - Mean for UPB columns (point-in-time balances)
        6. Handle UPB values for month before monthboard:
           - Remove rows before monthboard - 1 month
           - Validate zero values for non-UPB columns
           - Set UPB to initial sum
           
        Args:
            dfs: List of DataFrames to combine
            monthboard_date: The date of the monthboard being processed
        """
        if not dfs:
            return pd.DataFrame()
            
        # Calculate initial UPB sum at the start
        initial_upb_sum = sum(df['UPB'].iloc[0] for df in dfs)
            
        processed_dfs = []
        for df_idx, df in enumerate(dfs):
            print(f"\nProcessing DataFrame {df_idx + 1}/{len(dfs)}")
            print(f"Initial UPB sum: {initial_upb_sum}")
            
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            balance_columns = ['UPB', 'Performing UPB']
            agg_dict = {}
            for col in df.columns:
                if col in balance_columns:
                    agg_dict[col] = 'max'
                else:
                    agg_dict[col] = 'sum'

            # Create a new DataFrame after resampling
            resampled_df = df.resample('ME').agg(agg_dict)
            resampled_df.reset_index(inplace=True)
            
            # print("\nBefore forward filling:")
            # print(resampled_df[['Date', 'UPB', 'Performing UPB']].head())
            # print(resampled_df[['Date', 'UPB', 'Performing UPB']].tail())
            
            # Forward fill UPB values until we hit non-zero values
            for i in range(1, len(resampled_df)):
                # If we hit a non-zero UPB, stop forward filling
                if resampled_df.loc[i, 'UPB'] > 0:
                    print(f"\nStopping at row {i} due to non-zero UPB: {resampled_df.loc[i, 'UPB']}")
                    break
                    
                # Otherwise continue forward filling
                if pd.isna(resampled_df.loc[i, 'UPB']) or resampled_df.loc[i, 'UPB'] == 0:
                    print(f"Row {i}: Filling UPB from {resampled_df.loc[i, 'UPB']} to {initial_upb_sum}")
                    resampled_df.loc[i, 'UPB'] = initial_upb_sum
                if pd.isna(resampled_df.loc[i, 'Performing UPB']) or resampled_df.loc[i, 'Performing UPB'] == 0:
                    print(f"Row {i}: Filling Performing UPB from {resampled_df.loc[i, 'Performing UPB']} to {initial_upb_sum}")
                    resampled_df.loc[i, 'Performing UPB'] = initial_upb_sum
            
            # print("\nAfter forward filling:")
            # print(resampled_df[['Date', 'UPB', 'Performing UPB']].head())
            # print(resampled_df[['Date', 'UPB', 'Performing UPB']].tail())
            
            processed_dfs.append(resampled_df)

        # Combine DataFrames
        if not processed_dfs:
            return pd.DataFrame()
            
        combined = pd.concat(processed_dfs, ignore_index=True)
        
        # Set Date as index again for final resampling
        combined['Date'] = pd.to_datetime(combined['Date'])
        combined.set_index('Date', inplace=True)
        
        # Define how to aggregate each column
        balance_columns = ['UPB', 'Performing UPB']
        agg_dict = {}
        for col in combined.columns:
            if col in balance_columns:
                agg_dict[col] = 'sum'
            else:
                agg_dict[col] = 'sum'

        # Resample to month-end ('ME') with the specified aggregation
        monthly = combined.resample('ME').agg(agg_dict)

        # Reset the index so 'Date' is back as a column
        monthly.reset_index(inplace=True)
        
        # Calculate cutoff date (monthboard - 1 month)
        cutoff_date = monthboard_date - pd.DateOffset(months=1)
        
        # Check if the first date is already at or after the cutoff
        if not monthly.empty and monthly['Date'].iloc[0] >= cutoff_date:
            print(f"First date already cutoff for monthboard {monthboard_date.strftime('%Y-%m')}")
            return monthly
            
        # Remove rows before monthboard - 1 month
        monthly = monthly[monthly['Date'] >= cutoff_date]
        
        # Find the row for monthboard - 1 month by matching year and month
        pre_monthboard_mask = monthly['Date'].dt.to_period('M') == cutoff_date.to_period('M')
        if any(pre_monthboard_mask):
            # Verify all non-UPB columns are zero
            non_upb_cols = [col for col in monthly.columns if col not in ['Date', 'UPB', 'Performing UPB']]
            for col in non_upb_cols:
                if monthly.loc[pre_monthboard_mask, col].iloc[0] > 0:
                    print(f"Warning: Found non-zero value in {col} for month before monthboard")
            
            # Set UPB values to initial sum
            monthly.loc[pre_monthboard_mask, 'UPB'] = initial_upb_sum
            monthly.loc[pre_monthboard_mask, 'Performing UPB'] = initial_upb_sum
            print(f"Setting UPB and Perf UPB to: {initial_upb_sum} for {cutoff_date.strftime('%Y-%m')}")
        
        return monthly

class ExcelExporter:
    """Handles exporting results to Excel files."""
    
    def export_yields_and_prices(self, df: pd.DataFrame, filename: str = 'Original Yields and New Prices.xlsx'):
        """Export yields and prices to Excel."""
        df.to_excel(f'data/{filename}', index=False)

    def export_monthboards(self, results: Dict[int, Dict[str, pd.DataFrame]]):
        """Export monthboard cash flows to separate Excel files."""
        for monthboard, data in results.items():
            for flow_type, df in data.items():
                filename = f'data/monthboard_{monthboard}_{flow_type}.xlsx'
                df.to_excel(filename, index=False)

def analyze_sample_loans(df: pd.DataFrame, config: Config) -> None:
    """Analyze representative loans based on maturity dates.
    
    1. Find early, middle, and late maturity loans
    2. Generate October and original cash flows for each
    3. Output to Excel with descriptive names in data folder
    """
    # Filter valid maturity dates after target date
    target_date = pd.Timestamp(config.target_year, config.target_month, 1)
    valid_df = df[
        (df['AMTDTDT'] >= target_date) & 
        (df['PRICE'].notna())
    ]
    
    if len(valid_df) < 3:
        print("Not enough valid loans for sample analysis")
        return
        
    # Find representative indices
    sorted_df = valid_df.sort_values('AMTDTDT')
    early_idx = sorted_df.index[0]
    middle_idx = sorted_df.index[len(sorted_df)//2]
    late_idx = sorted_df.index[-1]
    
    # Create sample processor with same config
    processor = MonthboardProcessor(config)
    
    # Process each sample loan
    for loan_type, idx in [('early', early_idx), ('middle', middle_idx), ('late', late_idx)]:
        # Create sample DataFrame with just this loan
        sample_df = df.loc[[idx]].copy()
        
        # Set monthboard to October 202307 for current flows
        october_monthboard = 202307
        sample_df['MONTHBOARD'] = october_monthboard
        
        # Generate cash flows
        flows = processor.process_monthboard(sample_df, october_monthboard)
        
        # Export with descriptive names
        for flow_type, flow_df in flows.items():
            filename = f"data/{idx}_{loan_type}_{flow_type}.xlsx"
            flow_df.to_excel(filename, index=False)
            print(f"Generated {filename}")

def main():
    """Main execution flow.
    
    1. Initialize configuration with prepayment vectors
    2. Load and prepare loan data
    3. Calculate original yields
    4. Calculate new prices
    5. Process monthboard cash flows
    6. Export all results
    """
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)

    old = "0 0 0 0 0 2.37 4.7 6.97 8.75 9.08 9.41 9.74 10.06 10.39 10.71 11.04 11.36 11.68 12 12.32 12.64 12.96 13.28 13.59 13.91 14.22 14.53 14.84 15.15 15.46 15.77 16.08 16.38 16.69 16.99 17.3 17.6 17.9 18.2 18.5 18.8 19.09 19.39 19.68 19.98 20.27 20.56 20.85 21.14 21.53 21.53"

    # Best Case CDR values:
    best = "0.0000 0.0000 0.0000 0.0000 0.0000 1.9615 3.8892 5.7882 7.2821 7.5607 7.8401 8.1195 8.3997 8.6807 8.9617 9.2435 9.5253 9.8079 10.0906 10.3741 10.6577 10.9422 11.2267 11.5121 11.7976 12.0840 12.3705 12.6579 12.9454 13.2338 13.5224 13.8110 14.1005 14.3902 14.6808 14.9715 15.2632 15.5550 15.8477 16.1406 16.4345 16.7285 17.0235 17.3186 17.6497 17.6497 17.6497 17.6497 17.6497 17.6497"

    # Base Case CDR values:
    base = "0.0000 0.0000 0.0000 0.0000 0.0000 2.2157 4.3897 6.5316 8.2109 8.5219 8.8329 9.1439 9.4557 9.7683 10.0809 10.3943 10.7086 11.0229 11.3381 11.6533 11.9694 12.2856 12.6027 12.9207 13.2388 13.5578 13.8769 14.1969 14.5170 14.8380 15.1591 15.4812 15.8034 16.1265 16.4498 16.7740 17.0984 17.4237 17.7492 18.0756 18.4022 18.7297 19.0574 19.3860 19.7397 19.7397 19.7397 19.7397 19.7397 19.7397"

    # Worst Case CDR values:
    worst = "0.0000 0.0000 0.0000 0.0000 0.0000 2.4701 4.8907 7.2757 9.1404 9.4838 9.8272 10.1706 10.5148 10.8590 11.2040 11.5490 11.8948 12.2407 12.5874 12.9342 13.2818 13.6295 13.9780 14.3267 14.6762 15.0258 15.3762 15.7267 16.0780 16.4295 16.7818 17.1342 17.4874 17.8407 18.1949 18.5492 18.9043 19.2596 19.6157 19.9719 20.3289 20.6860 21.0440 21.4020 21.7947 21.7947 21.7947 21.7947 21.7947 21.7947"
    
    # Initialize configuration with prepayment vectors
    config = Config(
        cpr_vector="3.66 4.57 5.48 6.39 7.3 8.2 9.11 10.02 11.29 11.87 12.33 12.66 12.87 12.95 12.91 12.74 12.44 12.02 11.48 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5 11.5",
        cdr_vector=base,
        severity_vector="70",
        coupon_mult_vector="1.000 0.987 0.967 0.956 0.948 0.942 0.937 0.932 0.928 0.925 0.922 0.920 0.917 0.915 0.913 0.911 0.909 0.907 0.906 0.904 0.903 0.901 0.900 0.899 0.898 0.897 0.895 0.894 0.893 0.892 0.891 0.891 0.890 0.889 0.888 0.887 0.886 0.886 0.885 0.884 0.883 0.883 0.882 0.881 0.881 0.880 0.880 0.880 0.880",
        target_year=2024,
        target_month=11
    )

    # Initialize processors
    data_processor = LoanDataProcessor(config)
    monthboard_processor = MonthboardProcessor(config)
    exporter = ExcelExporter()

    # Load and prepare data
    print("Loading and preparing data...")
    df = data_processor.load_data('Yields and Prices.xlsx')
    df = data_processor.prepare_data(df)
    
    # Analyze sample loans first
    print("\nAnalyzing sample loans...")
    analyze_sample_loans(df, config)

    # Get performing loans (ASTAT = 0)
    performing_loans = df[df['ASTAT'] == 0].copy()

    # Calculate yields
    print("Calculating yields...")
    yields = data_processor.calculate_yields(performing_loans)
    performing_loans['Original Yields'] = yields

    # Calculate prices
    print("Calculating prices...")
    price_results = data_processor.calculate_prices(performing_loans)
    performing_loans['New Price'] = price_results['prices']
    performing_loans['Trimmed CPR'] = price_results['cpr']
    performing_loans['Trimmed CDR'] = price_results['cdr']
    performing_loans['Trimmed Severity'] = price_results['severity']
    performing_loans['Trimmed COUPON_MULT'] = price_results['coupon_mult']

    # Update original dataframe with results
    df.update(performing_loans)

    # Export yields and prices
    print("Exporting yields and prices...")
    exporter.export_yields_and_prices(performing_loans)

    # Process monthboards
    print("Processing monthboards...")
    monthboard_results = {}
    for monthboard in df['MONTHBOARD'].unique():
        monthboard_results[monthboard] = monthboard_processor.process_monthboard(df, monthboard)

    # Export monthboard results
    print("Exporting monthboard results...")
    exporter.export_monthboards(monthboard_results)

    print("Processing complete!")

if __name__ == "__main__":
    main()
