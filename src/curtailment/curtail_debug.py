import pandas as pd

A = pd.read_parquet("data/processed/availability_region_hourly_calibrated/availability_regions_2025_calibrated.parquet")
A["datetime"] = pd.to_datetime(A["datetime"])
times = pd.to_datetime([
 "2025-05-06 09:00:00","2025-05-06 12:00:00","2025-05-06 13:00:00","2025-05-06 14:00:00","2025-05-06 15:00:00",
 "2025-06-10 12:00:00","2025-06-10 13:00:00","2025-06-10 14:00:00"
])
print(A[A["datetime"].isin(times)].groupby("datetime").size())
