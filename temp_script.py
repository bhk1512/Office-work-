import duckdb
con=duckdb.connect()
df=con.execute("SELECT * FROM read_parquet('Parquets/Stringing/StringingDaily.parquet') LIMIT 5").df()
print(df)
