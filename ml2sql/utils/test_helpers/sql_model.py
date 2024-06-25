import duckdb


def execute_sql_script(sql_script, df, output_column):
    # Replace the <source_table> placeholder with the actual table name
    sql_script = sql_script.replace("<source_table>", "test_df")
    print(sql_script)
    # Create a DuckDB connection and register the DataFrame
    conn = duckdb.connect()
    conn.register("test_df", df)

    # Execute the SQL script
    results = conn.execute(sql_script).fetchdf()

    # Close the connection
    conn.close()

    # Return the selected column from the results
    return results[output_column]
