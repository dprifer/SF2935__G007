import pandas as pd


class removeRows:
    """
    Usage:
    import cleanup

    file_path = 'project_train.csv'
    rows_to_remove = [85, 95]  # These correspond to energy and loudness outliers

    prj_train_new, headers = cleanup.removeRows(file_path, rows_to_remove).process_rows()
    """
    def __init__(self, file_path, rows_to_remove=None):
        # Default
        if rows_to_remove is None:
            rows_to_remove = [85, 95]

        self.file_path = file_path
        self.rows_to_remove = rows_to_remove
        self.df = pd.read_csv(self.file_path)

    def process_rows(self):
        for row_num in self.rows_to_remove:
            # Check if the row exists
            if 0 <= row_num - 1 < len(self.df):
                print(f"\nRow {row_num} will be removed:")
                print(self.df.iloc[row_num - 1])
                # Drop the row
                self.df = self.df.drop(index=row_num - 1)
            else:
                print(f"\nRow {row_num} does not exist.")

        # Reset the index after row removal
        self.df = self.df.reset_index(drop=True)

        return self.df.to_numpy(), self.df.columns.to_numpy()


class modifyRows:
    """
    Usage:
    import cleanup

    file_path = 'project_train.csv'

    # (row, column, new value) defaults correspond to outliers
    rows_to_modify = [
        (85, 2, 0.734),
        (95, 4, -6.542)
    ]

    prj_train_new, headers = cleanup.modifyRows(file_path, rows_to_modify).modify_values()
    """
    def __init__(self, file_path, modifications=None):
        # Default
        if modifications is None:
            modifications = [
                (85, 2, 0.734),
                (95, 4, -6.542)
            ]

        self.file_path = file_path
        self.modifications = modifications
        self.df = pd.read_csv(self.file_path)

    def modify_values(self):
        for triplet in self.modifications:
            row_num, col_num, new_value = triplet

            if 0 <= row_num - 1 < len(self.df) and 0 <= col_num - 1 < len(self.df.columns):
                original_value = self.df.iat[row_num - 1, col_num - 1]
                print(f"\nRow {row_num}, Column {col_num} original value: {original_value}")

                # Modify the value in the DataFrame
                self.df.iat[row_num - 1, col_num - 1] = new_value
                print(f"Row {row_num}, Column {col_num} new value: {new_value}")
            else:
                print(f"\nInvalid index: Row {row_num}, Column {col_num} does not exist.")

        return self.df.to_numpy(), self.df.columns.to_numpy()
