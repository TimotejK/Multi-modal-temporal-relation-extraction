from torch.utils.data import Dataset


class DFDataset(Dataset):

    def __init__(self, df, row_converter):
        self.df = df
        self.row_converter = row_converter

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.row_converter(self.df.iloc[idx])