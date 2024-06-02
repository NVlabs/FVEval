import argparse
import numpy as np
import pandas as pd
import glob


def parse_results(dir_name, tag):
    csvs = glob.glob(dir_name + f'/eval/*_{tag}.csv')
    summary_df = []
    for csv in csvs:
        df = pd.read_csv(csv)
        summary_row = df.iloc[-1]
        summary_df.append(summary_row)
    summary_df = pd.concat(summary_df, axis=1).T
    return summary_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_name', type=str, default='results_nl2sva_human/0')
    parser.add_argument('--tag', type=str, default='human')
    args = parser.parse_args()
    # print(parse_results(args.dir_name, args.tag))
    # save to csv
    df = parse_results(args.dir_name, args.tag)
    for col in df.select_dtypes(include=['float64']):
        df[col] = df[col].map(lambda x: '{:.3g}'.format(x))
        print(df[col])
    df.to_csv(f'summary_{args.dir_name.split("/")[-1]}_{args.tag}.csv', index=False)

    
