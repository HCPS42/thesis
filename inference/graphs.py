import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson_binom
from collections import Counter


def get_bootstrapped_df(df, summary, lengths, max_tokens, num_tries, n_bootstrap):
    lengths = lengths.copy()

    num_tasks = len(df)
    real_num_tries = len([col for col in summary.columns if col.startswith('try ')])

    tries = summary.apply(lambda row: np.array([row[f'try {i}'] for i in range(real_num_tries)]), axis=1).to_numpy() % 1000
    lengths = lengths.apply(lambda row: np.array([row[f'try {i}'] for i in range(real_num_tries)]), axis=1).to_numpy()
    answers = summary['answer'].to_numpy() % 1000

    accs = []
    for i in range(num_tasks):
        possible_indices = [j for j in range(real_num_tries)]
        tries_bootstrap = np.random.choice(possible_indices, size=(n_bootstrap, num_tries), replace=True)

        rolls = []

        for roll in tries_bootstrap:
            cur = []
            for j in roll:
                if not np.isnan(tries[i][j]) and lengths[i][j] <= max_tokens:
                    cur.append(tries[i][j])
            if not cur:
                cur = [0]
            rolls.append(cur)

        cons = np.array([Counter(roll).most_common(1)[0][0] for roll in rolls])
        answer = answers[i]
        accuracy = (cons == answer).mean()
        accs.append(accuracy)

    df['accuracy'] = accs

    return df


def process_run(name, max_tokens=24576, num_tries=24, n_bootstrap=1000):
    df = pd.read_csv(f'data/eval/eval.csv')
    summary = pd.read_csv(f'logs/{name}/summary.csv')
    lengths = pd.read_csv(f'logs/{name}/lengths.csv')
    times = pd.read_csv(f'logs/{name}/times.csv')

    df = df[df['id'].isin(summary['id'])]

    if n_bootstrap is None:
        real_num_tries = len([col for col in summary.columns if col.startswith('try ')])

        df['accuracy'] = summary.apply(lambda row: sum(row[f'try {i}'] % 1000 == row['answer'] % 1000 for i in range(real_num_tries))/real_num_tries, axis=1)
        df['solved'] = df.apply(lambda row: summary.loc[summary['id'] == row['id'], 'consensus'].values[0] == row['answer'], axis=1)
    else:
        df = get_bootstrapped_df(df, summary, lengths, max_tokens, num_tries, n_bootstrap)

    df['level'] = 'obvious'

    return df, summary, lengths, times


def compare_runs(names):
    fig, axs = plt.subplots(2, len(names), figsize=(3 * len(names), 6))
    fig.suptitle(f'Comparison of runs')

    for idx, name in enumerate(names):
        df, summary, lengths, times = process_run(name, n_bootstrap=None)

        ax = axs[0, idx]

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 20)

        num_bars = 12
        
        bins = [idx/num_bars for idx in range(num_bars + 2)]
        solved_counts, _ = np.histogram(df[df['solved']]['accuracy'], bins=bins)
        not_solved_counts, _ = np.histogram(df[~df['solved']]['accuracy'], bins=bins)
                
        ax.bar(bins[:-1], solved_counts, width=1/num_bars, align='edge', color='blue', label='Solved', edgecolor='black')
        ax.bar(bins[:-1], not_solved_counts, width=1/num_bars, align='edge', bottom=solved_counts, color='orange', label='Not Solved', edgecolor='black')
        
        ax.set_title(name)
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Number of problems')
        ax.set_xticks(bins)
        ax.set_xticklabels([f'{i}/{num_bars}' for i in range(num_bars + 2)], rotation=90)

        ax.axvline(x=4/12, color='red', linestyle='--')
        ax.axvline(x=9/12, color='red', linestyle='--')

        ax.grid(True)
        ax.legend(fontsize='small')

        ax = axs[1, idx]

        source_solved_stats = df.groupby(['source', 'solved']).size().reset_index(name='count')
        pivot_stats = source_solved_stats.pivot(index='source', columns='solved', values='count').fillna(0)
        if True in pivot_stats.columns:
            pivot_stats.rename(columns={True: 'Solved', False: 'Not Solved'}, inplace=True)
        
        sources = pivot_stats.index
        solved_counts = pivot_stats['Solved'] if 'Solved' in pivot_stats.columns else pd.Series(0, index=sources)
        not_solved_counts = pivot_stats['Not Solved'] if 'Not Solved' in pivot_stats.columns else pd.Series(0, index=sources)
        
        bar_width = 0.8
        x = np.arange(len(sources))
        
        ax.bar(x, solved_counts, bar_width, label='Solved', color='blue', edgecolor='black')
        ax.bar(x, not_solved_counts, bar_width, bottom=solved_counts, label='Not Solved', color='orange', edgecolor='black')
        
        total_counts = solved_counts + not_solved_counts
        for i, (solved, total) in enumerate(zip(solved_counts, total_counts)):
            if total > 0:
                ax.text(i, total + 1, f'{solved}/{total}', ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Number of problems')
        ax.set_xticks(x)
        ax.set_xticklabels(sources)
        ax.set_ylim(0, 35)
        ax.legend(fontsize='small')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def compare_configs(configs, n_bootstrap=1000, count_suffixes=False):
    fig, axs = plt.subplots(2, len(configs), figsize=(3 * len(configs), 6))
    fig.suptitle(f'Comparison of bootstrapped configs')

    for idx, conf in enumerate(configs):
        name = conf['run_name']
        max_tokens = conf['max_tokens']
        num_tries = conf['num_tries']

        df, summary, lengths, times = process_run(name, max_tokens, num_tries, n_bootstrap)

        ax = axs[0, idx]

        x = np.arange(0, len(df)+1)
        pb = poisson_binom(df['accuracy'].to_numpy())
        pmf = pb.pmf(x)
        ax.bar(x, pmf, color='black', edgecolor='black')

        q_left = np.searchsorted(np.cumsum(pmf), 1e-3)
        q_right = np.searchsorted(np.cumsum(pmf), 1 - 1e-3)
        ax.set_xlim(q_left, q_right)

        alpha = 0.05
        q_left = np.searchsorted(np.cumsum(pmf), alpha / 2)
        q_right = np.searchsorted(np.cumsum(pmf), 1 - alpha / 2)
        ax.axvline(x=q_left, color='red', linestyle='--')
        ax.axvline(x=q_right, color='red', linestyle='--')

        ax.set_title(f'{name}\nmax_tokens={max_tokens}\nnum_tries={num_tries}\nSolved on average: {pb.mean():.2f}')
        ax.set_xlabel('Number of solved problems')
        ax.set_ylabel('Probability')
        ax.grid(True)
        
        ax = axs[1, idx]
        num_problems = len(df)
        num_bars = 12

        ax.set_xlim(0, 1)
        ax.set_ylim(0, num_problems)
        
        bins = [idx/num_bars for idx in range(num_bars + 2)]
        counts, _ = np.histogram(df['accuracy'], bins=bins)
        if count_suffixes:
            counts = np.cumsum(counts[::-1])[::-1]
        
        ax.bar(bins[:-1], counts, width=1/num_bars, align='edge', color='black', label='Solved', edgecolor='black')
        
        ax.set_xlabel('Probability of solving a problem')
        ax.set_ylabel('Number of problems')
        ax.set_xticks(bins)
        ax.set_xticklabels([f'{i}/{num_bars}' for i in range(num_bars + 2)], rotation=90)
        ax.axvline(x=4/12, color='red', linestyle='--')
        ax.axvline(x=9/12, color='red', linestyle='--')
        
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
