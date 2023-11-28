import wandb

api = wandb.Api()

# run is specified by <entity>/<project>/<run id>
run = api.run("querry/MA-nasuta/hhk0e3xk")

# save the metrics for the run to a csv file
metrics_dataframe = run.history(samples=1_000, keys=['num_timesteps', 'makespan_mean'])
df = metrics_dataframe[['num_timesteps', 'makespan_mean']]
df.to_csv("bad_icm_metrics.csv")

if __name__ == '__main__':
    pass
