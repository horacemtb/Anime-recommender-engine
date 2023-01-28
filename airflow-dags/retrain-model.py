from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.ssh.hooks.ssh import SSHHook
from airflow.providers.ssh.operators.ssh import SSHOperator
import pendulum

args = {'owner': 'airflow'}

dag = DAG(dag_id = 'retrain_model', default_args = args, schedule_interval = '@weekly', start_date = pendulum.datetime(2023, 1, 22, tz = "UTC"), tags = ['SSH'])

sshHook = SSHHook(remote_host = '84.201.153.30', username = 'dmitry-ds', key_file = r'/home/dmitryairflow/sshn/id_rsa')

run_retrain_model = SSHOperator(ssh_hook = sshHook, task_id = 'task2', command = 'sudo su - dmitry-ds -c "bash /home/dmitry-ds/rec-sys/Anime-recommender-engine/retrain-model/retrain-model.sh" ', dag=dag)

run_retrain_model